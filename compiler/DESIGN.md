# ONNX-C 编译器设计计划与 IR 评审

本文档用于确定当前编译器部分的总体架构、阶段职责、各层 IR 设计规范，以及基于当前实现的评审结果。

当前目标：

- 前端主要支持 `ONNX`
- 编译器为短生命周期程序
- 推理器优先面向一个自定义 `VM`
- 运行时追求低开销，尽量将复杂工作前移到编译期
- 编译器内存管理可以偏向"只申请、不释放"，适合使用 `Arena` 风格分配器

---

## 1. 总体结论

编译器采用 **三层结构**：

1. **Graph IR**：抽象计算图
2. **Execution IR**：抽象执行计划 / 线性化程序表示
3. **Target IR / VM Program**：目标平台程序表示，当前优先为虚拟机指令

这样划分的原因：

- `Graph IR` 适合表达模型语义、依赖关系、shape/type、图优化
- `Execution IR` 适合表达执行顺序、buffer 生命周期、内存规划、调试与 profile 信息
- `Target IR` 适合表达 opcode、操作数编码、序列化格式、runtime 装载接口

短期实现上，可以让第二层保持轻量；但从架构上，必须保留这三层职责边界。

---

## 2. 编译器分层设计

## 2.1 第一层：Graph IR

### 职责

`Graph IR` 负责表达"模型算什么"，主要承担：

- ONNX 解析导入
- 图中节点、值、边的统一表示
- 算子语义标准化
- 属性保存与规范化
- shape / dtype / rank 信息承载
- 常量 / 权重表示
- graph-level rewrite
- 常量折叠
- 死节点删除
- 可选的子图与控制流表示

### 不应承担的职责

`Graph IR` 不应直接承担：

- VM 指令编码
- 最终 opcode 选择
- 最终寄存器编号 / slot 编号
- 最终内存池偏移布局
- runtime ABI 细节

### 当前代码的一个额外提醒：第一层不要过早依赖第三层类型

从当前实现看，`graph/graph.h` 已经包含了 `common/type.h`，而 `common/type.h` 中同时放了：

- `Graph IR` 会用到的基础类型，例如 `DataType`
- `Target IR / VM Program` 会用到的类型，例如 `Opcode`、`Instruction`、`TensorMeta`、`EValue`

这说明当前第一层和第三层的类型边界有些混合。

这在原型阶段问题不大，但如果继续演进，会出现几个隐患：

- 第一层会被第三层的数据布局牵着走
- `Graph IR` 难以保持语义层面的纯净
- 后续如果更换 VM 表示，可能导致第一层被迫修改
- 第二层 `Execution IR` 的职责空间会被压缩

### 建议

尽快把公共类型拆成更明确的层次：

- `graph` 层公共类型：
  - `DataType`
  - shape / dimension 相关基础类型
  - attribute kind
- `execution` 层类型：
  - execution value kind
  - execution instruction kind
  - storage / slot / lifetime 相关类型
- `vm` 层类型：
  - `Opcode`
  - `Instruction`
  - `TensorMeta`
  - `EValue`

也就是说，`Graph IR` 最多依赖"真正跨层通用的基础类型"，不要直接依赖目标 VM 的程序格式定义。

### 当前实现状态

| 功能 | 状态 | 位置 |
|------|------|------|
| ONNX 解析导入 | **已实现** | `frontend/build_graph.cc` |
| Graph 构建 | **已实现** | `graph/graph.h`, `graph/graph.cc` |
| Graph 校验 | **已实现** | `graph/graph.cc` `Validate()` |
| 属性解析 | **已实现** | `frontend/build_graph.cc` `ParseAttribute()` |
| Shape/Type 推断 | 未实现 | — |
| 常量折叠 | 未实现 | — |
| 图优化 / Rewrite | 未实现 | — |

---

## 2.2 第二层：Execution IR

### 职责

`Execution IR` 负责表达"模型怎么执行"，主要承担：

- 节点拓扑线性化
- 执行顺序确定
- 值编号 / slot 编号
- 常量引用映射
- 中间张量生命周期分析
- buffer 复用与内存规划
- 外部 delegate 区域调用
- profile / debug 点插入
- source mapping：执行单元映射回 graph node

### 数据结构设计

#### ExecValueKind —— 值的种类

```
INPUT        图的外部输入，由调用者在运行时提供
CONSTANT     常量/权重，来自模型文件，编译期可确定
INTERMEDIATE 计算产生的中间值，需要运行时内存
```

#### ExecOpKind —— 指令的操作类型

```
KERNEL       标准算子调用（Conv、Relu、Gemm 等）
DELEGATE     硬件后端代理调用（NPU/DSP）
MOVE         数据搬移
NOP          空操作（debug/profile 占位）
```

#### ExecValue —— 执行值

每个 `ExecValue` 对应 Graph IR 中的一条 `Edge`，在 Execution IR 中被统一编号：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | `uint32_t` | 唯一值编号（slot ID） |
| `kind` | `ExecValueKind` | 值的种类 |
| `dtype` | `DataType` | 数据类型 |
| `shape` | `vector<int64_t>` | 形状（可能含动态维度 -1） |
| `first_def` | `uint32_t` | 定义该值的指令索引 |
| `last_use` | `uint32_t` | 最后使用该值的指令索引 |
| `buffer_id` | `uint32_t` | 所属内存池（0=常量池，1=运行时池） |
| `offset` | `uint64_t` | 在所属内存池中的字节偏移 |
| `size_bytes` | `uint64_t` | 占用字节数 |
| `mem_planned` | `bool` | 是否已完成内存规划 |
| `constant_data` | `const uint8_t*` | 常量数据指针（不拥有所有权） |
| `constant_size` | `uint64_t` | 常量数据大小 |
| `source_edge` | `const Edge*` | 溯源到 Graph IR 的原始 Edge |

#### ExecInstr —— 执行指令

每个 `ExecInstr` 对应 Graph IR 中的一个 `Node`，表示一步计算：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | `uint32_t` | 指令顺序编号（步骤索引） |
| `op_kind` | `ExecOpKind` | 操作类型 |
| `op_name` | `string` | 算子名称（如 "Conv"、"Gemm"） |
| `input_values` | `vector<uint32_t>` | 输入 ExecValue ID 列表 |
| `output_values` | `vector<uint32_t>` | 输出 ExecValue ID 列表 |
| `delegate_id` | `uint32_t` | 硬件代理 ID（仅 DELEGATE 类型有效） |
| `source_node` | `const Node*` | 溯源到 Graph IR 的原始 Node |

#### MemoryPlan —— 内存规划结果

| 字段 | 类型 | 说明 |
|------|------|------|
| `pools` | `vector<MemoryPoolPlan>` | 各内存池的规划信息 |
| `constant_pool_size` | `uint64_t` | 常量池总大小 |
| `runtime_pool_size` | `uint64_t` | 运行时池总大小 |

#### ExecProgram —— 执行计划

整体执行程序的顶层容器：

| 字段/方法 | 说明 |
|-----------|------|
| `name` | 执行计划名称（默认 "forward"） |
| `values_` | 所有 ExecValue 的有序列表（按 ID 索引） |
| `instructions_` | 按拓扑序排列的指令序列 |
| `input_value_ids_` | 程序输入对应的 ExecValue ID |
| `output_value_ids_` | 程序输出对应的 ExecValue ID |
| `memory_plan_` | 内存规划结果 |

### 降低流程（Graph IR → Execution IR）

`BuildFromGraph` 按以下阶段执行：

1. **值创建（输入）**：为每个 `graph_input` Edge 创建一个 `ExecValue`（kind=INPUT）
2. **值创建（常量）**：为每个 `is_constant` Edge 创建一个 `ExecValue`（kind=CONSTANT），引用原始权重数据
3. **拓扑排序**：使用 Kahn 算法（BFS）对图节点做拓扑线性化，检测环路
4. **指令生成**：按拓扑序遍历节点，为每个节点的输出 Edge 创建中间 `ExecValue`（kind=INTERMEDIATE），并生成对应的 `ExecInstr`
5. **输出记录**：标记哪些 ExecValue 是图的最终输出

### Pass 设计

#### AnalyzeLifetimes（生命周期分析）

遍历指令序列，为每个 `ExecValue` 标注：

- `first_def`：该值首次被定义的指令索引（输入/常量为 0）
- `last_use`：该值最后被使用的指令索引

特殊规则：
- 图输入的 `last_use` 延伸到程序末尾（外部管理）
- 图输出的 `last_use` 延伸到程序末尾（需要持久化到调用者读取）

#### PlanMemory（内存规划）

基于生命周期信息，为每个值分配物理内存位置：

- **常量池（buffer_id=0）**：所有 CONSTANT 值按顺序排列，16 字节对齐
- **运行时池（buffer_id=1）**：所有 INTERMEDIATE 值使用贪心首次适配（Greedy First-Fit）算法分配

首次适配算法：
1. 收集与当前值生命周期重叠的已分配区域
2. 按偏移排序
3. 在间隙中找到第一个足够大的位置
4. 分配并记录

输入值由外部管理，不参与内存池分配。

### 当前实现状态

| 功能 | 状态 | 位置 |
|------|------|------|
| 拓扑排序（Kahn's） | **已实现** | `execution/exec_program.cc` `BuildFromGraph()` |
| 值编号 | **已实现** | `execution/exec_program.cc` `BuildFromGraph()` |
| 指令生成 | **已实现** | `execution/exec_program.cc` `BuildFromGraph()` |
| 生命周期分析 | **已实现** | `execution/exec_program.cc` `AnalyzeLifetimes()` |
| 内存规划（First-Fit） | **已实现** | `execution/exec_program.cc` `PlanMemory()` |
| 校验 | **已实现** | `execution/exec_program.cc` `Validate()` |
| Dump 调试输出 | **已实现** | `execution/exec_program.cc` `Dump()` |
| Delegate 区域识别 | 未实现 | — |
| Debug/Profile 点插入 | 未实现 | — |
| ExecutionBlock 支持 | 未实现 | — |

### 当前已知限制

1. **中间值大小为零**：由于 Shape Inference 尚未实现，中间 Edge 的 shape 信息不完整，导致 `ComputeTensorBytes` 返回 0，运行时池大小为 0。这需要在 Graph IR 层实现 shape inference 后才能解决。
2. **图输入大小为零**：输入 Edge 含动态维度（-1），编译期无法确定大小。需要支持 Dynamic Bound 或静态化输入 shape。
3. **仅线性指令流**：当前不支持 `ExecutionBlock`（基本块），所有指令在单一线性序列中。
4. **无 Delegate 支持**：`ExecOpKind::DELEGATE` 类型已定义但未被使用。
5. **单内存池**：所有中间值共享一个运行时池，尚不支持多池策略。

---

## 2.3 第三层：Target IR / VM Program

### 职责

第三层负责将执行计划转成目标 VM 可直接消费的数据结构：

- opcode lowering
- operand 编码
- string/int/tensor/operator pool 组织
- instruction pool 组织
- 常量区与权重区布局
- 二进制序列化
- program / plan / metadata format

### 原则

这一层应尽量避免承载过多优化逻辑。  
主要工作应是：

- 降低抽象
- 扁平化
- 序列化
- 校验格式一致性

### 当前实现状态

| 功能 | 状态 | 位置 |
|------|------|------|
| VM 类型定义 | **已定义** | `common/vm_types.h` |
| Program / ProgramBuilder | **已声明** | `backend/vm_program.h` |
| Execution IR → VM 降低 | 未实现 | — |
| 二进制序列化 | 未实现 | — |
| Program 反序列化 | 未实现 | — |

---

## 3. 推荐的编译流程

建议编译流程如下：

1. **Import ONNX**
2. **Build Graph IR**
3. **Graph Normalize**
4. **Graph Validation**
5. **Shape / Type Inference**
6. **Constant Processing**
7. **Graph Rewrite / Optimization**
8. **Lower to Execution IR**
9. **Execution Planning**
10. **Memory Planning**
11. **Insert Debug / Profile Metadata**
12. **Lower to VM Program**
13. **Serialize Output**

---

## 4. 项目文件结构

```
compiler/
├── common/
│   ├── graph_types.h         # Graph 层基础类型：DataType, ShapeDynamism, AttributeKind
│   ├── exec_types.h          # Execution 层基础类型：ExecValueKind, ExecOpKind
│   ├── vm_types.h            # VM 层类型：Opcode, Instruction, TensorMeta, EValue, ...
│   ├── type.h                # 统一包含（待拆分）
│   └── log.h                 # 日志宏
│
├── frontend/
│   ├── build_graph.cc        # ONNX → Graph IR 解析
│   ├── onnx.proto            # ONNX protobuf 定义
│   ├── onnx.pb.h             # protobuf 生成
│   └── onnx.pb.cc            # protobuf 生成
│
├── graph/
│   ├── graph.h               # Graph IR：Graph, Node, Edge, Attribute
│   └── graph.cc              # Graph 方法实现
│
├── execution/
│   ├── exec_program.h        # Execution IR：ExecProgram, ExecValue, ExecInstr, MemoryPlan
│   └── exec_program.cc       # Execution IR 实现
│
├── backend/
│   └── vm_program.h          # VM Program：Program, ProgramBuilder, ExecutionPlanData
│
├── main.cc                   # 编译器入口
├── Makefile                  # 构建脚本
└── DESIGN.md                 # 本文档
```

---

## 5. 层间依赖关系

```
graph_types.h  ←── graph/graph.h  ←── frontend/build_graph.cc
                                   ←── execution/exec_program.cc

exec_types.h   ←── execution/exec_program.h

vm_types.h     ←── backend/vm_program.h
(依赖 graph_types.h)

main.cc ──→ graph/graph.h
        ──→ execution/exec_program.h
```

层级依赖方向：`Graph IR` → `Execution IR` → `Target IR`，不可反向依赖。

---
