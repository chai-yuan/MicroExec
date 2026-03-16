# MicroExec 编译器设计文档

本文档描述编译器的三层 IR 架构、各层职责与当前实现状态。

## 1. 架构概述

编译器采用 **三层 IR 结构**：

| 层级 | 名称 | 职责 |
|------|------|------|
| 第一层 | Graph IR | 表达模型语义、依赖关系、图优化 |
| 第二层 | Execution IR | 表达执行顺序、内存规划、生命周期管理 |
| 第三层 | Target IR / VM Program | 目标平台指令编码、序列化 |

**设计原则**：
- 编译器为短生命周期程序，内存管理采用 Arena 风格，不需要内存释放
- 运行时追求低开销，复杂工作前移到编译期
- 层级依赖方向：Graph IR → Execution IR → Target IR，不可反向依赖

---

## 2. Graph IR

### 职责

- ONNX 解析导入
- 节点、边、属性的统一表示
- Shape/Type 推断
- 图优化与常量折叠（待实现）

### 核心数据结构

**Edge**：数据流边，表示张量值。包含 id、name、shape、dtype、权重数据（常量）、producer 指针、consumers 列表。

**Node**：计算节点。包含 id、name、op_type、输入/输出边列表、属性映射。

**Graph**：计算图容器，内置 Arena 内存池管理节点和边的生命周期。

### 当前实现

- ONNX 解析导入：`frontend/build_graph.cc`
- ONNX 算子形状推断：`frontend/shape_infer.h`、`frontend/shape_infer.cc`，以注册表模式实现，支持 Conv、MaxPool、Relu、Reshape、Gemm 等算子，可通过 `RegisterShapeInferRule()` 扩展。可选启用 ONNX C++ 库内置推断（`make USE_ONNX=1`）
- Graph 构建/校验：`graph/graph.h`、`graph/graph.cc`，`InferShapes()` 负责拓扑排序并通过回调委托逐节点推断

---

## 3. Execution IR

### 职责

- 节点拓扑线性化
- 值编号与生命周期分析
- 内存规划（常量池 + 运行时池）
- 有界动态张量的编译期内存预分配（按最大值分配）
- 张量内存空间复用（生命周期不重叠的张量共享同一内存区域）

### 核心数据结构

**ExecValue**：执行值，对应 Graph IR 中的 Edge。包含：
- 值类型（INPUT / CONSTANT / INTERMEDIATE）
- 生命周期范围（first_def / last_use）
- 内存规划结果（buffer_id、offset、size_bytes）
- 有界动态维度标记（has_dynamic_bound_dims）

**ExecInstr**：执行指令，对应 Graph IR 中的 Node。包含操作类型（KERNEL / DELEGATE / MOVE / NOP）、输入输出值 ID 列表。

**MemoryPlan**：内存规划结果，记录常量池/运行时池大小及总内存需求。

### 当前实现

- 拓扑排序、值编号、指令生成：`execution/exec_program.cc` 的 `BuildFromGraph()`
- 生命周期分析：`AnalyzeLifetimes()`
- 内存规划：`PlanMemory()` 使用贪心首次适配算法复用内存，有界动态张量按最大值参与规划
- 编译完成后输出模型最大内存需求

---

## 4. Target IR / VM Program

### 职责

- Opcode lowering
- 操作数编码
- 程序序列化
- 输出供独立 VM 运行时消费的线格式数据

#### 4.1 指令码设计

- 指令仍保持 **16 字节固定长度**，便于在嵌入式环境顺序取指
- 指令头采用 4 字节紧凑布局：
  - `opcode`（1B）
  - `flags`（1B）
  - `input_count`（1B）
  - `output_count`（1B）
- 指令体保留 `arg1/arg2/arg3` 三个 `uint32_t` 操作数字段，兼容后续 lowering
- opcode 集合：`KERNEL_CALL`、`DELEGATE_CALL`、`MOVE_CALL`、`JUMP_FALSE_CALL`、`FREE_CALL`、`NOP_CALL`
- 约束：除权重外，所有 VM 元数据结构与段布局按 **4 字节对齐**
- 共享头要求：`common/vm_types.h` 必须仅包含 C / C++ 都可读取的 POD 线格式定义，不依赖编译器内部 C++ 类型

#### 4.2 二进制文件结构

采用“文件头 + 段表 + 数据段”的平坦布局：

1. `VMFileHeader`（64B）
2. `VMSectionDesc[]`（每项 16B）
3. 各数据段（字符串池、整型池、张量池、EValue 池、算子池、Delegate 池、指令池、执行计划池、权重段）

每个段通过 `offset + size_bytes + count` 描述，编译器后端只负责生成这些段；加载、解释执行和推理逻辑由独立 runtime 处理。

### 核心数据结构

**Opcode**：虚拟机指令操作码（KERNEL_CALL、DELEGATE_CALL、MOVE_CALL、JUMP_FALSE_CALL、FREE_CALL、NOP_CALL）

**Instruction**：16 字节指令结构（opcode + flags + in/out 计数 + arg1 + arg2 + arg3）

**ExecutionPlanData**：单个入口计划的线格式描述，记录输入/输出列表、指令区间和运行时内存池大小

**ProgramBuilder**：编译期构建器，内部维护 string/int/tensor/evalue/operator/delegate/instruction/plan 各类池，并负责统一序列化输出

### 当前实现

- VM 类型定义 + 文件格式定义：`common/vm_types.h`
- 后端仅保留 `ProgramBuilder` 与 `LowerToVMProgram()`，负责 lowering 与序列化：`backend/vm_program.h`、`backend/vm_program.cc`
- 编译器主流程不再回读 VM 文件，避免将推理期职责混入后端

---

## 5. 类型系统

类型按层级独立定义，避免跨层污染：

| 文件 | 层级 | 内容 |
|------|------|------|
| `common/graph_types.h` | Graph | DataType, ShapeDynamism, AttributeKind |
| `common/exec_types.h` | Execution | ExecValueKind, ExecOpKind |
| `common/vm_types.h` | VM | C / C++ 共享线格式：Opcode, Instruction, TensorMeta, EValue, ExecutionPlanData, VMFileHeader, VMSectionDesc |

`common/type.h` 作为统一包含入口。

---

## 6. 已知限制

1. **动态 Shape**：动态维度使用 `-x` 表示最大值为 `x` 的有界动态维度（默认 `-4`），编译期按最大值预分配内存，避免运行时动态分配。可通过 `--max-dynamic-size N` 编译选项调整默认上界
2. **线性指令流**：不支持 ExecutionBlock（基本块）
3. **Delegate 仅占位支持**：已完成编码与指令降级映射，但尚未接入真实后端 blob
4. **单内存池**：所有中间值共享一个运行时池
