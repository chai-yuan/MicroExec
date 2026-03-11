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

**Edge**：数据流边，表示张量值。包含 id、name、shape、dtype、权重数据（常量）、producer 指针。

**Node**：计算节点。包含 id、name、op_type、输入/输出边列表、属性映射。

**Graph**：计算图容器，内置 Arena 内存池管理节点和边的生命周期。

### 当前实现

- ONNX 解析导入：`frontend/build_graph.cc`
- Graph 构建/校验：`graph/graph.h`、`graph/graph.cc`
- Shape/Type 推断：支持 Conv、MaxPool、Relu、Reshape、Gemm 等算子的形状推断

---

## 3. Execution IR

### 职责

- 节点拓扑线性化
- 值编号与生命周期分析
- 内存规划（常量池 + 运行时池）
- 动态张量的延迟分配支持

### 核心数据结构

**ExecValue**：执行值，对应 Graph IR 中的 Edge。包含：
- 值类型（INPUT / CONSTANT / INTERMEDIATE）
- 生命周期范围（first_def / last_use）
- 内存规划结果（buffer_id、offset、size_bytes）
- 动态张量延迟分配标记

**ExecInstr**：执行指令，对应 Graph IR 中的 Node。包含操作类型（KERNEL / DELEGATE / MOVE / NOP）、输入输出值 ID 列表。

**MemoryPlan**：内存规划结果，记录常量池/运行时池大小及需延迟分配的值。

### 当前实现

- 拓扑排序、值编号、指令生成：`execution/exec_program.cc` 的 `BuildFromGraph()`
- 生命周期分析：`AnalyzeLifetimes()`
- 内存规划：`PlanMemory()` 使用贪心首次适配算法复用内存
- 动态张量延迟分配：为 shape 含 -1 的张量生成分配符号

---

## 4. Target IR / VM Program

### 职责

- Opcode lowering
- 操作数编码
- 程序序列化/反序列化（待实现）

### 核心数据结构

**Opcode**：虚拟机指令操作码（KERNEL_CALL、DELEGATE_CALL、MOVE_CALL、JUMP_FALSE_CALL、FREE_CALL）

**Instruction**：16 字节指令结构（opcode + arg1 + arg2 + arg3）

**Program**：序列化程序容器，包含全局池（string/int/evalue/tensor/operator/instruction）

### 当前实现

- VM 类型定义：`common/vm_types.h`
- Program/ProgramBuilder 声明：`backend/vm_program.h`

---

## 5. 类型系统

类型按层级独立定义，避免跨层污染：

| 文件 | 层级 | 内容 |
|------|------|------|
| `common/graph_types.h` | Graph | DataType, ShapeDynamism, AttributeKind |
| `common/exec_types.h` | Execution | ExecValueKind, ExecOpKind |
| `common/vm_types.h` | VM | Opcode, Instruction, TensorMeta, EValue |

`common/type.h` 作为统一包含入口。

---

## 6. 已知限制

1. **动态 Shape**：含动态维度（-1）的中间张量需运行时延迟分配
2. **线性指令流**：不支持 ExecutionBlock（基本块）
3. **无 Delegate 支持**：ExecOpKind::DELEGATE 已定义但未使用
4. **单内存池**：所有中间值共享一个运行时池
