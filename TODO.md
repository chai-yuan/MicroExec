# ONNX-C 编译器开发计划

按推荐优先级排列。每个阶段完成后应可编译运行、通过校验。

---

## 阶段 1：Shape Inference（推荐下一步）

**目标**：让所有 Edge 在 Graph IR 阶段就拥有完整的 shape 和 dtype，使 Execution IR 的内存规划真正生效。

**任务**：

- [ ] 实现 `Graph::InferShapes()` 方法
- [ ] 为每个 ONNX 算子定义 shape 推断规则（优先覆盖当前测试模型用到的算子）：
  - `Conv`：根据 input shape、kernel shape、pads、strides、dilations 计算输出 shape
  - `Relu`：输出 shape = 输入 shape
  - `MaxPool`：类似 Conv 的输出 shape 计算
  - `Reshape`：根据 shape 参数推断
  - `Gemm`：矩阵乘法的输出 shape 规则
- [ ] 处理动态维度（batch size = -1）的传播策略：
  - 方案 A：要求用户在编译时指定静态 input shape
  - 方案 B：传播 Dynamic Bound 标记，内存规划使用上限值
- [ ] 在 `main.cc` 中将 `InferShapes()` 插入 `BuildFromONNX()` 之后、`BuildFromGraph()` 之前
- [ ] 验证：Dump 输出中所有中间值的 `size` 字段应为非零值，运行时池大小应为合理值

**预计影响文件**：`graph/graph.h`, `graph/graph.cc`（或新建 `graph/shape_inference.cc`）, `main.cc`

---

## 阶段 2：Execution IR → VM Program 降低

**目标**：实现从 Execution IR 到 VM Program 的完整降低流程。

**任务**：

- [ ] 实现 `ProgramBuilder` 的所有方法（`AddString`, `AddTensorMeta`, `AddEValue`, `AddOperator`, `AppendInstruction`, `BuildExecutionPlan`）
- [ ] 实现降低函数 `LowerToVMProgram(const ExecProgram &exec, ProgramBuilder &builder)`：
  - 遍历 ExecProgram 的 values，为每个值创建对应的 `EValue` 和 `TensorMeta`
  - 遍历 ExecProgram 的 instructions，将每条 `ExecInstr` 降低为一条或多条 VM `Instruction`
  - `ExecOpKind::KERNEL` → `Opcode::KERNEL_CALL`
  - `ExecOpKind::DELEGATE` → `Opcode::DELEGATE_CALL`
  - 生成参数列表（args_list）放入 int_pool
  - 创建 `ExecutionPlanData`
- [ ] 实现 `ProgramBuilder::Serialize()` 将 Program 导出为二进制文件

**预计影响文件**：`backend/vm_program.h`, 新建 `backend/vm_program.cc`, 新建 `backend/lower_to_vm.cc`, `main.cc`

---

## 阶段 3：Graph 优化 Pass

**目标**：在 Graph IR 层实现基本的优化，减少不必要的计算和内存开销。

**任务**：

- [ ] 死节点消除（Dead Node Elimination）：删除输出未被任何后续节点或图输出使用的节点
- [ ] 常量折叠（Constant Folding）：如果一个节点的所有输入都是常量，在编译期计算结果
- [ ] 算子融合（Operator Fusion）：识别常见模式（如 Conv+Relu、Conv+BN+Relu）并合并
- [ ] 提供统一的 Pass 接口：`int RunPass(Graph &graph)` 风格，便于组合和排序

**预计影响文件**：新建 `graph/passes/` 目录

---

## 阶段 4：Execution IR 增强

**目标**：为 Execution IR 补充进阶能力。

**任务**：

- [ ] **内存规划优化**：将 First-Fit 升级为更高效的算法（如基于图着色的寄存器分配思路）
- [ ] **多内存池支持**：区分 SRAM / DRAM / 外部存储等不同内存层级
- [ ] **Delegate 区域识别**：扫描图中可被硬件后端加速的子图，标记为 DELEGATE 指令
- [ ] **Debug/Profile 元数据插入**：在关键指令前后插入 NOP 指令携带调试信息
- [ ] **ExecutionBlock 支持**：引入基本块概念，为未来控制流（if/loop）做准备
- [ ] **FREE 指令生成**：基于生命周期分析，在值最后使用后插入显式释放指令

**预计影响文件**：`execution/exec_program.h`, `execution/exec_program.cc`

---

## 阶段 5：VM 运行时

**目标**：实现一个最小可用的 VM 解释器，能够加载编译产物并执行推理。

**任务**：

- [ ] 实现 `Program::LoadFromFile()` 反序列化
- [ ] 实现 `ExecutionPlan` 执行引擎
- [ ] 实现基本算子 kernel（Conv, Relu, MaxPool, Gemm, Reshape）
- [ ] 内存池的运行时分配与管理
- [ ] 输入/输出数据的绑定接口
- [ ] 基本的错误处理与边界检查

**预计影响文件**：新建 `runtime/` 目录

---

## 阶段 6：测试与工具

**目标**：建立测试基础设施，确保每个阶段的正确性。

**任务**：

- [ ] 单元测试框架集成（如 Google Test 或简单的自制测试宏）
- [ ] Graph IR 构建和校验的测试用例
- [ ] Execution IR 各 Pass 的测试用例（拓扑排序、生命周期、内存规划）
- [ ] 端到端测试：ONNX 模型 → 编译 → VM 执行 → 结果对比
- [ ] 多模型覆盖：除当前 LeNet 外，增加 ResNet-18、MobileNetV2 等测试模型
- [ ] IR Dump 的可读性改进（可选：graphviz DOT 输出）

---

## 架构提醒

- 保持三层之间的依赖方向：`Graph IR` → `Execution IR` → `Target IR`
- `common/type.h` 仍同时包含 graph 和 vm 类型，待阶段 2 完成后考虑拆分
- 编译器是短生命周期程序，Arena 风格分配仍然适用，但 Execution IR 层当前使用 `std::vector`，这在性能上可接受
