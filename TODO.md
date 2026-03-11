# MicroExec 编译器与运行时早期开发计划

按推荐优先级排列。每个阶段完成后应可编译运行、通过校验，早期开发计划当中优先关注功能实现，暂不考虑性能优化。

---

## 阶段 0：优化设计VM指令码与文件结构

**目标**：优化当前的VM指令码设计，参考Pytorch成熟的指令码设计。

---

## 阶段 1：Execution IR → VM Program 降低

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

## 阶段 2：Graph 优化 Pass

**目标**：在 Graph IR 层实现基本的优化，减少不必要的计算和内存开销。

**任务**：

- [ ] 死节点消除（Dead Node Elimination）：删除输出未被任何后续节点或图输出使用的节点
- [ ] 常量折叠（Constant Folding）：如果一个节点的所有输入都是常量，在编译期计算结果
- [ ] 算子融合（Operator Fusion）：识别常见模式（如 Conv+Relu、Conv+BN+Relu）并合并
- [ ] 提供统一的 Pass 接口：`int RunPass(Graph &graph)` 风格，便于组合和排序

**预计影响文件**：新建 `graph/passes/` 目录

---

## 阶段 3：Execution IR 增强

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

## 阶段 4：VM 运行时

**目标**：实现一个最小可用的 VM 解释器，能够加载编译产物并执行推理，为了方便部署于嵌入式环境，这部分使用C语言实现，以C语言代码库的形式分发。

**任务**：

- [ ] 实现 `Program::LoadFromFile()` 反序列化
- [ ] 实现 `ExecutionPlan` 执行引擎
- [ ] 实现基本算子 kernel（Conv, Relu, MaxPool, Gemm, Reshape）
- [ ] 内存池的运行时分配与管理
- [ ] 输入/输出数据的绑定接口
- [ ] 基本的错误处理与边界检查

**预计影响文件**：新建 `runtime/` 目录

---

## 阶段 5：测试与工具

**目标**：建立测试基础设施，确保每个阶段的正确性。

**任务**：

- [ ] 端到端测试：ONNX 模型 → 编译 → VM 执行 → 结果对比
- [ ] 集成差分测试：编写python脚本，利用onnx官方运行时，运行同一个onnx模型并对比输出
- [ ] 多模型覆盖：除当前 LeNet 外，增加 ResNet-18、MobileNetV2 等测试模型
- [ ] IR Dump 的可读性改进（可选：graphviz DOT 输出）

---

## 架构提醒

- 保持三层之间的依赖方向：`Graph IR` → `Execution IR` → `Target IR`
- 类型系统已按层级拆分：`graph_types.h`、`exec_types.h`、`vm_types.h`
- 编译器是短生命周期程序，Arena 风格分配仍然适用
