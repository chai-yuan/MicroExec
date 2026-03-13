# MicroExec 编译器与运行时早期开发计划

按推荐优先级排列。每个阶段完成后应可编译运行、通过校验，早期开发计划当中优先关注功能实现，暂不考虑性能优化。

---

## 阶段 0：优化设计VM指令码与文件结构

**目标**：优化当前的VM指令码设计，参考Pytorch成熟的指令码设计。VM指令码主要运行于带有低功耗NPU的嵌入式环境，除储存的权重等数据以外，应追求整齐的4字节对齐设计

- [x] 指令头改为紧凑 4 字节（`opcode/flags/input_count/output_count`），并保持整条指令 16 字节
- [x] 增补 `NOP_CALL` 与指令 flags 语义，为 debug/profile 与后续扩展预留编码位
- [x] 建立 VM 二进制文件布局（`VMFileHeader + VMSectionDesc + SectionData`）
- [x] 统一约束 VM 元数据结构 4 字节对齐，并用 `static_assert` 做编译期校验
- [x] 将 VM 线格式定义收敛到 `common/vm_types.h`，补齐 `ExecutionPlanData`，并改造成可被 C / C++ 共同包含的共享头
- [x] 同步 `DESIGN.md` 的 Target IR 设计说明，明确阶段0编码和文件结构规范

---

## 阶段 1：Execution IR → VM Program 降低

**目标**：实现从 Execution IR 到 VM Program 的完整降低流程。

**任务**：

- [x] 实现 `ProgramBuilder` 的所有方法（`AddString`, `AddTensorMeta`, `AddEValue`, `AddOperator`, `AppendInstruction`, `BuildExecutionPlan`）
- [x] 实现降低函数 `LowerToVMProgram(const ExecProgram &exec, ProgramBuilder &builder)`：
  - 遍历 ExecProgram 的 values，为每个值创建对应的 `EValue` 和 `TensorMeta`
  - 遍历 ExecProgram 的 instructions，将每条 `ExecInstr` 降低为一条或多条 VM `Instruction`
  - `ExecOpKind::KERNEL` → `Opcode::KERNEL_CALL`
  - `ExecOpKind::DELEGATE` → `Opcode::DELEGATE_CALL`
  - 生成参数列表（args_list）放入 int_pool
  - 创建 `ExecutionPlanData`
- [x] 实现 `ProgramBuilder::Serialize()` 将 Program 导出为二进制文件
- [x] 移除编译器后端中仅供 VM 推理期使用的回读/查询逻辑，保留生成与序列化职责

**预计影响文件**：`backend/vm_program.h`, `backend/vm_program.cc`, `main.cc`

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

## 架构提醒

- 保持三层之间的依赖方向：`Graph IR` → `Execution IR` → `Target IR`
- 类型系统已按层级拆分：`graph_types.h`、`exec_types.h`、`vm_types.h`
- 编译器是短生命周期程序，Arena 风格分配仍然适用
