# ONNX-C 编译器设计计划与 Graph IR 评审

本文档用于确定当前编译器部分的总体架构、阶段职责、Graph IR 设计规范，以及基于当前实现的第一层抽象图设计评审结果。

当前目标：

- 前端主要支持 `ONNX`
- 编译器为短生命周期程序
- 推理器优先面向一个自定义 `VM`
- 运行时追求低开销，尽量将复杂工作前移到编译期
- 编译器内存管理可以偏向“只申请、不释放”，适合使用 `Arena` 风格分配器

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

`Graph IR` 负责表达“模型算什么”，主要承担：

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

也就是说，`Graph IR` 最多依赖“真正跨层通用的基础类型”，不要直接依赖目标 VM 的程序格式定义。

---

## 2.2 第二层：Execution IR

### 职责

`Execution IR` 负责表达“模型怎么执行”，主要承担：

- 节点拓扑线性化
- 执行顺序确定
- 值编号 / slot 编号
- 常量引用映射
- 中间张量生命周期分析
- buffer 复用与内存规划
- 外部 delegate 区域调用
- profile / debug 点插入
- source mapping：执行单元映射回 graph node

### 表示建议

建议优先采用：

- **以顺序执行为主**
- **保留未来扩展到 basic block 的能力**

也就是说，短期可以是“线性指令流”，但类型命名建议使用：

- `ExecutionProgram`
- `ExecutionValue`
- `ExecutionInstr`
- `ExecutionBlock`（可选）

而不是过早把整个中间层强行命名为“纯 linear IR”。

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

## 4. 现有第一层 Graph 设计评审

结合当前代码，现有第一层已经具备了基础骨架：

- `Graph`
- `Node`
- `Edge`
- `Attribute`
- `Arena`
- `BuildFromONNX()`

这说明当前工程方向是正确的，已经具备继续扩展的基础。

但如果你准备把这一层作为正式的第一抽象层，我认为还需要做几项重要改进。

---

## 5. 当前实现的优点

### 5.1 结构简单，容易推进

当前图结构很简洁：

- `Node` 表示算子节点
- `Edge` 表示张量流动
- `Graph` 统一管理
- `producer / consumers` 关系已建立

这个设计很适合快速搭建原型。

### 5.2 已经具备 Arena 思路

当前通过内部 `Arena` 集中分配：

- `Node`
- `Edge`

这很符合“编译器是短生命程序”的目标。  
对于你的项目，这是一个非常合理的方向。

### 5.3 已具备 ONNX 导入关键路径

已经能完成：

- initializer 导入
- graph input/output 导入
- node 导入
- attribute 解析
- value_info 形状信息补充

这说明前端第一版已经跑通了最核心路径。

---

## 6. 当前 Graph IR 的核心问题与改进建议

---

## 6.0 ONNX 导入实现层面的额外注意事项

除了数据结构本身，我还建议尽快关注当前 `BuildFromONNX()` 的几个实现问题，因为这些问题会直接影响第一层 IR 的可靠性。

### 6.0.1 `google::protobuf::ShutdownProtobufLibrary()` 不应放在单次导入函数末尾

当前实现会在 `BuildFromONNX()` 结束时调用全局 protobuf shutdown。  
这通常不适合作为普通库函数的一部分，因为：

- 它是进程级别资源清理
- 如果未来同一进程内还要再次解析 protobuf / ONNX，行为会变得脆弱
- 编译器后续如果支持多次加载模型、单测或多模块编译，这里会埋下问题

### 建议

将 protobuf 的全局 shutdown 从导入函数中移除。  
如果确实需要，应该由进程退出阶段统一处理，而不是放在单个图导入接口里。

### 6.0.2 `value_info` 的 shape 补充条件过于保守

当前逻辑只有在 `edge->shape.empty()` 时才补充 shape/type。  
这会导致下面这种情况无法被后续信息修正：

- shape 已有，但 dtype 仍未知
- shape 部分存在，但不完整
- input/output/value_info 中后出现的信息更完整

### 建议

把“是否更新元信息”的判断拆开：

- dtype 未知时可补 dtype
- shape 未知时可补 shape
- 如需更稳，可以定义“信息质量优先级”并进行合并

不要只依赖 `shape.empty()` 作为唯一条件。

### 6.0.3 initializer 非 `raw_data` 格式目前直接报错，兼容性不足

当前如果权重不是 `raw_data()` 存储，而是使用：

- `float_data`
- `int32_data`
- `int64_data`
- 其他 typed field

导入就会失败。

这对于很多 ONNX 模型来说是一个真实兼容性问题。

### 建议

尽快补齐常见 typed field 的导入逻辑。  
至少建议优先支持：

- `FLOAT`
- `INT32`
- `INT64`
- `UINT8`
- `INT8`

即使内部最后仍统一转成原始 byte buffer，也应在前端导入阶段完成转换。

### 6.0.4 当前缺少“单个值被多个 producer 定义”的导入期保护

现在输出边通过名称查找并复用对象，如果同名值被多个节点输出重复定义，当前实现会直接覆盖：

- `output_edge->producer = node`

这会导致 IR 进入不一致状态，而且问题可能直到后续阶段才暴露。

### 建议

在导入时立即检查：

- 若某个非空输出名已经有 producer，则直接报错
- 这样可以保证第一层 IR 的 SSA-like 约束更明确

### 6.0.5 空输入目前被静默跳过，但需要在设计上明确语义

ONNX 某些算子允许 optional input 为空字符串。  
当前 `get_or_create_edge()` 对空名字返回 `nullptr`，这是合理的，但要注意：

- `input_edges` 中不会保留占位
- 输入位置索引信息会丢失
- 对于依赖“第几个输入为空”的算子，后续 pass 会变得模糊

### 建议

短期可以继续保持当前行为，但文档中应明确：

- 现阶段空输入被视为“缺失输入并且不保留显式占位”

如果你后续要做更严谨的 schema 校验，建议保留输入槽位语义，而不是简单省略。

### 6.0.6 节点名可能为空，不能依赖 `name` 作为稳定标识

ONNX 中 `node.name()` 并不保证一定存在。  
因此当前设计中若未来使用 `name` 作为调试或唯一索引依据，会不够稳。

### 建议

继续保留 `name` 作为可选调试字段，但真正的内部稳定标识应尽快改成：

- `node id`
- `value id`

这一点和后续 `Execution IR` source mapping 也直接相关。

---

## 6.1 最大问题：当前 `Edge` 同时承担了“值”和“边”两种语义

当前 `Edge` 实际上表示的是：

- 一个具名 tensor value
- 带 shape / dtype / constant 数据
- 有 producer
- 有 consumers

从编译器语义上看，它更像是 **Value**，而不是传统图论意义上的 Edge。

### 当前问题

名称叫 `Edge`，但实际职责是：

- 张量值定义
- 张量元信息载体
- 数据依赖连接点

这会在后续设计中带来混淆：

- Graph pass 容易混淆“边”和“值”
- Execution IR 降低时语义映射不自然
- 未来支持 tuple/list/optional/control-flow 时概念扩展不顺

### 建议

将 `Edge` 在概念上重命名为：

- `Value`
- 或 `TensorValue`

如果短期不想大改代码，至少在设计文档中明确：

> 当前 `Edge` 的真实语义是 SSA 风格的 value，而不是单纯的图边。

### 推荐方向

- `Node`：算子
- `Value`：张量/标量/字符串/列表等值
- `Graph`：容器

这是更稳的长期方案。

---

## 6.2 `Attribute` 的类型系统过弱

当前：

- `Attribute.type` 使用 `DataType`
- `Attribute.value` 使用 `std::variant<int64_t, float, std::string, std::vector<int64_t>, std::vector<float>, Graph *>`

这个方案短期可用，但语义上不够准确。

### 问题

`DataType` 是张量元素类型，不适合直接表示 attribute 类型。  
例如：

- attribute 是 `INT`
- attribute 是 `FLOAT`
- attribute 是 `STRING`
- attribute 是 `GRAPH`
- attribute 是 `INTS`
- attribute 是 `FLOATS`

这些不是 tensor dtype，而是 **attribute kind**。

### 建议

新增独立枚举，例如：

- `AttributeKind`

可取值如：

- `UNDEFINED`
- `INT`
- `FLOAT`
- `STRING`
- `INTS`
- `FLOATS`
- `STRINGS`
- `TENSOR`
- `GRAPH`
- `GRAPHS`

这样可以避免把 attribute 和 tensor dtype 混在一起。

### 建议结构

`Attribute` 应更像：

- `name`
- `kind`
- `value`

而不是：

- `type`
- `value`

---

## 6.3 Graph 对“值种类”的区分不够明确

当前 `Edge` 只有：

- `is_constant`
- `producer`
- `consumers`

但从图语义上，值通常至少应区分：

- graph input
- graph output
- initializer / constant
- intermediate activation
- optional empty input
- node output alias / view

### 当前问题

例如一个值是否是：

- 用户输入
- 常量权重
- 中间结果
- 最终输出

目前主要依赖外部容器和关系推断，缺少内嵌标记，后续 pass 使用起来会不够直接。

### 建议

为 `Value` 增加分类，例如：

- `ValueKind::INPUT`
- `ValueKind::CONSTANT`
- `ValueKind::INTERMEDIATE`
- `ValueKind::OUTPUT`

注意：一个值可能同时是 `INPUT` 和 `OUTPUT` 吗？  
从语义上最好不要用单值枚举强行表达，建议改成 flag，或者至少保留多个 bool：

- `is_graph_input`
- `is_graph_output`
- `is_initializer`

对编译期判断会更方便。

---

## 6.4 当前只支持 Tensor 风格值，扩展性不足

ONNX 虽然主流是 tensor，但也会出现：

- tensor
- sequence
- map
- optional
- sparse tensor
- subgraph
- 标量属性与常量

如果你的长期目标是稳健前端，建议图 IR 从一开始至少保留“值类型”扩展点。

### 建议

为值增加高层 kind：

- `ValueTypeKind::TENSOR`
- `ValueTypeKind::SCALAR`
- `ValueTypeKind::SEQUENCE`
- `ValueTypeKind::OPTIONAL`
- `ValueTypeKind::OPAQUE`

短期可以只真正实现 `TENSOR`，但接口上保留扩展位。

---

## 6.5 形状表达目前不够完整

当前 shape 表示为：

- `std::vector<int64_t> shape`
- 动态维度用 `-1`

这个方案可用，但表达能力偏弱。

### 问题

动态维度在 ONNX 里可能有几种情况：

- 固定值
- `dim_param`
- 完全未知
- 上界动态
- 下界/区间信息（未来可能扩展）

而单纯使用 `-1` 会丢失：

- 原始符号名
- 维度是否未知还是符号
- 是否来自 shape inference
- 是否已经部分约束

### 建议

后续引入专门的维度结构，例如：

- `Dimension`
  - `is_static`
  - `value`
  - `symbol`

短期如果不想扩展太快，也建议至少保留：

- `shape`
- `shape_dynamism`
- `has_complete_shape`

---

## 6.6 缺少 rank / shape / type 是否已知的显式状态

当前如果：

- `shape.empty()`，可能表示标量，也可能表示未知
- `dtype == UNKNOWN`，表示未知
- 没有显式记录是否做过 shape/type inference

这会让后续 pass 出现歧义。

### 建议

给值增加状态字段，例如：

- `has_dtype`
- `has_shape`
- `rank`
- `shape_is_complete`

这样可以避免用“空容器”来承担多重语义。

---

## 6.7 `weight_data` 直接存在图值中，短期合理，长期需要抽象常量对象

当前常量权重直接存放在：

- `std::vector<uint8_t> weight_data`

这个做法对于第一版编译器是完全合理的，尤其编译器是短生命程序。

但长期会出现几个问题：

- 常量 tensor 元数据与常量实际 bytes 混在一个对象里
- 后续常量池管理不够清晰
- 多图共享常量、外部大权重文件映射不方便
- 不利于后端做只读常量段布局

### 建议

短期保留当前方式即可。  
中期可以抽象出：

- `ConstantData`
- 或 `TensorStorage`

让 `Value` 只引用它。

### 结论

这个点不是当前最优先修改项，但在第二层和第三层打通前，需要尽早考虑。

---

## 6.8 缺少节点输出位置语义

当前 `Node` 只有：

- `input_edges`
- `output_edges`

这在 ONNX 中一般够用，因为 ONNX 输出是有序列表。  
但后续如果你做：

- kernel schema 校验
- operator normalization
- named input/output
- optional input handling

建议增加端口语义。

### 建议

至少在逻辑上支持：

- input index
- output index

比如约定：

- `input_edges[i]` 对应第 `i` 个输入
- `output_edges[i]` 对应第 `i` 个输出

并在文档中写明。  
未来可以进一步扩展为显式 `Use` / `Def` 结构。

---

## 6.9 缺少图级唯一命名 / ID 体系

当前图中节点和值目前主要依赖：

- 指针身份
- 字符串名字

这对调试方便，但不够适合编译 pass。

### 问题

后续你会很快需要：

- 稳定的 node id
- 稳定的 value id
- debug 输出
- pass 前后映射
- source mapping 到 execution IR

### 建议

增加：

- `uint32_t id` 到 `Node`
- `uint32_t id` 到 `Value`

名字仍然保留，但名字不应作为内部唯一标识的唯一手段。

---

## 6.10 缺少 `opset` / domain / overload 等算子规范信息

当前节点只有：

- `name`
- `op_type`

但 ONNX 节点还涉及：

- domain
- opset version（通常图级持有）
- 未来内部标准化算子名
- 可能的 overload / kernel key

### 建议

至少为图或模块保留：

- model opset imports
- node domain

这样未来做标准化和兼容时会更稳。

---

## 6.11 目前 Graph 缺少校验接口

`BuildFromONNX()` 当前完成了解析和搭图，但缺少显式验证阶段。

### 建议新增

- `Validate()`
- `InferTypesAndShapes()`
- `DumpSummary()`

最起码应检查：

- 非空输出是否存在生产者或是图输入/常量
- 节点输出不应被多个 producer 重复定义
- graph inputs / outputs 合法
- 常量 shape / dtype 合法
- 节点输入输出数量基础合法性

---

## 6.12 ONNX 子图属性当前只是占位

当前 `GRAPH` attribute 只记录为 `nullptr`，并打印日志。  
这对于第一版是合理的，但设计上需要明确：

- 当前版本不支持控制流图
- 后续必须允许 `Graph` 嵌套 `Graph`

### 建议

文档中明确现阶段限制：

- 暂不支持 `If`
- 暂不支持 `Loop`
- 暂不支持携带子图 attribute 的节点深度导入

并在类型上为后续保留扩展。

---

## 6.13 `std::map<std::string, Attribute>` 可用，但不一定是最佳结构

当前节点属性存储为：

- `std::map<std::string, Attribute>`

优点：

- 稳定
- 便于按名查找
- 简单

缺点：

- 分配较多
- 查找和遍历成本不必要偏高
- 编译器短生命周期场景下，不一定最优

### 建议

短期保留没问题。  
如果后续你开始关注编译速度和内存碎片，可以考虑：

- `std::vector<NamedAttribute>`
- 或 arena-backed 小对象容器

但这不是第一优先级。

---

## 7. 基于当前项目目标的 Graph IR 推荐规范

下面给出一个适合你当前阶段的推荐方向。

---

## 7.1 建议的核心对象

### Graph

图容器，负责持有：

- 节点列表
- 值列表
- graph inputs
- graph outputs
- arena / allocator
- opset / model metadata

### Node

表示一个算子调用，包含：

- `id`
- `name`
- `op_type`
- `domain`
- `inputs`
- `outputs`
- `attributes`

### Value

表示一个图中的值，包含：

- `id`
- `name`
- `kind flags`
- `dtype`
- `shape`
- `producer`
- `uses`
- `constant payload`（可选）

---

## 7.2 推荐的最小字段集合

### Value 建议字段

- `id`
- `name`
- `dtype`
- `shape`
- `is_graph_input`
- `is_graph_output`
- `is_constant`
- `producer`
- `consumers`
- `data`（仅常量）
- `has_dtype`
- `has_shape`

### Node 建议字段

- `id`
- `name`
- `op_type`
- `domain`
- `inputs`
- `outputs`
- `attributes`

### Graph 建议字段

- `nodes`
- `values`
- `graph_inputs`
- `graph_outputs`
- `initializer_values`
- `arena`
- `opset_imports`

---

## 8. 内存管理建议：Arena 是正确方向

你特别说明了：

> 编译器项目是一个短生命程序，大多数内存只考虑申请，不考虑释放，可以使用类似 Arena alloc 的管理器

我完全认同。

---

## 8.1 为什么适合 Arena

编译器前端和中端数据通常具有以下特点：

- 生命周期接近整个编译过程
- 节点和值对象数量多
- 单个对象尺寸小且分配频繁
- 很少做单独销毁
- pass 之间共享大量对象指针

这非常适合 Arena。

---

## 8.2 对当前 Arena 的建议

当前 `Arena` 用：

- `std::vector<std::unique_ptr<Node>>`
- `std::vector<std::unique_ptr<Edge>>`

它实现简单，已经具备“集中持有、统一释放”的效果。

但从严格意义上，它更像：

- “对象集中托管”

而不是高效的块式 arena。

### 短期建议

先保留当前做法，完全没问题。  
因为它已经满足：

- 生命周期统一管理
- 外部拿裸指针使用
- 不需要单对象释放

### 中期建议

后续如果节点和值数量变大，可以升级为：

- chunk-based arena
- bump allocator

例如按块申请：

- `Node` 块
- `Value` 块
- 小字符串 / 小属性块

这样可以减少：

- 大量 `new`
- `unique_ptr` 元数据成本
- 分配器碎片

---

## 8.3 Arena 设计原则

Graph IR 中适合 arena 化的对象：

- `Node`
- `Value`
- `Attribute`
- shape/dim 小对象
- source location 元数据
- pass 临时分析对象（可用临时 arena）

不建议一开始就 arena 化的对象：

- 大块权重数据 bytes
- 可能需要单独搬移/序列化的大 buffer

对于权重数据，更适合：

- 直接保留在 `std::vector<uint8_t>`
- 或单独的只读数据池

---

## 9. 针对当前代码的优先改进顺序

我建议按下面顺序推进。

### P0：立即建议改

1. 明确 `Edge` 实际语义是 `Value`
2. 给 `Node` / `Edge(Value)` 增加稳定 `id` ✅ 已完成
3. 给值增加：
   - `is_graph_input` ✅ 已完成
   - `is_graph_output` ✅ 已完成
   - `has_shape` ✅ 已完成
   - `has_dtype` ✅ 已完成
4. 将 `Attribute.type` 改为独立的 `AttributeKind` ✅ 已完成
5. 为图增加基本 `Validate()` 接口 ✅ 已完成
6. 将 protobuf 全局 shutdown 从 `BuildFromONNX()` 中移除 ✅ 已完成
7. 在导入阶段检查重复 producer 定义 ✅ 已完成
8. 补齐非 `raw_data` initializer 的常见解析路径 ✅ 已完成（当前已支持 `FLOAT` / `INT32` / `INT64`）
9. 将第一层公共类型与第三层 VM 类型解耦 ✅ 已完成（已拆分 `graph_types.h` 与 `vm_types.h`）

### 本轮已落地的第一层优化

本轮已经对第一层 `Graph IR` 完成如下改造：

- 为 `Node` / `Edge` 增加了稳定的 `id`
- 为 `Edge` 增加了：
  - `has_shape`
  - `has_dtype`
  - `is_graph_input`
  - `is_graph_output`
- 为 `Node` 增加了 `domain`
- 将 attribute 类型从复用 `DataType` 改为独立的 `AttributeKind`
- 新增了 `Graph::Validate()` 和 `Graph::DumpSummary()`
- 在 `BuildFromONNX()` 中：
  - 移除了 protobuf 全局 shutdown
  - 增加了重复 producer 检查
  - 增强了 shape / dtype 合并逻辑
  - 支持了常见 typed initializer 数据导入
  - 在导入完成后执行图校验与摘要输出
- 将图层和 VM 层公共类型拆分为：
  - `common/graph_types.h`
  - `common/vm_types.h`

### 当前仍然保留、但尚未彻底完成的点

以下事项在设计上已经明确，但当前实现仍属于“过渡状态”：

1. `Edge` 还未正式重命名为 `Value`
2. `Graph` 还未保存更完整的模型级元数据，例如：
   - `opset imports`
3. shape 表达仍使用 `std::vector<int64_t>` + `-1` 表示动态维度
4. optional input 仍未显式保留输入槽位占位
5. `initializer` 的 typed data 支持还未覆盖全部 ONNX 类型

### P1：近期建议改

1. 为 `Node` 增加 `domain`
2. 为 `Graph` 增加 `opset import` 信息
3. 改善 shape 表达
4. 抽出图打印和调试接口
5. 增加 graph normalization pass 骨架

### P2：中期建议改

1. 将 `Edge` 正式重命名为 `Value`
2. 引入更正规的 arena allocator
3. 常量数据抽象为独立存储对象
4. 为子图与控制流预留接口
5. 支持 source mapping 与 pass annotation

---

## 10. 第一层 Graph IR 的阶段性目标

为了避免一次设计过重，建议第一层先做到以下程度即可：

### MVP 能力

- 支持 ONNX 静态图导入
- 支持常见 tensor op
- 支持 initializer
- 支持 graph input/output
- 支持基础 attribute
- 支持 shape/type 基础信息
- 支持 producer/consumer 关系
- 支持图遍历与拓扑排序基础

### 暂缓支持

- 完整控制流
- sequence / map / optional
- 高级别 shape constraints
- 稀疏张量
- 复杂别名分析
- 子图内联与跨图优化

---

## 11. 下一阶段规划建议

在 Graph IR 完善后，下一步建议进入 `Execution IR` 设计，重点包括：

1. 如何从 Graph 节点生成执行单元
2. 如何为每个值分配 execution slot
3. 如何做 activation lifetime 分析
4. 如何规划一块或多块内存池
5. 如何把常量映射到常量区
6. 如何保留 graph node 到 execution instr 的映射关系

---

## 12. 总结

当前第一层图设计方向是正确的，而且经过这一轮实现后，已经从“可运行原型”进一步提升到了“可作为正式 `Graph IR` 起点”的状态。  
目前已经完成的关键强化包括：

- attribute 类型系统和 tensor dtype 解耦
- 值状态显式化
- 稳定 ID
- 基本图校验接口
- ONNX 导入稳健性增强
- 图层与 VM 层基础类型解耦

但如果它要继续承担长期演进的 `Graph IR` 角色，后续仍建议继续推进以下事项：

- 把 `Edge` 的语义从“边”进一步落实为“值”
- 增加更完整的图级元数据
- 改善 shape / dynamic dimension 表达
- 为 optional input、控制流和子图预留更稳的接口
- 保持 `Arena` 风格内存管理，后续再升级为更高效实现

简要结论如下：

- **三层结构选择是正确的**
- **当前 Graph IR 已经具备了更正式的第一层抽象能力**
- **但在值语义命名、shape 表达、模型元数据方面仍有继续演进空间**
- **Arena 内存管理非常适合这个项目，应坚持这个方向**

---

## 13. 建议的后续落地顺序

1. 完成 `Edge -> Value` 的正式语义收敛与命名调整
2. 为 `Graph` 增加更完整的模型级元数据，例如：
   - `opset imports`
   - 其他必要的 module / model metadata
3. 改善 shape / dynamic dimension 表达
4. 设计 `Execution IR`
5. 设计从 `Graph IR -> Execution IR` 的 lowering
6. 再对接现有 `vm_program` 结构，形成第三层

如果后续继续推进，建议下一份设计文档聚焦于：

- `Execution IR` 的最小数据结构
- `Graph -> Execution` 的 lowering 规则
- activation memory planning 策略
- `VM Program` 与 `Execution IR` 的边界定义