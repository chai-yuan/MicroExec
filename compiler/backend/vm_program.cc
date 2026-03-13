#include "backend/vm_program.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/exec_types.h"
#include "common/log.h"
#include "execution/exec_program.h"

namespace {

// 将值按指定对齐方式向上对齐到边界。
static uint32_t AlignUpU32(uint32_t value, uint32_t alignment) {
    if (alignment == 0u) {
        return value;
    }
    return (value + alignment - 1u) & ~(alignment - 1u);
}

// 将 POD 类型的 vector 转换为字节序列并追加到输出缓冲区。
template <typename T>
static void AppendPodVectorAsBytes(const std::vector<T> &items, std::vector<uint8_t> &out) {
    if (items.empty()) {
        return;
    }
    const uint8_t *bytes = reinterpret_cast<const uint8_t *>(items.data());
    out.insert(out.end(), bytes, bytes + sizeof(T) * items.size());
}

// 将字符串池构建为字节序列（每个字符串前带长度前缀）。
static std::vector<uint8_t> BuildStringSection(const std::vector<std::string> &pool) {
    std::vector<uint8_t> out;
    for (const std::string &item : pool) {
        uint32_t len = static_cast<uint32_t>(item.size());
        const uint8_t *len_bytes = reinterpret_cast<const uint8_t *>(&len);
        out.insert(out.end(), len_bytes, len_bytes + sizeof(uint32_t));
        out.insert(out.end(), item.begin(), item.end());
    }
    return out;
}

// 将 int64 维度值归一化为 uint32，负值或溢出时返回最大值。
static uint32_t NormalizeDimToU32(int64_t dim) {
    if (dim < 0) {
        return std::numeric_limits<uint32_t>::max();
    }
    if (static_cast<uint64_t>(dim) > std::numeric_limits<uint32_t>::max()) {
        return std::numeric_limits<uint32_t>::max();
    }
    return static_cast<uint32_t>(dim);
}

// 将执行操作类型转换为对应的 VM 操作码。
static uint8_t ToOpcode(ExecOpKind kind) {
    switch (kind) {
    case ExecOpKind::KERNEL:
        return static_cast<uint8_t>(OPCODE_KERNEL_CALL);
    case ExecOpKind::DELEGATE:
        return static_cast<uint8_t>(OPCODE_DELEGATE_CALL);
    case ExecOpKind::MOVE:
        return static_cast<uint8_t>(OPCODE_MOVE_CALL);
    case ExecOpKind::NOP:
    default:
        return static_cast<uint8_t>(OPCODE_NOP_CALL);
    }
}

// 将数据类型转换为 VM 张量标量类型枚举值。
static uint32_t ToVMTensorScalarType(DataType dtype) {
    switch (dtype) {
    case DataType::FLOAT32:
        return VM_TENSOR_SCALAR_FLOAT32;
    case DataType::FLOAT16:
        return VM_TENSOR_SCALAR_FLOAT16;
    case DataType::INT64:
        return VM_TENSOR_SCALAR_INT64;
    case DataType::INT32:
        return VM_TENSOR_SCALAR_INT32;
    case DataType::INT8:
        return VM_TENSOR_SCALAR_INT8;
    case DataType::UINT8:
        return VM_TENSOR_SCALAR_UINT8;
    case DataType::BOOL:
        return VM_TENSOR_SCALAR_BOOL;
    case DataType::STRING:
        return VM_TENSOR_SCALAR_STRING;
    case DataType::UNKNOWN:
    default:
        return VM_TENSOR_SCALAR_UNKNOWN;
    }
}

// 根据执行值判断张量形状的动态性类型。
static uint32_t ToVMTensorShapeDynamism(const ExecValue &value) {
    if (value.deferred_runtime_alloc) {
        return VM_TENSOR_SHAPE_DYNAMIC_UNBOUND;
    }
    return VM_TENSOR_SHAPE_STATIC;
}

} // namespace

// 默认构造函数。
ProgramBuilder::ProgramBuilder() = default;

// 向字符串池添加字符串并返回其索引。
uint32_t ProgramBuilder::AddString(const std::string &str) {
    uint32_t idx = static_cast<uint32_t>(string_pool_.size());
    string_pool_.push_back(str);
    return idx;
}

// 向整数池添加数组并返回其起始偏移量。
uint32_t ProgramBuilder::AddIntArray(const std::vector<uint32_t> &arr) {
    uint32_t offset = static_cast<uint32_t>(int_pool_.size());
    int_pool_.insert(int_pool_.end(), arr.begin(), arr.end());
    return offset;
}

// 向张量元数据池添加张量描述并返回其索引。
uint32_t ProgramBuilder::AddTensorMeta(const TensorMeta &meta) {
    uint32_t idx = static_cast<uint32_t>(tensor_pool_.size());
    tensor_pool_.push_back(meta);
    return idx;
}

// 向 EValue 池添加值并返回其索引。
uint32_t ProgramBuilder::AddEValue(const EValue &evalue) {
    uint32_t idx = static_cast<uint32_t>(evalue_pool_.size());
    evalue_pool_.push_back(evalue);
    return idx;
}

// 向算子池添加算子定义并返回其索引。
uint32_t ProgramBuilder::AddOperator(const std::string &name, const std::string &overload) {
    OperatorDef op{};
    op.name_idx     = AddString(name);
    op.overload_idx = AddString(overload);

    uint32_t idx = static_cast<uint32_t>(operator_pool_.size());
    operator_pool_.push_back(op);
    return idx;
}

// 向委托池添加后端委托定义并返回其索引。
uint32_t ProgramBuilder::AddDelegate(const std::string &backend_id, uint32_t blob_offset, uint32_t blob_size) {
    BackendDelegate delegate{};
    delegate.id_idx      = AddString(backend_id);
    delegate.blob_offset = blob_offset;
    delegate.blob_size   = blob_size;

    uint32_t idx = static_cast<uint32_t>(delegate_pool_.size());
    delegate_pool_.push_back(delegate);
    return idx;
}

// 将指令追加到指令池。
void ProgramBuilder::AppendInstruction(const Instruction &inst) { instruction_pool_.push_back(inst); }

// 构建执行计划，包含输入输出和指令范围信息。
void ProgramBuilder::BuildExecutionPlan(const std::string &name, const std::vector<uint32_t> &inputs,
                                        const std::vector<uint32_t> &outputs, uint32_t memory_pool_size) {
    ExecutionPlanData plan{};
    plan.name_idx       = AddString(name);
    plan.inputs_offset  = AddIntArray(inputs);
    plan.inputs_count   = static_cast<uint32_t>(inputs.size());
    plan.outputs_offset = AddIntArray(outputs);
    plan.outputs_count  = static_cast<uint32_t>(outputs.size());

    uint32_t instruction_begin = 0;
    if (!plan_pool_.empty()) {
        const ExecutionPlanData &last_plan = plan_pool_.back();
        instruction_begin = last_plan.instructions_offset + last_plan.instructions_count;
    }
    plan.instructions_offset = instruction_begin;
    plan.instructions_count = static_cast<uint32_t>(instruction_pool_.size()) - instruction_begin;
    plan.memory_pool_size = memory_pool_size;

    plan_pool_.push_back(plan);
}

// 将所有构建的数据序列化为 VM 二进制文件。
int ProgramBuilder::Serialize(const std::string &output_file) {
    struct SectionPayload {
        VMSectionKind kind;
        std::vector<uint8_t> bytes;
        uint32_t count;
    };

    std::vector<SectionPayload> sections;
    sections.reserve(9);

    sections.push_back(
        {VM_SECTION_STRINGS, BuildStringSection(string_pool_), static_cast<uint32_t>(string_pool_.size())});

    SectionPayload ints{VM_SECTION_INTS, {}, static_cast<uint32_t>(int_pool_.size())};
    AppendPodVectorAsBytes(int_pool_, ints.bytes);
    sections.push_back(std::move(ints));

    SectionPayload tensors{VM_SECTION_TENSORS, {}, static_cast<uint32_t>(tensor_pool_.size())};
    AppendPodVectorAsBytes(tensor_pool_, tensors.bytes);
    sections.push_back(std::move(tensors));

    SectionPayload evalues{VM_SECTION_EVALUES, {}, static_cast<uint32_t>(evalue_pool_.size())};
    AppendPodVectorAsBytes(evalue_pool_, evalues.bytes);
    sections.push_back(std::move(evalues));

    SectionPayload ops{VM_SECTION_OPERATORS, {}, static_cast<uint32_t>(operator_pool_.size())};
    AppendPodVectorAsBytes(operator_pool_, ops.bytes);
    sections.push_back(std::move(ops));

    SectionPayload delegates{VM_SECTION_DELEGATES, {}, static_cast<uint32_t>(delegate_pool_.size())};
    AppendPodVectorAsBytes(delegate_pool_, delegates.bytes);
    sections.push_back(std::move(delegates));

    SectionPayload instrs{VM_SECTION_INSTRUCTIONS, {}, static_cast<uint32_t>(instruction_pool_.size())};
    AppendPodVectorAsBytes(instruction_pool_, instrs.bytes);
    sections.push_back(std::move(instrs));

    SectionPayload plans{VM_SECTION_EXEC_PLANS, {}, static_cast<uint32_t>(plan_pool_.size())};
    AppendPodVectorAsBytes(plan_pool_, plans.bytes);
    sections.push_back(std::move(plans));

    sections.push_back({VM_SECTION_WEIGHTS, {}, 0});

    std::vector<VMSectionDesc> section_descs(sections.size());
    uint32_t section_table_ofs = sizeof(VMFileHeader);
    uint32_t data_ofs =
        AlignUpU32(section_table_ofs + static_cast<uint32_t>(sections.size() * sizeof(VMSectionDesc)), kVMDataAlignment);

    for (size_t i = 0; i < sections.size(); ++i) {
        VMSectionDesc desc{};
        desc.kind = static_cast<uint32_t>(sections[i].kind);
        desc.offset = data_ofs;
        desc.size_bytes = static_cast<uint32_t>(sections[i].bytes.size());
        desc.count = sections[i].count;
        section_descs[i] = desc;
        data_ofs = AlignUpU32(data_ofs + desc.size_bytes, kVMDataAlignment);
    }

    VMFileHeader header{};
    header.magic = kVMFileMagic;
    header.version_major = kVMFileVersionMajor;
    header.version_minor = kVMFileVersionMinor;
    header.header_size = sizeof(VMFileHeader);
    header.file_size = data_ofs;
    header.section_count = static_cast<uint32_t>(section_descs.size());
    header.section_table_ofs = section_table_ofs;
    header.entry_plan_idx = 0;
    header.flags = 0;

    std::vector<uint8_t> file_bytes(header.file_size, 0);
    std::memcpy(file_bytes.data(), &header, sizeof(header));
    std::memcpy(file_bytes.data() + header.section_table_ofs, section_descs.data(),
                section_descs.size() * sizeof(VMSectionDesc));

    for (size_t i = 0; i < sections.size(); ++i) {
        const VMSectionDesc &desc = section_descs[i];
        if (!sections[i].bytes.empty()) {
            std::memcpy(file_bytes.data() + desc.offset, sections[i].bytes.data(), sections[i].bytes.size());
        }
    }

    std::ofstream ofs(output_file, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!ofs.is_open()) {
        LOG_ERROR("ProgramBuilder::Serialize failed to open file: %s", output_file.c_str());
        return -1;
    }

    ofs.write(reinterpret_cast<const char *>(file_bytes.data()), static_cast<std::streamsize>(file_bytes.size()));
    if (!ofs.good()) {
        LOG_ERROR("ProgramBuilder::Serialize failed to write file: %s", output_file.c_str());
        return -1;
    }

    LOG_INFO("VM program serialized: %s (%u bytes, %u sections)", output_file.c_str(), header.file_size,
             header.section_count);
    return 0;
}

// 从执行程序 IR 构建 VM 程序数据。
int ProgramBuilder::BuildFromExecProgram(const ExecProgram &exec) {
    const auto &values = exec.GetValues();
    const auto &instructions = exec.GetInstructions();
    const auto &inputs = exec.GetInputValueIds();
    const auto &outputs = exec.GetOutputValueIds();

    std::vector<uint32_t> exec_value_ids;
    exec_value_ids.reserve(values.size());
    for (const ExecValue &value : values) {
        exec_value_ids.push_back(value.id);
    }
    std::sort(exec_value_ids.begin(), exec_value_ids.end());

    std::unordered_map<uint32_t, const ExecValue *> id_to_value;
    id_to_value.reserve(values.size());
    for (const ExecValue &value : values) {
        id_to_value[value.id] = &value;
    }

    std::unordered_map<uint32_t, uint32_t> exec_to_evalue_idx;
    exec_to_evalue_idx.reserve(values.size());

    for (uint32_t value_id : exec_value_ids) {
        auto value_it = id_to_value.find(value_id);
        if (value_it == id_to_value.end()) {
            LOG_ERROR("LowerToVM: missing ExecValue id %u", value_id);
            return -1;
        }

        const ExecValue &value = *value_it->second;
        if (value.offset > std::numeric_limits<uint32_t>::max()) {
            LOG_ERROR("LowerToVM: value %u offset too large: %lu", value.id, (unsigned long)value.offset);
            return -1;
        }

        std::vector<uint32_t> shape_u32;
        shape_u32.reserve(value.shape.size());
        std::vector<uint32_t> dim_order;
        dim_order.reserve(value.shape.size());
        for (size_t i = 0; i < value.shape.size(); ++i) {
            shape_u32.push_back(NormalizeDimToU32(value.shape[i]));
            dim_order.push_back(static_cast<uint32_t>(i));
        }

        TensorMeta meta{};
        meta.scalar_type = ToVMTensorScalarType(value.dtype);
        meta.shape_dynamism = ToVMTensorShapeDynamism(value);
        meta.ndim = static_cast<uint32_t>(value.shape.size());
        meta.shape_offset = AddIntArray(shape_u32);
        meta.dim_order_offset = AddIntArray(dim_order);
        meta.buffer_id = value.buffer_id;
        meta.data_offset = static_cast<uint32_t>(value.offset);

        uint32_t tensor_meta_idx = AddTensorMeta(meta);

        EValue evalue{};
        evalue.type = EVALUE_TYPE_TENSOR;
        evalue.payload = tensor_meta_idx;

        uint32_t evalue_idx = AddEValue(evalue);
        exec_to_evalue_idx[value.id] = evalue_idx;
    }

    std::unordered_map<std::string, uint32_t> op_to_idx;
    op_to_idx.reserve(instructions.size());
    std::unordered_map<uint32_t, uint32_t> delegate_to_idx;
    delegate_to_idx.reserve(instructions.size());

    for (const ExecInstr &instr : instructions) {
        Instruction vm_instr{};
        vm_instr.opcode = ToOpcode(instr.op_kind);
        vm_instr.flags = INSTR_FLAG_NONE;
        vm_instr.arg1 = 0;
        vm_instr.arg2 = 0;
        vm_instr.arg3 = 0;

        switch (instr.op_kind) {
        case ExecOpKind::KERNEL:
        case ExecOpKind::DELEGATE: {
            std::vector<uint32_t> args;
            args.reserve(instr.input_values.size() + instr.output_values.size());

            for (uint32_t input_id : instr.input_values) {
                auto value_it = exec_to_evalue_idx.find(input_id);
                if (value_it == exec_to_evalue_idx.end()) {
                    LOG_ERROR("LowerToVM: input value %u not mapped", input_id);
                    return -1;
                }
                args.push_back(value_it->second);
            }

            for (uint32_t output_id : instr.output_values) {
                auto value_it = exec_to_evalue_idx.find(output_id);
                if (value_it == exec_to_evalue_idx.end()) {
                    LOG_ERROR("LowerToVM: output value %u not mapped", output_id);
                    return -1;
                }
                args.push_back(value_it->second);
            }

            vm_instr.input_count = static_cast<uint8_t>(std::min<size_t>(instr.input_values.size(), 255));
            vm_instr.output_count = static_cast<uint8_t>(std::min<size_t>(instr.output_values.size(), 255));
            vm_instr.arg2 = AddIntArray(args);
            vm_instr.arg3 = static_cast<uint32_t>(args.size());

            if (instr.op_kind == ExecOpKind::KERNEL) {
                auto op_it = op_to_idx.find(instr.op_name);
                if (op_it == op_to_idx.end()) {
                    uint32_t op_idx = AddOperator(instr.op_name, "");
                    op_to_idx[instr.op_name] = op_idx;
                    vm_instr.arg1 = op_idx;
                } else {
                    vm_instr.arg1 = op_it->second;
                }
            } else {
                auto delegate_it = delegate_to_idx.find(instr.delegate_id);
                if (delegate_it == delegate_to_idx.end()) {
                    std::string backend_id = "delegate_" + std::to_string(instr.delegate_id);
                    uint32_t delegate_idx = AddDelegate(backend_id, 0, 0);
                    delegate_to_idx[instr.delegate_id] = delegate_idx;
                    vm_instr.arg1 = delegate_idx;
                } else {
                    vm_instr.arg1 = delegate_it->second;
                }
            }
            break;
        }
        case ExecOpKind::MOVE:
            vm_instr.input_count = 1;
            vm_instr.output_count = 1;
            if (!instr.input_values.empty()) {
                auto input_it = exec_to_evalue_idx.find(instr.input_values[0]);
                if (input_it != exec_to_evalue_idx.end()) {
                    vm_instr.arg1 = input_it->second;
                }
            }
            if (!instr.output_values.empty()) {
                auto output_it = exec_to_evalue_idx.find(instr.output_values[0]);
                if (output_it != exec_to_evalue_idx.end()) {
                    vm_instr.arg2 = output_it->second;
                }
            }
            break;
        case ExecOpKind::NOP:
        default:
            vm_instr.input_count = 0;
            vm_instr.output_count = 0;
            break;
        }

        AppendInstruction(vm_instr);
    }

    std::vector<uint32_t> vm_inputs;
    vm_inputs.reserve(inputs.size());
    for (uint32_t input_id : inputs) {
        auto input_it = exec_to_evalue_idx.find(input_id);
        if (input_it == exec_to_evalue_idx.end()) {
            LOG_ERROR("LowerToVM: graph input value %u not mapped", input_id);
            return -1;
        }
        vm_inputs.push_back(input_it->second);
    }

    std::vector<uint32_t> vm_outputs;
    vm_outputs.reserve(outputs.size());
    for (uint32_t output_id : outputs) {
        auto output_it = exec_to_evalue_idx.find(output_id);
        if (output_it == exec_to_evalue_idx.end()) {
            LOG_ERROR("LowerToVM: graph output value %u not mapped", output_id);
            return -1;
        }
        vm_outputs.push_back(output_it->second);
    }

    uint64_t runtime_pool_size_u64 = exec.GetMemoryPlan().runtime_pool_size;
    if (runtime_pool_size_u64 > std::numeric_limits<uint32_t>::max()) {
        LOG_ERROR("LowerToVM: runtime pool too large: %lu", (unsigned long)runtime_pool_size_u64);
        return -1;
    }

    BuildExecutionPlan(exec.name, vm_inputs, vm_outputs, static_cast<uint32_t>(runtime_pool_size_u64));
    return 0;
}
