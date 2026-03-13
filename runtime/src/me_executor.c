#include "me_internal.h"

/* ---- Internal: Instruction Dispatch ----------------------------------- */

MeStatus me_executor_run_plan(MeProgram prog, uint32_t plan_idx) {
    if (!prog || plan_idx >= prog->plan_count)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;

    /* TODO: implement the main dispatch loop:
     *
     * const ExecutionPlanData *plan = &prog->plan_pool[plan_idx];
     * for (uint32_t pc = 0; pc < plan->instructions_count; ++pc) {
     *     const Instruction *instr =
     *         &prog->instruction_pool[plan->instructions_offset + pc];
     *     switch (instr->opcode) {
     *     case OPCODE_KERNEL_CALL:
     *         // build MeOpContext from args_list
     *         // call resolved_kernels[instr->arg1]
     *         break;
     *     case OPCODE_DELEGATE_CALL:
     *         break;
     *     case OPCODE_MOVE_CALL:
     *         break;
     *     case OPCODE_JUMP_FALSE_CALL:
     *         break;
     *     case OPCODE_FREE_CALL:
     *         break;
     *     case OPCODE_NOP_CALL:
     *         break;
     *     }
     * }
     */

    (void)plan_idx;
    return ME_STATUS_ERROR_UNSUPPORTED;
}

/* ---- Public: Execution ------------------------------------------------ */

MeStatus me_program_set_input(MeProgram prog, uint32_t index,
                              MeTensor tensor) {
    if (!prog || !tensor) return ME_STATUS_ERROR_INVALID_ARGUMENT;

    /* TODO: validate index against plan input count, store binding */
    (void)index;
    return ME_STATUS_ERROR_UNSUPPORTED;
}

MeStatus me_program_execute(MeProgram prog) {
    if (!prog) return ME_STATUS_ERROR_INVALID_ARGUMENT;

    uint32_t entry = 0;
    if (prog->header)
        entry = prog->header->entry_plan_idx;

    return me_executor_run_plan(prog, entry);
}

MeStatus me_program_get_output(MeProgram prog, uint32_t index,
                               MeTensor *out) {
    if (!prog || !out) return ME_STATUS_ERROR_INVALID_ARGUMENT;

    /* TODO: return borrowed reference from execution state */
    (void)index;
    return ME_STATUS_ERROR_UNSUPPORTED;
}
