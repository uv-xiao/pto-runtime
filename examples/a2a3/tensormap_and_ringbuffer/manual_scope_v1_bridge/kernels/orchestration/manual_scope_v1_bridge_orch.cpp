/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"

#define FUNC_AIV_NOOP 0

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
aicpu_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 0,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;

    uint32_t shapes[1] = {1};
    TensorCreateInfo ci(shapes, 1, DataType::FLOAT32);
    ci.set_initial_value(1.0f);

    TaskOutputTensors manual_outs;
    PTO2_SCOPE(PTO2ScopeMode::MANUAL) { manual_outs = alloc_tensors(ci); }

    Arg args;
    args.add_input(manual_outs.get_ref(0));
    args.add_dep(manual_outs.task_id());
    (void)pto2_rt_submit_aiv_task(FUNC_AIV_NOOP, args);
}

}  // extern "C"
