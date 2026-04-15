#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Scene test covering MANUAL -> AUTO explicit dependency bridging."""

from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, scene_test


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestManualScopeV1Bridge(SceneTestCase):
    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/manual_scope_v1_bridge_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "kernels/aiv/aiv_noop.cpp",
                "core_type": "aiv",
                "signature": [D.IN],
            },
        ],
    }

    CASES = [
        {
            "name": "default",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 2, "block_dim": 1},
            "params": {},
        },
    ]

    def generate_args(self, params):
        return TaskArgsBuilder()

    def compute_golden(self, args, params):
        del args, params


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
