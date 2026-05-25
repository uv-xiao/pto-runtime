# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for CUDA persistent_device source generation."""

from __future__ import annotations

import pytest

from simpler_setup.cuda_callable_compiler import CudaPersistentTaskFunction, render_persistent_dag_source


def test_render_persistent_dag_source_generates_dispatch_switch():
    source = render_persistent_dag_source(
        [
            CudaPersistentTaskFunction(
                func_id=2,
                name="mul_f32",
                body="task->out[i] = task->a[i] * task->b[i];",
            ),
            CudaPersistentTaskFunction(
                func_id=1,
                name="add_f32",
                body="task->out[i] = task->a[i] + task->b[i];",
            ),
        ]
    )

    assert 'extern "C" __global__ void pto_persistent_dag_f32_executor' in source
    assert "__device__ void pto_task_add_f32" in source
    assert "__device__ void pto_task_mul_f32" in source
    assert "switch (task->func_id)" in source
    assert "case 1U:" in source
    assert "pto_task_add_f32(task);" in source
    assert "case 2U:" in source
    assert "pto_task_mul_f32(task);" in source
    assert source.index("case 1U:") < source.index("case 2U:")


def test_render_persistent_dag_source_rejects_duplicate_func_id():
    with pytest.raises(ValueError, match="duplicate func_id"):
        render_persistent_dag_source(
            [
                CudaPersistentTaskFunction(func_id=1, name="add_a", body=""),
                CudaPersistentTaskFunction(func_id=1, name="add_b", body=""),
            ]
        )
