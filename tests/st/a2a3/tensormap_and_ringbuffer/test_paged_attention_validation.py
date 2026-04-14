#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Negative ST for paged-attention shape validation."""

from pathlib import Path
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[4]
KERNELS_DIR = PROJECT_ROOT / "tests/st/a2a3/tensormap_and_ringbuffer/paged_attention/kernels"
GOLDEN_PATH = PROJECT_ROOT / "tests/st/a2a3/tensormap_and_ringbuffer/paged_attention_invalid_shape/golden.py"


@pytest.mark.platforms(["a2a3"])
@pytest.mark.device_count(1)
def test_paged_attention_rejects_unsupported_shapes(st_platform, st_device_ids, monkeypatch):
    monkeypatch.setenv("PTO_LOG_LEVEL", "error")

    cmd = [
        sys.executable,
        "examples/scripts/run_example.py",
        "--build",
        "-k",
        str(KERNELS_DIR),
        "-g",
        str(GOLDEN_PATH),
        "-p",
        st_platform,
        "-d",
        str(st_device_ids[0]),
        "--case",
        "InvalidHeadDim",
    ]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    output = result.stdout + result.stderr

    assert result.returncode != 0, output
    assert "orch_error_code=5" in output, output
    assert "runtime_status=-5" in output, output
