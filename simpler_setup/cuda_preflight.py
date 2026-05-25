# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CUDA test preflight checks."""

from __future__ import annotations

import shutil
import subprocess


def cuda_skip_reason(*, require_nvcc: bool = True) -> str | None:
    """Return a pytest skip reason when CUDA smoke tests cannot run."""

    if require_nvcc and shutil.which("nvcc") is None:
        return "nvcc is required for CUDA tests"
    if shutil.which("nvidia-smi") is None:
        return "nvidia-smi is required for CUDA driver checks"

    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version,memory.total", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "nvidia-smi returned a non-zero exit status").strip()
        return f"CUDA driver check failed: {detail}"
    if not result.stdout.strip():
        return "CUDA driver check found no visible NVIDIA GPUs"
    return None
