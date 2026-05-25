# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for CUDA hardware preflight skip reporting."""

from __future__ import annotations

import subprocess

from simpler_setup import cuda_preflight


def test_cuda_preflight_reports_missing_nvcc(monkeypatch):
    monkeypatch.setattr(cuda_preflight.shutil, "which", lambda name: None)

    assert cuda_preflight.cuda_skip_reason(require_nvcc=True) == "nvcc is required for CUDA tests"


def test_cuda_preflight_reports_missing_nvidia_smi(monkeypatch):
    monkeypatch.setattr(cuda_preflight.shutil, "which", lambda name: "/usr/bin/nvcc" if name == "nvcc" else None)

    assert cuda_preflight.cuda_skip_reason(require_nvcc=True) == "nvidia-smi is required for CUDA driver checks"


def test_cuda_preflight_reports_unusable_driver(monkeypatch):
    def fake_which(name):
        return f"/usr/bin/{name}"

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=9, stdout="", stderr="driver not loaded")

    monkeypatch.setattr(cuda_preflight.shutil, "which", fake_which)
    monkeypatch.setattr(cuda_preflight.subprocess, "run", fake_run)

    assert cuda_preflight.cuda_skip_reason(require_nvcc=True) == "CUDA driver check failed: driver not loaded"


def test_cuda_preflight_returns_none_when_toolkit_and_driver_are_available(monkeypatch):
    def fake_which(name):
        return f"/usr/bin/{name}"

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="NVIDIA A100-SXM4-80GB, 8.0, 535.0, 81920 MiB",
            stderr="",
        )

    monkeypatch.setattr(cuda_preflight.shutil, "which", fake_which)
    monkeypatch.setattr(cuda_preflight.subprocess, "run", fake_run)

    assert cuda_preflight.cuda_skip_reason(require_nvcc=True) is None
