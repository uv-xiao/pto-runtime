"""
Paged Attention (aicpu_build_graph) Kernel and Orchestration Configuration

This example matches `examples/host_build_graph/paged_attention`, but builds the
task graph on AICPU via a dlopen'd orchestration plugin.

Test cases are selected via:
  PA_CASE=Case1|Case2
"""

from pathlib import Path

_KERNELS_ROOT = Path(__file__).parent

ORCHESTRATION = {
    # This .so runs on AICPU. The aicpu_build_graph host runtime embeds its bytes
    # and the AICPU executor loads it via dlopen+dlsym.
    "source": str(_KERNELS_ROOT / "orchestration" / "orchestration.cpp"),
    "function_name": "orchestration",
}

KERNELS = [
    {"func_id": 0, "name": "QK", "source": str(_KERNELS_ROOT / "aic" / "aic_qk_matmul.cpp"), "core_type": "aic"},
    {"func_id": 2, "name": "PV", "source": str(_KERNELS_ROOT / "aic" / "aic_pv_matmul.cpp"), "core_type": "aic"},
    {"func_id": 1, "name": "SF", "source": str(_KERNELS_ROOT / "aiv" / "aiv_softmax_prepare.cpp"), "core_type": "aiv"},
    {"func_id": 3, "name": "UP", "source": str(_KERNELS_ROOT / "aiv" / "aiv_online_update.cpp"), "core_type": "aiv"},
]

RUNTIME_CONFIG = {
    "runtime": "aicpu_build_graph",
    # Default split: 1 AICPU thread builds tasks while 3 AICPU threads schedule/execute.
    "aicpu_thread_num": 4,
    "block_dim": 24,
}

RUNTIME_ENV = {
    # 1: concurrent build||schedule (default); 0: sequential build->schedule
    "PTO_AICPU_BUILD_GRAPH_BUILD_MODE": "1",
}
