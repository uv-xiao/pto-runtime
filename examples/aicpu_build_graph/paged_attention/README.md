# Paged Attention Example - AICPU Builds Graph (aicpu_build_graph)

This example matches the kernels and golden reference from `examples/host_build_graph/paged_attention`,but builds the task graph on **AICPU** via the `aicpu_build_graph` runtime.

Unlike `host_build_graph`, there is no “host orchestration that builds tasks”. The example provides a single orchestration plugin `.so` (configured by `kernels/kernel_config.py::ORCHESTRATION`) which is embedded by the host runtime and `dlopen()`'d on AICPU to build/publish tasks.

## Overview

Paged Attention is an efficient attention mechanism that processes KV cache in fixed-size blocks, enabling memory-efficient inference for long sequences.

### Algorithm

For each query token, the attention is computed incrementally across KV cache blocks:

```
For each block j:
    sij = Qi @ Kj^T                    # QK MatMul (AIC)
    mij, lij, pij = softmax_prepare(sij)  # Softmax (AIV)
    oi_new = pij @ Vj                  # PV MatMul (AIC)
    oi = online_update(oi, oi_new, mij, lij)  # Accumulate (AIV)
```

### Kernel Design (AIC/AIV Split)

| Kernel | Core Type | Operation | Key Instructions |
|--------|-----------|-----------|------------------|
| aic_qk_matmul | AIC (Cube) | Q @ K^T | TLOAD/TMOV/TMATMUL/TSTORE |
| aiv_softmax_prepare | AIV (Vector) | scale, rowmax, exp, rowsum | TMULS/TROWMAX/TROWEXPANDSUB/TEXP/TROWSUM |
| aic_pv_matmul | AIC (Cube) | P @ V | TLOAD/TMOV/TMATMUL/TSTORE |
| aiv_online_update | AIV (Vector) | Online Softmax + normalize | TMAX/TSUB/TEXP/TROWEXPANDMUL/TROWEXPANDDIV |

### Memory Hierarchy (AIC Matmul)

```
GM -> L1 (Mat tiles) -> L0A/L0B -> L0C (Accumulator) -> GM
```

### Task Graph Structure

For each batch, the task dependency pattern is:

```
Block 0: QK -> SF -> PV --+
Block 1: QK -> SF -> PV --+--> UP[0] -> UP[1] -> ... -> UP[n]
Block n: QK -> SF -> PV --+
```

- **QK/SF/PV chains**: Run in parallel across blocks
- **UP (Online Update)**: Serialized within batch due to accumulator dependency

## Quick Start

```bash
# Case1 (default)
python examples/scripts/run_example.py \
  -k examples/aicpu_build_graph/paged_attention/kernels \
  -g examples/aicpu_build_graph/paged_attention/golden.py \
  -p a2a3sim # or a2a3

# Case2
PA_CASE=Case2 python examples/scripts/run_example.py \
  -k examples/aicpu_build_graph/paged_attention/kernels \
  -g examples/aicpu_build_graph/paged_attention/golden.py \
  -p a2a3sim # or a3a3
```

## Directory Structure

```
examples/aicpu_build_graph/paged_attention/
├── README.md                    # This file
├── golden.py                    # Input generation and expected output
└── kernels/
    ├── kernel_config.py         # Kernel registration config
    ├── aic/                      # AIC kernels (CCE codegen style)
    │   ├── aic_qk_matmul.cpp     # Q @ K^T matmul
    │   └── aic_pv_matmul.cpp     # P @ V matmul
    ├── aiv/                      # AIV kernels (PTO Tile API)
    │   ├── aiv_softmax_prepare.cpp  # Softmax preparation
    │   └── aiv_online_update.cpp    # Online Softmax update + normalize
    └── orchestration/
        └── orchestration.cpp        # AICPU orchestration (graph builder)
```
