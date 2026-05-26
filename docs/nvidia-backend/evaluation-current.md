# CUDA Current Evaluation Capture

This page summarizes the current paired A100/H200 CUDA backend capture from
commit `9c99ae8a`. The raw JSON, Markdown, and SVG reports are generated
locally under `tmp/cuda-backend/` and intentionally remain uncommitted.

The capture uses `nvcc` for target-specific PTX on both machines:

- A100: `compute_80`
- H200: `compute_90`
- tensor descriptor: `8x4x12`
- sizes: `1024,65536,1048576`
- repeats: `3`
- batch tasks: `2,6,12`
- worker blocks per task: `32,64,128,256`
- samples in combined JSON: `702`

## Artifact Paths

- `tmp/cuda-backend/a100-current-9c99ae8a/cuda-benchmark.json`
- `tmp/cuda-backend/a100-current-9c99ae8a/cuda-benchmark.md`
- `tmp/cuda-backend/h200-current-9c99ae8a/cuda-benchmark.json`
- `tmp/cuda-backend/h200-current-9c99ae8a/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-9c99ae8a/cuda-benchmark.json`
- `tmp/cuda-backend/combined-current-9c99ae8a/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-9c99ae8a/cuda-benchmark.svg`
- `tmp/cuda-backend/combined-current-9c99ae8a/cuda-benchmark-ratios.svg`

## Launch Baselines

CUDA Graph replay remains a useful phase-one host-launch baseline. It can
reduce launch overhead in some rows, but it is still host-owned replay rather
than a device-side scheduler.

The `pto_host_schedule_compiler` row validates the task-body compiler path.
It uses the same host runtime path as `pto_host_schedule`, but the PTX comes
from `KernelCompiler(platform="cuda").compile_cuda_host_schedule(...)` and
the shared task wrapper generator.

| GPU | N | PTO host ns | Compiler ns | Driver ns | Graph ns | Compiler/PTO | Graph/PTO |
| --- | - | ----------- | ----------- | --------- | -------- | ------------ | --------- |
| A100 | 1024 | 7168 | 7168 | 9216 | 20479 | 1.00x | 2.86x |
| A100 | 65536 | 422176 | 429344 | 424192 | 707328 | 1.02x | 1.68x |
| A100 | 1048576 | 24672 | 23616 | 28767 | 24127 | 0.96x | 0.98x |
| H200 | 1024 | 30592 | 32928 | 22048 | 16095 | 1.08x | 0.53x |
| H200 | 65536 | 15360 | 16640 | 24032 | 18303 | 1.08x | 1.19x |
| H200 | 1048576 | 20320 | 20448 | 27327 | 19360 | 1.01x | 0.95x |

The compiler row stays in the same launch-latency band as the handwritten
host-schedule PTX. That is the important signal for this slice: the shared
task-body wrapper path can feed the existing host runtime without changing
the launch ABI or adding a separate generated-kernel calling convention.
The A100 medium-vector launch rows are noisy in this capture,
so use the paired report as current smoke/evaluation evidence rather than a
final launch-overhead ranking.

## Host-Schedule Shape Rows

The `pto_host_schedule_unary_square` row validates the generated unary
`(a, out, n)` ABI in the full paired benchmark. The
`pto_host_schedule_quad` row validates the generated four-input
`(a, b, c, d, out, n)` ABI. Both rows compare against CUDA `float32`
goldens, which matters at larger sizes where exact Python integer arithmetic
does not match single-precision output.

| GPU | N | Unary square ns | Quad ns |
| --- | - | --------------- | ------- |
| A100 | 1024 | 7168 | 7168 |
| A100 | 65536 | 297024 | 304128 |
| A100 | 1048576 | 23968 | 27680 |
| H200 | 1024 | 33120 | 30848 |
| H200 | 65536 | 20544 | 15904 |
| H200 | 1048576 | 19264 | 22240 |

## Worker Grid Rows

The plain one-block persistent batch rows are still too serial for larger
vectors. The worker-grid row is the current useful persistent-device
throughput slice because it keeps one persistent launch but assigns multiple
worker blocks to each task descriptor.

| GPU | N | Tasks | Best worker blocks/task | Device ns | Vs host batch |
| --- | - | ----- | ----------------------- | --------- | ------------- |
| A100 | 1024 | 2 | 64 | 9216 | 0.39x |
| A100 | 1024 | 6 | 32 | 7168 | 0.15x |
| A100 | 1024 | 12 | 32 | 8192 | 0.09x |
| A100 | 65536 | 2 | 32 | 19456 | 0.57x |
| A100 | 65536 | 6 | 128 | 16384 | 0.23x |
| A100 | 65536 | 12 | 32 | 15360 | 0.16x |
| A100 | 1048576 | 2 | 256 | 30720 | 0.86x |
| A100 | 1048576 | 6 | 256 | 44032 | 0.62x |
| A100 | 1048576 | 12 | 256 | 61440 | 0.47x |
| H200 | 1024 | 2 | 64 | 30400 | 1.31x |
| H200 | 1024 | 6 | 64 | 29760 | 0.34x |
| H200 | 1024 | 12 | 128 | 28928 | 0.22x |
| H200 | 65536 | 2 | 256 | 17472 | 0.78x |
| H200 | 65536 | 6 | 128 | 15520 | 0.29x |
| H200 | 65536 | 12 | 32 | 16672 | 0.21x |
| H200 | 1048576 | 2 | 256 | 19232 | 0.72x |
| H200 | 1048576 | 6 | 128 | 24960 | 0.43x |
| H200 | 1048576 | 12 | 256 | 36960 | 0.34x |

The H200 worker-grid rows keep the same broad signal as prior captures:
larger worker-block counts help larger vectors, but the best count is not
monotonic across GPUs, vector sizes, and descriptor counts. The A100 worker
grid rows stay below matched host-schedule batch rows for most multi-task
rows in this capture, with the H200 two-task small-vector point as the
current exception.

## Persistent DAG Shapes

The DAG rows validate the persistent-device scheduler path rather than equal
work throughput. Chain and reuse add dependency levels and extra arithmetic.
The tensor row replaces one elementwise task with tiled GEMM work, so its
large-vector ratio is expected to be several times slower than the simple DAG.
The scalar affine, triad, quad, generic-args, and unary-square rows use
generated dispatch and different descriptor fields/task-body arities without
changing the persistent launch path.

| GPU | N | Chain/DAG | Reuse/DAG | Scalar AXPY/DAG | Scalar Affine/DAG | Triad/DAG | Quad/DAG | Generic Args/DAG | Unary Square/DAG | Tensor/DAG |
| --- | - | --------- | --------- | --------------- | ----------------- | --------- | -------- | ---------------- | ---------------- | ---------- |
| A100 | 1024 | 1.29x | 1.29x | 0.88x | 0.76x | 0.97x | 0.74x | 0.82x | 0.91x | 1.18x |
| A100 | 65536 | 0.14x | 0.12x | 0.07x | 0.09x | 0.09x | 0.10x | 0.09x | 1.13x | 1.32x |
| A100 | 1048576 | 1.81x | 1.94x | 1.00x | 0.99x | 1.09x | 1.16x | 1.15x | 1.52x | 4.63x |
| H200 | 1024 | 1.34x | 1.37x | 1.03x | 1.02x | 1.00x | 1.07x | 1.00x | 1.10x | 1.19x |
| H200 | 65536 | 1.77x | 1.79x | 0.98x | 0.99x | 0.99x | 1.06x | 1.06x | 1.44x | 2.96x |
| H200 | 1048576 | 1.79x | 1.79x | 0.99x | 1.00x | 1.00x | 1.01x | 0.99x | 1.37x | 3.06x |

The key correctness signal is that all DAG variants use generated dispatch
and runtime graph descriptors without changing the persistent launch path.
The scalar AXPY, scalar affine, triad, quad, generic-args, and unary-square
rows prove mixed tensor/scalar fields, extra tensor pointers through a fourth
tensor task descriptor field, generic indexed argument slots, and unary
task-body lowering.
The tensor row also proves the descriptor metadata path for non-square
`8x4x12` tiles. Treat these DAG-shape rows as correctness and scheduler-shape
evidence first; throughput conclusions require a tuned tensor workload.

## Reproduction Commands

Local A100:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --device 0 --sizes 1024,65536,1048576 --repeats 3 \
    --arch compute_80 --include-persistent --batch-tasks 2,6,12 \
    --worker-blocks-per-task 32,64,128,256 \
    --tensor-rows 8 --tensor-cols 4 --tensor-inner 12 \
    --label a100-current-$(git rev-parse --short HEAD) \
    --output-dir tmp/cuda-backend/a100-current-$(git rev-parse --short HEAD)
```

Paired A100/H200:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_pair_benchmark.py \
    --sync-remote-tree
```

Merge reports:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --merge-json \
    tmp/cuda-backend/a100-current-9c99ae8a/cuda-benchmark.json \
    tmp/cuda-backend/h200-current-9c99ae8a/cuda-benchmark.json \
    --label combined-current-9c99ae8a \
    --output-dir tmp/cuda-backend/combined-current-9c99ae8a
```
