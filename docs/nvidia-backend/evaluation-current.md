# CUDA Current Evaluation Capture

This page summarizes the current paired A100/H200 CUDA backend capture from
commit `832d24bf`. The raw JSON, Markdown, and SVG reports are generated
locally under `tmp/cuda-backend/` and intentionally remain uncommitted.

The capture uses `nvcc` for target-specific PTX on both machines:

- A100: `compute_80`
- H200: `compute_90`
- tensor descriptor: `8x4x12`
- sizes: `1024,65536,1048576`
- repeats: `3`
- batch tasks: `2,6,12`
- worker blocks per task: `32,64,128,256`
- samples in combined JSON: `648`

## Artifact Paths

- `tmp/cuda-backend/a100-current-832d24bf/cuda-benchmark.json`
- `tmp/cuda-backend/a100-current-832d24bf/cuda-benchmark.md`
- `tmp/cuda-backend/h200-current-832d24bf/cuda-benchmark.json`
- `tmp/cuda-backend/h200-current-832d24bf/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-832d24bf/cuda-benchmark.json`
- `tmp/cuda-backend/combined-current-832d24bf/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-832d24bf/cuda-benchmark.svg`
- `tmp/cuda-backend/combined-current-832d24bf/cuda-benchmark-ratios.svg`

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
| A100 | 1024 | 7168 | 6144 | 8191 | 8191 | 0.86x | 1.14x |
| A100 | 65536 | 330560 | 345120 | 1208415 | 1468384 | 1.04x | 4.44x |
| A100 | 1048576 | 309984 | 309280 | 1257375 | 1524608 | 1.00x | 4.92x |
| H200 | 1024 | 28992 | 30752 | 24607 | 18239 | 1.06x | 0.63x |
| H200 | 65536 | 15552 | 17312 | 24351 | 18495 | 1.11x | 1.19x |
| H200 | 1048576 | 18944 | 20352 | 26303 | 21183 | 1.07x | 1.12x |

The compiler row stays in the same launch-latency band as the handwritten
host-schedule PTX. That is the important signal for this slice: the shared
task-body wrapper path can feed the existing host runtime without changing
the launch ABI or adding a separate generated-kernel calling convention.
The A100 raw Driver API and CUDA Graph rows are slower than the PTO
host-schedule row in this capture for larger vectors, so use the paired report
as current evidence rather than a final launch-overhead ranking.

## Unary Host-Schedule Row

The `pto_host_schedule_unary_square` row validates the generated unary
`(a, out, n)` ABI in the full paired benchmark. The validator compares
against CUDA `float32` square results, which matters at the larger sizes
because exact Python integer squares no longer match single-precision output.

| GPU | N | Unary square ns |
| --- | - | --------------- |
| A100 | 1024 | 7168 |
| A100 | 65536 | 301952 |
| A100 | 1048576 | 306208 |
| H200 | 1024 | 30144 |
| H200 | 65536 | 19776 |
| H200 | 1048576 | 19392 |

## Worker Grid Rows

The plain one-block persistent batch rows are still too serial for larger
vectors. The worker-grid row is the current useful persistent-device
throughput slice because it keeps one persistent launch but assigns multiple
worker blocks to each task descriptor.

| GPU | N | Tasks | Best worker blocks/task | Device ns | Vs host batch |
| --- | - | ----- | ----------------------- | --------- | ------------- |
| A100 | 1024 | 2 | 32 | 7168 | 0.50x |
| A100 | 1024 | 6 | 64 | 7168 | 0.18x |
| A100 | 1024 | 12 | 64 | 7168 | 0.09x |
| A100 | 65536 | 2 | 128 | 8192 | 0.03x |
| A100 | 65536 | 6 | 128 | 9216 | 0.02x |
| A100 | 65536 | 12 | 64 | 10240 | 0.03x |
| A100 | 1048576 | 2 | 256 | 19456 | 0.02x |
| A100 | 1048576 | 6 | 128 | 35840 | 0.04x |
| A100 | 1048576 | 12 | 256 | 55296 | 0.07x |
| H200 | 1024 | 2 | 256 | 29056 | 1.34x |
| H200 | 1024 | 6 | 32 | 30464 | 0.38x |
| H200 | 1024 | 12 | 64 | 28384 | 0.21x |
| H200 | 65536 | 2 | 32 | 16800 | 0.80x |
| H200 | 65536 | 6 | 64 | 15296 | 0.28x |
| H200 | 65536 | 12 | 64 | 15104 | 0.19x |
| H200 | 1048576 | 2 | 256 | 19104 | 0.74x |
| H200 | 1048576 | 6 | 128 | 24352 | 0.42x |
| H200 | 1048576 | 12 | 256 | 37088 | 0.35x |

The H200 worker-grid rows keep the same broad signal as prior captures:
larger worker-block counts help larger vectors, but the best count is not
monotonic across GPUs, vector sizes, and descriptor counts. The A100 worker
grid rows stay well below matched host-schedule batch rows for medium and
large vectors in this capture.

## Persistent DAG Shapes

The DAG rows validate the persistent-device scheduler path rather than equal
work throughput. Chain and reuse add dependency levels and extra arithmetic.
The tensor row replaces one elementwise task with tiled GEMM work, so its
large-vector ratio is expected to be several times slower than the simple DAG.
The scalar affine, triad, and unary-square rows use generated dispatch and
different descriptor fields/task-body arities without changing the persistent
launch path.

| GPU | N | Chain/DAG | Reuse/DAG | Scalar AXPY/DAG | Scalar Affine/DAG | Triad/DAG | Unary Square/DAG | Tensor/DAG |
| --- | - | --------- | --------- | --------------- | ----------------- | --------- | ---------------- | ---------- |
| A100 | 1024 | 1.45x | 1.50x | 1.00x | 1.00x | 1.00x | 1.20x | 1.70x |
| A100 | 65536 | 1.37x | 1.37x | 1.02x | 6.62x | 6.88x | 7.18x | 8.41x |
| A100 | 1048576 | 1.54x | 1.54x | 1.01x | 1.00x | 0.99x | 1.38x | 3.21x |
| H200 | 1024 | 1.32x | 1.54x | 0.95x | 0.93x | 0.89x | 1.01x | 1.17x |
| H200 | 65536 | 1.81x | 1.82x | 1.00x | 1.01x | 1.02x | 1.47x | 2.97x |
| H200 | 1048576 | 1.79x | 1.79x | 0.99x | 1.00x | 1.00x | 1.37x | 3.08x |

The key correctness signal is that all DAG variants use generated dispatch
and runtime graph descriptors without changing the persistent launch path.
The scalar AXPY, scalar affine, triad, and unary-square rows prove mixed
tensor/scalar fields, extra tensor pointers, and unary task-body lowering.
The tensor row also proves the descriptor metadata path for non-square
`8x4x12` tiles. The A100 `N=65536` scalar affine, triad, unary-square, and
tensor rows are much slower than the base DAG in this capture while the H200
rows stay close except for tensor and unary-square. Treat those A100 rows as
correctness evidence and recheck before drawing a throughput conclusion from
them.

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
    tmp/cuda-backend/a100-current-832d24bf/cuda-benchmark.json \
    tmp/cuda-backend/h200-current-832d24bf/cuda-benchmark.json \
    --label combined-current-832d24bf \
    --output-dir tmp/cuda-backend/combined-current-832d24bf
```
