# CUDA Current Evaluation Capture

This page summarizes the current paired A100/H200 CUDA backend capture from
commit `0eed34ff`. The raw JSON, Markdown, and SVG reports are generated
locally under `tmp/cuda-backend/` and intentionally remain uncommitted.

The capture uses `nvcc` for target-specific PTX on both machines:

- A100: `compute_80`
- H200: `compute_90`
- tensor descriptor: `8x4x12`
- sizes: `1024,65536,1048576`
- repeats: `3`
- batch tasks: `2,6,12`
- worker blocks per task: `32,64,128,256`
- samples in combined JSON: `630`

## Artifact Paths

- `tmp/cuda-backend/a100-current-0eed34ff/cuda-benchmark.json`
- `tmp/cuda-backend/a100-current-0eed34ff/cuda-benchmark.md`
- `tmp/cuda-backend/h200-current-0eed34ff/cuda-benchmark.json`
- `tmp/cuda-backend/h200-current-0eed34ff/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-0eed34ff/cuda-benchmark.json`
- `tmp/cuda-backend/combined-current-0eed34ff/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-0eed34ff/cuda-benchmark.svg`
- `tmp/cuda-backend/combined-current-0eed34ff/cuda-benchmark-ratios.svg`

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
| A100 | 1024 | 7168 | 7168 | 8191 | 8191 | 1.00x | 1.14x |
| A100 | 65536 | 693888 | 347936 | 296032 | 774688 | 0.50x | 1.12x |
| A100 | 1048576 | 751296 | 309280 | 814848 | 549023 | 0.41x | 0.73x |
| H200 | 1024 | 37216 | 36288 | 38368 | 29120 | 0.98x | 0.78x |
| H200 | 65536 | 25600 | 27232 | 39391 | 31295 | 1.06x | 1.22x |
| H200 | 1048576 | 23360 | 32063 | 28384 | 20864 | 1.37x | 0.89x |

The compiler row stays in the same launch-latency band as the handwritten
host-schedule PTX. That is the important signal for this slice: the shared
task-body wrapper path can feed the existing host runtime without changing
the launch ABI or adding a separate generated-kernel calling convention.
The A100 host-schedule rows in this capture show larger host-side variance
than the prior capture, so use the paired report as current evidence rather
than a final launch-overhead ranking.

## Unary Host-Schedule Row

The `pto_host_schedule_unary_square` row validates the generated unary
`(a, out, n)` ABI in the full paired benchmark. The validator compares
against CUDA `float32` square results, which matters at the larger sizes
because exact Python integer squares no longer match single-precision output.

| GPU | N | Unary square ns |
| --- | - | --------------- |
| A100 | 1024 | 7168 |
| A100 | 65536 | 300128 |
| A100 | 1048576 | 347232 |
| H200 | 1024 | 38432 |
| H200 | 65536 | 29728 |
| H200 | 1048576 | 27232 |

## Worker Grid Rows

The plain one-block persistent batch rows are still too serial for larger
vectors. The worker-grid row is the current useful persistent-device
throughput slice because it keeps one persistent launch but assigns multiple
worker blocks to each task descriptor.

| GPU | N | Tasks | Best worker blocks/task | Device ns | Vs host batch |
| --- | - | ----- | ----------------------- | --------- | ------------- |
| A100 | 1024 | 2 | 32 | 7168 | 0.54x |
| A100 | 1024 | 6 | 32 | 7168 | 0.18x |
| A100 | 1024 | 12 | 32 | 7168 | 0.09x |
| A100 | 65536 | 2 | 256 | 8192 | 0.03x |
| A100 | 65536 | 6 | 64 | 9216 | 0.03x |
| A100 | 65536 | 12 | 64 | 10240 | 0.01x |
| A100 | 1048576 | 2 | 256 | 18432 | 0.05x |
| A100 | 1048576 | 6 | 128 | 31744 | 0.04x |
| A100 | 1048576 | 12 | 256 | 53248 | 0.12x |
| H200 | 1024 | 2 | 256 | 36640 | 1.14x |
| H200 | 1024 | 6 | 256 | 35744 | 0.33x |
| H200 | 1024 | 12 | 32 | 35712 | 0.21x |
| H200 | 65536 | 2 | 256 | 24000 | 0.72x |
| H200 | 65536 | 6 | 128 | 23072 | 0.34x |
| H200 | 65536 | 12 | 32 | 23840 | 0.22x |
| H200 | 1048576 | 2 | 256 | 21568 | 0.69x |
| H200 | 1048576 | 6 | 128 | 26944 | 0.44x |
| H200 | 1048576 | 12 | 256 | 39840 | 0.36x |

The H200 worker-grid rows keep the same broad signal as prior captures:
larger worker-block counts help larger vectors, but the best count is not
monotonic across GPUs, vector sizes, and descriptor counts. The A100 host
batch rows show run-to-run noise in this capture, so ratios should be read as
capture evidence rather than a final resource policy.

## Persistent DAG Shapes

The DAG rows validate the persistent-device scheduler path rather than equal
work throughput. Chain and reuse add dependency levels and extra arithmetic.
The tensor row replaces one elementwise task with tiled GEMM work, so its
large-vector ratio is expected to be several times slower than the simple DAG.
The scalar affine and triad rows use the same runtime graph shape as the base
DAG while reading additional descriptor fields, so they should track the base
DAG closely.

| GPU | N | Chain/DAG | Reuse/DAG | Scalar AXPY/DAG | Scalar Affine/DAG | Triad/DAG | Tensor/DAG |
| --- | - | --------- | --------- | --------------- | ----------------- | --------- | ---------- |
| A100 | 1024 | 1.40x | 1.50x | 1.00x | 1.00x | 1.00x | 1.70x |
| A100 | 65536 | 1.05x | 1.05x | 1.00x | 0.99x | 0.14x | 1.22x |
| A100 | 1048576 | 1.59x | 1.61x | 1.04x | 1.02x | 1.02x | 3.27x |
| H200 | 1024 | 1.30x | 1.33x | 1.10x | 1.04x | 0.93x | 1.17x |
| H200 | 65536 | 1.74x | 1.75x | 0.99x | 0.99x | 1.00x | 2.89x |
| H200 | 1048576 | 1.79x | 1.79x | 0.99x | 1.00x | 1.00x | 3.11x |

The key correctness signal is that all DAG variants use generated dispatch
and runtime graph descriptors without changing the persistent launch path.
The scalar AXPY, scalar affine, and triad rows prove mixed tensor/scalar and
extra tensor-pointer descriptor lowering while tracking the base DAG closely.
The tensor row also proves the descriptor metadata path for non-square
`8x4x12` tiles. The A100 `N=65536` triad row is unusually faster than the
base DAG in this capture, so treat that row as correctness evidence and
recheck before drawing a throughput conclusion from it. The `N=1024`
DAG-shape ratios are small-launch rows and should be treated as
scheduling-smoke evidence rather than throughput signal.

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
    tmp/cuda-backend/a100-current-0eed34ff/cuda-benchmark.json \
    tmp/cuda-backend/h200-current-0eed34ff/cuda-benchmark.json \
    --label combined-current-0eed34ff \
    --output-dir tmp/cuda-backend/combined-current-0eed34ff
```
