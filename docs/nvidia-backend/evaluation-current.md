# CUDA Current Evaluation Capture

This page summarizes the current paired A100/H200 CUDA backend capture from
commit `32744245`. The raw JSON, Markdown, and SVG reports are generated
locally under `tmp/cuda-backend/` and intentionally remain uncommitted.

The capture uses `nvcc` for target-specific PTX on both machines:

- A100: `compute_80`
- H200: `compute_90`
- tensor descriptor: `8x4x12`
- sizes: `1024,65536,1048576`
- repeats: `3`
- batch tasks: `2,6,12`
- worker blocks per task: `32,64,128,256`

## Artifact Paths

- `tmp/cuda-backend/a100-current-32744245/cuda-benchmark.json`
- `tmp/cuda-backend/a100-current-32744245/cuda-benchmark.md`
- `tmp/cuda-backend/h200-current-32744245/cuda-benchmark.json`
- `tmp/cuda-backend/h200-current-32744245/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-32744245/cuda-benchmark.json`
- `tmp/cuda-backend/combined-current-32744245/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-32744245/cuda-benchmark.svg`
- `tmp/cuda-backend/combined-current-32744245/cuda-benchmark-ratios.svg`

## Launch Baselines

CUDA Graph replay remains a useful phase-one host-launch baseline. It is
faster than PTO host scheduling for most captured rows, but it is still
host-owned replay rather than a device-side scheduler.

The `pto_host_schedule_compiler` row validates the new task-body compiler
path. It uses the same host runtime path as `pto_host_schedule`, but the PTX
comes from `KernelCompiler(platform="cuda").compile_cuda_host_schedule(...)`
and the shared task wrapper generator.

| GPU | N | PTO host ns | Compiler ns | Driver ns | Graph ns | Compiler/PTO | Graph/PTO |
| --- | - | ----------- | ----------- | --------- | -------- | ------------ | --------- |
| A100 | 1024 | 33792 | 35840 | 36864 | 19455 | 1.06x | 0.58x |
| A100 | 65536 | 25440 | 29920 | 29472 | 28704 | 1.18x | 1.13x |
| A100 | 1048576 | 25728 | 24992 | 41919 | 24863 | 0.97x | 0.97x |
| H200 | 1024 | 35488 | 36736 | 28799 | 23840 | 1.04x | 0.67x |
| H200 | 65536 | 24448 | 24128 | 36191 | 27264 | 0.99x | 1.12x |
| H200 | 1048576 | 23488 | 24896 | 30656 | 17888 | 1.06x | 0.76x |

The compiler row is within the same launch-latency band as the handwritten
host-schedule PTX. That is the important signal for this slice: the shared
task-body wrapper path can feed the existing host runtime without changing
the launch ABI or adding a separate generated-kernel calling convention.

## Worker Grid Rows

The plain one-block persistent batch rows are still too serial for larger
vectors. The worker-grid row is the current useful persistent-device
throughput slice because it keeps one persistent launch but assigns multiple
worker blocks to each task descriptor.

| GPU | N | Tasks | Best worker blocks/task | Device ns | Vs host batch |
| --- | - | ----- | ----------------------- | --------- | ------------- |
| A100 | 1024 | 2 | 64 | 34816 | 0.79x |
| A100 | 1024 | 6 | 256 | 43008 | 0.42x |
| A100 | 1024 | 12 | 64 | 29696 | 0.19x |
| A100 | 65536 | 2 | 256 | 29696 | 0.77x |
| A100 | 65536 | 6 | 128 | 23552 | 0.34x |
| A100 | 65536 | 12 | 128 | 29696 | 0.27x |
| A100 | 1048576 | 2 | 256 | 27648 | 0.70x |
| A100 | 1048576 | 6 | 128 | 40960 | 0.55x |
| A100 | 1048576 | 12 | 64 | 61440 | 0.45x |
| H200 | 1024 | 2 | 32 | 35296 | 0.94x |
| H200 | 1024 | 6 | 64 | 35776 | 0.36x |
| H200 | 1024 | 12 | 128 | 37152 | 0.22x |
| H200 | 65536 | 2 | 32 | 22688 | 0.74x |
| H200 | 65536 | 6 | 32 | 23392 | 0.31x |
| H200 | 65536 | 12 | 128 | 22304 | 0.20x |
| H200 | 1048576 | 2 | 256 | 20640 | 0.52x |
| H200 | 1048576 | 6 | 128 | 30176 | 0.43x |
| H200 | 1048576 | 12 | 256 | 40096 | 0.36x |

The strongest large-vector launch-amortization rows are still the 12-task
rows: `0.45x` on A100 and `0.36x` on H200 at `N=1048576`. The best
worker-block count is
not monotonic, so these rows support a tunable policy rather than a fixed
default.

## Persistent DAG Shapes

The DAG rows validate the persistent-device scheduler path rather than equal
work throughput. Chain and reuse add dependency levels and extra arithmetic.
The tensor row replaces one elementwise task with tiled GEMM work, so its
large-vector ratio is expected to be several times slower than the simple DAG.

| GPU | N | Chain/DAG | Reuse/DAG | Tensor/DAG |
| --- | - | --------- | --------- | ---------- |
| A100 | 1024 | 1.21x | 1.36x | 1.32x |
| A100 | 65536 | 1.78x | 1.79x | 3.84x |
| A100 | 1048576 | 1.80x | 1.71x | 4.18x |
| H200 | 1024 | 1.17x | 1.30x | 1.28x |
| H200 | 65536 | 1.76x | 1.78x | 2.89x |
| H200 | 1048576 | 1.80x | 1.79x | 3.03x |

The key correctness signal is that all DAG variants use generated dispatch
and runtime graph descriptors without changing the persistent launch path.
The tensor row also proves the descriptor metadata path for non-square
`8x4x12` tiles.

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
    tmp/cuda-backend/a100-current-32744245/cuda-benchmark.json \
    tmp/cuda-backend/h200-current-32744245/cuda-benchmark.json \
    --label combined-current-32744245 \
    --output-dir tmp/cuda-backend/combined-current-32744245
```
