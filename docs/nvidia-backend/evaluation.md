# CUDA Backend Evaluation Notes

This page summarizes the current evaluation evidence for the CUDA backend.
The measurements are early runtime microbenchmarks, not end-to-end LLM serving
results. They are shaped by the VDCores and MPK papers only at the evaluation
structure level: fixed GPU work, repeated problem sizes, selected launch
baselines, local A100 runs, and remote H200 runs.

The latest captured raw reports are under `tmp/`:

- `tmp/cuda-backend/a100-grid-37d75192/cuda-benchmark.md`
- `tmp/cuda-backend/h200-grid-37d75192/cuda-benchmark.md`
- `tmp/cuda-backend/combined-grid-37d75192/cuda-benchmark.md`
- `tmp/cuda-backend/combined-grid-37d75192/cuda-benchmark.svg`

The data was captured from commit `37d75192`.

## Current Baselines

- `direct_driver`: thin CUDA Driver API launch path for the same vector-add
  PTX kernel.
- `pto_host_schedule`: PTO CUDA host runtime C API and manifest dispatch.
- `pto_persistent_device`: descriptor-array persistent executor.
- `pto_persistent_queue`: scheduler block plus bounded device ring queue.
- `pto_persistent_dag`: generated-dispatch-like task selection with fan-in
  counters.
- `*_batch`: same-work rows with six vector-add task descriptors. These rows
  compare repeated host launches with one persistent launch over the same
  descriptor count.
- `pto_persistent_device_grid_batch`: direct persistent-device batch row with
  four CUDA worker blocks assigned to each task descriptor.

Ratios are relative to the matched host-schedule row for the same GPU, vector
length, and task count. For batch rows, the reference is
`pto_host_schedule_batch`, not the one-task `pto_host_schedule` row.

## Headline Results

| GPU | N | `pto_host_schedule_batch` ns | `persistent_device_batch` | `persistent_grid_batch` | `persistent_queue_batch` |
| --- | - | ---------------------------- | ------------------------- | ----------------------- | ------------------------ |
| A100 | 1024 | 81920 | 0.62x | 0.53x | 0.57x |
| H200 | 1024 | 75072 | 0.49x | 0.51x | 0.57x |
| A100 | 1048576 | 73248 | 17.28x | 4.85x | 16.83x |
| H200 | 1048576 | 90272 | 13.74x | 3.60x | 13.66x |

The small-vector rows show launch-amortization benefit from the persistent
paths. The large-vector rows show why the worker-grid variant matters: giving
each descriptor four worker blocks reduces the A100 large-vector direct
persistent row from `17.28x` to `4.85x` versus the matched host-schedule batch
row, and reduces the H200 row from `13.74x` to `3.60x`. It is still not a
full parity comparison with `pto_host_schedule`, which launches a full grid
per vector-add task.

## PTX Sources

The A100 rows compiled PTX with local `nvcc` for `compute_80`. The H200 rows
compiled PTX with remote `nvcc` for `compute_90`, discovered from the
`/usr/local/cuda*` toolkit path. The report still marks embedded PTX rows when
fallback PTX is used, but the latest H200 report does not use that fallback.

## Reproduction Commands

Local A100:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --device 0 --sizes 1024,1048576 --repeats 3 --arch compute_80 \
    --include-persistent --batch-tasks 6 --worker-blocks-per-task 4 \
    --label a100-grid-$(git rev-parse --short HEAD) \
    --output-dir tmp/cuda-backend/a100-grid-$(git rev-parse --short HEAD)
```

Remote H200:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd /data/shibizhao/pto-cu && git pull --ff-only && \
   PYTHONPATH=$PWD:$PWD/python \
   python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
     --device 0 --sizes 1024,1048576 --repeats 3 --arch compute_90 \
     --include-persistent --batch-tasks 6 --worker-blocks-per-task 4 \
     --label h200-grid-$(git rev-parse --short HEAD) \
     --output-dir tmp/cuda-backend/h200-grid-$(git rev-parse --short HEAD)'
```

Merge reports:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --merge-json tmp/cuda-backend/a100-grid-37d75192/cuda-benchmark.json \
    tmp/cuda-backend/h200-grid-37d75192/cuda-benchmark.json \
    --label cuda-grid-a100-h200-37d75192 \
    --output-dir tmp/cuda-backend/combined-grid-37d75192
```

## Next Evaluation Gaps

- Sweep `--worker-blocks-per-task` to find a better large-vector point before
  treating the grid row as a tuned baseline.
- Add a higher-level task graph workload beyond vector add once the runtime
  ABI is stable enough.
