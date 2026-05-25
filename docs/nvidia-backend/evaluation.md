# CUDA Backend Evaluation Notes

This page summarizes the current evaluation evidence for the CUDA backend.
The measurements are early runtime microbenchmarks, not end-to-end LLM serving
results. They are shaped by the VDCores and MPK papers only at the evaluation
structure level: fixed GPU work, repeated problem sizes, selected launch
baselines, local A100 runs, and remote H200 runs.

The latest captured raw reports are under `tmp/`:

- `tmp/cuda-backend/a100-batch-66c83aba/cuda-benchmark.md`
- `tmp/cuda-backend/h200-batch-66c83aba/cuda-benchmark.md`
- `tmp/cuda-backend/combined-batch-66c83aba/cuda-benchmark.md`
- `tmp/cuda-backend/combined-batch-66c83aba/cuda-benchmark.svg`

The data was captured from commit `66c83aba`.

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

Ratios are relative to the matched host-schedule row for the same GPU, vector
length, and task count. For batch rows, the reference is
`pto_host_schedule_batch`, not the one-task `pto_host_schedule` row.

## Headline Results

| GPU | N | `pto_host_schedule_batch` ns | `persistent_device_batch` | `persistent_queue_batch` |
| --- | - | ---------------------------- | ------------------------- | ------------------------ |
| A100 | 1024 | 91136 | 0.55x | 0.54x |
| H200 | 1024 | 71232 | 0.47x | 0.57x |
| A100 | 1048576 | 73568 | 17.29x | 16.98x |
| H200 | 1048576 | 86528 | 14.31x | 14.27x |

The small-vector rows show launch-amortization benefit from the persistent
paths. The large-vector rows expose the current tracer-bullet limitation:
batch rows match descriptor count, not intra-task grid shape. The persistent
executor currently uses one worker block per descriptor, while
`pto_host_schedule` vector-add uses a full grid.

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
    --include-persistent --batch-tasks 6 \
    --label a100-batch-$(git rev-parse --short HEAD) \
    --output-dir tmp/cuda-backend/a100-batch-$(git rev-parse --short HEAD)
```

Remote H200:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd /data/shibizhao/pto-cu && git pull --ff-only && \
   PYTHONPATH=$PWD:$PWD/python \
   python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
     --device 0 --sizes 1024,1048576 --repeats 3 --arch compute_90 \
     --include-persistent --batch-tasks 6 \
     --label h200-batch-$(git rev-parse --short HEAD) \
     --output-dir tmp/cuda-backend/h200-batch-$(git rev-parse --short HEAD)'
```

Merge reports:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --merge-json tmp/cuda-backend/a100-batch-66c83aba/cuda-benchmark.json \
    tmp/cuda-backend/h200-batch-66c83aba/cuda-benchmark.json \
    --label cuda-batch-a100-h200-66c83aba \
    --output-dir tmp/cuda-backend/combined-batch-66c83aba
```

## Next Evaluation Gaps

- Add a persistent worker-grid variant so large-vector rows compare similar
  intra-task parallelism.
- Add a higher-level task graph workload beyond vector add once the runtime
  ABI is stable enough.
