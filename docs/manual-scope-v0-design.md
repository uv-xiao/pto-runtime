# Manual Scope V0 Design

Date: 2026-04-15

Branch: `manual_scope_v0`

## Goal

Add a lighter manual-scope mode to `a2a3/tensormap_and_ringbuffer` that:

- keeps the same submit API shape as AUTO mode
- does not introduce a separate manual submit API family
- does not support delayed dependency wiring
- publishes tasks at submit time, like AUTO mode
- allows explicit same-scope task ordering without using TensorMap for
  current-manual-scope-local tensors

This is intentionally smaller than the previous manual-scope branch.

## Constraints

The v0 branch must follow these rules:

1. Use the same submit API as AUTO mode.
2. Append explicit dependencies into `Arg`, for example `args.add_dep(task_id)`.
3. Do not support delayed wiring and delayed linking.
4. Publish at submit time, same as AUTO mode.
5. Treat tensor allocation as a task for manual dependency building.
6. Keep manual-scope state minimal: only track the scope depth at which manual
   mode begins.

## Non-goals

- no nested manual scopes in v0
- no post-submit `add_dependency(...)`
- no delayed explicit-edge replay or scope-end linking
- no batch publish barrier at manual `scope_end()`
- no implicit TensorMap fallback for current-manual-scope-local tensors
- no attempt to redesign AUTO mode

## User-Facing API

### Scope

Manual mode remains an explicit scope:

```cpp
PTO2_SCOPE(PTO2ScopeMode::MANUAL) {
    ...
}
```

Nested manual scopes are rejected.

### Submit API

Keep the existing AUTO-mode submit entry points:

```cpp
auto out = pto2_rt_submit_aic_task(FUNC_ID, args);
auto out = pto2_rt_submit_aiv_task(FUNC_ID, args);
auto out = pto2_rt_submit_task(mixed_kernels, args);
```

No `*_manual(...)` or `*_manual_with_deps(...)` APIs in v0.

Each submit result should carry:

- a standalone `task_id`
- zero or more materialized output tensors

The returned task id must not depend on whether the task created any new
`OUTPUT` tensors.

### Explicit dependencies

`Arg` grows explicit dependency support:

```cpp
Arg args;
args.add_input(...);
args.add_dep(task_id);
```

Rules:

- `Arg.add_dep(...)` may be used by any consumer that needs an explicit edge
- when the consumer is inside manual scope, explicit deps must point to tasks
  from the current top scope
- when the consumer is outside manual scope, explicit deps may still be used to
  express boundary edges from earlier producers
- explicit deps are attached to the consumer before submit
- no delayed wiring after submit

### Tensor alloc

`alloc_tensors(...)` stays output-only, but returns a producer task id:

```cpp
auto alloc = alloc_tensors(create_info0, create_info1);
// alloc.task_id()
// alloc.get_ref(...)
```

This allows later manual tasks to depend on the allocation task explicitly.

`alloc_tensors(...)` itself does not accept `Arg.add_dep(...)`.

## Runtime Model

### High-level behavior

Manual mode in v0 is:

- AUTO-style submit and publish
- plus explicit deps from `Arg`
- plus full TensorMap bypass for current-manual-scope-local tensors

There is no hidden manual subgraph and no delayed publish.

### Submit-time flow

Submit should do this:

1. Allocate the task slot and task id.
2. Read explicit deps from `Arg`.
3. Validate explicit deps:
   - for an in-manual-scope consumer: dep task ids must belong to the current
     top scope
   - for an outside-manual-scope consumer: dep task ids must resolve to valid
     producer tasks
4. Turn explicit deps into ordinary fanins immediately.
5. While manual mode is active, skip TensorMap lookup and TensorMap insert on
   the submit path.
6. Treat explicit task ids as the ordering source for manual-scope-local work.
7. Keep creator retention through existing `owner_task_id` metadata on tensors.
8. For boundary cases, allow explicit deps to carry ordering that TensorMap
   bypass would otherwise lose.
9. Publish the task immediately, same as AUTO mode.

### Scope-end behavior

`scope_end()` should keep only normal scope-lifetime behavior.

It should not do any of the old manual-specific work:

- no deferred explicit-edge linking
- no explicit-edge replay
- no batch publish of manual tasks

## Scope-State Model

V0 does not need per-tensor or per-task scope metadata.

The runtime only needs one manual-specific state value:

- `manual_begin_depth`

Meaning:

- default value is `PTO2_MAX_SCOPE_DEPTH`
- when `scope_stack_top < manual_begin_depth`, runtime is in AUTO behavior
- when `scope_stack_top >= manual_begin_depth`, runtime is in MANUAL behavior

Important details:

- nested `AUTO` scopes under an active manual scope still execute with manual
  submit-path behavior, because the active depth is still
  `scope_stack_top >= manual_begin_depth`
- even in that case, `Arg.add_dep(...)` remains constrained to the current top
  scope, not any earlier scope inside the active manual region
- once the consumer is outside manual scope, `Arg.add_dep(...)` is still
  allowed for explicit boundary edges

This is enough because v0 rejects nested manual scopes and does not support
delayed wiring/linking. The older metadata-heavy state from the previous branch
is intentionally removed:

- no tensor `producer_scope_depth`
- no tensor `producer_manual_scope_depth`
- no per-slot `scope_depth`
- no per-slot `scope_epochs`

## Tensor Dependency Rule

When a task is submitted inside manual scope:

- the runtime does not use TensorMap lookup
- the runtime does not use TensorMap insert
- explicit `Arg.add_dep(...)` edges are the dependency source for manual-scope
  ordering
- existing tensor `owner_task_id` still provides creator retention for tensors
  that already carry an owning task

This is the current v0 implementation model after removing the earlier
metadata-based locality check.

When a task is submitted outside manual scope:

- normal AUTO creator-retention and TensorMap behavior still applies
- explicit `Arg.add_dep(...)` may still be used for boundary edges from earlier
  producers, including producers created inside manual scope

## `manual_dep=true`

`manual_dep=true` keeps its existing meaning:

- skip TensorMap lookup/insert for that tensor
- keep creator retention through task ownership metadata

It is orthogonal to manual scope.

Inside manual scope, `manual_dep=true` is mostly redundant because the manual
submit path already bypasses TensorMap. Outside manual scope, it keeps its
original creator-only meaning.

## Representation Change From Previous Design

V0 intentionally narrows manual scope.

Previous heavier direction:

- submit first
- wire later
- link later
- publish later

V0:

- consumer must know explicit deps at submit time
- no post-submit dependency wiring
- no delayed linking
- no delayed publish

This means manual scope in v0 is not a general explicit-graph construction API.
It is a lighter explicit-dependency annotation on top of normal submit.

## Practical Example

```cpp
PTO2_SCOPE(PTO2ScopeMode::MANUAL) {
    Arg qk = make_qk_args(...);
    auto qk_out = pto2_rt_submit_aic_task(FUNC_QK_MATMUL, qk);

    Arg sf = make_sf_args(qk_out.outputs.tensor(0), ...);
    sf.add_dep(qk_out.task_id);
    auto sf_out = pto2_rt_submit_aiv_task(FUNC_SOFTMAX_PREPARE, sf);

    Arg pv = make_pv_args(sf_out.outputs.tensor(0), ...);
    pv.add_dep(sf_out.task_id);
    auto pv_out = pto2_rt_submit_aic_task(FUNC_PV_MATMUL, pv);

    Arg up = make_update_args(...);
    up.add_dep(sf_out.task_id);
    up.add_dep(pv_out.task_id);
    (void)pto2_rt_submit_aiv_task(FUNC_ONLINE_UPDATE, up);
}
```

This keeps the orchestration shape readable while avoiding a separate manual
submit API family.

For repeated zero-output updater chains, orchestration must still carry the
chain explicitly:

```cpp
PTO2TaskId prev_update = PTO2TaskId::invalid();

for (...) {
    Arg up = make_update_args(...);
    if (prev_update.is_valid()) {
        up.add_dep(prev_update);
    }
    TaskOutputTensors update_out = pto2_rt_submit_aiv_task(FUNC_ONLINE_UPDATE, up);
    prev_update = update_out.task_id();
}
```

This is the intended v0 use of the direct submit-result task id after removing
manual-local TensorMap fallback.

## Important Limitation

V0 still rejects nested manual scopes and still requires explicit deps to be
known at submit time.

With a standalone task id in `TaskOutputTensors`, a zero-output updater can
still be named by later `Arg.add_dep(...)` calls. That keeps manual-local
dependency expression explicit without relying on TensorMap fallback.

## Validation Plan

### Unit / behavior

- reject invalid task ids in manual scope
- reject explicit deps that are not from the current manual scope
- reject nested manual scopes
- verify manual-local tensors skip both TensorMap lookup and TensorMap insert
- verify boundary tensors still use TensorMap
- allow `Arg.add_dep(...)` outside manual scope for boundary edges from a
  manual-scope producer to a later outside-manual consumer

### Allocation behavior

- verify `alloc_tensors(...)` returns `{task_id, outputs}`
- verify manual tasks can depend on alloc task ids

### Regression / examples

- add explicit-manual variants for the two paged-attention workloads:
  `paged_attention_manual_scope` and `paged_attention_unroll_manual_scope`
- keep the existing AUTO paths as the comparison baseline
- verify `Case1` and `Case2` against golden outputs on real hardware
- benchmark AUTO vs manual-scope with the same 30-round trimmed-average method
  used by `tools/benchmark_rounds.sh`

## Validation Status

This document reflects the cleaned `manual_scope_v0` branch state. The latest
real-device validation below was refreshed on 2026-04-20, including direct
submit-result task ids, manual-local TensorMap bypass, example-side explicit
updater chaining, and the simplified manual-scope state model.

### Automated checks

- C++ UT passed:
  - `test_a2a3_pto2_manual_scope_api`
  - `test_a2a3_pto2_manual_scope_runtime`
- simulation negative test passed:
  - `tests/st/a2a3/tensormap_and_ringbuffer/test_manual_scope_validation.py`
- rerun commands:
  - `ctest --test-dir tests/ut/cpp/build -R 'test_a2a3_pto2_manual_scope_(api|runtime)' --output-on-failure`
  - `python -m pytest tests/st/a2a3/tensormap_and_ringbuffer/test_manual_scope_validation.py --platform a2a3sim --device 0 -q`

### Real-device golden checks

Environment:

- platform: `a2a3`
- device: `9`
- PTO-ISA commit: `d96c8784`
- local package rebuilt with `pip install --no-build-isolation -e .`

Fresh hardware reruns passed for all four kept paged-attention paths:

- `examples/a2a3/tensormap_and_ringbuffer/paged_attention`
  - `Case1`
  - `Case2`
- `examples/a2a3/tensormap_and_ringbuffer/paged_attention_manual_scope`
  - `Case1`
  - `Case2`
- `tests/st/a2a3/tensormap_and_ringbuffer/paged_attention_unroll`
  - `Case1`
  - `Case2`
- `examples/a2a3/tensormap_and_ringbuffer/paged_attention_unroll_manual_scope`
  - `Case1`
  - `Case2`

## Fresh Golden Validation

This checkpoint validates output correctness separately from benchmark timing.
The benchmark table below still uses `--skip-golden` to avoid CPU golden
generation affecting measurement, so this validation must be kept as a separate
correctness gate.

- current validation tree: `2298653`
- platform: `a2a3`
- device: `9`
- PTO-ISA commit: `d96c8784`
- rounds: one golden-checked run per case
- result: all TMR AUTO and TMR manual-scope checked paths passed
  output-vs-golden comparison

Validated paths:

| Example | Mode | Case1 | Case2 |
| --- | --- | --- | --- |
| `paged_attention` | AUTO | PASS | PASS |
| `paged_attention_manual_scope` | manual scope | PASS | PASS |
| `paged_attention_unroll` | AUTO | PASS | PASS |
| `paged_attention_unroll_manual_scope` | manual scope | PASS | PASS |

ABG baseline checks were also run for context:

| Example | Mode | Case1 | Case2 |
| --- | --- | --- | --- |
| `paged_attention` | ABG | PASS | PASS |
| `paged_attention_unroll` | ABG | PASS | FAIL |

The ABG unroll `Case2` failure is a baseline correctness issue observed in the
comparison path, not a manual-scope failure. Its benchmark command can still run
with `--skip-golden`, but the timing should not be interpreted as a
correctness-clean baseline for that case.

## Fresh Benchmark Status

### Method

This benchmark block is the authoritative batch for the current aligned branch
state.

It includes:

- direct `TaskOutputTensors::task_id()` storage on submit results
- manual-local TensorMap lookup / insert bypass in the runtime
- explicit `update(i) -> update(i+1)` chaining in the two manual paged-attention
  examples
- last-commit alignment with the colleague patch, while keeping the negative
  allocator failure sentinel documented below

- current benchmark tree: `2298653`
- base commit before the example-side chaining fix: `3d36370`
- platform: `a2a3`
- device: `9`
- PTO-ISA commit: `d96c8784`
- rounds: `100`
- aggregation: trimmed average, dropping `10` low and `10` high rounds
- runner:
  - scene-test entrypoint for AUTO paths
  - `run_example.py` for manual-scope example paths
- benchmark mode: `--skip-golden`
- timing source: device log parsing with the same `orch_start/orch_end` logic
  as `tools/benchmark_rounds.sh`

Important reading rule:

- compare AUTO vs manual only within the same row
- do not compare `paged_attention` rows directly against
  `paged_attention_unroll` rows as if they were the same workload

### Results

| Example | Case | Auto Elapsed Trim (us) | Auto Orch Trim (us) | Manual Elapsed Trim (us) | Manual Orch Trim (us) | Elapsed Delta | Orch Delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `paged_attention` | `Case1` | 77.9 | 62.1 | 124.5 | 108.3 | +46.6 us (+59.8%) | +46.2 us (+74.4%) |
| `paged_attention` | `Case2` | 93.9 | 72.9 | 141.9 | 118.9 | +48.0 us (+51.1%) | +46.0 us (+63.1%) |
| `paged_attention_unroll` | `Case1` | 1135.8 | 762.2 | 1124.6 | 638.3 | -11.2 us (-1.0%) | -123.9 us (-16.3%) |
| `paged_attention_unroll` | `Case2` | 517.1 | 305.9 | 494.5 | 251.2 | -22.6 us (-4.4%) | -54.7 us (-17.9%) |

ABG context from the same run:

| Example | Case | ABG Elapsed Trim (us) | Golden Status |
| --- | --- | ---: | --- |
| `paged_attention` | `Case1` | 31625.5 | PASS |
| `paged_attention` | `Case2` | 16611.7 | PASS |
| `paged_attention_unroll` | `Case1` | 1384.1 | PASS |
| `paged_attention_unroll` | `Case2` | 675.7 | FAIL |

### Reading The Batch

- Non-unroll manual scope is still materially slower than AUTO in this fresh
  batch. The regression is almost entirely orchestration time.
- Unroll manual scope is now slightly faster than AUTO in total elapsed time
  and materially lower in orchestration time on both kept cases.
- The current benchmark batch runs successfully on hardware, but the non-unroll
  manual path still does not meet the earlier performance target.

## Why Raw Non-unroll Looks Faster Than Unroll

The raw elapsed columns above are not a fair cross-workload comparison.
`paged_attention` and `paged_attention_unroll` are intentionally very different
problem sizes.

### Case Shape Comparison

| Example | Case | Batch | Num Heads | Head Dim | Block Size | Context Len |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `paged_attention` | `Case1` | 1 | 16 | 16 | 16 | 33 |
| `paged_attention_unroll` | `Case1` | 256 | 16 | 128 | 128 | 8192 |
| `paged_attention` | `Case2` | 1 | 16 | 16 | 16 | 128 |
| `paged_attention_unroll` | `Case2` | 64 | 64 | 128 | 64 | 8192 |

So the unroll example is not a tuned version of the same small test. It is the
production-scale workload.

### Device-log Evidence

From the fresh benchmark logs on device `9`:

- `paged_attention Case1` submitted `13` tasks total
- `paged_attention_unroll Case1` submitted `1280` tasks total

These numbers match the orchestration structure:

- non-unroll `Case1`
  - `batch=1`
  - `bn_this_batch=ceil(33 / 16)=3`
  - `q_loop=1`
  - per batch: `1 alloc + 3 * 4 kernel tasks = 13`
- unroll `Case1`
  - `batch=256`
  - `bn_this_batch=ceil(8192 / 128)=64`
  - `N_UNROLL=64`, so one unrolled group per batch
  - per batch: `1 alloc + 4 grouped kernel tasks = 5`
  - total: `256 * 5 = 1280`

The important result is therefore not:

- "non-unroll is faster than unroll"

The important result is:

- unroll processes a vastly larger workload, yet its runtime does not scale
  remotely in proportion to raw work size
- unroll dramatically reduces orchestration cost per unit of work

One simple normalization is orchestration time per submitted task:

| Example | Case | Orch Trim (us) | Submitted Tasks | Orch per Task (us) |
| --- | --- | ---: | ---: | ---: |
| `paged_attention` | `Case1` | 59.2 | 13 | 4.55 |
| `paged_attention_unroll` | `Case1` | 774.1 | 1280 | 0.60 |

So unroll is not "worse" in the meaningful sense. Its absolute latency is
larger because the workload is massively larger, while its orchestration cost
per submitted task is much lower.

## Fresh TensorMap Profiling

This section is the required profiling checkpoint for the current branch.
Any optimization that touches the manual-scope runtime hot path or the manual
paged-attention orchestration should refresh both this section and the benchmark
table above with new real-device data.

### Method

- local profiling-only rebuild with:
  - `PTO2_ORCH_PROFILING=1`
  - `PTO2_TENSORMAP_PROFILING=1`
- platform: `a2a3`
- device: `9`
- PTO-ISA commit: `d96c8784`
- rounds: `30`
- mode: `--skip-golden`
- AUTO runner:
  - `python examples/a2a3/tensormap_and_ringbuffer/paged_attention/test_paged_attention.py -p a2a3 -d 9 -n 30 --case <Case> --skip-golden`
- manual runner:
  - `python examples/scripts/run_example.py -k examples/a2a3/tensormap_and_ringbuffer/paged_attention_manual_scope/kernels -g examples/a2a3/tensormap_and_ringbuffer/paged_attention_manual_scope/golden.py -p a2a3 -d 9 -c d96c8784 -n 30 --case <Case> --skip-golden`
- parsing:
  - per-round device-log `=== Orchestrator Profiling ===` blocks
  - per-round device-log `=== TensorMap Lookup Stats ===` blocks
  - trimmed average for time fields, mean for lookup / insert counts

### Results

| Case | Mode | Tasks | `lookup+dep` Trim (us) | `tensormap_ins` Trim (us) | TensorMap Lookups Avg | TensorMap Inserts Avg | Profiled Submit Trim (us) | Full Orch Trim (us) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Case1` | AUTO | 13 | 4.132 | 1.842 | 40.0 | 12.0 | 16.422 | 194.508 |
| `Case1` | MANUAL | 13 | 1.944 | 1.414 | 16.0 | 3.0 | 14.458 | 259.318 |
| `Case2` | AUTO | 33 | 6.320 | 2.638 | 105.0 | 32.0 | 24.368 | 210.274 |
| `Case2` | MANUAL | 33 | 2.598 | 1.728 | 41.0 | 8.0 | 21.560 | 285.182 |

### What The Numbers Prove

- The manual-local TensorMap bypass is working.
  - `Case1`: lookups dropped from `40.0` to `16.0` (`-60.0%`), inserts
    dropped from `12.0` to `3.0` (`-75.0%`), and `lookup+dep` time dropped
    from `4.132us` to `1.944us` (`-53.0%`).
  - `Case2`: lookups dropped from `105.0` to `41.0` (`-61.0%`), inserts
    dropped from `32.0` to `8.0` (`-75.0%`), and `lookup+dep` time dropped
    from `6.320us` to `2.598us` (`-58.9%`).
- The manual path still shows non-zero TensorMap traffic because boundary
  tensors still use TensorMap in v0. That is expected.
- The remaining non-unroll regression is no longer explained by TensorMap.
  - The profiled submit buckets are lower in manual mode
    (`-12.0%` in `Case1`, `-11.5%` in `Case2`).
  - But full orchestration time is still much higher
    (`+33.3%` in `Case1`, `+35.6%` in `Case2`).

### Why Full Orch Is Still Worse

The gap has moved out of the TensorMap buckets.

The current profiling points to two more likely hot regions:

1. Orchestration-side explicit-dep construction.
   The manual paged-attention orchestration adds many `Arg.add_dep(...)`
   calls and threads task ids explicitly through the loop body.
2. Runtime explicit-dep validation and dedupe before the first profiled phase.
   `pto2_submit_mixed_task()` validates every explicit dep against the current
   scope and deduplicates it before the first `alloc/sync/lookup/insert` lap is
   recorded.

So the next optimization target is no longer "remove more TensorMap work".
It is "make explicit-dep construction and validation cheaper".

### Code Pointers For The Current Design

- Current-manual-scope-local classification:
  - [pto_orchestrator.cpp](/data/uvxiao/pto-runtime/src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp#L143)
- Manual-local tensors bypass TensorMap lookup / insert here:
  - [pto_orchestrator.cpp](/data/uvxiao/pto-runtime/src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp#L742)
- Explicit deps are validated and deduplicated here:
  - [pto_orchestrator.cpp](/data/uvxiao/pto-runtime/src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp#L627)
- `Arg.add_dep(...)` storage is a simple append here:
  - [pto_types.h](/data/uvxiao/pto-runtime/src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_types.h#L228)
- The manual paged-attention example that exercises this path is here:
  - [paged_attention_orch.cpp](/data/uvxiao/pto-runtime/examples/a2a3/tensormap_and_ringbuffer/paged_attention_manual_scope/kernels/orchestration/paged_attention_orch.cpp#L131)

## Current Implementation Gap

The runtime and manual paged-attention examples now match the core v0 alignment
rules:

- `TaskOutputTensors` carries `task_id` directly, including for zero-output
  updater tasks
- manual-local tensors bypass TensorMap lookup and TensorMap insert in the
  submit path
- manual examples explicitly chain repeated zero-output updater tasks with
  `add_dep(prev_update_task)`

The remaining work is performance-focused:

- reduce non-unroll manual orchestration cost without reintroducing TensorMap
  fallback for manual-local tensors
- focus the next optimization round on explicit-dep construction / validation,
  not on TensorMap lookup / insert removal
- keep the unroll gains while tightening the small-workload path

### Colleague Patch Alignment Note

The colleague patch returned `{0, 0, nullptr, nullptr}` from
`PTO2TaskAllocator::alloc()` on allocator failure, while `failed()` still used
`task_id < 0`. This branch keeps the old negative failure sentinel:
`{-1, -1, nullptr, nullptr}`.

Reason: task id `0` is a valid first task id. If failure returns `task_id == 0`
but `failed()` checks `task_id < 0`, the first allocator failure can be treated
as a successful allocation with null packed-buffer pointers. Keeping the
negative sentinel preserves one invariant across A2A3 and A5:

```cpp
bool failed() const { return task_id < 0; }
```

## Historical Optimization Notes

The current PR intentionally does not carry the broader optimization branch.
Those attempts were removed from the code to keep the PR scope small, but the
history is still useful and should stay documented here as historical context.

These entries are not claims about the current branch state. They record what
was tried on the earlier heavier line of work and whether it was kept there,
reverted there, or later dropped from this PR scope.

### Historical Kept Attempts On The Earlier Branch

| Optimization | Files / Functions Touched | Historical Effect |
| --- | --- | --- |
| O(1) explicit-dep membership check | `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp` `pto2_get_current_scope_task_slot()` `pto2_submit_mixed_task()` `pto2_scope_begin()` `pto2_prepare_task()`; `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.h`; `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_ring_buffer.h` | Large win on the old branch because it removed a linear current-scope scan from each `add_dep(...)` validation. |
| Exact per-scope epoch validation | `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp` `pto2_get_current_scope_task_slot()` `pto2_scope_begin()` `pto2_prepare_task()`; `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.h`; `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_ring_buffer.h`; `tests/ut/cpp/test_a2a3_pto2_manual_scope_runtime.cpp` | Correctness fix that preserved the O(1) validation path while closing the stale-scope hole. |
| Skip redundant creator-retention when owner is already explicit | `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp` `pto2_submit_mixed_task()` `pto2_append_explicit_manual_deps()` | Reduced duplicate submit-path work when explicit deps already named the producer. |
| Canonicalize explicit dep ids once per submit | `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp` `pto2_submit_mixed_task()` | Small hotpath cleanup on the earlier branch. |
| Eager `start_offset` caching, remove hot-path recompute from payload init | `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/tensor.h` `Tensor::init()` `Tensor::init_with_view()` `Tensor::init_from_create_info()`; `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_runtime2_types.h` `PTO2TaskPayload::init()`; `tests/ut/cpp/test_a2a3_pto2_manual_scope_runtime.cpp` | Reduced the earlier branch's `param_copy` hotspot substantially, but this was later dropped from the PR to keep the diff tightly coupled to manual-scope v0 itself. |
| Cached allocator-state fast path before shared reload | `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_ring_buffer.h` `PTO2TaskAllocator::alloc()` `has_task_slot_capacity()` `try_alloc_with_last_alive()` | Reduced allocator overhead on the earlier branch, but this was also dropped from the PR because it is a general runtime optimization, not a manual-scope-v0-specific change. |

### Historical Reverted / Rejected Attempts

| Attempt | Files / Functions Touched | Historical Result |
| --- | --- | --- |
| Fold output owner metadata into `payload->init()` to save the post-init output loop | `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/tensor.h`; `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_runtime2_types.h`; `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp` | Reverted on the earlier branch. End-to-end gains were not reliable and the code became harder to reason about. |
| Change `out_view` in paged-attention update from `INOUT` to `NO_DEP` | earlier `paged_attention` orchestration files on the heavy branch | Reverted. Real-device golden failed. |
| Change `out_view` in paged-attention update from `INOUT` to `OUTPUT_EXISTING` | earlier `paged_attention` orchestration files on the heavy branch | Reverted. Real-device golden also failed. |

## Risks

1. Manual scope still relies on orchestration authors to supply producer-flow
   explicit deps correctly.
2. Wrong scope metadata would still cause incorrect locality classification.
3. The current non-unroll benchmark gap is large enough that performance should
   not be described as improved relative to AUTO.

## Recommendation

Keep v0 narrow and explicit:

- same submit APIs as AUTO
- `Arg.add_dep(task_id)` only inside manual scope
- no delayed wiring
- no delayed linking
- immediate publish
- alloc returns `task_id`
- submit results carry `task_id` directly, independent of outputs
- scope-depth metadata decides locality
- manual-local tensors bypass TensorMap entirely

This is the smallest v0 target design that keeps the API narrow. The current
branch is functionally correct on the current hardware batch, but it is not yet
fully aligned with the stricter "manual-local tensors bypass TensorMap
entirely" rule above. The next implementation work should close that gap first,
then focus on the non-unroll orchestration regression without widening the API
again.
