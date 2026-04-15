# Manual Scope V0 Design

Date: 2026-04-15

Branch: `manual_scope_v0`

Base: current branch fork point `e3e4bd5`

Fetched `upstream/main` while writing this document: `6800c38`

## Goal

Add a lighter manual-scope mode to `a2a3/tensormap_and_ringbuffer` that:

- keeps the same submit API shape as AUTO mode
- does not introduce a separate manual submit API family
- does not support delayed dependency wiring
- publishes tasks at submit time, like AUTO mode
- allows explicit same-scope task ordering without relying entirely on
  TensorMap rediscovery

This is intentionally smaller than the previous manual-scope branch.

## Constraints

The v0 branch must follow these rules:

1. Use the same submit API as AUTO mode.
2. Append explicit dependencies into `Arg`, for example `args.add_dep(task_id)`.
3. Do not support delayed wiring and delayed linking.
4. Publish at submit time, same as AUTO mode.
5. Treat tensor allocation as a task for manual dependency building.
6. Determine whether TensorMap lookup is required from tensor scope metadata
   first, not from ring id.

## Non-goals

- no nested manual scopes in v0
- no post-submit `add_dependency(...)`
- no delayed explicit-edge replay or scope-end linking
- no batch publish barrier at manual `scope_end()`
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

### Explicit dependencies

`Arg` grows explicit dependency support:

```cpp
Arg args;
args.add_input(...);
args.add_dep(task_id);
```

Rules:

- `Arg.add_dep(...)` is valid only inside `PTO2_SCOPE(PTO2ScopeMode::MANUAL)`
- explicit deps must point to tasks created in the current manual scope
- explicit deps are attached to the consumer before submit
- no delayed wiring after submit

### Tensor alloc

`alloc_tensors(...)` stays output-only, but returns a producer task id:

```cpp
auto alloc = alloc_tensors(create_info0, create_info1);
// alloc.task_id
// alloc.outputs
```

This allows later manual tasks to depend on the allocation task explicitly.

`alloc_tensors(...)` itself does not accept `Arg.add_dep(...)`.

## Runtime Model

### High-level behavior

Manual mode in v0 is:

- AUTO-style submit and publish
- plus explicit deps from `Arg`
- plus reduced TensorMap work for current-manual-scope-local tensors

There is no hidden manual subgraph and no delayed publish.

### Submit-time flow

Inside manual scope, submit should do this:

1. Allocate the task slot and task id.
2. Read explicit deps from `Arg`.
3. Validate that explicit deps belong to the current manual scope.
4. Turn explicit deps into ordinary fanins immediately.
5. Classify tensor args as current-manual-scope-local or boundary.
6. Skip TensorMap lookup for manual-local producer-flow tensors only when the
   explicit task ids already cover the producing tasks.
7. Keep TensorMap lookup/insert for modifier tensors (`INOUT`,
   `OUTPUT_EXISTING`) because v0 does not expose a returned task id for
   zero-output updater tasks.
8. Keep normal creator-retention and TensorMap behavior for boundary tensors.
9. Publish the task immediately, same as AUTO mode.

### Scope-end behavior

`scope_end()` should keep only normal scope-lifetime behavior.

It should not do any of the old manual-specific work:

- no deferred explicit-edge linking
- no explicit-edge replay
- no batch publish of manual tasks

## Metadata Model

Ring id is not the primary locality test in v0.

Each produced task/tensor should carry scope metadata:

- `producer_scope_depth`
- `producer_manual_scope_depth`

For v0, nested manual scopes are still rejected, but storing
`producer_manual_scope_depth` now gives a clean upgrade path later for
distinguishing outer-manual-scope tensors.

External tensors use invalid producer scope metadata.

## Tensor Lookup Rule

When a tensor is used inside manual scope:

### Manual-local tensor

Treat a tensor as manual-local only when:

- it was produced in the current manual scope
- its stored producer manual-scope depth matches the current manual-scope depth

Behavior:

- for producer-flow inputs, explicit task ids are the primary ordering source
- if the producing task is already covered by `Arg.add_dep(...)`, lookup can
  skip the manual-local creator / modifier rediscovery
- modifier tensors (`INOUT`, `OUTPUT_EXISTING`) still publish through
  TensorMap so later updates can chain correctly

### Boundary tensor

Treat everything else as boundary:

- external tensors
- tensors from AUTO scope
- tensors from outer scopes
- tensors from outer manual scopes

Behavior:

- keep creator retention
- keep normal TensorMap lookup/insert behavior unless `manual_dep=true`

This is intentionally conservative.

## `manual_dep=true`

`manual_dep=true` keeps its existing meaning:

- skip TensorMap lookup/insert for that tensor
- keep creator retention through task ownership metadata

It is orthogonal to manual scope.

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

## Important Limitation

V0 cannot express every same-scope chain only with `Arg.add_dep(...)`.

Reason:

- `alloc_tensors(...)` returns a task id because it returns materialized outputs
- normal producer tasks also expose task ids through their returned outputs
- a zero-output updater does not expose a returned task-id handle in
  `TaskOutputTensors`

Practical consequence:

- explicit deps are enough for `qk -> softmax -> pv -> update`
- they are not enough to express `update(i) -> update(i+1)` when the updater
  only modifies existing tensors
- v0 therefore keeps TensorMap publication / lookup for modifier tensors even
  inside manual scope

## Validation Plan

### Unit / behavior

- reject `Arg.add_dep(...)` outside manual scope
- reject invalid task ids in manual scope
- reject explicit deps that are not from the current manual scope
- reject nested manual scopes
- verify manual-local tensors skip TensorMap lookup when explicit deps are
  present
- verify boundary tensors still use TensorMap

### Allocation behavior

- verify `alloc_tensors(...)` returns `{task_id, outputs}`
- verify manual tasks can depend on alloc task ids

### Regression / examples

- paged attention partial-manual example rewritten to `Arg.add_dep(...)`
- compare against AUTO and `aicpu_build_graph`
- verify correctness against golden outputs

## Validation Status

This document now reflects the rebased v0 state actually validated on
2026-04-15.

### Automated checks

- C++ UT passed:
  - `test_a2a3_pto2_manual_scope_api`
  - `test_a2a3_pto2_manual_scope_runtime`
- These were rerun on the rebased branch before the device pass.

### Real-device golden checks

Environment:

- platform: `a2a3`
- device: `3`
- `PTO_ISA_ROOT=/data/uvxiao/pto-runtime/build/pto-isa`

Fresh hardware reruns passed for all four rebased paged-attention paths:

- `examples/a2a3/tensormap_and_ringbuffer/paged_attention`
  - `Case1`
  - `Case2`
- `examples/a2a3/tensormap_and_ringbuffer/paged_attention_manual_scope`
  - `Case1`
  - `Case2`
- `examples/a2a3/tensormap_and_ringbuffer/paged_attention_unroll`
  - `Case1`
  - `Case2`
- `examples/a2a3/tensormap_and_ringbuffer/paged_attention_unroll_manual_scope`
  - `Case1`
  - `Case2`

### What broke during rebased validation

`paged_attention_unroll_manual_scope` `Case2` initially failed on hardware.

Root cause:

- the example uses repeated modifier tensors (`mi_update`, `li_update`, `oi`,
  `out_view`) across multiple update iterations
- v0 explicit deps can describe `qk -> softmax -> pv -> update`
- v0 cannot directly describe `update(i) -> update(i+1)` when `update` returns
  no output tensor handle and therefore no returned task id
- the runtime was skipping TensorMap publication / lookup too aggressively for
  manual-local tensors once any explicit dep existed

Fix:

- keep manual-scope producer-flow fast paths
- but preserve TensorMap publication / lookup for modifier tensors
  (`INOUT`, `OUTPUT_EXISTING`) so repeated updates still serialize correctly

## Rebased Benchmark Status

### Method

This benchmark block is the authoritative rebased batch for the current branch.
Older pre-rebase numbers were removed because they are no longer comparable.

- commit under test: `b9deae0`
- platform: `a2a3`
- device: `3`
- rounds: `30`
- aggregation: trimmed average, dropping `10` low and `10` high rounds
- runner:
  - scene-test entrypoint for `paged_attention`
  - `run_example.py` for `*_manual_scope` and `*_unroll*`
- benchmark mode: `--skip-golden`
- timing source: device log parsing with the same `orch_start/orch_end` logic
  as `tools/benchmark_rounds.sh`

### Results

| Example | Case | Auto Elapsed Trim (us) | Auto Orch Trim (us) | Manual Elapsed Trim (us) | Manual Orch Trim (us) | Manual Delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `paged_attention` | `Case1` | 127.2 | 112.8 | 125.6 | 110.9 | -1.6 us (-1.3%) |
| `paged_attention` | `Case2` | 146.2 | 126.9 | 136.2 | 118.1 | -10.0 us (-6.8%) |
| `paged_attention_unroll` | `Case1` | 1137.1 | 784.2 | 1141.5 | 795.4 | +4.4 us (+0.4%) |
| `paged_attention_unroll` | `Case2` | 521.8 | 322.4 | 525.8 | 330.8 | +4.0 us (+0.8%) |

### Reading The Batch

- Non-unroll manual scope is slightly faster than auto in this rebased batch.
- Unroll manual scope is still slightly slower than auto, and the gap is mainly
  orchestration (`+11.2us` / `+8.4us` on orch trim).
- The correctness fix for repeated modifier tensors is not free. It restores
  TensorMap work on the modifier path, which is exactly where the unroll cases
  still pay a small penalty.

## Rebased Runtime Change

### Kept change

| Change | Files / Functions Touched | Why It Matters | Observed Effect |
| --- | --- | --- | --- |
| Keep TensorMap for manual-scope modifier tensors | `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp` `pto2_submit_mixed_task()` | V0 has no returned task-id handle for zero-output updater tasks. Without TensorMap publication / lookup on `INOUT` / `OUTPUT_EXISTING`, repeated same-scope updates are under-constrained even when producer-flow deps are explicit. | Fixed the rebased hardware failure in `paged_attention_unroll_manual_scope` `Case2`. Current batch shows correctness restored across all eight device checks; performance remains slightly slower than auto on unroll because modifier tensors still pay TensorMap cost. |

### Removed stale history

The earlier optimization ledger in this file referred to:

- pre-rebase commits
- reverted runtime changes
- benchmark batches gathered with different runners and different workloads

Those entries were removed instead of being carried forward as historical noise.

## Risks

1. Manual scope still relies on orchestration authors to supply producer-flow
   explicit deps correctly.
2. V0 cannot fully replace TensorMap for modifier chains when the updater task
   does not return an output-backed task id.
3. Wrong scope metadata would still cause incorrect locality classification.
4. The lighter API remains intentionally less expressive than the old
   delayed-wiring design.

## Recommendation

Keep v0 narrow and explicit:

- same submit APIs as AUTO
- `Arg.add_dep(task_id)` only inside manual scope
- no delayed wiring
- no delayed linking
- immediate publish
- alloc returns `task_id`
- scope-depth metadata decides locality
- modifier tensors keep TensorMap chaining until the API can expose zero-output
  updater task ids in a lighter way

This is the smallest rebased design that is functionally correct on the current
hardware batch. The next optimization work should focus on reducing the
remaining unroll orchestration gap without widening the API again.
