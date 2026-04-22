# Manual Scope V0 Design

Date: 2026-04-23

Branch checkpoint:

- local rebased head: `f19fa37`
- rebased onto `upstream/main`: `cb1a948`
- PR `#568` still points at older remote head `efab669`, so the PR page does not
  yet reflect this forward-ported checkpoint

## Goal

Keep manual scope small and explicit:

- same submit API family as AUTO mode
- explicit task-to-task deps via `Arg.add_dep(task_id)`
- no delayed wiring
- no delayed publish barrier at `scope_end()`
- publish at submit time, same as AUTO mode
- support allocation-as-task through `alloc_tensors(...).task_id()`

This is intentionally narrower than the older manual-dep branch.

## Constraints

The v0 design keeps these rules:

1. `PTO2_SCOPE(PTO2ScopeMode::MANUAL)` opts a scope into manual mode.
2. Nested manual scopes are rejected.
3. Manual deps are attached before submit through `Arg.add_dep(...)`.
4. No post-submit `add_dependency(...)` API exists.
5. `alloc_tensors(...)` remains output-only and returns a task id.
6. Scope handling stays close to upstream AUTO mode.

## User-Facing API

### Scope

```cpp
PTO2_SCOPE(PTO2ScopeMode::MANUAL) {
    ...
}
```

### Submit

Manual mode still uses the normal submit calls:

```cpp
auto out = pto2_rt_submit_aic_task(FUNC_ID, args);
auto out = pto2_rt_submit_aiv_task(FUNC_ID, args);
auto out = pto2_rt_submit_task(mixed_kernels, args);
```

There is no extra `*_manual(...)` submit family in v0.

### Explicit deps

```cpp
Arg args;
args.add_input(tensor);
args.add_dep(prev_task_id);
```

Rules:

- inside manual scope, `add_dep(...)` must refer to a task from the current
  top scope
- outside manual scope, `add_dep(...)` is still allowed for valid boundary
  edges from earlier producers
- explicit deps become ordinary fanins immediately at submit time

### Allocation task ids

```cpp
auto alloc = alloc_tensors(ci0, ci1, ci2);
PTO2TaskId alloc_tid = alloc.task_id();
const Tensor &tmp = alloc.get_ref(0);
```

This is required for manual-local tensors because the alloc task itself may be
the producer that later tasks must explicitly depend on.

## Runtime Model

### High-Level Behavior

Manual scope v0 is:

- AUTO-style submit and publish
- plus explicit fanins from `Arg.add_dep(...)`
- plus full TensorMap lookup / insert bypass for tasks submitted inside manual
  scope

It is not:

- a delayed graph builder
- a second runtime path with a manual-only publish barrier
- a TensorMap fallback mode for boundary dependencies

### Scope State

The runtime keeps two manual-scope-specific pieces of state:

- `manual_begin_depth`
- the current scope task list (`scope_tasks[...]`)

`manual_begin_depth` decides whether the current submit is in manual mode.
`scope_tasks[...]` is then used to validate that explicit deps inside manual
scope come from the current scope.

There is no per-tensor manual-local classification on the submit path. If the
consumer task is inside manual scope, that submit skips TensorMap lookup and
insert for all tensor args.

There is no delayed-link metadata, per-slot epoch replay, or scope-end explicit
dependency rebuild in v0.

### How Manual Scope Decides TensorMap Use

The rule is submit-scoped, not tensor-scoped:

- inside manual scope: skip the entire TensorMap lookup section
- inside manual scope: skip the entire TensorMap insert section
- outside manual scope: keep normal AUTO TensorMap behavior

Current helper usage:

- `pto2_find_task_slot_by_task_id(...)`
- `pto2_task_slot_is_in_current_scope(...)`

These helpers validate explicit dependency task ids. They do not classify
tensors for partial TensorMap fallback.

This matches the pre-rebase aligned PR head `efab669` and the colleague design
at `poursoul/simpler@dd76880`: manual-scope tasks express dependencies through
`Arg.add_dep(...)`, not through TensorMap lookup / insert.

### Dependency Handling

For a submitted task:

- explicit `add_dep(...)` edges are always checked first
- if the task is inside manual scope, TensorMap lookup / insert is skipped
- if the task is outside manual scope, normal AUTO TensorMap behavior is used

That gives the actual v0 dependency split:

- manual-local tensor producer -> manual-local consumer:
  explicit dep, no TensorMap lookup / insert
- external tensor -> manual consumer:
  no TensorMap lookup; the orchestration must add an explicit dep if an
  external producer task must be ordered before the manual consumer
- manual producer -> later external consumer:
  no manual-scope TensorMap insert was published; the later consumer must use
  `Arg.add_dep(...)` when it needs to order after the manual producer

### Scope End

`scope_end()` keeps only normal scope-lifetime behavior:

- call scheduler `on_scope_end(...)`
- pop scope bookkeeping
- clear `manual_begin_depth` when leaving the manual region

It does not:

- replay explicit deps
- batch publish manual tasks
- rebuild delayed edges

## Implementation Notes From The Rebased Cleanup

The current forward-port intentionally shrinks the runtime diff against
`upstream/main`:

- `pto_orchestrator.cpp`
  - restores the upstream-style allocator-driven `pto2_prepare_task(...)`
  - restores normal profiling lap recording
  - keeps the pre-rebase aligned rule that manual-scope submits skip TensorMap
    lookup / insert entirely
- `pto_types.h`
  - restores upstream variadic `Arg` helpers (`add_input`, `add_output`,
    `add_inout`, `add_no_dep`, `add_scalar`)
  - keeps only the v0-specific explicit dep storage and `add_dep(...)`
- `pto_runtime2_types.h`
  - removes the extra `alignas(64)` / size assertion on `PTO2TaskDescriptor`
- `pto_ring_buffer.h`
  - removes allocator helper accessors that were only supporting the older,
    more intrusive prepare-task path

These changes are the main response to the rebased review feedback that the
runtime hot path had drifted too far from upstream.

## Example Pattern

```cpp
PTO2_SCOPE(PTO2ScopeMode::MANUAL) {
    auto alloc = alloc_tensors(tmp_ci);

    Arg qk;
    qk.add_input(qi, kj);
    qk.add_output(sij_ci);
    qk.add_dep(alloc.task_id());
    auto qk_out = pto2_rt_submit_aic_task(FUNC_QK, qk);

    Arg sf;
    sf.add_input(qk_out.get_ref(0));
    sf.add_output(pij_ci, li_ci, mi_ci);
    sf.add_dep(qk_out.task_id());
    auto sf_out = pto2_rt_submit_aiv_task(FUNC_SF, sf);

    Arg up;
    up.add_input(sf_out.get_ref(1), sf_out.get_ref(2), pv_out.get_ref(0));
    up.add_inout(mi, li, out_view, tmp);
    up.add_dep(sf_out.task_id());
    up.add_dep(pv_out.task_id());
    auto up_out = pto2_rt_submit_aiv_task(FUNC_UP, up);
}
```

Repeated zero-output updater chains must explicitly thread the returned task id:

```cpp
PTO2TaskId prev_update = PTO2TaskId::invalid();
for (...) {
    Arg up = ...;
    if (prev_update.is_valid()) {
        up.add_dep(prev_update);
    }
    prev_update = pto2_rt_submit_aiv_task(FUNC_UP, up).task_id();
}
```

## Rebased Validation

Environment:

- platform: `a2a3`
- device: `10`
- PTO-ISA commit: `478daadb`
- benchmark date: `2026-04-23`

### Golden Checks

One-round hardware reruns without `--skip-golden`:

| Path | Case1 | Case2 |
| --- | --- | --- |
| TMR `paged_attention` AUTO | PASS | PASS |
| TMR `paged_attention_manual_scope` | PASS | PASS |
| TMR `paged_attention_unroll` AUTO | PASS | PASS |
| TMR `paged_attention_unroll_manual_scope` | PASS | PASS |
| ABG `paged_attention_unroll` | PASS | FAIL |

Notes:

- The rebased TMR AUTO and manual-scope paths are correctness-clean on the
  kept `Case1`/`Case2` workloads.
- ABG unroll `Case2` still fails golden on the rebased branch with
  `max_diff=0.01536`, so its timing remains a non-correctness-clean baseline.
- A temporary small-shape ABG non-unroll wrapper was also tried to make the
  non-unroll comparison apples-to-apples with TMR `CaseSmall1/CaseSmall2`.
  Both small ABG runs produced `NaN` golden mismatches, so those timings are
  useful for rough performance context only, not as a correctness-clean
  baseline.

## Fresh 100-Round Benchmark

Method:

- hardware: `a2a3`, device `10`
- rounds: `100`
- trimmed average: drop `10` low + `10` high
- TMR manual rows were rerun after restoring the explicit-only manual-scope
  TensorMap policy
- TMR benchmark mode uses `--skip-golden`; correctness is tracked separately in
  the golden table above

### Comparable Table

This is the main comparison table to read.

- `paged_attention` compares the kept small-shape non-unroll cases
- `paged_attention_unroll` compares the production unroll cases
- ABG small non-unroll rows are timing-only because their temporary wrapper is
  not correctness-clean
- ABG unroll `Case2` is also timing-only because its in-tree baseline still
  fails golden

| Example | Case | Runtime | Elapsed Trim (us) | Orch Trim (us) | Golden |
| --- | --- | --- | ---: | ---: | --- |
| `paged_attention` | `Case1` | TMR AUTO | 72.082 | 53.513 | PASS |
| `paged_attention` | `Case1` | TMR manual scope | 80.897 | 61.755 | PASS |
| `paged_attention` | `Case1` | ABG small-shape wrapper | 96.916 | 13.399 | FAIL |
| `paged_attention` | `Case2` | TMR AUTO | 90.645 | 62.805 | PASS |
| `paged_attention` | `Case2` | TMR manual scope | 97.781 | 67.749 | PASS |
| `paged_attention` | `Case2` | ABG small-shape wrapper | 110.935 | 22.776 | FAIL |
| `paged_attention_unroll` | `Case1` | TMR AUTO | 1139.615 | 718.983 | PASS |
| `paged_attention_unroll` | `Case1` | TMR manual scope | 1130.907 | 609.279 | PASS |
| `paged_attention_unroll` | `Case1` | ABG | 1385.087 | 706.679 | PASS |
| `paged_attention_unroll` | `Case2` | TMR AUTO | 514.259 | 279.321 | PASS |
| `paged_attention_unroll` | `Case2` | TMR manual scope | 492.723 | 229.459 | PASS |
| `paged_attention_unroll` | `Case2` | ABG | 664.744 | 284.575 | FAIL |

### Readout

The fresh rebased numbers show:

- non-unroll manual scope is still slower than TMR AUTO, but the gap is much
  smaller than the older pre-forward-port table
  - `Case1`: `+8.815us` elapsed, `+8.242us` orch
  - `Case2`: `+7.136us` elapsed, `+4.944us` orch
- unroll manual scope remains better than TMR AUTO on both kept cases
  - `Case1`: `-8.709us` elapsed, `-109.704us` orch
  - `Case2`: `-21.536us` elapsed, `-49.863us` orch
- the forward-ported cleanup therefore did not regress the manual path; it
  materially improved the rebased non-unroll result versus the old doc table

## ABG Production-Scale Context

The in-tree ABG non-unroll scene test still uses larger production cases. Those
rows are not directly comparable to the kept small TMR non-unroll cases, but
they are recorded here for completeness:

| Example | Case | Runtime | Elapsed Trim (us) | Orch Trim (us) | Golden |
| --- | --- | --- | ---: | ---: | --- |
| `paged_attention` | `case1` | ABG production | 32772.205 | 32476.044 | PASS |
| `paged_attention` | `case2` | ABG production | 17288.330 | 16718.342 | PASS |

## Risks

1. The PR page still shows the old remote branch head, so GitHub review state is
   stale until the rebased branch is pushed.
2. Non-unroll manual scope is not yet performance-neutral versus TMR AUTO.
3. The only correctness-clean non-unroll comparison on this branch today is TMR
   AUTO vs TMR manual scope; ABG small-shape comparison is still timing-only.
4. ABG unroll `Case2` remains a failing baseline and should not be used as a
   correctness-clean target.

## Recommendation

Treat this rebased checkpoint as:

- functionally validated for TMR AUTO/manual v0 on real hardware
- benchmarked with fresh 100-round data on device `10`
- ready for PR refresh after pushing the rebased branch and updating the PR body

But do not describe the non-unroll manual path as zero-overhead yet. The
remaining gap is much smaller than before, not eliminated.
