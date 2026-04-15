# Manual Scope V1 Explicit-Only Design

Date: 2026-04-16

Branch: `manual_scope_v1_explicit_only`

## Goal

Add a stricter manual-scope mode to `a2a3/tensormap_and_ringbuffer` where:

- tasks in manual scope do zero TensorMap lookup
- tasks in manual scope do zero TensorMap insert
- tasks in manual scope do zero implicit creator-retention
- any dependency that touches manual-scope work is expressed explicitly with
  `Arg.add_dep(task_id)`
- `Arg.add_dep(task_id)` is allowed both inside and outside manual scope

This is intentionally stricter than v0.

The core idea is:

- v0 = manual scope is a lighter submit path, but boundary tensors still keep
  part of the implicit runtime dependency machinery
- v1 = manual scope becomes an explicit-edge island; the runtime keeps metadata
  for provenance and validation, but it never infers scheduling edges for
  manual-scope work

## Motivation

The v0 branch proved two things:

1. manual-local TensorMap bypass is mechanically feasible
2. partial implicit behavior makes the model harder to reason about

The remaining ambiguity in v0 is that dependency behavior still differs across
manual-local tensors and boundary tensors. That keeps the runtime semantics
mixed:

- some edges are explicit
- some edges are still inferred
- some missing `add_dep(...)` mistakes can still pass silently

V1 removes that ambiguity.

## Core Rule

If a task is part of the manual-scope provenance chain, its dependencies are
explicit-only.

That means the runtime must not build dependency edges for that work from:

- TensorMap lookup
- TensorMap insert / later rediscovery
- implicit creator-retention

The runtime may still maintain provenance metadata on tensors and submit
results, but that metadata is used only for:

- validation
- diagnostics
- later explicit wiring by orchestration authors

It is not used to silently add fanins.

## Constraints

This version follows these rules:

1. `PTO2_SCOPE(PTO2ScopeMode::MANUAL)` remains the explicit manual-scope entry.
2. Submit APIs stay the same as AUTO mode.
3. `Arg.add_dep(task_id)` is allowed inside and outside manual scope.
4. Manual-scope tasks publish immediately at submit time.
5. Delayed wiring / delayed linking / scope-end replay remain unsupported.
6. Nested manual scopes remain unsupported.
7. Manual-scope dependency correctness depends entirely on explicit task ids.
8. The runtime rejects missing explicit deps when a task touches manual-scope
   provenance that still requires ordering.

## Non-goals

- no post-submit dependency API
- no delayed graph construction API
- no recovery path that silently falls back to TensorMap for manual work
- no automatic conversion of tensor provenance metadata back into fanins
- no redesign of pure AUTO mode

## User-Facing API

### Scope

Manual mode remains:

```cpp
PTO2_SCOPE(PTO2ScopeMode::MANUAL) {
    ...
}
```

AUTO remains the default:

```cpp
PTO2_SCOPE() {
    ...
}
```

### Submit API

Submit entry points stay unchanged:

```cpp
auto out = pto2_rt_submit_aic_task(FUNC_ID, args);
auto out = pto2_rt_submit_aiv_task(FUNC_ID, args);
auto out = pto2_rt_submit_task(mixed_kernels, args);
```

### Explicit dependencies

`Arg.add_dep(task_id)` becomes generally available:

```cpp
Arg args;
args.add_input(tensor);
args.add_dep(task_id);
```

V1 change relative to v0:

- v0: `add_dep(...)` valid only inside manual scope
- v1: `add_dep(...)` valid everywhere

Reason:

- later AUTO consumers of manual-produced tensors must also wire explicit deps
- without this change, `manual -> later external` cannot be expressed cleanly

### `alloc_tensors(...)`

`alloc_tensors(...)` remains output-only and still returns a standalone task id:

```cpp
auto alloc = alloc_tensors(ci0, ci1);
alloc.task_id();
alloc.get_ref(0);
```

That allocation task id is also explicit-only in v1 when the allocation happens
inside manual scope.

## Provenance Model

V1 keeps tensor/task provenance metadata, but only as metadata.

Each produced or updated tensor should continue to carry enough provenance to
answer:

- was this tensor last produced or updated by manual-scope work?
- what is the latest task id that logically owns the current contents?
- what scope depth / manual-scope depth did that happen in?

This metadata is not a hidden dependency mechanism.

It exists so the runtime can validate that orchestration supplied the required
explicit dep instead of silently patching the graph.

## Latest-writer Rule

V1 needs a precise rule for tensors that are updated in place.

For any tensor used as `INOUT` or `OUTPUT_EXISTING` by a manual-scope task:

- the tensor's provenance must advance to the new manual writer task id
- later consumers must depend on that latest writer explicitly

This is required for both:

- manual -> manual updater chains
- manual -> later external consumers

Without a latest-writer provenance rule, the runtime cannot validate that the
consumer depends on the correct final writer.

## Dependency Semantics By Case

### 1. AUTO -> AUTO

Unchanged.

Pure AUTO behavior keeps the existing implicit dependency rules:

- TensorMap lookup / insert
- creator-retention
- `manual_dep=true` special cases

V1 does not redesign this path.

### 2. External host input -> MANUAL

If a tensor is truly host-external and has no runtime producer task id, there is
no PTO task to depend on.

So:

- no explicit dep is required for host inputs with invalid producer task id
- manual submit still does no TensorMap work for them

### 3. AUTO-produced tensor -> MANUAL consumer

Explicit dep is required.

If a manual-scope task consumes a tensor whose latest writer is a prior PTO task
outside manual scope, the manual consumer must include:

```cpp
args.add_dep(latest_writer_task_id);
```

The runtime validates this and does not add the edge implicitly.

### 4. MANUAL -> MANUAL

Explicit dep is required.

No TensorMap lookup / insert.
No implicit creator-retention.
No hidden edge recovery.

This is the cleanest v1 path.

### 5. MANUAL -> later AUTO consumer

Explicit dep is required.

This is the main reason `Arg.add_dep(...)` must be allowed outside manual scope.

If an AUTO-scope task consumes a tensor whose latest writer came from manual
scope, AUTO-mode TensorMap logic must not silently rediscover and wire that
manual producer.

Instead, the caller must provide:

```cpp
args.add_dep(manual_writer_task_id);
```

The runtime validates this before submit.

### 6. MANUAL -> later host readback

No extra PTO dependency is needed if the host reads only after runtime/device
synchronization. This is outside the PTO task graph.

## Runtime Submit Rules

### Manual-scope submit

For a task submitted inside manual scope:

1. allocate task slot and task id
2. read explicit deps from `Arg`
3. validate them
4. append them as ordinary fanins
5. copy params / publish immediately
6. stamp output and updated tensors with latest-writer provenance
7. do not do TensorMap lookup
8. do not do TensorMap insert
9. do not do implicit creator-retention

### Non-manual submit

For a task submitted outside manual scope:

1. keep current AUTO behavior for pure AUTO tensors
2. if a consumed tensor has manual-scope provenance, require explicit dep to its
   latest writer
3. do not silently infer a dep from that manual provenance
4. continue normal TensorMap / creator-retention behavior only for non-manual
   provenance tensors

So v1 does not mean "AUTO mode becomes explicit-only".
It means "manual provenance is explicit-only everywhere".

## Validation Rules

The runtime should reject at submit time when:

1. a task touches a manual-provenance tensor but does not include the tensor's
   latest writer task id in explicit deps
2. an explicit dep task id is invalid
3. an explicit dep references a task that is not alive / not resolvable
4. nested manual scope is requested
5. `alloc_tensors(...)` is asked to infer dependencies implicitly

The runtime should not reject when:

- a host-external tensor has no producer task id
- a task adds extra explicit deps that are redundant but valid

## Example Patterns

### AUTO producer -> MANUAL consumer

```cpp
PTO2_SCOPE() {
    auto qk = pto2_rt_submit_aic_task(FUNC_QK, qk_args);

    PTO2_SCOPE(PTO2ScopeMode::MANUAL) {
        Arg sf;
        sf.add_input(qk.get_ref(0));
        sf.add_dep(qk.task_id());
        auto sf_out = pto2_rt_submit_aiv_task(FUNC_SF, sf);
    }
}
```

### MANUAL updater chain

```cpp
PTO2_SCOPE(PTO2ScopeMode::MANUAL) {
    auto alloc = alloc_tensors(ci);
    PTO2TaskId prev = alloc.task_id();

    for (...) {
        Arg up;
        up.add_inout(alloc.get_ref(0));
        up.add_dep(prev);
        auto out = pto2_rt_submit_aiv_task(FUNC_UPDATE, up);
        prev = out.task_id();
    }
}
```

### MANUAL producer -> later AUTO consumer

```cpp
TaskOutputTensors manual_out;

PTO2_SCOPE(PTO2ScopeMode::MANUAL) {
    Arg mm;
    manual_out = pto2_rt_submit_aic_task(FUNC_MM, mm);
}

PTO2_SCOPE() {
    Arg consumer;
    consumer.add_input(manual_out.get_ref(0));
    consumer.add_dep(manual_out.task_id());
    auto out = pto2_rt_submit_aiv_task(FUNC_USE, consumer);
}
```

## Migration From V0

Relative to v0, v1 changes these rules:

1. `Arg.add_dep(...)` becomes valid outside manual scope.
2. Boundary tensors are no longer allowed to rely on implicit TensorMap wiring
   when they carry manual provenance.
3. Manual-scope submit no longer keeps creator-retention for any consumed
   tensors.
4. Missing deps on manual-produced / manual-updated tensors become hard submit
   errors instead of implicit fallback behavior.

## Main Benefits

- simpler mental model
- cleaner zero-overhead manual runtime path
- fewer mixed implicit / explicit semantics
- easier to reason about correctness from orchestration source alone

## Main Risks

1. orchestration burden increases
2. missing `add_dep(...)` becomes easier to write
3. validation logic becomes more important and more invasive
4. example code may become noticeably more verbose
5. latest-writer provenance for in-place updates must be correct, or validation
   will be wrong

## Testing Strategy

### Unit / runtime validation

Add tests that cover:

- `add_dep(...)` outside manual scope is accepted
- manual submit does not use TensorMap lookup or insert
- manual submit does not build implicit creator-retention fanins
- AUTO consumer of manual-produced tensor is rejected without explicit dep
- AUTO consumer of manual-produced tensor passes with explicit dep
- manual updater chain advances latest-writer provenance correctly
- host-external tensor with invalid producer task id does not require a dep

### Example validation

Add or adapt examples that explicitly exercise:

- AUTO -> MANUAL
- MANUAL -> MANUAL
- MANUAL -> AUTO

### Benchmarking

Benchmark separately against v0 because v1 changes semantics, not just
performance tuning.

The first benchmark question is not "is v1 faster?".
It is:

- does the fully explicit model eliminate the remaining TensorMap-related
  ambiguity without introducing unacceptable orchestration overhead?

## Recommendation

Implement v1 as a new branch line, not as an in-place mutation of v0.

V0 and v1 represent different product choices:

- v0 = mixed explicit + implicit model
- v1 = explicit-only model for manual provenance

Keeping both lines separate makes evaluation clearer and reduces the risk of
confusing correctness or benchmark results across the two designs.
