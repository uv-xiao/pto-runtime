# Manual Scope V1 Explicit-Only Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement manual-scope v1 so manual provenance is explicit-only everywhere: no TensorMap lookup/insert and no implicit creator-retention for manual-scope work, while `Arg.add_dep(task_id)` is accepted both inside and outside manual scope.

**Architecture:** The implementation keeps the existing submit APIs and task/tensor provenance metadata, but changes runtime dependency construction rules. Manual-scope submit becomes explicit-edge only, and non-manual submit gains validation for tensors whose latest writer came from manual scope. The work is split across API validation, runtime wiring/validation, examples, and focused tests so correctness is proven before any benchmark tuning.

**Tech Stack:** C++ runtime (`src/a2a3/runtime/tensormap_and_ringbuffer/runtime/`), C++ unit tests (`tests/ut/cpp/`), Python scene tests (`tests/st/a2a3/tensormap_and_ringbuffer/`), AICPU orchestration examples under `examples/a2a3/tensormap_and_ringbuffer/`.

---

## File Structure

**Runtime core**
- Modify: `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_types.h`
  - Relax `Arg.add_dep(...)` API contract and keep explicit dep storage unchanged.
- Modify: `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/tensor.h`
  - Clarify and update latest-writer provenance semantics on tensors.
- Modify: `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.h`
  - Add helpers / comments for manual-provenance validation and explicit-only behavior.
- Modify: `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp`
  - Remove manual-scope implicit creator-retention and TensorMap work.
  - Add validation for manual-provenance tensors consumed outside manual scope.
  - Update latest-writer stamping for manual in-place updates.
- Modify: `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_runtime2.cpp`
  - Review host/runtime helper functions that currently wait on `owner_task_id` so they match latest-writer semantics if needed.

**Unit tests**
- Modify: `tests/ut/cpp/test_a2a3_pto2_manual_scope_api.cpp`
  - Update API expectation: `add_dep(...)` outside manual scope is valid.
- Modify: `tests/ut/cpp/test_a2a3_pto2_manual_scope_runtime.cpp`
  - Add failing/passing runtime tests for explicit-only manual provenance.

**Scene / negative tests**
- Modify: `tests/st/a2a3/tensormap_and_ringbuffer/test_manual_scope_validation.py`
  - Replace the old “outside manual scope add_dep is invalid” case.
- Modify or add: `tests/st/a2a3/tensormap_and_ringbuffer/manual_scope_validation/kernels/orchestration/*.cpp`
  - Add negative orchestration cases for missing dep on manual-produced tensors.

**Examples**
- Modify: `examples/a2a3/tensormap_and_ringbuffer/paged_attention_manual_scope/kernels/orchestration/paged_attention_orch.cpp`
  - Keep manual path fully explicit and verify it still matches v1 rules.
- Modify: `examples/a2a3/tensormap_and_ringbuffer/paged_attention_unroll_manual_scope/kernels/orchestration/paged_attention_orch.cpp`
  - Same as above for unroll.
- Create or modify a small AUTO-after-manual example under `examples/a2a3/tensormap_and_ringbuffer/` if current paged-attention coverage is insufficient to prove `MANUAL -> AUTO`.

**Documentation**
- Modify: `docs/manual-scope-v1-explicit-only-design.md`
  - Keep design aligned with implementation findings and verification notes.

### Task 1: API Contract And Failing Validation Tests

**Files:**
- Modify: `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_types.h`
- Modify: `tests/ut/cpp/test_a2a3_pto2_manual_scope_api.cpp`
- Modify: `tests/st/a2a3/tensormap_and_ringbuffer/test_manual_scope_validation.py`
- Modify: `tests/st/a2a3/tensormap_and_ringbuffer/manual_scope_validation/kernels/orchestration/outside_scope_add_dep.cpp`
- Create: `tests/st/a2a3/tensormap_and_ringbuffer/manual_scope_validation/kernels/orchestration/missing_dep_on_manual_tensor.cpp`

- [ ] **Step 1: Write the failing API/runtime expectation updates**

Update the API unit test so `add_dep(...)` outside manual scope is accepted, and replace the old invalid scene-test case with a new missing-dep case.

Add this expectation in `tests/ut/cpp/test_a2a3_pto2_manual_scope_api.cpp` near the existing `ArgAddDepOutsideManualScopeSetsError` test by replacing it with:

```cpp
TEST(A2A3ManualScopeApi, ArgAddDepOutsideManualScopeIsAccepted) {
    Arg args;
    args.add_dep(PTO2TaskId::make(0, 9));
    args.add_dep(PTO2TaskId::make(1, 11));

    EXPECT_FALSE(args.has_error);
    ASSERT_EQ(args.explicit_dep_count(), 2u);
    EXPECT_EQ(args.explicit_dep(0), PTO2TaskId::make(0, 9));
    EXPECT_EQ(args.explicit_dep(1), PTO2TaskId::make(1, 11));
}
```

Add a new negative orchestration file `tests/st/a2a3/tensormap_and_ringbuffer/manual_scope_validation/kernels/orchestration/missing_dep_on_manual_tensor.cpp`:

```cpp
#include "pto_orchestration_api.h"

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
 aicpu_orchestration_config(const ChipStorageTaskArgs &) {
    return PTO2OrchestrationConfig{.expected_arg_count = 0};
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipStorageTaskArgs &) {
    uint32_t shape[1] = {1};
    TensorCreateInfo ci(shape, 1, DataType::FLOAT32);

    TaskOutputTensors manual_out;
    PTO2_SCOPE(PTO2ScopeMode::MANUAL) {
        Arg alloc_args;
        alloc_args.add_output(ci);
        manual_out = alloc_tensors(alloc_args);
    }

    PTO2_SCOPE() {
        Arg use_args;
        use_args.add_input(manual_out.get_ref(0));
        (void)pto2_rt_submit_aiv_task(0, use_args);
    }
}

}
```

Replace the old invalid `outside_scope_add_dep.cpp` expectation with a positive orchestration that proves `add_dep(...)` outside manual scope is accepted by the actual submit path, not just by the `Arg` container:

```cpp
#include "pto_orchestration_api.h"

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
 aicpu_orchestration_config(const ChipStorageTaskArgs &) {
    return PTO2OrchestrationConfig{.expected_arg_count = 0};
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipStorageTaskArgs &) {
    uint32_t shape[1] = {1};
    TensorCreateInfo ci(shape, 1, DataType::FLOAT32);

    Arg alloc_args;
    alloc_args.add_output(ci);
    TaskOutputTensors produced = alloc_tensors(alloc_args);

    Arg use_args;
    use_args.add_input(produced.get_ref(0));
    use_args.add_dep(produced.task_id());
    (void)pto2_rt_submit_aiv_task(0, use_args);
}

}
```

Update `tests/st/a2a3/tensormap_and_ringbuffer/test_manual_scope_validation.py` so the failing matrix drops the old “outside scope add_dep” invalid case, adds the new `missing_dep_on_manual_tensor.cpp` case with invalid-args expectation, and runs `outside_scope_add_dep.cpp` as a positive scene if the validation harness has a positive-case matrix.

- [ ] **Step 2: Run the focused tests to confirm they fail for the right reason**

Run:

```bash
source .venv/bin/activate
cmake -B tests/ut/cpp/build -S tests/ut/cpp >/dev/null
cmake --build tests/ut/cpp/build --target test_a2a3_pto2_manual_scope_api >/dev/null
ctest --test-dir tests/ut/cpp/build -R test_a2a3_pto2_manual_scope_api --output-on-failure
python -m pytest tests/st/a2a3/tensormap_and_ringbuffer/test_manual_scope_validation.py --platform a2a3sim --device 0 -q
```

Expected:
- the API test still fails because runtime/API still rejects `add_dep(...)` outside manual scope
- the new negative case fails because runtime does not yet reject missing dep on manual-produced tensors

- [ ] **Step 3: Implement the minimal API acceptance change**

In `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_types.h`, keep storage unchanged and only relax the contract commentary if present. The actual runtime-side reject path must be removed later in Task 2.

No functional code is required in `Arg::add_dep(...)` besides preserving the current append behavior:

```cpp
void add_dep(PTO2TaskId task_id) {
    if (explicit_dep_count_ >= kMaxExplicitDeps) {
        set_error("Too many explicit deps (exceeds Arg::kMaxExplicitDeps=16)");
        return;
    }
    explicit_deps_[explicit_dep_count_++] = task_id;
}
```

- [ ] **Step 4: Re-run the same focused tests**

Run:

```bash
source .venv/bin/activate
cmake --build tests/ut/cpp/build --target test_a2a3_pto2_manual_scope_api >/dev/null
ctest --test-dir tests/ut/cpp/build -R test_a2a3_pto2_manual_scope_api --output-on-failure
python -m pytest tests/st/a2a3/tensormap_and_ringbuffer/test_manual_scope_validation.py --platform a2a3sim --device 0 -q
```

Expected:
- API unit test still fails until Task 2 removes the orchestrator-side rejection
- negative scene test still fails until Task 2 adds runtime validation

- [ ] **Step 5: Commit the test-first/API setup checkpoint**

Run:

```bash
git add \
  src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_types.h \
  tests/ut/cpp/test_a2a3_pto2_manual_scope_api.cpp \
  tests/st/a2a3/tensormap_and_ringbuffer/test_manual_scope_validation.py \
  tests/st/a2a3/tensormap_and_ringbuffer/manual_scope_validation/kernels/orchestration/outside_scope_add_dep.cpp \
  tests/st/a2a3/tensormap_and_ringbuffer/manual_scope_validation/kernels/orchestration/missing_dep_on_manual_tensor.cpp
git commit -m "Update: prepare manual-scope v1 API validation"
```

### Task 2: Runtime Explicit-Only Semantics For Manual Provenance

**Files:**
- Modify: `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp`
- Modify: `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.h`
- Modify: `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/tensor.h`
- Modify: `tests/ut/cpp/test_a2a3_pto2_manual_scope_runtime.cpp`

- [ ] **Step 1: Write failing runtime tests for the new rules**

Add these tests to `tests/ut/cpp/test_a2a3_pto2_manual_scope_runtime.cpp`.

Test 1: AUTO consumer of manual-produced tensor is rejected without explicit dep.

```cpp
TEST_F(ManualScopeRuntimeTest, AutoConsumerOfManualTensorRequiresExplicitDep) {
    pto2_scope_begin(&rt_->orchestrator, PTO2ScopeMode::MANUAL);

    TensorCreateInfo ci = make_create_info();
    Arg producer_args;
    producer_args.add_output(ci);
    TaskSubmitResult producer = pto2_alloc_tensors(&rt_->orchestrator, producer_args);
    ASSERT_TRUE(producer.task_id().is_valid());
    pto2_scope_end(&rt_->orchestrator);

    pto2_scope_begin(&rt_->orchestrator, PTO2ScopeMode::AUTO);
    Arg consumer_args;
    consumer_args.add_input(producer.get_ref(0));
    MixedKernels kernels{};
    kernels.aiv0_kernel_id = 0;
    TaskSubmitResult consumer = pto2_submit_mixed_task(&rt_->orchestrator, kernels, consumer_args);

    EXPECT_TRUE(consumer.empty());
    EXPECT_TRUE(rt_->orchestrator.fatal);
}
```

Test 2: AUTO consumer passes with explicit dep.

```cpp
TEST_F(ManualScopeRuntimeTest, AutoConsumerOfManualTensorPassesWithExplicitDep) {
    pto2_scope_begin(&rt_->orchestrator, PTO2ScopeMode::MANUAL);

    TensorCreateInfo ci = make_create_info();
    Arg producer_args;
    producer_args.add_output(ci);
    TaskSubmitResult producer = pto2_alloc_tensors(&rt_->orchestrator, producer_args);
    ASSERT_TRUE(producer.task_id().is_valid());
    pto2_scope_end(&rt_->orchestrator);

    pto2_scope_begin(&rt_->orchestrator, PTO2ScopeMode::AUTO);
    Arg consumer_args;
    consumer_args.add_input(producer.get_ref(0));
    consumer_args.add_dep(producer.task_id());
    MixedKernels kernels{};
    kernels.aiv0_kernel_id = 0;
    TaskSubmitResult consumer = pto2_submit_mixed_task(&rt_->orchestrator, kernels, consumer_args);

    EXPECT_FALSE(rt_->orchestrator.fatal);
    EXPECT_TRUE(consumer.task_id().is_valid());
}
```

Test 3: manual submit does not create owner-based fanin automatically.

```cpp
TEST_F(ManualScopeRuntimeTest, ManualSubmitDoesNotAddImplicitCreatorRetention) {
    pto2_scope_begin(&rt_->orchestrator, PTO2ScopeMode::MANUAL);

    TensorCreateInfo ci = make_create_info();
    Arg alloc_args;
    alloc_args.add_output(ci);
    TaskSubmitResult producer = pto2_alloc_tensors(&rt_->orchestrator, alloc_args);
    ASSERT_TRUE(producer.task_id().is_valid());

    Arg consumer_args;
    consumer_args.add_input(producer.get_ref(0));
    MixedKernels kernels{};
    kernels.aiv0_kernel_id = 0;
    TaskSubmitResult consumer = pto2_submit_mixed_task(&rt_->orchestrator, kernels, consumer_args);

    EXPECT_TRUE(consumer.empty());
    EXPECT_TRUE(rt_->orchestrator.fatal);
}
```

- [ ] **Step 2: Run the focused runtime test target and confirm failure**

Run:

```bash
source .venv/bin/activate
cmake --build tests/ut/cpp/build --target test_a2a3_pto2_manual_scope_runtime >/dev/null
ctest --test-dir tests/ut/cpp/build -R test_a2a3_pto2_manual_scope_runtime --output-on-failure
```

Expected:
- new tests fail because runtime still performs mixed implicit behavior

- [ ] **Step 3: Implement explicit-only dependency construction and validation**

In `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp` make these changes.

A. Remove the reject path that bans `add_dep(...)` outside manual scope. Replace:

```cpp
if (!orch->in_manual_scope()) {
    pto2_orch_report_fatal(
        orch, PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "Arg.add_dep(...) is only valid inside manual scope"
    );
    return result;
}
```

with logic that only validates task-id resolvability and dedupe.

B. Split explicit dep validation into two policies:
- inside manual scope: dep must still resolve to a live task and, for the first cut, remain restricted to current/known reachable tasks
- outside manual scope: dep must resolve to a live task; do not require current-manual-scope membership

C. Remove all implicit creator-retention for manual-scope submit. Replace the existing owner-based fanin block with:

```cpp
PTO2TaskId owner = tensor->owner_task_id;
if (!orch->in_manual_scope() && !tensor_has_manual_provenance(*tensor) && owner.is_valid() && sched != nullptr) {
    PTO2TaskSlotState *prod_state =
        &sched->ring_sched_states[owner.ring()].get_slot_state_by_task_id(owner.local());
    if (!pto2_append_fanin_or_fail(
            orch, task_id, i, ptype, prod_state, &fanin_builder, sched, fc, ring_id, "creator retention"
        )) {
        return result;
    }
}
```

D. Add a helper, declared in `pto_orchestrator.h` and defined in `pto_orchestrator.cpp`, to recognize manual provenance from tensor metadata:

```cpp
static bool tensor_has_manual_provenance(const Tensor &tensor) {
    return tensor.producer_manual_scope_depth >= 0 && tensor.owner_task_id.is_valid();
}
```

E. In the manual-scope submit path, skip TensorMap sync/lookup/insert work for dependency construction. For the first implementation, it is acceptable to keep the global `sync_tensormap(...)` call if needed for runtime housekeeping, but lookup/insert and owner-based fanin inference must not affect manual-scope dependencies.

F. In the non-manual submit path, before any implicit AUTO wiring from tensor metadata, require explicit dep for any tensor with manual provenance:

```cpp
if (!orch->in_manual_scope() && tensor_has_manual_provenance(*tensor)) {
    PTO2TaskId latest_writer = tensor->owner_task_id;
    if (!pto2_explicit_dep_ids_contains(explicit_dep_ids, explicit_dep_count, latest_writer)) {
        pto2_orch_report_fatal(
            orch, PTO2_ERROR_INVALID_ARGS, __FUNCTION__,
            "tensor produced by manual scope requires explicit Arg.add_dep(latest_writer_task_id)"
        );
        return result;
    }
    continue;
}
```

G. Update latest-writer provenance for manual `INOUT` / `OUTPUT_EXISTING` tensors after successful submit. In the tensor-argument loop after `payload->init(...)`, add:

```cpp
if (orch->in_manual_scope()) {
    for (int i = 0; i < args.tensor_count(); i++) {
        TensorArgType ptype = args.tag(i);
        if (ptype == TensorArgType::INOUT || ptype == TensorArgType::OUTPUT_EXISTING) {
            Tensor *mutable_tensor = const_cast<Tensor *>(args.tensor(i).ptr);
            mutable_tensor->owner_task_id = prepared.task_id;
            mutable_tensor->set_producer_scope_metadata(
                static_cast<int16_t>(orch->scope_stack_top),
                static_cast<int16_t>(orch->current_manual_scope_depth)
            );
        }
    }
}
```

H. Keep `OUTPUT` tensor stamping unchanged so materialized outputs still carry latest-writer provenance.

- [ ] **Step 4: Re-run the focused runtime/API validation tests**

Run:

```bash
source .venv/bin/activate
cmake --build tests/ut/cpp/build --target \
  test_a2a3_pto2_manual_scope_api \
  test_a2a3_pto2_manual_scope_runtime >/dev/null
ctest --test-dir tests/ut/cpp/build -R 'test_a2a3_pto2_manual_scope_(api|runtime)' --output-on-failure
python -m pytest tests/st/a2a3/tensormap_and_ringbuffer/test_manual_scope_validation.py --platform a2a3sim --device 0 -q
```

Expected:
- API/runtime tests pass
- scene negative tests pass with the new missing-dep behavior

- [ ] **Step 5: Commit the runtime semantic change**

Run:

```bash
git add \
  src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp \
  src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.h \
  src/a2a3/runtime/tensormap_and_ringbuffer/runtime/tensor.h \
  tests/ut/cpp/test_a2a3_pto2_manual_scope_runtime.cpp \
  tests/ut/cpp/test_a2a3_pto2_manual_scope_api.cpp \
  tests/st/a2a3/tensormap_and_ringbuffer/test_manual_scope_validation.py \
  tests/st/a2a3/tensormap_and_ringbuffer/manual_scope_validation/kernels/orchestration/outside_scope_add_dep.cpp \
  tests/st/a2a3/tensormap_and_ringbuffer/manual_scope_validation/kernels/orchestration/missing_dep_on_manual_tensor.cpp
git commit -m "Add: enforce explicit-only manual provenance"
```

### Task 3: Example Coverage For AUTO -> MANUAL, MANUAL -> MANUAL, MANUAL -> AUTO

**Files:**
- Modify: `examples/a2a3/tensormap_and_ringbuffer/paged_attention_manual_scope/kernels/orchestration/paged_attention_orch.cpp`
- Modify: `examples/a2a3/tensormap_and_ringbuffer/paged_attention_unroll_manual_scope/kernels/orchestration/paged_attention_orch.cpp`
- Create: `examples/a2a3/tensormap_and_ringbuffer/manual_scope_v1_bridge/`
- Test: `examples/a2a3/tensormap_and_ringbuffer/manual_scope_v1_bridge/test_manual_scope_v1_bridge.py`

- [ ] **Step 1: Add a minimal bridge example that proves `MANUAL -> AUTO`**

Create `examples/a2a3/tensormap_and_ringbuffer/manual_scope_v1_bridge/test_manual_scope_v1_bridge.py` with a scene-test that:
- allocates/produces a tensor in manual scope
- consumes it in AUTO scope without dep in one failing case
- consumes it with `add_dep(...)` in one passing case

Use this callable skeleton:

```python
@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestManualScopeV1Bridge(SceneTestCase):
    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/manual_scope_v1_bridge_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "kernels/aiv/aiv_noop.cpp",
                "core_type": "aiv",
                "signature": [D.IN],
            },
        ],
    }
```

Create orchestration `manual_scope_v1_bridge_orch.cpp` with one valid path that uses explicit dep:

```cpp
PTO2_SCOPE(PTO2ScopeMode::MANUAL) {
    Arg alloc_args;
    alloc_args.add_output(ci);
    manual_out = alloc_tensors(alloc_args);
}

PTO2_SCOPE() {
    Arg use_args;
    use_args.add_input(manual_out.get_ref(0));
    use_args.add_dep(manual_out.task_id());
    (void)pto2_rt_submit_aiv_task(0, use_args);
}
```

- [ ] **Step 2: Align existing manual paged-attention examples with v1 assumptions**

Review both manual paged-attention orchestration files and ensure every task that consumes manual-produced or manual-updated state already carries explicit deps to the latest writer task id.

Specifically confirm in both files:
- `alloc_task` is explicitly depended on where needed
- updater chains use `prev_update_task`
- no behavior depends on TensorMap fallback for boundary tensors

If the current code already satisfies this, document that no functional change was required and leave the files untouched.

- [ ] **Step 3: Run focused example validation**

Run:

```bash
source .venv/bin/activate
python examples/scripts/run_example.py \
  -k examples/a2a3/tensormap_and_ringbuffer/paged_attention_manual_scope/kernels \
  -g examples/a2a3/tensormap_and_ringbuffer/paged_attention_manual_scope/golden.py \
  -p a2a3sim -d 0 --case Case1
python examples/scripts/run_example.py \
  -k examples/a2a3/tensormap_and_ringbuffer/paged_attention_unroll_manual_scope/kernels \
  -g examples/a2a3/tensormap_and_ringbuffer/paged_attention_unroll_manual_scope/golden.py \
  -p a2a3sim -d 0 --case Case1
python examples/a2a3/tensormap_and_ringbuffer/manual_scope_v1_bridge/test_manual_scope_v1_bridge.py -p a2a3sim -d 0
```

Expected:
- both manual paged-attention examples still pass in sim for at least one representative case
- bridge example passes on the explicit-dep path

- [ ] **Step 4: Commit the example coverage checkpoint**

Run:

```bash
git add \
  examples/a2a3/tensormap_and_ringbuffer/paged_attention_manual_scope/kernels/orchestration/paged_attention_orch.cpp \
  examples/a2a3/tensormap_and_ringbuffer/paged_attention_unroll_manual_scope/kernels/orchestration/paged_attention_orch.cpp \
  examples/a2a3/tensormap_and_ringbuffer/manual_scope_v1_bridge
git commit -m "Add: cover manual-scope v1 bridge cases"
```

### Task 4: Full Verification, Benchmark Baseline, And Doc Refresh

**Files:**
- Modify: `docs/manual-scope-v1-explicit-only-design.md`

- [ ] **Step 1: Run focused hardware correctness checks**

Run:

```bash
source .venv/bin/activate
ctest --test-dir tests/ut/cpp/build -R 'test_a2a3_pto2_manual_scope_(api|runtime)' --output-on-failure
python -m pytest tests/st/a2a3/tensormap_and_ringbuffer/test_manual_scope_validation.py --platform a2a3sim --device 0 -q
python examples/scripts/run_example.py \
  -k examples/a2a3/tensormap_and_ringbuffer/paged_attention_manual_scope/kernels \
  -g examples/a2a3/tensormap_and_ringbuffer/paged_attention_manual_scope/golden.py \
  -p a2a3 -d 9 -c d96c8784 --case Case1
python examples/scripts/run_example.py \
  -k examples/a2a3/tensormap_and_ringbuffer/paged_attention_unroll_manual_scope/kernels \
  -g examples/a2a3/tensormap_and_ringbuffer/paged_attention_unroll_manual_scope/golden.py \
  -p a2a3 -d 9 -c d96c8784 --case Case1
```

Expected:
- focused C++ tests pass
- negative sim test passes
- representative hardware examples pass

- [ ] **Step 2: Gather the first v1 benchmark snapshot**

Run the same four-row batch used for v0 so v0/v1 remain comparable:

```bash
source .venv/bin/activate
python examples/a2a3/tensormap_and_ringbuffer/paged_attention/test_paged_attention.py -p a2a3 -d 9 -n 30 --case Case1 --skip-golden
python examples/scripts/run_example.py -k examples/a2a3/tensormap_and_ringbuffer/paged_attention_manual_scope/kernels -g examples/a2a3/tensormap_and_ringbuffer/paged_attention_manual_scope/golden.py -p a2a3 -d 9 -c d96c8784 -n 30 --case Case1 --skip-golden
python examples/a2a3/tensormap_and_ringbuffer/paged_attention/test_paged_attention.py -p a2a3 -d 9 -n 30 --case Case2 --skip-golden
python examples/scripts/run_example.py -k examples/a2a3/tensormap_and_ringbuffer/paged_attention_manual_scope/kernels -g examples/a2a3/tensormap_and_ringbuffer/paged_attention_manual_scope/golden.py -p a2a3 -d 9 -c d96c8784 -n 30 --case Case2 --skip-golden
python tests/st/a2a3/tensormap_and_ringbuffer/paged_attention_unroll/test_paged_attention_unroll.py -p a2a3 -d 9 -n 30 --case Case1 --skip-golden
python examples/scripts/run_example.py -k examples/a2a3/tensormap_and_ringbuffer/paged_attention_unroll_manual_scope/kernels -g examples/a2a3/tensormap_and_ringbuffer/paged_attention_unroll_manual_scope/golden.py -p a2a3 -d 9 -c d96c8784 -n 30 --case Case1 --skip-golden
python tests/st/a2a3/tensormap_and_ringbuffer/paged_attention_unroll/test_paged_attention_unroll.py -p a2a3 -d 9 -n 30 --case Case2 --skip-golden
python examples/scripts/run_example.py -k examples/a2a3/tensormap_and_ringbuffer/paged_attention_unroll_manual_scope/kernels -g examples/a2a3/tensormap_and_ringbuffer/paged_attention_unroll_manual_scope/golden.py -p a2a3 -d 9 -c d96c8784 -n 30 --case Case2 --skip-golden
```

Expected:
- a fresh v1 benchmark table with the same shape as v0

- [ ] **Step 3: Update the v1 design doc with implementation status**

Append these sections to `docs/manual-scope-v1-explicit-only-design.md` after implementation is verified:
- validation status
- benchmark method
- fresh benchmark results
- any deviations from the original spec discovered during implementation

Use this table shape:

```md
| Example | Case | Auto Elapsed Trim (us) | Auto Orch Trim (us) | Manual Elapsed Trim (us) | Manual Orch Trim (us) | Elapsed Delta | Orch Delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
```

- [ ] **Step 4: Run final doc-only verification**

Run:

```bash
git diff -- docs/manual-scope-v1-explicit-only-design.md
```

Expected:
- doc reflects the actual final implementation and measured status

- [ ] **Step 5: Commit the verification/documentation checkpoint**

Run:

```bash
git add docs/manual-scope-v1-explicit-only-design.md
git commit -m "Update: verify manual-scope v1 explicit-only"
```

## Self-Review

### Spec coverage

This plan covers all approved v1 requirements from `docs/manual-scope-v1-explicit-only-design.md`:
- `add_dep(...)` outside manual scope
- no TensorMap lookup/insert for manual-scope submit
- no implicit creator-retention for manual-scope submit
- explicit dep requirement for `MANUAL -> AUTO`
- latest-writer provenance on in-place manual updates
- new examples/tests for the bridge cases
- verification and benchmark refresh

No approved requirement is left without a task.

### Placeholder scan

Checked for `TODO`, `TBD`, “similar to”, and unspecified test steps.
Each task names exact files, concrete code to add/change, and exact commands.

### Type consistency

The plan consistently uses:
- `TaskSubmitResult` / `TaskOutputTensors`
- `Arg.add_dep(PTO2TaskId)`
- latest-writer provenance via `Tensor::owner_task_id`
- manual provenance via `producer_manual_scope_depth`

No mismatched function names or competing terminology remain.
