# Annotated Walkthrough: `pto_orchestrator.cpp` (tensormap_and_ringbuffer)

Last verified against repo state on **2026-02-26**.

This is a **very detailed, function-by-function** explanation of:
- `src/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp`
- the closely related pieces it interacts with (TensorMap, rings, shared memory header, device scheduler “parallel mode”).

It is written to answer: “What does each line/step do, and why?”

Companion docs:
- Architecture + overview + profiling usage: `docs/tensormap-ringbuffer-runtime-guide.md`
- Full runtime code tour: `docs/tensormap-ringbuffer-runtime-codewalk.md`
- Line-numbered submit hot-path (`pto2_submit_task`): `docs/linebyline-pto2-submit-task.md`
- Platform layer (a2a3 vs a2a3sim glue): `docs/platform-codewalk.md`
- Platform deep-dive (DeviceRunner + regs): `docs/annotated-platform-device-runner.md`

---

## 0. The Orchestrator’s Job (in one paragraph)

The orchestrator is the **graph builder**. It runs the orchestration `.so` code (which is arbitrary control flow) and turns each `submit_task(...)` call into:
- a new `PTO2TaskDescriptor` slot in device shared memory
- optional GM heap allocations for outputs
- dependency edges (fanin + fanout lists) discovered by **TensorMap overlap**
- a published `current_task_index` so scheduler threads know new tasks exist

It does *not* execute kernels itself; it sets up metadata so the scheduler can dispatch them.

---

## 1. File-Level Switch: `PTO2_ORCH_PROFILING`

At the top of `pto_orchestrator.cpp` you’ll see:

- compile-time toggle `#if PTO2_ORCH_PROFILING`
- accumulation variables:
  - `g_orch_sync_cycle`, `g_orch_alloc_cycle`, `g_orch_params_cycle`, …

These are used to measure **submit-time overhead** on the AICPU:
- how much time is spent in TensorMap sync
- task ring allocation
- param copy
- dependency lookup
- heap allocation
- tensormap insert
- fanin finalization + “already ready” check
- finalize (scheduler init + shared memory publish)

Important nuance:
- These counters measure the orchestrator’s *CPU time* and can be printed after orchestration completes (device path does this in `aicpu_executor.cpp`).

---

## 2. The Fanout Spinlock Helpers

Functions (static inline):
- `task_fanout_lock(PTO2TaskDescriptor*)`
- `task_fanout_unlock(PTO2TaskDescriptor*)`

What they protect:
- `producer->fanout_head`
- `producer->fanout_count`

Why a per-task lock exists:
- While orchestration is still submitting tasks, scheduler threads can concurrently complete tasks and traverse fanout lists (device scheduler does a snapshot under this lock).
- So: the lock is the rendezvous point between “graph building” and “completion propagation”.

Key detail:
- It uses `PTO2_EXCHANGE(&task->fanout_lock, 1)` and spins with `PTO2_SPIN_PAUSE_LIGHT()`.
- Unlock uses `PTO2_STORE_RELEASE(&task->fanout_lock, 0)` to ensure the fanout writes are visible before releasing.

---

## 3. `pto2_orchestrator_init(...)`

Signature:
- `bool pto2_orchestrator_init(PTO2OrchestratorState* orch, PTO2SharedMemoryHandle* sm_handle, void* gm_heap, int32_t heap_size)`

“Line-by-line” intent, grouped:

### 3.1 Zero and store pointers
- `memset(orch, 0, sizeof(...))`
  - ensures every field starts clean (including stats and pointer fields)
- stores:
  - `orch->sm_handle = sm_handle`
  - `orch->gm_heap_base = gm_heap`
  - `orch->gm_heap_size = heap_size`

### 3.2 Initialize ring structures
These structures are pure metadata. They don’t allocate device memory (they *use* existing regions):

- HeapRing:
  - `pto2_heap_ring_init(&orch->heap_ring, gm_heap, heap_size, &sm_handle->header->heap_tail)`
  - `heap_tail` is **written by scheduler** (back-pressure); orchestrator reads it when allocating.

- TaskRing:
  - `pto2_task_ring_init(&orch->task_ring, sm_handle->task_descriptors, header->task_window_size, &header->last_task_alive)`
  - `last_task_alive` is **written by scheduler** to indicate which slots can be reused.

- DepListPool:
  - `pto2_dep_pool_init(&orch->dep_pool, sm_handle->dep_list_pool, header->dep_list_pool_size)`
  - Entry 0 is reserved as NULL; the pool typically allocates from 1.

### 3.3 Initialize TensorMap
- `pto2_tensormap_init_default(&orch->tensor_map)`
  - allocates CPU-side buckets + entry pool + free list + per-task entry heads
  - the TensorMap itself is not in shared memory; it’s private to orchestrator
- `orch->tensor_map.orch = orch`
  - required so `pto2_orchestrator_sync_tensormap` can read `header->last_task_alive`

### 3.4 Initialize scope stack storage
Orchestrator keeps:
- `scope_tasks[]`: a flat vector of task IDs
- `scope_begins[]`: begin offsets per nested scope
- `scope_stack_top`: current depth

So that `scope_end` can “release” all tasks created in that scope.

Failure behavior:
- if malloc fails, it frees what it can and returns false.

---

## 4. `pto2_orchestrator_reset(...)`

This resets:
- local rings (`heap_ring`, `task_ring`, `dep_pool`)
- TensorMap
- scope state and counters

It also writes shared memory header fields:
- `current_task_index = 0`
- `heap_top = 0`
- `orchestrator_done = 0`

Why this matters:
- the scheduler discovers tasks by scanning `[0, current_task_index)`.
- resetting must “rewind” that to prevent reading stale task descriptors.

---

## 5. Scope Management

### 5.1 `pto2_scope_begin(orch)`

This is purely stack bookkeeping:
- increments scope depth
- stores `scope_begins[top] = scope_tasks_size`

No scheduler state is touched here.

### 5.2 `scope_tasks_push(orch, task_id)` (static helper)

This pushes task IDs into a flat vector and grows it by `realloc` when needed.

This is called from `pto2_submit_task(...)` after allocating the task id.

### 5.3 `pto2_scope_end(orch)`

What it does:
- computes `[begin, end)` task ID range belonging to the current scope
- if `orch->scheduler != nullptr`, calls:
  - `pto2_scheduler_on_scope_end(orch->scheduler, &scope_tasks[begin], count)`
- rewinds the flat buffer:
  - `scope_tasks_size = begin`

**Important device-path caveat**

On real `a2a3`, the scheduler loop is implemented in `aicpu_executor.cpp`, not `pto_scheduler.cpp`.

However, in the device orchestrator thread, `PTO2Runtime` is created via `pto2_runtime_create_from_sm(...)` which initializes the *generic* scheduler (`pto_scheduler.cpp`) and sets `orch->scheduler = &rt->scheduler`.

So:
- `pto2_scope_end` will call `pto2_scheduler_on_scope_end` on the generic scheduler data structures.
- That generic scheduler is not the thing completing tasks on device, so it doesn’t advance device reclamation.

Practical takeaway:
- `scope_end` still matters for **logical lifetime modeling**, but **device resource reclamation** is not fully wired to it in the current device scheduler implementation.
- For the current examples (task_count < task_window_size; heap big), this is “OK”; for long-running graphs, this becomes a correctness/performance point to fix.

---

## 6. `pto2_add_consumer_to_producer(...)`

Signature:
- `void pto2_add_consumer_to_producer(PTO2OrchestratorState* orch, PTO2TaskDescriptor* producer, int32_t producer_id, int32_t consumer_id)`

This is the “connect the graph” primitive:
- consumer depends on producer
- so producer must list consumer in its fanout list
- and producer must increase its fanout_count

### 6.1 Lock the producer fanout fields
- `task_fanout_lock(producer)`

This synchronizes with:
- device scheduler completion code that snapshots `fanout_head` under this same lock.

### 6.2 Fast path: producer already completed (device parallel mode)

If `orch->aicpu_task_completed` is non-null:
- compute `prod_slot = producer_id & orch->aicpu_window_mask`
- if `aicpu_task_completed[prod_slot] >= 2` (meaning “completed”):
  - increment consumer’s device-side fanin refcount:
    - `__atomic_fetch_add(&orch->aicpu_fanin_refcount[cons_slot], 1, __ATOMIC_ACQ_REL)`
  - unlock and return

What this does:
- avoids pushing into producer’s fanout list (since the producer’s completion has already happened)
- ensures the consumer still accounts for this producer dependency

Why it’s needed:
- Orchestration and scheduling run concurrently on device.
- A producer may finish before a consumer is even submitted.

### 6.3 Normal path: prepend into fanout list

- `producer->fanout_head = pto2_dep_list_prepend(&orch->dep_pool, producer->fanout_head, consumer_id)`
- `producer->fanout_count++`

Semantics:
- fanout list is a singly-linked list in DepListPool.
- prepend is O(1) but produces a “reverse submission order” list.

### 6.4 “Scheduler mode” completion check (generic scheduler)

This block:

```cpp
if (orch->scheduler) {
  // check sched->task_state[prod_slot] >= COMPLETED
  // if so, sched->fanin_refcount[cons_slot]++
}
```

is designed for the **generic runtime2 scheduler**.

On the real device scheduler path:
- this typically does nothing because the generic scheduler is not updated to COMPLETED.
- the **actual** completion-aware logic is the device-parallel-mode branch above (`aicpu_task_completed`).

### 6.5 Unlock
- `task_fanout_unlock(producer)`

---

## 7. `pto2_alloc_packed_buffer(...)`

Signature:
- `void* pto2_alloc_packed_buffer(PTO2OrchestratorState* orch, int32_t total_size)`

Behavior:
- allocates from HeapRing with alignment inside `pto2_heap_ring_alloc`
- increments stats
- publishes:
  - `header->heap_top = orch->heap_ring.top` (release store)

Why write `heap_top`:
- Scheduler can use it (in a fully wired design) to understand how far heap has advanced.
- Orchestrator uses `heap_tail` for back-pressure; writing `heap_top` makes the header symmetric.

---

## 8. `pto2_submit_task(...)` (the core of everything)

Signature:
- `void pto2_submit_task(PTO2OrchestratorState* orch, int32_t kernel_id, PTO2WorkerType worker_type, PTOParam* params, int32_t num_params)`

This function is long, but its logic is very structured. Below is a “step-by-step” mapping to the code’s `STEP 0..7` comments.

### Step 0 — Sync TensorMap validity (and occasionally cleanup)

Code:
- `pto2_orchestrator_sync_tensormap(&orch->tensor_map)`

What it does (see `src/runtime/tensormap_and_ringbuffer/runtime/pto_tensormap.cpp`):
- reads `header->last_task_alive` from shared memory
- updates `tm->last_task_alive`
- periodically removes stale entries from bucket chains (per-task cleanup)

Why it matters:
- TensorMap lookup must not build dependencies to tasks that have been “retired” and whose slots may be reused.

### Step 0.5 — Enforce scope discipline

- `assert(orch->scope_stack_top >= 0 && "Cannot submit task outside a scope");`

This runtime requires every submission to be inside at least one scope, because:
- tasks start with `fanout_count = 1` (scope owns a lifetime reference)
- scope_end releases that reference
- without scopes, tasks would either leak or deadlock ring reclamation logic

### Step 1 — Allocate a task id (TaskRing)

- `int32_t task_id = pto2_task_ring_alloc(&orch->task_ring);`
- `PTO2TaskDescriptor* task = pto2_task_ring_get(&orch->task_ring, task_id);`

What TaskRing enforces:
- there’s a maximum number of “active” tasks within `task_window_size`.
- if window is full, orchestrator spins until scheduler advances `last_task_alive`.

### Step 1.5 — Initialize descriptor fields

It sets:
- identity:
  - `task_id`, `kernel_id`, `worker_type`
- fanin:
  - `fanin_head=0`, `fanin_count=0`
- fanout:
  - `fanout_head=0`, `fanout_lock=0`, `fanout_count=1`
- outputs:
  - `packed_buffer_base/end = NULL`, `num_outputs=0`
- status:
  - `is_active=true`

Then:
- `scope_tasks_push(orch, task_id)`

### Step 2 — Copy params into task-owned storage

Two separate arrays are involved:

1. `task->params[]` receives a raw memcpy of `PTOParam` structures.
2. `task->tensor_copies[]` receives a copy of the `Tensor` structs from the caller.
   - and then `task->params[i].tensor` is redirected to point into `task->tensor_copies[i]`.

Why this is crucial:
- On the device, AICore kernels will receive `Tensor*` pointers from `PTO2DispatchPayload`.
- Those pointers must remain valid after orchestration stack frames unwind.
- So they must point into shared-memory-backed task descriptor storage.

### Step 2 (first pass) — For each param: build deps + compute output allocation size

Loop: `for i in [0..num_params)`

#### INPUT / INOUT

For each INPUT or INOUT:
1. `pto2_tensormap_lookup(&orch->tensor_map, params[i].tensor, &lookup_result)`
2. For each overlapping entry in `lookup_result`:
   - identify producer task id
   - deduplicate in `fanin_temp[]`
   - link consumer into producer via `pto2_add_consumer_to_producer(...)`

Dedup logic:
- it’s a small O(N) scan because `PTO2_MAX_INPUTS` is small (16).

INOUT COVERED pruning:
- only for INOUT, if overlap status is `COVERED`:
  - if `!entry.with_alloc`, remove the older TensorMap entry.

What this means operationally:
- repeated INOUT updates to the same state tensor become a near-linear chain
- TensorMap stays compact and lookup doesn’t grow unbounded

#### OUTPUT

For each OUTPUT:
- if `params[i].tensor->buffer.addr == 0`, the runtime will allocate it.
- so it adds:
  - `ALIGN_UP(buffer.size, PTO2_PACKED_OUTPUT_ALIGN)`
to `total_output_size`.

### Step 3 — Allocate packed output buffer (HeapRing)

If `total_output_size > 0`:
- allocate one contiguous range from HeapRing
- assign each “needs alloc” OUTPUT a slice of this packed buffer at 1024B alignment
- write back addresses:
  - into `task->tensor_copies[i].buffer.addr`
  - and into the caller’s `params[i].tensor->buffer.addr`

Why write back to the caller `Tensor`:
- orchestration code often needs the allocated address for later `Tensor.view(...)` or downstream tasks.

### Step 4 — Insert outputs into TensorMap (make this task the producer)

For each OUTPUT or INOUT param:
- `pto2_tensormap_insert(&orch->tensor_map, &task->tensor_copies[i], task_id, is_output)`

This is what makes **future** consumers depend on this task.

### Step 5 — Finalize the fanin list

For each producer id in `fanin_temp[]`:
- `task->fanin_head = pto2_dep_list_prepend(&orch->dep_pool, task->fanin_head, producer_task_id)`

Then:
- `__atomic_store_n(&task->fanin_count, fanin_count, __ATOMIC_RELEASE);`

The release store is important:
- it ensures the scheduler won’t see a new `fanin_count` before the list nodes are visible.

### Step 5b — “Already ready” check (device parallel mode)

This exists for a very specific race:
- if `pto2_add_consumer_to_producer` saw some producers already completed,
  it incremented `aicpu_fanin_refcount[consumer_slot]` *before* `fanin_count` was finalized.

So after `fanin_count` is stored, orchestrator checks:
- if `refcount >= fanin_count`, push this task id into `orch_ready_queue`.

This is a performance optimization:
- it prevents scheduler threads from having to do a costly scan for readiness.

### Step 6 — Initialize in the generic scheduler (optional)

If `orch->scheduler && orch->init_task_on_submit`:
- `pto2_scheduler_init_task(orch->scheduler, task_id, task);`

Device-path note:
- The real device scheduler loop does not use `pto_scheduler.cpp` state.
- So this step is mostly overhead on device unless you explicitly rely on the generic scheduler for something.

### Step 7 — Publish to shared memory (`current_task_index`)

- `PTO2_STORE_RELEASE(&header->current_task_index, orch->task_ring.current_index);`

This is the “task visibility fence”:
- scheduler threads discover “how many tasks exist” by reading `current_task_index`.

---

## 9. `pto2_orchestrator_done(...)`

Writes:
- `header->orchestrator_done = 1` (release store)

Scheduler threads typically use this to decide whether “no more tasks will appear”.

---

## 10. `pto2_orchestrator_wait_all(...)`

This function only makes sense when:
- `orch->scheduler` points at an actively-updated scheduler implementation

On real device:
- completion tracking is done by the `aicpu_executor.cpp` scheduler loop.
- so this wait function is not the thing you’d call to wait for device completion.

---

## 11. Practical Debug Tips (Orchestrator bugs)

If you suspect “graph build” issues, the fastest checks:

1. **Verify `fanin_count` and `fanin_head` consistency**
   - A consumer that never runs often has `fanin_count > fanin_refcount` forever.
2. **Check TensorMap overlap costs**
   - excessive `complex_overlap()` calls can make orchestration extremely slow.
3. **Look for INOUT chains**
   - state tensors updated by many tasks should show INOUT “COVERED pruning” behavior; otherwise TensorMap grows.
4. **Watch for ring deadlocks**
   - TaskRing/HeapRing spin messages in `pto_ring_buffer.cpp` are designed to surface deadlock cycles.
