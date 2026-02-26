# Tensormap–Ringbuffer Runtime: Code Walk (Architecture → Functions)

Last verified against repo state on **2026-02-26**.

This document is the “deep code tour” companion to `docs/tensormap-ringbuffer-runtime-guide.md`. It focuses on **what each module/function actually does**, and how the paged-attention example drives the runtime.

---

## 0. Where to Start (Entry Points)

If you want to understand the runtime end-to-end, read code in this order:

1. **Orchestration ABI (what the orchestration `.so` can call)**
   - `src/runtime/tensormap_and_ringbuffer/orchestration/pto_orchestration_api.h`
2. **Orchestration (paged-attention DAG builder)**
   - `tests/device_tests/tensormap_and_ringbuffer/paged_attention/kernels/orchestration/paged_attention_orch.cpp`
3. **Runtime2 (in-process orchestrator + scheduler building blocks)**
   - `src/runtime/tensormap_and_ringbuffer/runtime/pto_runtime2.h`
   - `src/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.h`
   - `src/runtime/tensormap_and_ringbuffer/runtime/pto_tensormap.h`
   - `src/runtime/tensormap_and_ringbuffer/runtime/pto_ring_buffer.h`
   - `src/runtime/tensormap_and_ringbuffer/runtime/pto_shared_memory.h`
4. **Device execution path (what runs on real a2a3)**
   - AICPU: `src/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp`
   - AICore: `src/runtime/tensormap_and_ringbuffer/aicore/aicore_executor.cpp`
   - Handshake struct: `src/runtime/tensormap_and_ringbuffer/runtime/runtime.h`
5. **Tensor overlap semantics (dependency discovery correctness & cost)**
   - Methods used by orchestration `.so`: `src/runtime/tensormap_and_ringbuffer/orchestration/tensor_orch.cpp`
   - Methods only in runtime targets: `src/runtime/tensormap_and_ringbuffer/runtime/tensor.cpp`

---

## 1. Macro Architecture (What Runs Where)

At runtime there are three “actors”:

- **Host CPU** (Python runner)
  - Compiles orchestration + kernels, loads binaries to device, and collects perf traces.
  - Driver script: `examples/scripts/run_example.py`
- **AICPU** (device CPU cores)
  - Runs the orchestration entry (`aicpu_orchestration_entry`) and the scheduler loop.
  - Main file: `src/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp`
- **AICore** (compute cores: AIC + AIV)
  - Polls registers for dispatch; executes kernel entry points and writes perf records.
  - Main file: `src/runtime/tensormap_and_ringbuffer/aicore/aicore_executor.cpp`

```mermaid
flowchart LR
  Host[Host CPU\nexamples/scripts/run_example.py] -->|upload .o + .so\nallocate GM buffers| Dev[Device]
  subgraph Dev[Device / a2a3]
    Orchestrator[AICPU Orchestrator\n(aicpu_orchestration_entry)] -->|writes TaskDescriptor + DepList\nupdates current_task_index| SM[(GM Shared Memory\nHeader + TaskDescriptor[] + DepListPool)]
    Scheduler[AICPU Scheduler Threads\n(resolve_and_dispatch_pto2)] -->|scan/drain/complete/dispatch| SM
    Scheduler -->|write regs\nDATA_MAIN_BASE / COND| Regs[(Per-core registers)]
    Regs --> AICore[AICore workers\n(aicore_execute)]
    AICore -->|writes perf records| Perf[(Perf buffers)]
    Scheduler -->|patch dispatch/finish ts\n+ export scheduler phases| Perf
  end
  Host -->|poll perf buffers\nexport JSON + reports| Out[outputs/\nmerged_swimlane_*.json\npto2_schedule_report_*.md]
```

### 1.1 What is “tensormap_and_ringbuffer”?

It’s a task-graph runtime with two core ideas:

1. **Dependency discovery by memory overlap**
   - “Who produces the buffer region I’m reading/writing?” is discovered by looking up the `Tensor` in a **TensorMap**.
2. **Bounded memory / bounded metadata**
   - Task descriptors are in a ring window (**TaskRing**) and output buffers come from a bump-pointer ring (**HeapRing**).
   - Dependency lists are stored in a ring allocator (**DepListPool**).

The intent is: submit many small tasks and let the scheduler keep AIC/AIV cores busy with minimal host involvement.

---

## 2. Key Data Types (These Drive Everything)

### 2.1 `Tensor` (the dependency “key”)

File: `src/runtime/tensormap_and_ringbuffer/runtime/tensor.h`

`Tensor` is a *strided access descriptor* over an underlying allocation:
- `buffer.addr` / `buffer.size`: base address + total bytes of the allocation
- `start_offset`, `strides[]`, `repeats[]`, `ndims`: access pattern (in **elements**, not bytes)
- `dtype`: element size matters because overlap checks are done in **bytes**
- `version`, `overlap_type`: extra semantics; `OverlapType::Fuzzy` forces conservative dependency behavior

Two very important implementation details:
- **`Tensor::optimize()` sorts strides** so validation and overlap checks get simpler.
  - Implemented in `src/runtime/tensormap_and_ringbuffer/orchestration/tensor_orch.cpp` (`resort_strides()`).
- **Overlap checks can get expensive** for complex layouts:
  - Fast-ish path: 1D contiguous or “hyper-rectangle” overlap for matching dtype/ndims/strides.
  - Slow path: `complex_overlap()` iterates segments (`Tensor::ContiguousMemSegIterator`) in `src/runtime/tensormap_and_ringbuffer/runtime/tensor.cpp`.

### 2.2 `PTOParam` (submit-time semantics)

File: `src/runtime/tensormap_and_ringbuffer/runtime/pto_types.h`

Each task argument is a `PTOParam`:
- `INPUT`: read-only (creates dependency on overlapping producers)
- `OUTPUT`: write-only (registers a produced region in TensorMap)
- `INOUT`: read-then-write (both depends-on and becomes-new-producer; used for state updates)
- `SCALAR`: no TensorMap tracking

Critical rule:
- `OUTPUT` can have `tensor->buffer.addr == 0` to request runtime allocation (address is written back).

### 2.3 `PTO2TaskDescriptor` (device-visible task metadata)

File: `src/runtime/tensormap_and_ringbuffer/runtime/pto_runtime2_types.h`

This struct lives in **device shared memory** (`TaskDescriptor[]`) and is written by the orchestrator and read by the scheduler.

Fields to care about:
- `task_id`, `kernel_id`, `worker_type`
- `fanin_head`, `fanin_count` (dependency list)
- `fanout_lock`, `fanout_head`, `fanout_count` (dependents list + scope lifetime reference)
- `packed_buffer_base/end` (heap allocation range for outputs)
- `params[]`, `tensor_copies[]`, `param_count` (task-owned argument storage)

### 2.4 Shared memory header (flow control only)

File: `src/runtime/tensormap_and_ringbuffer/runtime/pto_shared_memory.h`

The shared-memory region contains:
- `PTO2SharedMemoryHeader` (flow control + offsets)
- `PTO2TaskDescriptor[]` (task ring window)
- `PTO2DepListEntry[]` (fanin/fanout linked-list nodes)

The **GM heap storage is separate** (not inside shared memory). Shared memory only carries `heap_top`/`heap_tail` integers for back-pressure.

Implementation of the layout builder:
- `src/runtime/tensormap_and_ringbuffer/runtime/pto_shared_memory.cpp`

---

## 3. Orchestration ABI (How `.so` Talks to Runtime)

File: `src/runtime/tensormap_and_ringbuffer/orchestration/pto_orchestration_api.h`

The orchestration `.so` does *not* link against runtime `.cpp` code. Instead:
- It sees `PTO2Runtime` as an **opaque** struct whose first field is an ops table pointer.
- All calls go through `PTO2RuntimeOps` function pointers:
  - `submit_task`, `scope_begin`, `scope_end`, `orchestration_done`

The macro you see in orchestration code:

- `PTO2_SCOPE(rt) { ... }`

is implemented as a C++17 “if-init” trick that instantiates an RAII guard:
- `PTO2ScopeGuard` calls `scope_begin` in ctor and `scope_end` in dtor.

So in paged-attention, a `PTO2_SCOPE(rt) { ... }` block means:
- all tasks submitted inside are “owned” by that scope, and scope end releases one lifetime reference per task.

Important: the scope mechanism is implemented in the orchestrator (`pto_orchestrator.cpp`) by holding a list of task IDs and calling a “release” hook on scope end. Whether that release affects memory reclamation depends on which scheduler implementation is being used (see Section 6).

---

## 4. Orchestrator Internals (`pto2_submit_task` in Slow Motion)

Files:
- `src/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.h`
- `src/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp`

The orchestrator is responsible for:
- allocating task IDs (TaskRing)
- allocating output buffers (HeapRing)
- discovering dependencies (TensorMap lookup)
- registering produced regions (TensorMap insert)
- publishing the task so scheduler threads can see it (`current_task_index`)

### 4.1 `pto2_submit_task(...)` step-by-step

Function: `void pto2_submit_task(PTO2OrchestratorState* orch, int32_t kernel_id, PTO2WorkerType worker_type, PTOParam* params, int32_t num_params)`

```mermaid
flowchart TD
  A[pto2_submit_task] --> S0[Step 0: sync TensorMap validity\n(read last_task_alive; optional cleanup)]
  S0 --> S1[Step 1: TaskRing alloc task_id\n(blocks if window full)]
  S1 --> S2[Step 2: init PTO2TaskDescriptor\nfanin=0, fanout_count=1, etc]
  S2 --> S3[Step 3: copy PTOParams + Tensor structs\nredirect params[i].tensor -> tensor_copies[i]]
  S3 --> S4[Step 4: build deps\nTensorMap lookup for INPUT/INOUT\nlink into producers' fanout lists]
  S4 --> S5[Step 5: allocate OUTPUT buffers\nHeapRing alloc packed_buffer\nwrite back buffer.addr]
  S5 --> S6[Step 6: insert OUTPUT/INOUT\ninto TensorMap as new producer]
  S6 --> S7[Step 7: finalize fanin list\nDepListPool prepend\nstore fanin_count with release]
  S7 --> S8[Step 8: publish\nheader.current_task_index = current_index]
```

**Step 0: sync tensormap validity**
- Calls `pto2_orchestrator_sync_tensormap(&orch->tensor_map)` (defined in `pto_tensormap.cpp`).
- Reads `header->last_task_alive` from shared memory and updates TensorMap’s “stale threshold”.
- Periodically calls `pto2_tensormap_cleanup_retired(...)` to remove entries belonging to retired tasks.

**Step 1: allocate a task ID**
- Calls `pto2_task_ring_alloc(&orch->task_ring)`.
- This can spin if `current_index - last_task_alive` reaches the window size.

**Step 2: initialize `PTO2TaskDescriptor`**
- Sets identity fields, initializes fanin/fanout heads to 0, and sets:
  - `fanout_count = 1` (the owning scope holds one reference)

**Step 3: copy parameters into task-owned storage**
- `params[]` is memcpy’d into `task->params[]`.
- Each `Tensor` pointed to by `params[i].tensor` is copied into `task->tensor_copies[i]`,
  then `task->params[i].tensor` is redirected to `&task->tensor_copies[i]`.

Why this matters:
- Scheduler/AICore will later see stable pointers (into shared-memory task descriptor), not transient stack pointers from orchestration code.

**Step 4: dependency discovery (fanin + fanout linking)**
- For each `INPUT` or `INOUT` param:
  1. `pto2_tensormap_lookup(&orch->tensor_map, params[i].tensor, &lookup_result)`
  2. For each overlapping producer entry:
     - add producer task_id to a local `fanin_temp[]` set (deduplicate)
     - link “consumer into producer fanout list” via `pto2_add_consumer_to_producer(...)`

Early-ready / “producer already completed” optimization:
- In device parallel mode, `pto2_add_consumer_to_producer` can early-return by checking `orch->aicpu_task_completed[producer_slot] >= 2`.
  - If producer already completed, it increments `orch->aicpu_fanin_refcount[consumer_slot]` and returns **without** linking into fanout.

INOUT “COVERED pruning”:
- If the new INOUT region fully covers a previous INOUT region and that old entry has `with_alloc == false`,
  the old entry is removed from TensorMap (`pto2_tensormap_remove_entry(...)`).
- The goal is to keep “state updates” from creating huge overlapping histories in TensorMap.

**Step 5: output allocation**
- For each `OUTPUT` param where `tensor->buffer.addr == 0`, compute aligned sizes and allocate one packed buffer:
  - `task->packed_buffer_base = pto2_alloc_packed_buffer(...)`
  - outputs get assigned offsets aligned to `PTO2_PACKED_OUTPUT_ALIGN` (1024B)
- The allocated address is written back both into:
  - `task->tensor_copies[i].buffer.addr`
  - and the caller-visible `params[i].tensor->buffer.addr`

**Step 6: insert produced regions into TensorMap**
- For each `OUTPUT` and `INOUT`, call:
  - `pto2_tensormap_insert(&orch->tensor_map, &task->tensor_copies[i], task_id, with_alloc)`
- This is what makes *future* consumers depend on this task.

**Step 7: finalize fanin list**
- Builds the linked list `task->fanin_head` by prepending entries in `fanin_temp[]` into DepListPool.
- Stores `fanin_count` with release semantics so scheduler sees a consistent list.

**Step 8: publish**
- Updates shared memory:
  - `header->current_task_index = orch->task_ring.current_index` (release store)

This is the “visibility barrier” the device scheduler uses to know which task IDs exist.

### 4.2 Scope begin/end

Functions:
- `pto2_scope_begin(PTO2OrchestratorState*)`
- `pto2_scope_end(PTO2OrchestratorState*)`

Implementation:
- `scope_begin` records a begin offset into a flat `scope_tasks[]` array.
- Each submitted task pushes its `task_id` into `scope_tasks[]`.
- `scope_end`:
  - computes the list of tasks in the scope
  - calls `pto2_scheduler_on_scope_end(...)` **if `orch->scheduler` is non-null**
  - then rewinds `scope_tasks_size` back to the begin offset

Whether `pto2_scheduler_on_scope_end` is meaningful depends on which scheduler path you’re using (Section 6).

---

## 5. TensorMap Internals (Why Lookup Works, and When It’s Expensive)

Files:
- `src/runtime/tensormap_and_ringbuffer/runtime/pto_tensormap.h`
- `src/runtime/tensormap_and_ringbuffer/runtime/pto_tensormap.cpp`

### 5.1 Hashing rule (correctness-critical)

`pto2_tensormap_hash(...)` hashes **only by `tensor->buffer.addr`**.

Reason:
- overlap detection must compare *all subregions* of the same base allocation in one bucket.

### 5.2 Bucket chain truncation (performance-critical)

Bucket chains are maintained in “newest first” order because insertion is at the head.

During lookup:
- if the first stale entry is found (`producer_task_id < last_task_alive`), the lookup:
  - truncates the chain at that point (`*prev_ptr = -1`)
  - marks all remaining entries as “not in bucket”
  - returns early

This is a key reason TensorMap stays performant under steady retirement.

### 5.3 Per-task cleanup

TensorMap also maintains a per-task linked list of entries (`task_entry_head[task_slot]`).

When orchestrator sees `last_task_alive` advance, it can call:
- `pto2_tensormap_cleanup_retired(...)`
to free all entries belonging to retired tasks in O(entries-of-retired-tasks).

---

## 6. Scheduler: Two Implementations (Don’t Mix Them Up)

There are two schedulers in this repo:

1. **Generic “runtime2” scheduler** (primarily used for simulation/in-process execution)
   - `src/runtime/tensormap_and_ringbuffer/runtime/pto_scheduler.cpp`
2. **Device scheduler loop** (the one used on real `a2a3`)
   - `src/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp`

They share the same shared-memory task descriptors, but differ in:
- how they track task state
- how they reclaim ring resources
- how they detect completion

### 6.1 Device scheduler loop (`aicpu_executor.cpp`)

High-level model:
- AICPU spawns `thread_num_` threads.
  - When `thread_num_ == 4`: **3 scheduler threads + 1 orchestrator thread**.
- Scheduler threads execute `resolve_and_dispatch_pto2(...)`:
  - SCAN: discover root tasks (`fanin_count==0`) up to `header->current_task_index`
  - ORCH_DRAIN: drain an “early-ready” queue written by the orchestrator
  - COMPLETE: poll per-core `RegId::COND` and, on completion, traverse fanout to ready consumers
  - DISPATCH: write registers to dispatch ready tasks to idle cores
- YIELD: if no progress, `std::this_thread::yield()`

```mermaid
flowchart TD
  Loop[while not done] --> Scan[SCAN\nclaim next_scan_index < current_task_index\nenqueue roots where fanin_count==0]
  Scan --> Drain[ORCH_DRAIN\npull tasks from orch_ready_queue]
  Drain --> Complete[COMPLETE\npoll COND for IDLE\non completion: traverse fanout\nfanin_refcount++ and enqueue when satisfied]
  Complete --> Dispatch[DISPATCH\nfor each idle core:\nready_queue pop\nbuild payload\nwrite COND=BUSY\nwrite DATA_MAIN_BASE=task_id+1]
  Dispatch -->|made progress?| MP{made_progress?}
  MP -- no --> Yield[YIELD\nthread::yield(); idle_iterations++]
  MP -- yes --> Loop
  Yield --> Loop
```

State tracking data structures:
- `s_pto2_task_completed[slot]`
  - used as a lightweight state marker:
    - `0`: not enqueued (and not known complete)
    - `1`: enqueued / ready
    - `2`: completed
- `s_pto2_fanin_refcount[slot]`
  - counts how many producer completions have arrived for that consumer

Completion propagation (core idea):
- When task `T` completes:
  - mark `T` complete
  - traverse its `fanout_head` linked list
  - for each consumer `C`:
    - `prev = fetch_add(fanin_refcount[C], 1)`
    - if `prev + 1 == fanin_count[C]`, enqueue `C`

Tail latency root cause (from profiling):
- COMPLETE phase depends on register polling (`RegId::COND`), which drives the large “tail” portion of schedule window time.

### 6.2 Generic scheduler (`pto_scheduler.cpp`)

This scheduler keeps richer bookkeeping:
- `task_state[slot]`: PENDING/READY/RUNNING/COMPLETED/CONSUMED
- `fanin_refcount[slot]` and `fanout_refcount[slot]`

It can also advance:
- `last_task_alive` (task ring reclamation)
- `heap_tail` (heap reclamation)

Important caveat for device path:
- The device scheduler loop does **not** currently reuse the `pto_scheduler.cpp` consumption logic, so shared-memory `last_task_alive` / `heap_tail` advancement is limited.
- In practice, current examples rely on:
  - large default `task_window_size` (65536)
  - large default `heap_size` (1 GiB)
  - task counts small enough to fit without needing reclamation mid-graph

---

## 7. AICore Execution (Dispatch Protocol + Perf)

File: `src/runtime/tensormap_and_ringbuffer/aicore/aicore_executor.cpp`

Per-core loop:
1. Wait for `Handshake.aicpu_ready`
2. Write `physical_core_id`, `core_type`, `aicore_done`
3. Set `RegId::COND = IDLE`
4. Poll `RegId::DATA_MAIN_BASE`:
   - `0`: no task
   - `AICORE_EXIT_SIGNAL`: exit
   - otherwise: task id encoded as `task_id + 1`
5. On new task:
   - set `COND = BUSY`
   - `dcci()` invalidate to read fresh payload
   - call kernel function pointer `function_bin_addr(args)`
   - optionally record perf record
   - set `COND = IDLE`

The payload object:
- `src/runtime/tensormap_and_ringbuffer/runtime/pto2_dispatch_payload.h`

Note:
- Kernels expect each “tensor arg” to be a `Tensor*` pointer (not a raw buffer address). The AICPU packs those pointers into `payload->args[]`.

---

## 8. Profiling & Reports (What’s Collected, Where It Comes From)

If you run with `--enable-profiling`, artifacts are produced under `outputs/`:
- `perf_swimlane_*.json` / `merged_swimlane_*.json`
- `pto2_schedule_report_*.md`

The schedule report is generated by:
- `tools/pto2_schedule_report.py`

and consumes:
- AICore perf records (task start/end)
- AICPU scheduler phase counters exported into perf header (scan/orch_drain/complete/dispatch/yield)

Repro command is documented in `docs/tensormap-ringbuffer-runtime-guide.md` Section 7.0.

---

## 9. Paged Attention Deep Dive

For a very detailed, code-level explanation of the example DAG and kernels, see:
- `docs/paged-attention-example-codewalk.md`

---

## Appendix A. Function Index (What to Read When You’re Debugging)

This appendix is intentionally “function-by-function”: it’s meant for people who want to jump straight to the implementation details.

### A.1 Shared memory (`pto_shared_memory.*`)

Files:
- `src/runtime/tensormap_and_ringbuffer/runtime/pto_shared_memory.h`
- `src/runtime/tensormap_and_ringbuffer/runtime/pto_shared_memory.cpp`

Functions:
- `pto2_sm_calculate_size(task_window_size, dep_list_pool_size)`
  - Computes required bytes for: header + task descriptors + dep list pool.
  - Note: heap storage is not included (heap is separate GM memory).
- `pto2_sm_create(task_window_size, heap_size, dep_list_pool_size)`
  - Allocates an aligned buffer, sets `handle->header/task_descriptors/dep_list_pool` pointers, then calls `pto2_sm_init_header`.
- `pto2_sm_create_from_buffer(sm_base, sm_size, task_window_size, heap_size, dep_list_pool_size)`
  - Wraps an existing buffer (device GM) as shared memory; validates size; sets pointers and initializes header.
- `pto2_sm_init_header(handle, task_window_size, heap_size, dep_list_pool_size)`
  - Writes the flow-control fields (`current_task_index`, `last_task_alive`, …) and computes offsets.
  - Initializes dep_list entry 0 as a NULL marker (`task_id=-1`, `next_offset=0`).
- `pto2_sm_reset(handle)`
  - Resets flow-control fields and clears task descriptors + dep list pool contents.
- `pto2_sm_validate(handle)`
  - Sanity checks offsets, alignment, and pointer ranges.

### A.2 Ring buffers (`pto_ring_buffer.*`)

Files:
- `src/runtime/tensormap_and_ringbuffer/runtime/pto_ring_buffer.h`
- `src/runtime/tensormap_and_ringbuffer/runtime/pto_ring_buffer.cpp`

HeapRing:
- `pto2_heap_ring_init(ring, base, size, tail_ptr)`
  - Initializes the bump-pointer heap allocator with back-pressure on `*tail_ptr`.
- `pto2_heap_ring_try_alloc(ring, size)` / `pto2_heap_ring_alloc(ring, size)`
  - Implements wrap-around allocation without splitting buffers across the wrap boundary.
  - `alloc` spins until `try_alloc` succeeds; emits periodic “BLOCKED” logs and can `exit(1)` on deadlock.

TaskRing:
- `pto2_task_ring_init(ring, descriptors, window_size, last_alive_ptr)`
  - Sliding window allocator for task IDs; back-pressure on `*last_alive_ptr`.
- `pto2_task_ring_try_alloc(ring)` / `pto2_task_ring_alloc(ring)`
  - Allocates a new task_id if `current_index - last_task_alive < window_size - 1`.
  - `alloc` spins with deadlock detection message that explains the “scope reference” cycle risk.

DepListPool:
- `pto2_dep_pool_init(pool, base, capacity)`
  - Initializes `top=1` and reserves entry 0 as NULL.
- `pto2_dep_pool_alloc_one(pool)`
  - Wraps to 1 at capacity; assumes old entries are reclaimed together with task retirement.
- `pto2_dep_list_prepend(pool, head, task_id)`
  - Push-front list node allocator; returns new head offset.

### A.3 TensorMap (`pto_tensormap.*`)

Files:
- `src/runtime/tensormap_and_ringbuffer/runtime/pto_tensormap.h`
- `src/runtime/tensormap_and_ringbuffer/runtime/pto_tensormap.cpp`

Init/reset:
- `pto2_tensormap_init(tm, num_buckets, pool_size)` / `pto2_tensormap_init_default(tm)`
  - Allocates buckets + entry pool + free list + per-task entry heads.
- `pto2_tensormap_reset(tm)` / `pto2_tensormap_destroy(tm)`

Core algorithms:
- `pto2_tensormap_hash(tm, tensor)`
  - Hashes **only** `tensor->buffer.addr` (correctness requirement).
- `pto2_tensormap_lookup(tm, tensor, result)`
  - Scans the bucket chain and calls `tensor->is_overlap(entry->tensor)` for each valid entry.
  - Truncates bucket chain when the first stale entry is encountered (performance-critical).
- `pto2_tensormap_insert(tm, tensor, producer_task_id, with_alloc)`
  - Inserts at bucket head; links into the producer’s per-task entry chain.
- `pto2_tensormap_cleanup_retired(tm, old_last_task_alive, new_last_task_alive)`
  - Uses per-task chains to free entries belonging to retired tasks.
- `pto2_tensormap_remove_entry(tm, entry_idx)`
  - Removes an entry from per-task chain + bucket chain (used by INOUT COVERED pruning).

Orchestrator-driven synchronization:
- `pto2_orchestrator_sync_tensormap(tm, force=false)`
  - Reads `header->last_task_alive` and optionally triggers `cleanup_retired` periodically.

### A.4 Orchestrator (`pto_orchestrator.*`)

Files:
- `src/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.h`
- `src/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp`

Lifecycle:
- `pto2_orchestrator_init(orch, sm_handle, gm_heap, heap_size)` / `destroy` / `reset`

Scopes:
- `pto2_scope_begin(orch)`
  - Pushes a begin index into the flat `scope_tasks` buffer.
- `pto2_scope_end(orch)`
  - Pops a scope, and (if `orch->scheduler` is set) calls `pto2_scheduler_on_scope_end(...)`.

Submission:
- `pto2_submit_task(...)`
  - The main “build DAG + publish task” pipeline (Section 4).
- `pto2_add_consumer_to_producer(orch, producer_desc, producer_id, consumer_id)`
  - Takes `producer_desc->fanout_lock` and prepends consumer into producer’s fanout list.
  - In device parallel mode, can early-return by checking `aicpu_task_completed` and directly incrementing `aicpu_fanin_refcount`.
- `pto2_alloc_packed_buffer(orch, total_size)`
  - Allocates from HeapRing and updates `header->heap_top`.

### A.5 Device executor / scheduler (`aicpu_executor.cpp`)

File:
- `src/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp`

Core responsibilities:
- Core discovery and register address caching
  - `handshake_all_cores(runtime)`
  - `assign_cores_to_threads()`
- Scheduling loop and dispatch protocol (device runtime)
  - `resolve_and_dispatch_pto2(runtime, thread_idx, cur_thread_cores, core_num)`
    - Implements SCAN / ORCH_DRAIN / COMPLETE / DISPATCH / YIELD.
- Dispatch payload packing
  - `build_pto2_payload(out, runtime, task_desc, ...)`
    - Packs `Tensor*` pointers and scalars into `PTO2DispatchPayload.args[]`.

Orchestrator thread (when thread_num==4):
- Dynamically loads the orchestration `.so`, creates a `PTO2Runtime` bound to device shared memory,
  then runs `PTO2_SCOPE(rt) { aicpu_orchestration_entry(rt, args, arg_count); }`.

### A.6 AICore executor (`aicore_executor.cpp`)

File:
- `src/runtime/tensormap_and_ringbuffer/aicore/aicore_executor.cpp`

Functions:
- `aicore_execute(runtime, block_idx, core_type)`
  - Handshake; then poll `RegId::DATA_MAIN_BASE`; run kernels; write perf records.
- `execute_task(task_ptr)` (inline)
  - Reads `payload->function_bin_addr` and calls it as `kernel(args)`.
