# Annotated Walkthrough: `aicpu_executor.cpp` (Device Scheduler + Orchestrator Thread)

Last verified against repo state on **2026-02-26**.

This document is a deep, code-level explanation of:
- `src/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp`

It focuses on **the real `a2a3` device execution path**:
- core handshake and register mapping
- AICPU thread model (3 scheduler threads + 1 orchestrator thread when `sche_cpu_num==4`)
- the scheduler loop phases (SCAN / ORCH_DRAIN / COMPLETE / DISPATCH / YIELD)
- profiling export (scheduler phase cycles + perf timestamp patching)

Companion docs:
- Architecture + profiling usage: `docs/tensormap-ringbuffer-runtime-guide.md`
- Runtime code tour: `docs/tensormap-ringbuffer-runtime-codewalk.md`
- Orchestrator internals: `docs/annotated-pto-orchestrator.md`
- Line-numbered scheduler loop (`resolve_and_dispatch_pto2`): `docs/linebyline-aicpu-resolve-and-dispatch-pto2.md`
- Platform layer (a2a3 vs a2a3sim glue): `docs/platform-codewalk.md`
- Platform deep-dive (profiling subsystem): `docs/annotated-platform-profiling.md`

---

## 0. What This File Owns

`aicpu_executor.cpp` is the “device runtime glue” that:

1. **Discovers AICore workers** (AIC vs AIV) via handshake.
2. **Maps worker IDs to register addresses** so the scheduler can dispatch tasks quickly.
3. **Runs a scheduling loop** on multiple AICPU threads:
   - detects ready tasks
   - dispatches tasks to idle cores
   - detects completion by polling `RegId::COND`
   - propagates readiness to dependents by traversing fanout lists
4. Optionally (when `thread_num==4`) runs **device-side orchestration** on a dedicated AICPU thread by `dlopen`ing the orchestration `.so` and calling `aicpu_orchestration_entry`.

This is *the* code path that produced your `a2a3` schedule profile report.

---

## 1. Thread Model and Global State

### 1.1 `AicpuExecutor` fields (what they mean)

Key fields grouped by purpose:

**Threading / lifecycle**
- `thread_idx_`: assigns a unique index to each AICPU thread entering `run()`
- `initialized_`, `init_done_`, `init_failed_`: init barrier
- `finished_`, `finished_count_`: last-thread cleanup detection

**Core discovery**
- `aic_cores_[]`, `aiv_cores_[]`: discovered worker ids with core type and register address
- `core_id_to_reg_addr_[]`: quick map `worker_id -> reg_addr`
- `regs_`: platform “register address table” base (from `get_platform_regs()`)

**Per-thread core assignments**
- `core_assignments_[t][]`: list of worker IDs owned by thread `t`
- `core_count_per_thread_[t]`: number of cores owned by thread `t`
- when `thread_num_==4`, thread 3 is orchestrator and owns 0 cores.

**Ready queues**
- Two queues, one per core type:
  - `ready_queue_aic_[]` + head/tail + `ready_queue_aic_lock_`
  - `ready_queue_aiv_[]` + head/tail + `ready_queue_aiv_lock_`
- They are simple arrays with monotonic head/tail indices and a mask for wrap.

**Scheduling progress**
- `total_tasks_`: final task count; in device orchestration it is set by thread 3 at the end.
- `completed_tasks_`: global completion counter across all scheduler threads.
- `executing_task_ids_[worker_id]`: local view of what task each owned worker is running.

**Orchestrator-to-scheduler “early-ready” queue**
- `orch_ready_queue_`, `orch_ready_head_`, `orch_ready_tail_`, `orch_ready_capacity_`
- Set by thread 3 after creating `PTO2Runtime` and pointing at `rt->orchestrator.orch_ready_queue`.

**Profiling**
- `dispatch_timestamps_[core]`: AICPU dispatch timestamp used to patch perf records.
- `core_dispatch_counts_[core]`: used to decide when to switch perf buffers.

### 1.2 Global arrays: the device scheduler’s task state

At file scope:
- `s_pto2_fanin_refcount[PTO2_MAX_SLOTS]`
- `s_pto2_task_completed[PTO2_MAX_SLOTS]`
- `s_pto2_payload_per_core[RUNTIME_MAX_WORKER]`

Interpretation:
- `fanin_refcount[slot]`: how many producer completions have been observed for a consumer task.
- `task_completed[slot]` is used as a small state machine:
  - `0`: not enqueued
  - `1`: enqueued/ready (or already scanned as root)
  - `2`: completed

**Important constraint:** this device scheduler does not fully implement “slot reuse”.
It assumes `task_count_final` stays within `task_window_size` (default 65536), so wrap/reuse isn’t needed for current workloads.

---

## 2. Entry Point: `aicpu_execute(Runtime* runtime)`

This is the public C entry point.

High-level flow:
1. validate `runtime != nullptr`
2. set `g_aicpu_executor.regs_ = get_platform_regs()`
3. call `g_aicpu_executor.init(runtime)` once (thread-safe)
4. wait for `init_done_` or abort on `init_failed_`
5. call `g_aicpu_executor.run(runtime)` (per-thread behavior)
6. last finishing thread calls `deinit()`

Key idea:
- every AICPU thread enters `aicpu_execute`, but only one performs initialization (CAS on `initialized_`).

---

## 3. Initialization Path

### 3.1 `AicpuExecutor::init(Runtime* runtime)`

This function:

1. One-time init guard:
   - `initialized_.compare_exchange_strong(expected=false, true)`
2. Reads configuration:
   - `thread_num_ = runtime->sche_cpu_num` (defaults to 1 if 0)
3. Discovers cores:
   - `handshake_all_cores(runtime)`
4. Assigns cores to threads:
   - `assign_cores_to_threads()`
5. Initializes per-core tracking:
   - `executing_task_ids_[i] = -1`
6. Initializes task counters:
   - reads `total_tasks_` from shared memory *if already built* (host-orch case)
   - sets `orchestrator_done_ = runtime->get_orch_built_on_host()`
7. Resets ready queues and profiling counters.

Device orchestration note:
- if the graph is built on device, thread 3 will later set `total_tasks_` and `orchestrator_done_`.

### 3.2 `AicpuExecutor::handshake_all_cores(Runtime* runtime)`

This is core discovery + register mapping.

Two phases:

**Phase A: wake all cores**
- For each worker `i`:
  - write `workers[i].task = &s_pto2_payload_per_core[i]`
  - write `workers[i].aicpu_ready = 1`

**Phase B: wait for each core to respond**
- Spin until `workers[i].aicore_done != 0`
- Read:
  - `core_type` (AIC or AIV)
  - `physical_core_id`
- Convert physical_core_id to register address:
  - `reg_addr = regs_[physical_core_id]`
- Store mappings:
  - per-type arrays (`aic_cores_[]`, `aiv_cores_[]`)
  - `core_id_to_reg_addr_[worker_id] = reg_addr`
- Enable “fast path”:
  - `write_reg(reg_addr, FAST_PATH_ENABLE, OPEN)`
  - `write_reg(reg_addr, DATA_MAIN_BASE, 0)`

Why `workers[i].task` is set during handshake:
- AICore reads this pointer once and uses it as “my payload location”.
- Later, on each dispatch, AICPU only needs to update the payload contents and write a register signal.

### 3.3 `AicpuExecutor::assign_cores_to_threads()`

When `thread_num_ == 4`:
- scheduler threads = 3 (threads 0..2)
- orchestrator thread = 1 (thread 3; assigned 0 cores)

For scheduler threads:
- assigns `aic_per_thread = aic_count / scheduler_thread_num`
- assigns `aiv_per_thread = aiv_count / scheduler_thread_num`
- each scheduler thread gets a contiguous slice of AIC cores then a slice of AIV cores.

Note:
- this is a simple static partition; it does not attempt to balance based on kernel mix.

---

## 4. Dispatch Payload Packing

Static helper:
- `build_pto2_payload(PTO2DispatchPayload* out, Runtime* runtime, PTO2TaskDescriptor* task, ...)`

What it writes:
- `out->task_id = task->task_id`
- `out->kernel_id = task->kernel_id`
- `out->core_type = (task->worker_type == PTO2_WORKER_CUBE) ? AIC : AIV`
- `out->function_bin_addr = runtime->get_function_bin_addr(task->kernel_id)`
- `out->args[]`:
  - for SCALAR params: `scalar_value`
  - otherwise: pointer to the **task-owned `Tensor` struct** (`task->params[i].tensor`)

Why kernels want `Tensor*` not raw addresses:
- kernels read `tensor->buffer.addr` and `tensor->start_offset` etc.
- the runtime can represent views/slices without copying.

---

## 5. Main Scheduler Loop: `resolve_and_dispatch_pto2(...)`

Signature:
- `int resolve_and_dispatch_pto2(Runtime* runtime, int thread_idx, const int* cur_thread_cores, int core_num)`

### 5.1 Shared memory binding

It loads:
- `sm_base = runtime->get_pto2_gm_sm_ptr()`
- `header = (PTO2SharedMemoryHeader*)sm_base`
- `task_descriptors = sm_base + header->task_descriptors_offset`
- `dep_list_pool = sm_base + header->dep_list_pool_offset`

If device orchestration and `thread_num==4`:
- scheduler threads wait for `sm_header_ready_` set by thread 3.

It sets:
- `window_size = header->task_window_size` (clamped to `PTO2_MAX_SLOTS`)
- `window_mask = window_size - 1`

### 5.2 One-time init: reset state arrays + init perf

Only one scheduler thread does:
- `memset(s_pto2_fanin_refcount, 0, ...)`
- `memset(s_pto2_task_completed, 0, ...)`
- `perf_aicpu_init_profiling(runtime)` (if enabled)

Other threads spin until `pto2_init_complete_ == true`.

Why this is necessary:
- these arrays are static globals; they must not carry state between runs.

### 5.3 Loop termination condition

At top of each iteration it reads:
- `task_count = total_tasks_`
- `orch_done = orchestrator_done_`

Stopping logic:
- if `orch_done && task_count == 0`: empty graph
- if `completed_tasks_ >= task_count`:
  - still checks that all owned cores are idle
  - and `orch_done` is true

This avoids exiting while cores are still running.

### 5.4 Phase: SCAN (discover roots)

Key idea:
- roots are tasks with `fanin_count == 0`.
- those tasks become ready *without* waiting for any completion.

Mechanism:
- read `visible = header->current_task_index`
- claim indices with:
  - `idx = next_scan_index_`
  - CAS `idx -> idx+1`
  - ensures each task id is scanned once across threads
- for each claimed `idx`:
  - `slot = idx & window_mask`
  - load `fanin_count = t->fanin_count` (acquire)
  - if `fanin_count == 0`:
    - mark `s_pto2_task_completed[slot] = 1` (enqueued)
    - push `idx` into AIC or AIV ready queue based on `t->worker_type`

Profiling behavior:
- if profiling enabled, whenever `visible` changes it calls:
  - `perf_aicpu_update_total_tasks(runtime, visible)`

### 5.5 Phase: ORCH_DRAIN (early-ready queue)

This handles a specific race:
- a producer completes before a consumer is submitted.
- orchestrator increments consumer’s fanin_refcount directly.
- after finalizing consumer’s fanin_count, orchestrator may push the consumer into `orch_ready_queue`.

Scheduler threads drain that queue:
- load `head` and `tail`
- CAS `orch_ready_head` to claim a slot
- read `task_id = orch_ready_queue[head & mask]`
- CAS `s_pto2_task_completed[slot]` from `0 -> 1` to avoid double enqueue
- push to ready queue based on worker_type

### 5.6 Phase: COMPLETE (poll cores, propagate readiness)

Completion detection:
- for each owned `core_id`:
  - read `status = read_reg(reg_addr, COND)`
  - if `status == IDLE` and `executing_task_ids_[core_id] >= 0`:
    - treat that tracked task as completed

On completion, it does three important actions:

1. **Perf timestamp patching** (if profiling enabled)
   - `dispatch_ts` is recorded at dispatch time (`dispatch_timestamps_[core_id]`)
   - `finish_ts = get_sys_cnt_aicpu()`
   - it finds the matching `PerfRecord` for `payload->task_id` near the tail of perf buffers and writes dispatch/finish timestamps.
   - this is deliberately robust (search window + retry + both double buffers).

2. **Mark task completed and snapshot its fanout list**
   - acquires `pto2_task->fanout_lock`
   - writes `s_pto2_task_completed[task_slot] = 2`
   - snapshots `fanout_head`
   - releases lock

3. **Traverse fanout list and update consumers**
   - iterates `PTO2DepListEntry` nodes starting from `fanout_head`
   - for each consumer:
     - `prev = fetch_add(fanin_refcount[consumer_slot], 1)`
     - load `fanin_count` from the consumer descriptor
     - if `prev + 1 == fanin_count`, the consumer is now ready:
       - set `task_completed[consumer_slot] = 1`
       - push consumer_id into appropriate ready queue

This is the core DAG scheduling mechanism:
- completion of a producer drives readiness of its dependents.

### 5.7 Phase: DISPATCH (send ready tasks to idle cores)

Dispatch condition:
- only attempts dispatch when `cur_thread_tasks_in_flight < core_num`

For each owned core:
- read `COND`
- if `IDLE` and `executing_task_ids_[core_id] == -1`:
  - pick a task from the queue matching that core’s type:
    - if `Handshake.core_type == AIC`: pop from AIC queue
    - else pop from AIV queue
  - build payload into `s_pto2_payload_per_core[core_id]`
  - profiling:
    - record `dispatch_timestamps_[core_id] = get_sys_cnt_aicpu()`
    - possibly switch perf buffers if dispatch count hits `PLATFORM_PROF_BUFFER_SIZE`
  - write registers:
    - `write_reg(COND, BUSY)` first (prevents false completion detection)
    - `write_reg(DATA_MAIN_BASE, task_id + 1)` (the “doorbell”)
  - track:
    - `executing_task_ids_[core_id] = task_id`
    - `cur_thread_tasks_in_flight++`

### 5.8 Phase: YIELD (no progress)

If no progress was made in an iteration:
- `idle_iterations++`
- warns every `WARN_INTERVAL`
- errors out after `MAX_IDLE_ITERATIONS`
- yields the thread (`std::this_thread::yield()`)

This is a safety net against deadlocks/livelocks.

### 5.9 End-of-loop: export scheduler profiles

At loop end (per scheduler thread), if profiling enabled:
- prints a machine-readable JSON line `PTO2_SCHED_PROFILE_JSON {...}`
- writes `SchedulerProfile` into perf shared memory (`PerfDataHeader.sched_profiles[thread_idx]`)
- sets a bit in `sched_profiles_ready_mask`
- flushes perf buffers for owned cores

This is what enables `tools/pto2_schedule_report.py` to work on real hardware even if device logs aren’t captured.

---

## 6. Orchestrator Thread: `AicpuExecutor::run(...)` when `thread_idx==3`

When `thread_num_ == 4` and `thread_idx == 3`:
- this thread runs device orchestration (unless `orch_built_on_host` is true).

High-level steps:

1. Pull orchestration SO bytes from `runtime`
2. Write them to an executable path on AICPU FS (tries multiple directories)
3. `dlopen()` the SO
4. `dlsym()`:
   - `aicpu_orchestration_config` (optional)
   - `aicpu_orchestration_entry` (required)
5. Validate `arg_count >= expected_arg_count`
6. Wrap device shared memory:
   - `sm_handle = pto2_sm_create_from_buffer(sm_ptr, sm_size, task_window_size, heap_size, dep_list_pool_size)`
   - then `sm_header_ready_ = true` so scheduler threads may begin
7. Create `PTO2Runtime` bound to that shared memory + GM heap:
   - `rt = pto2_runtime_create_from_sm(...)`
8. Wait for scheduler one-time init (`pto2_init_complete_`) so global arrays are cleared
9. Configure orchestrator “parallel mode” pointers:
   - `rt->orchestrator.aicpu_fanin_refcount = s_pto2_fanin_refcount`
   - `rt->orchestrator.aicpu_task_completed = s_pto2_task_completed`
10. Expose orchestrator ready queue pointers for ORCH_DRAIN
11. Call orchestration inside an outer `PTO2_SCOPE(rt) { ... }`
12. Signal orchestration done and destroy runtime
13. Read `pto2_task_count = *(volatile int32_t*)sm_base` (that’s `header->current_task_index`)
14. Publish:
   - `total_tasks_ = pto2_task_count`
   - `orchestrator_done_ = true`

Why step 13 uses `*(volatile int32_t*)sm`:
- `current_task_index` is the first field in `PTO2SharedMemoryHeader`, so offset 0.

---

## 7. Shutdown and Cleanup

Scheduler threads (0..2) after finishing dispatch:
- call `shutdown_aicore(...)` to send `AICORE_EXIT_SIGNAL` via register
- closes FAST_PATH

All threads:
- increment `finished_count_`
- last one sets `finished_=true`

Then `aicpu_execute` sees `finished_` and calls `deinit()` once:
- clears queue heads/tails
- resets profiling counters and state flags
- resets discovery state

---

## 8. Practical Debug Tips (device scheduler bugs)

1. “All cores idle, no ready tasks, incomplete tasks”
   - suggests a dependency refcount mismatch (fanin_count never reached).
2. “Some cores busy forever”
   - suggests a hung kernel or register protocol mismatch.
3. “Large tail latency”
   - expected with COND polling; consider completion queue/bitmap design.
4. “perf_ts_update_fail grows”
   - means AICPU could not find matching perf record; tune search window/retry or fix record visibility ordering.
