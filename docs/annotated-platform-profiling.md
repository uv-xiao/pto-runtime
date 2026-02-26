# Annotated Walkthrough: Platform Profiling (Perf Buffers + Ready Queues + Scheduler Profiles)

Last verified against repo state on **2026-02-26**.

This document is a deep explanation of how profiling works across Host/AICPU/AICore, focusing on the code under `src/platform/`.

Primary files:
- Data layout: `src/platform/include/common/perf_profiling.h`
- Host collector: `src/platform/src/host/performance_collector.cpp`
- AICPU buffer manager: `src/platform/src/aicpu/performance_collector_aicpu.cpp`
- AICore record writer: `src/platform/include/aicore/performance_collector_aicore.h`

Runtime integration points (important but not platform-owned):
- PTO2 scheduler patches dispatch/finish timestamps and exports phase stats:
  - `src/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp`

Companion docs:
- Platform overview: `docs/platform-codewalk.md`
- Scheduler loop details: `docs/linebyline-aicpu-resolve-and-dispatch-pto2.md`

> Note on line numbers: `Lxxx` references below match 2026-02-26. Refresh with `nl -ba` if code moves.

---

## 0. The goal: a low-overhead trace that works on real hardware

Constraints:
- AICore tasks are high-frequency; writing perf records must be *very cheap*.
- Host must be able to read records while kernels run (to avoid buffer starvation).
- On real hardware, device logs may not be captured reliably, so the profiling channel must be **data-driven**, not log-driven.

This design provides:
1. per-core ping/pong perf record buffers
2. per-AICPU-thread ready queues for “buffer full / ready to read”
3. a host-side poller that drains these queues and exports JSON
4. optional scheduler phase stats exported via the same shared memory

---

## 1. Data layout (`perf_profiling.h`): what lives in shared memory

File: `src/platform/include/common/perf_profiling.h`

### 1.1 The core state machine: `BufferStatus` (L57–L61)

```
IDLE (0) -> WRITING (1) -> READY (2) -> IDLE (0)
```

- AICPU owns transitions `IDLE→WRITING` and `WRITING→READY`
- Host owns transition `READY→IDLE`
- AICore **never** changes status; it only writes records and increments `count`

This separation is intentional: AICore code stays minimal and doesn’t contend on status.

### 1.2 `PerfRecord`: who writes what (L70–L92)

Fields are split by producer:

- AICore writes:
  - `start_time`, `end_time`, `kernel_ready_time`
  - identifiers: `task_id`, `func_id`, `core_id`, `core_type`
- AICPU patches later:
  - `dispatch_time` (when AICPU dispatched)
  - `finish_time` (when AICPU observed completion)
- Host just reads and exports.

### 1.3 `PerfDataHeader`: two important “extra channels”

Besides ready queues and metadata, header also includes:

1. `total_tasks` (L212):
   - AICPU can update this dynamically as orchestration progresses.
   - Host poller uses it to determine when to stop.
2. `sched_profiles[]` + `sched_profiles_ready_mask` (L214–L217):
   - per-AICPU-thread scheduler phase stats, written at end by the runtime scheduler
   - host exports into JSON under `"scheduler_profiles"`

This is how you get scheduler timing on real hardware without relying on AICPU logs.

---

## 2. Host collector (`performance_collector.cpp`): allocate → poll → export

File: `src/platform/src/host/performance_collector.cpp`

### 2.1 Initialize: allocate + optional host-mapping (L26–L116)

Key steps:

1. compute size: `calc_perf_data_size(num_aicore)` (L41–L54)
2. allocate device memory via callback (L56–L63)
3. map to host if `register_cb != nullptr` (L64–L77)
   - a2a3 uses `halHostRegister` (SVM map)
   - a2a3sim passes `nullptr` and uses the same pointer as both dev+host
4. initialize `PerfDataHeader` (L79–L90) and all `DoubleBuffer`s to IDLE (L92–L103)
5. write `runtime.perf_data_base = perf_dev_ptr` (L107–L112) so AICPU can locate the shared memory

Memory barriers:
- `wmb()` at L105 ensures header/buffers init is globally visible before device reads.

### 2.2 Poll: drain per-thread ready queues (L118–L257)

Core idea:
- AICPU enqueues `(core_index, buffer_id)` into `header->queues[thread_idx]`
- Host loops over threads round-robin (L165–L205), and when it sees `head!=tail`:
  - locate the `DoubleBuffer` for that core (L220–L223)
  - read `buf->count` (L225–L227)
  - copy records into `collected_perf_records_` (L229–L233)
  - reset `buf->count = 0` and `status = IDLE` (L234–L237)
  - advance queue head (L236)

Stop condition:
- loop ends once `total_records_collected >= expected_tasks` (L167)

Dynamic expected_tasks:
- even if you pass a fixed `expected_tasks`, the poller will raise it if it sees `header->total_tasks` increase (L170–L177).
  - this is critical for “device orchestration” where task count grows incrementally.

Timeout behavior:
- if no buffers appear ready for long enough, it bails with a timeout (L185–L201).
  - this protects you from hard deadlocks but can also fire if host polling is accidentally disabled.

### 2.3 Export JSON: embed scheduler profiles + tasks (L259–L417)

Important behavior:
- It embeds `PerfDataHeader.sched_profiles[]` only if `sched_profiles_ready_mask != 0` (L314–L363).
- It exports each record with `dispatch_time_us` and `finish_time_us` too (L374–L389), allowing you to visualize:
  - queueing / scheduling gaps (dispatch delays)
  - completion detection delays (finish vs end)

---

## 3. AICPU buffer manager (`performance_collector_aicpu.cpp`)

File: `src/platform/src/aicpu/performance_collector_aicpu.cpp`

### 3.1 Initialize perf buffers on device (L41–L70)

`perf_aicpu_init_profiling(runtime)`:
- reads `runtime->perf_data_base` (L42–L46)
- sets `header->total_tasks = runtime->get_task_count()` (L51–L53)
- for each core:
  - sets `Handshake.perf_records_addr = &db->buffer1` (L61–L63)
  - sets `buffer1_status = WRITING` (L62–L63)

`wmb()` at L67 makes sure AICore sees the buffer pointer and status changes.

### 3.2 Patch dispatch/finish timestamps (L72–L81)

`perf_aicpu_record_dispatch_and_finish_time(...)`:
- `rmb()` then writes `dispatch_time/finish_time` then `wmb()`.

This is deliberately tiny, because it’s called on completion hot path by the runtime scheduler.

### 3.3 Buffer switch on “full” (L84–L203)

`perf_aicpu_switch_buffer(runtime, core_id, thread_idx)`:

1. identify which buffer is “current” by comparing `Handshake.perf_records_addr` to `&db->buffer1`/`&db->buffer2` (L96–L121)
2. call `runtime->complete_perf_records(full_buf)` (L130–L133)
   - this is where fanout lists get filled into records
3. wait until alternate buffer becomes `IDLE` (host has consumed it) (L136–L181)
   - includes a timeout to avoid deadlock; on timeout it discards data and reuses the full buffer (L169–L179)
4. set:
   - `full_status = READY`
   - `alternate_status = WRITING` (L183–L185)
5. enqueue `(core_id, full_buffer_id)` into this AICPU thread’s ready queue (L186–L195)
6. set `Handshake.perf_records_addr = alternate_buf` so AICore writes into the alternate buffer next (L199–L203)

Why enqueue is per-thread:
- avoids contention between multiple AICPU threads when multiple cores fill buffers concurrently.

### 3.4 Flush at end (L205–L281)

`perf_aicpu_flush_buffers(...)` walks cores owned by this scheduler thread:
- if current buffer has `count > 0`, it calls `complete_perf_records`
- marks the buffer READY and enqueues it

This ensures the host can export a complete trace even if buffers never got “full”.

### 3.5 Dynamic task count updates (L283–L292)

`perf_aicpu_update_total_tasks(runtime, total_tasks)` writes `header->total_tasks = total_tasks` + `wmb()`.

This is used by the PTO2 scheduler scan phase to reflect orchestration progress.

---

## 4. AICore record write helper (`performance_collector_aicore.h`)

File: `src/platform/include/aicore/performance_collector_aicore.h`

`perf_aicore_record_task(...)`:
- reads `perf_buf->count` (with cache management on real HW)
- writes one record
- increments count
- uses `dcci` flushes on real HW to ensure visibility to AICPU/Host

It intentionally does not do:
- queueing
- status updates
- any locks

This keeps the AICore path minimal.

---

## 5. Where scheduler phase stats come from (PTO2)

Not platform-owned, but critical to understand:
- runtime AICPU scheduler writes phase cycles into `PerfDataHeader.sched_profiles[thread_idx]`
- then sets `sched_profiles_ready_mask` bit `thread_idx`
- host exports them under `"scheduler_profiles"`

See:
- `docs/linebyline-aicpu-resolve-and-dispatch-pto2.md` (“End-of-run exports” section)

---

## 6. “Extreme micro-task” tuning checklist (profiling impact)

When your scheduling target is ~25µs, profiling can become a significant perturbation. Common knobs:

1. **Reduce AICPU-side record patching cost**
   - The scheduler currently searches near the tail + may retry (runtime-level).
   - For stability experiments, consider patching 1/N tasks (sampling) or reducing retry window.
2. **Avoid frequent buffer switches**
   - With `PLATFORM_PROF_BUFFER_SIZE=1000`, switches are rare for large tasks but can be frequent for micro-tasks.
   - If switches are frequent, the enqueue/host-drain path becomes hot.
3. **Ensure host poller runs concurrently**
   - If host polling is delayed, AICPU can spin waiting for alternate buffer to become IDLE (switch path L136–L181).
4. **Watch for ready queue full**
   - `enqueue_ready_buffer` returns -1 if full (L28–L32); current code logs and drops data.
   - If you see drops, increase queue capacity or drain more aggressively.
5. **Use scheduler phase stats to localize overhead**
   - If `yield_us` dominates, you’re starving ready work.
   - If `complete_us` dominates, fanout traversal or perf patching is hot.

