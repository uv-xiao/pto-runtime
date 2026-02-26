# Line-by-line: `pto2_submit_task` (TensorMap + RingBuffer Orchestrator)

Last verified against repo state on **2026-02-26**.

This document is a *line-numbered*, code-grounded explanation of:
- `src/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp:255` (`pto2_submit_task`)

It complements (does not replace):
- `docs/annotated-pto-orchestrator.md` (broader file walkthrough)
- `docs/tensormap-ringbuffer-runtime-guide.md` (architecture + profiling report)

> Note on line numbers: all `Lxxx` references below match the repo state on 2026-02-26. If code changes, re-run `nl -ba` to refresh.

---

## 0. What this function *really* does

`pto2_submit_task(...)` is the *graph builder hot path* executed by the orchestrator thread (AICPU side, either “device orchestration” or host-built graph depending on mode). For every “task submission”, it:

1. **Allocates a new task slot** from the task ring (shared memory window).
2. **Copies params into task-owned storage** (so later stages don’t depend on caller memory).
3. **Discovers dependencies** by TensorMap overlap on INPUT/INOUT tensors:
   - builds this task’s **fanin** list (the producers it must wait for)
   - appends this task to each producer’s **fanout** list (who depends on the producer)
4. **Allocates OUTPUT buffers** from the GM heap ring (optional; only if caller didn’t supply an addr).
5. **Inserts OUTPUT/INOUT tensors into TensorMap** so later tasks depend on this task.
6. **Publishes “new tasks are visible”** via `header->current_task_index` (release store).

Visually, submission is “turn parameters into a node in a dependency DAG”:

```mermaid
flowchart LR
  Submit[pto2_submit_task] --> TR[TaskRing alloc + TaskDescriptor init]
  Submit --> TM1[TensorMap lookup on INPUT/INOUT]
  TM1 --> FanIn[consumer.fanin list]
  TM1 --> FanOut[producer.fanout list]
  Submit --> Heap[HeapRing alloc for OUTPUT (optional)]
  Submit --> TM2[TensorMap insert for OUTPUT/INOUT]
  Submit --> Pub[Publish current_task_index]
```

---

## 1. Full function listing (with line numbers)

File: `src/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp`

```cpp
255 void pto2_submit_task(PTO2OrchestratorState* orch,
256     int32_t kernel_id,
257     PTO2WorkerType worker_type,
258     PTOParam* params,
259     int32_t num_params) {
260     CYCLE_COUNT_START();
261
262     // === STEP 0: Sync TensorMap validity and optional cleanup ===
263     pto2_orchestrator_sync_tensormap(&orch->tensor_map);
264
265     CYCLE_COUNT_LAP(g_orch_sync_cycle);
266
267     // Submission without an open scope is illegal
268     assert(orch->scope_stack_top >= 0 && "Cannot submit task outside a scope");
269
270     // === STEP 1: Allocate task slot from Task Ring (blocks until available) ===
271     int32_t task_id = pto2_task_ring_alloc(&orch->task_ring);
272
273     CYCLE_COUNT_LAP(g_orch_alloc_cycle);
274
275     PTO2TaskDescriptor* task = pto2_task_ring_get(&orch->task_ring, task_id);
276
277     // Initialize task descriptor
278     task->task_id = task_id;
279     task->kernel_id = kernel_id;
280     task->worker_type = worker_type;
281     task->fanin_head = 0;
282     task->fanin_count = 0;
283     task->fanout_head = 0;
284     task->fanout_lock = 0;
285     // Initial fanout_count = 1 (the owning scope holds one reference)
286     task->fanout_count = 1;
287     task->packed_buffer_base = NULL;
288     task->packed_buffer_end = NULL;
289     task->num_outputs = 0;
290     task->is_active = true;
291
292     // Register this task in its owning scope
293     scope_tasks_push(orch, task_id);
294
295     // Temporary storage for collecting output sizes
296     int32_t total_output_size = 0;
297
298     // Temporary storage for fanin
299     int32_t fanin_temp[PTO2_MAX_INPUTS];
300     int32_t fanin_count = 0;
301
302     task->param_count = num_params;
303     // Bulk copy all params at once
304     memcpy(task->params, params, num_params * sizeof(PTOParam));
305     // Copy tensor data into task-owned storage; redirect pointers
306     for (int i = 0; i < num_params; i++) {
307         if (params[i].tensor) {
308             task->tensor_copies[i] = *params[i].tensor;
309             task->params[i].tensor = &task->tensor_copies[i];
310         }
311     }
312
313     CYCLE_COUNT_LAP(g_orch_params_cycle);
314
315     // === STEP 2: First pass - collect output sizes and process inputs ===
316
317     for (int i = 0; i < num_params; i++) {
318         PTOParam* p = &params[i];
319
320         switch (p->type) {
321             case PTOParamType::INOUT:
322             case PTOParamType::INPUT: {
323                 // Look up producer via TensorMap
324                 PTO2LookupResult lookup_result;
325                 pto2_tensormap_lookup(&orch->tensor_map, params[i].tensor, &lookup_result);
326
327                 for (int r = 0; r < lookup_result.count; r++) {
328                     int32_t entry_idx = lookup_result.entries[r].entry_idx;
329                     auto &entry = orch->tensor_map.entry_pool[entry_idx];
330                     auto overlap_status = lookup_result.entries[r].overlap_status;
331                     // Check if this producer is already in fanin list (avoid duplicates)
332                     int producer_task_id = entry.producer_task_id;
333                     bool already_added = false;
334                     for (int j = 0; j < fanin_count; j++) {
335                         if (fanin_temp[j] == producer_task_id) {
336                             already_added = true;
337                             break;
338                         }
339                     }
340
341                     if (!already_added) {
342                         // Add to fanin list (this task depends on producer)
343                         if (fanin_count < PTO2_MAX_INPUTS) {
344                             fanin_temp[fanin_count++] = producer_task_id;
345                         }
346
347                         // Add this task to producer's fanout list (with spinlock)
348                         PTO2TaskDescriptor* producer = pto2_task_ring_get(&orch->task_ring, producer_task_id);
349                         pto2_add_consumer_to_producer(orch, producer, producer_task_id, task_id);
350                     }
351                     if (p->type == PTOParamType::INOUT && overlap_status == OverlapStatus::COVERED) {
352                         // inout因为会再次insert进tensor map，
353                         // 因此为了尽量减少依赖构建个数（尽可能构造链式依赖），当该tensor完全覆盖前面的tensor时，
354                         // 应将前面的tensor从tensor map中剔除。
355                         // 但是最开始的tensor除外，因为必须建立和最开始的task的依赖关系以保证tensor生命周期的正确管理
356                         if (!entry.with_alloc) {
357                             pto2_tensormap_remove_entry(orch->tensor_map, entry_idx);
358                         }
359                     }
360                 }
361                 break;
362             }
363
364             case PTOParamType::OUTPUT: {
365                 // Only allocate from ring buffer when caller did not provide an address
366                 if (params[i].tensor->buffer.addr == 0) {
367                     total_output_size += PTO2_ALIGN_UP(params[i].tensor->buffer.size, PTO2_PACKED_OUTPUT_ALIGN);
368                 }
369                 break;
370             }
371             default:
372                 break;
373         }
374     }
375
376     CYCLE_COUNT_LAP(g_orch_lookup_cycle);
377
378     // === STEP 3: Allocate packed buffer from Heap Ring (may stall) ===
379     // Each output slot is aligned to PTO2_PACKED_OUTPUT_ALIGN (1024B); gap after data is padding.
380     if (total_output_size > 0) {
381         task->packed_buffer_base = pto2_alloc_packed_buffer(orch, total_output_size);
382         task->packed_buffer_end = (char*)task->packed_buffer_base + total_output_size;
383
384         // Offsets: each output at 1024B-aligned slot; slot size = ALIGN_UP(size, 1024)
385         int32_t offset = 0;
386         for (int i = 0; i < task->param_count; i++) {
387             if (task->params[i].type == PTOParamType::OUTPUT) {
388                 if (task->tensor_copies[i].buffer.addr == 0) {
389                     uint64_t alloc_addr = reinterpret_cast<uint64_t>((char*)task->packed_buffer_base + offset);
390                     task->tensor_copies[i].buffer.addr = alloc_addr;
391                     // Write back through caller's pointer (implicit update)
392                     params[i].tensor->buffer.addr = alloc_addr;
393                     offset += PTO2_ALIGN_UP(task->tensor_copies[i].buffer.size, PTO2_PACKED_OUTPUT_ALIGN);
394                 }
395                 task->output_index[task->num_outputs++] = i;
396             }
397         }
398     }
399
400     CYCLE_COUNT_LAP(g_orch_heap_cycle);
401
402     // === STEP 4: Second pass - register outputs in TensorMap ===
403     int32_t output_idx = 0;
404     for (int i = 0; i < num_params; i++) {
405         PTOParam* p = &params[i];
406
407         if (p->type == PTOParamType::OUTPUT || p->type == PTOParamType::INOUT) {
408             // Register in TensorMap: this tensor is produced by task_id
409             // Use task's tensor_copies (which has the heap-allocated address for outputs)
410             pto2_tensormap_insert(&orch->tensor_map, &task->tensor_copies[i], task_id, p->type == PTOParamType::OUTPUT);
411             output_idx++;
412         }
413     }
414
415     CYCLE_COUNT_LAP(g_orch_insert_cycle);
416
417     // === STEP 5: Finalize fanin list ===
418     // First build the fanin list
419     for (int i = 0; i < fanin_count; i++) {
420         task->fanin_head = pto2_dep_list_prepend(&orch->dep_pool, task->fanin_head, fanin_temp[i]);
421     }
422     // Use release semantics to ensure fanin list is visible before fanin_count
423     __atomic_store_n(&task->fanin_count, fanin_count, __ATOMIC_RELEASE);
424
425     CYCLE_COUNT_LAP(g_orch_fanin_cycle);
426
427     // === STEP 5b: Check if task is already ready (all producers completed via early-return) ===
428     // In AICPU parallel mode, early-return in pto2_add_consumer_to_producer may have
429     // already incremented aicpu_fanin_refcount for this task.  Now that fanin_count is
430     // finalized, check if the task is already satisfied and push it to the orchestrator
431     // ready queue so scheduler threads can pick it up without an O(N) scan.
432     if (orch->aicpu_fanin_refcount && fanin_count > 0) {
433         int32_t slot = task_id & orch->aicpu_window_mask;
434         int32_t refcount = __atomic_load_n(&orch->aicpu_fanin_refcount[slot], __ATOMIC_ACQUIRE);
435         if (refcount >= fanin_count) {
436             // All producers already completed — push to orch ready queue
437             int32_t tail = orch->orch_ready_tail;
438             int32_t capacity = PTO2OrchestratorState::ORCH_READY_QUEUE_SIZE;
439             int32_t head = __atomic_load_n(&orch->orch_ready_head, __ATOMIC_ACQUIRE);
440             if (((tail + 1) & (capacity - 1)) != (head & (capacity - 1))) {
441                 orch->orch_ready_queue[tail & (capacity - 1)] = task_id;
442                 __atomic_store_n(&orch->orch_ready_tail, tail + 1, __ATOMIC_RELEASE);
443             }
444         }
445     }
446
447     // === STEP 6: Initialize task in scheduler ===
448     // In multi-threaded mode, scheduler thread handles task initialization via polling
449     if (orch->scheduler && orch->init_task_on_submit) {
450         pto2_scheduler_init_task(orch->scheduler, task_id, task);
451     }
452
453     // === STEP 7: Update shared memory with current task index ===
454     PTO2_STORE_RELEASE(&orch->sm_handle->header->current_task_index, orch->task_ring.current_index);
455
456     CYCLE_COUNT_LAP(g_orch_finalize_cycle);
457
458     orch->tasks_submitted++;
459 #if PTO2_ORCH_PROFILING
460     g_orch_submit_count++;
461 #endif
462 }
```

---

## 2. Step-by-step interpretation (what each block implies)

### Step 0 (L260–L266): TensorMap validity sync

- **What it touches:** the orchestrator’s private `orch->tensor_map`.
- **Why it exists:** TensorMap uses *lazy invalidation* tied to “which task slots are still alive”. When the scheduler advances `last_task_alive`, orchestrator periodically trims “retired tasks’” entries so lookups stay fast.
- **What it costs:** `g_orch_sync_cycle` measures the time spent keeping TensorMap chains from growing without bound.

### Step 1 (L270–L301): Allocate and initialize TaskDescriptor

- `pto2_task_ring_alloc(...)` (L271) is the **back-pressure point**: it may spin/wait if the ring window is full.
- Descriptor fields (L277–L291) are the contract the scheduler relies on:
  - `fanin_head` + `fanin_count` define dependency edges for readiness.
  - `fanout_head` + `fanout_count` are used for completion propagation.
  - `fanout_count` starts at `1` (L285–L286): “owning scope holds one ref”, so scopes can safely free resources only when fanout reaches 0.
- `scope_tasks_push(...)` (L292–L293) records this task in the current scope so `pto2_scope_end` can later tell the scheduler “these tasks’ outputs can be retired”.

### Step 1.5 (L302–L313): Make params task-owned (critical correctness detail)

Two copies happen:

1. `memcpy(task->params, params, ...)` (L303–L304) copies the **PTOParam** metadata.
2. For tensor params, it copies the `Tensor` object into `task->tensor_copies[i]` and rewrites `task->params[i].tensor` to point to that copy (L306–L310).

Why this is critical:
- the scheduler and kernels must not depend on caller-side param memory staying alive
- OUTPUT buffer addresses are written *into these task-owned Tensor copies* (Step 3)

### Step 2 (L315–L377): Dependency discovery + output-size accumulation

This loop does two different jobs depending on param type:

**(A) INPUT / INOUT**
- `pto2_tensormap_lookup(...)` (L323–L325) returns “all potentially overlapping previous producers for this base buffer”.
- For each overlapping producer entry:
  - it adds `producer_task_id` to this task’s temporary fanin list (L331–L345), with an O(fanin_count) duplicate check
  - it appends this consumer into the producer’s fanout list (L347–L349) via `pto2_add_consumer_to_producer` (spinlocked)

**(B) OUTPUT**
- If caller didn’t provide a buffer addr (`addr==0`), accumulate a packed allocation size (L365–L368).

#### The INOUT “COVERED truncation” optimization (L351–L359)

If an INOUT tensor **fully covers** a previous TensorMap entry, the code removes the older entry from TensorMap (unless `entry.with_alloc`).

Intuition:
- INOUT means “this task produces a new version of this region”.
- If the new region completely covers the old region, future lookups don’t need the old entry; they should depend on the newest producer to form a *chain* rather than a fan-in explosion.

Safety nuance: `with_alloc`
- `with_alloc==true` indicates the entry is tied to a scope-managed allocation/lifetime that must not be dropped early.
- So the code only truncates when `!entry.with_alloc`, preserving lifetime correctness.

### Step 3 (L378–L401): Allocate packed outputs (optional) and write back addresses

If `total_output_size>0`:
- allocate one contiguous “packed” region from HeapRing (L381–L382)
- assign each OUTPUT tensor a slice aligned to `PTO2_PACKED_OUTPUT_ALIGN` (1024B) (L384–L397)

Two extremely important side-effects:

1. `task->tensor_copies[i].buffer.addr` is filled with the allocated device address (L388–L390).
2. **Caller-visible write-back:** it writes the address back into `params[i].tensor->buffer.addr` (L391–L392).
   - this is how orchestration code that passed an “empty OUTPUT tensor” learns the allocated address.

Also: `task->output_index[]` (L395) records which param indices are outputs, so other components can iterate outputs quickly.

### Step 4 (L402–L415): Insert produced tensors into TensorMap

- OUTPUT and INOUT are inserted with `pto2_tensormap_insert(...)` (L407–L411).
- The insert uses `&task->tensor_copies[i]` (task-owned), which now contains allocated addresses for outputs.
- The last boolean argument marks whether this is “with_alloc” (OUTPUT) vs not (INOUT), which affects later truncation behavior and lifetime semantics.

### Step 5 (L417–L426): Build the actual fanin linked-list and publish fanin_count

- The function first builds a linked list of producer task IDs in `task->fanin_head` (L419–L421) using `pto2_dep_list_prepend`.
- It then stores `fanin_count` with `__ATOMIC_RELEASE` (L423).

This release-store is the key publish point for “this task’s dependency list is now valid”.
Schedulers can do acquire-loads on `fanin_count` and safely traverse `fanin_head`.

### Step 5b (L427–L446): “Already ready” fast-path for parallel scheduler mode

When orchestration and scheduling run concurrently, it’s possible that:
- producer completes *before* we managed to attach this consumer to its fanout list
- `pto2_add_consumer_to_producer` detects this under lock and increments `orch->aicpu_fanin_refcount[consumer_slot]` directly (early return)

So once `fanin_count` becomes final, the orchestrator checks if `refcount >= fanin_count` (L432–L445).
If so, it pushes `task_id` into `orch_ready_queue` so scheduler threads can pick it up without doing a global scan.

This block is performance-critical when:
- orchestration produces tasks “slowly”
- scheduler threads are fast and may complete producers early

### Step 6 (L447–L451): Optional scheduler init-on-submit

- If `orch->scheduler && orch->init_task_on_submit` it calls `pto2_scheduler_init_task`.
- In the a2a3 device scheduler path, most init is done lazily by scheduler scanning, so this can be disabled for lower submit overhead.

### Step 7 (L453–L456): Publish `current_task_index`

`PTO2_STORE_RELEASE(&header->current_task_index, task_ring.current_index)` is the global “new tasks are visible” flag.

Schedulers use an acquire-load of `current_task_index` to know how many tasks exist and which slots are safe to read.

---

## 3. The hidden concurrency primitive: `pto2_add_consumer_to_producer`

File: `src/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp`

This helper is the rendezvous between:
- “graph is being built” (orchestrator)
- “tasks are completing” (scheduler threads)

```cpp
201 void pto2_add_consumer_to_producer(
202     PTO2OrchestratorState* orch, PTO2TaskDescriptor* producer, int32_t producer_id, int32_t consumer_id) {
203     task_fanout_lock(producer);
204
205     // AICPU parallel mode: producer might already be completed
206     if (orch->aicpu_task_completed) {
207         int32_t prod_slot = producer_id & orch->aicpu_window_mask;
208         if (__atomic_load_n(&orch->aicpu_task_completed[prod_slot], __ATOMIC_ACQUIRE) >= 2) {
209             int32_t cons_slot = consumer_id & orch->aicpu_window_mask;
210             __atomic_fetch_add(&orch->aicpu_fanin_refcount[cons_slot], 1, __ATOMIC_ACQ_REL);
211             task_fanout_unlock(producer);
212             return;
213         }
214     }
215
216     producer->fanout_head = pto2_dep_list_prepend(&orch->dep_pool, producer->fanout_head, consumer_id);
217     producer->fanout_count++;
218
219     // Scheduler mode: if producer already completed, increment scheduler fanin_refcount too
220     ...
221
222     task_fanout_unlock(producer);
223 }
```

Interpretation:
- The lock protects the producer’s fanout list against concurrent traversal during completion.
- The “early return” is what makes Step 5b necessary: sometimes the edge is accounted for via `fanin_refcount` without ever being stored in the producer’s fanout list.

---

## 4. Profiling mapping: which counters correspond to which steps

This function calls `CYCLE_COUNT_LAP(...)` after major sections, feeding:
- `g_orch_sync_cycle` (Step 0)
- `g_orch_alloc_cycle` (Step 1 ring alloc + descriptor init)
- `g_orch_params_cycle` (param copy / redirect)
- `g_orch_lookup_cycle` (TensorMap lookup + dependency build + output-size scan)
- `g_orch_heap_cycle` (HeapRing packed alloc + output addr assignment)
- `g_orch_insert_cycle` (TensorMap insert outputs)
- `g_orch_fanin_cycle` (fanin list build + publish)
- `g_orch_finalize_cycle` (ready-queue check + scheduler init + current_task_index publish)

These are the “submit path” costs you see summarized in the scheduling report.

---

## 5. Optimization checklist (submit path)

Ordered by “likely to matter in extreme micro-task graphs”:

1. **Reduce TensorMap lookup cost**
   - Bucket chain length dominates; `pto2_orchestrator_sync_tensormap` frequency and truncation policy matter.
   - Consider more aggressive chain truncation when INOUT covered is common and `with_alloc` is false.
2. **Avoid O(k²) fanin duplicate detection**
   - Current duplicate check is linear search inside the producer loop (L331–L339).
   - For tasks with many inputs/overlaps, consider a small fixed-size bitmap/hash for fanin_temp within a submission.
3. **Reduce fanout lock contention**
   - `pto2_add_consumer_to_producer` takes a per-producer lock; in patterns where many consumers share one producer, this becomes a hotspot.
   - Consider batching fanout appends per producer (build local list then append once).
4. **Shrink packed output padding waste**
   - 1024B alignment is great for simplicity but can explode wasted bytes for many small outputs.
   - If output sizes are typically small and numerous, evaluate a smaller alignment (requires checking AICore kernel alignment requirements).
5. **Tune init-on-submit**
   - If scheduler threads already scan/incrementally discover tasks, init-on-submit might be redundant.
   - If enabled, ensure it doesn’t duplicate work the scheduler does anyway.

