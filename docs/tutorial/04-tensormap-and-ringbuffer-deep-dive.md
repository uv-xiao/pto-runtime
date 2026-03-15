# Tensormap and Ringbuffer Runtime Deep Dive

This chapter is the core tutorial for the `tensormap_and_ringbuffer` runtime, often called `PTO2` in the code. The goal is not just to name the components, but to explain how they cooperate when a real example runs.

This chapter answers five questions:

1. Why does this runtime exist?
2. What are the critical shared data structures?
3. How does a task move from orchestration to scheduling to execution?
4. How does the `paged_attention` example map onto those mechanisms?
5. How do you profile the runtime when it stalls, spins, or simply looks slow?

## What PTO2 Is Trying To Solve

The earlier runtimes in this repository are easier to understand:

- `host_build_graph`: build the graph on the host, then launch it.
- `aicpu_build_graph`: build the graph on device AICPU, but still with a more direct graph-building model.

`tensormap_and_ringbuffer` exists because the repository wants a more aggressive on-device runtime:

- orchestration runs on AICPU, not on the host
- dependencies are inferred automatically from tensor overlap
- task slots come from a sliding ring window
- output buffers come from a heap ring
- dependency edges come from a ring-allocated pool
- scheduler state is split from task descriptors so the hot path stays small
- mixed AIC/AIV tasks can be completed in two stages

That is why PTO2 feels more like a small operating system than a thin runtime wrapper.

## Start With A Mental Model

Read PTO2 as four cooperating actors:

```text
Host CPU
  |
  | init_runtime_impl() uploads kernels, allocates device GM heap and PTO2 shared memory,
  | copies orchestration .so, and passes device pointers as orch args.
  v
AICPU Orchestrator Thread(s)
  |
  | aicpu_orchestration_entry() calls PTO2 submit APIs.
  | For each submitted task:
  |   - allocate a task slot
  |   - infer dependencies via TensorMap
  |   - allocate outputs from heap ring
  |   - register outputs back into TensorMap
  |   - attach consumer edges to producer fanout lists
  v
AICPU Scheduler Thread(s)
  |
  | ready queue pop -> choose cluster -> build payload -> write dispatch registers
  | completion -> notify consumers -> release producers -> reclaim ring space
  v
AIC / AIV Cores
  |
  | poll register -> read payload -> run kernel -> ACK -> FIN
  v
Shared Memory + GM Heap
```

The key design split is:

- the orchestrator decides what should run
- the scheduler decides when it can run
- AIC/AIV cores only execute already-decided work

## A Beginner Attention Primer

Before diving into `paged_attention`, we need the attention algorithm in plain language.

Assume one query vector `q`, many key vectors `k_j`, and many value vectors `v_j`.

Standard attention computes:

```text
score_j = q · k_j
prob_j  = softmax(score_j)
out     = sum_j(prob_j * v_j)
```

Now scale this to a matrix of 16 query heads against many blocks of cached keys and values:

- `QK` computes score tiles
- `SF` turns scores into normalized probabilities
- `PV` multiplies probabilities by value tiles
- `UP` keeps an online running result so you do not need to materialize the full long-sequence softmax in one shot

That last point matters. In paged attention, the sequence is broken into cache blocks. PTO2 therefore processes:

```text
for each KV block:
  compute scores for this block
  softmax this block
  compute this block's value contribution
  merge this contribution into a running accumulator
```

The runtime chapter matters because the example does not directly call "attention". It submits many small AIC and AIV tasks that together implement that loop.

## The Files To Read In Order

If you want the shortest path to understanding, read in this order:

1. `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_runtime2_types.h`
2. `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_ring_buffer.h`
3. `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_tensormap.h`
4. `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_tensormap.cpp`
5. `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp`
6. `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_scheduler.h`
7. `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_scheduler.cpp`
8. `src/a2a3/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp`
9. `src/a2a3/runtime/tensormap_and_ringbuffer/aicore/aicore_executor.cpp`
10. `examples/a2a3/tensormap_and_ringbuffer/paged_attention/kernels/orchestration/paged_attention_orch.cpp`

The rest of this chapter follows that order.

## Public Runtime Surface: What The Orchestration SO Sees

The orchestration shared object does not link directly against runtime `.cpp` objects. Instead, it talks through a function-pointer table defined in `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_runtime2.h:52-74`.

Important lines in `pto_runtime2.h`:

- `:61-74` define `PTO2RuntimeOps`, the ABI between the orchestration `.so` and the runtime.
- `:82-102` define `PTO2Runtime`, which owns orchestrators, scheduler, shared memory, and GM heap.
- `:173-188` expose the orchestration-facing scope and submit APIs.
- `:235-269` implement `PTO2ScopeGuard` and `PTO2_SCOPE(rt)`, which are heavily used by examples.

This design is why the example orchestration code can be compiled as a standalone `.so` and later loaded by AICPU with `dlopen`.

## Host-Side Setup: How PTO2 Gets Ready Before Any Task Exists

The host-side setup lives in `src/a2a3/runtime/tensormap_and_ringbuffer/host/runtime_maker.cpp`.

Read `init_runtime_impl` in this order:

- `:85-119` uploads kernel binaries and records their device addresses by `func_id`.
- `:140-223` converts host arguments into device arguments.
  - input pointers are copied to device
  - output pointers are allocated on device and remembered for copy-back
  - inout pointers do both
- `:226-245` copies the orchestration `.so` to device-visible memory and stores a host copy inside the `Runtime` object for AICPU-side loading.
- `:264-279` reads ring override environment variables:
  - `PTO2_RING_TASK_WINDOW`
  - `PTO2_RING_HEAP`
  - `PTO2_RING_DEP_POOL`
- `:281-302` allocates the device GM heap and PTO2 shared memory block.
- `:304-317` marks orchestration as device-built and stores the device argument vector.

Two consequences matter for the rest of the chapter:

1. The orchestrator is not inventing buffers from nowhere. The host already allocated PTO2's global heap and shared memory.
2. The ring sizes are partly compile-time defaults and partly runtime-overridable through environment variables.

## Core Type And Task State Invariants

The file `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_runtime2_types.h` defines the rules of the whole runtime.

### Profiling Gates

Start at `pto_runtime2_types.h:30-56`.

- `PTO2_PROFILING` is the umbrella switch.
- `PTO2_ORCH_PROFILING` and `PTO2_SCHED_PROFILING` require `PTO2_PROFILING=1`.
- `PTO2_TENSORMAP_PROFILING` additionally requires orchestrator profiling.

This explains why some logs appear only in certain builds even if `--enable-profiling` is passed at runtime.

### Capacity Defaults

Read `:62-92`.

These are not random constants. They define the runtime's operating envelope:

- `PTO2_TASK_WINDOW_SIZE`: maximum in-flight task ID window
- `PTO2_HEAP_SIZE`: bytes in the packed output heap
- `PTO2_DEP_LIST_POOL_SIZE`: dependency-edge storage
- `PTO2_TENSORMAP_POOL_SIZE` and `PTO2_TENSORMAP_NUM_BUCKETS`: TensorMap capacity
- `PTO2_READY_QUEUE_SIZE`: per-shape ready queue capacity

If PTO2 deadlocks because a ring is full, the error logs point back to exactly these knobs.

### Worker Types

Read `:102-108`.

PTO2 distinguishes workers by execution resource:

- `PTO2_WORKER_CUBE`: matrix-heavy AIC work
- `PTO2_WORKER_VECTOR`: vector AIV work
- `PTO2_WORKER_AI_CPU`: scalar/control work
- `PTO2_WORKER_ACCELERATOR`: reserved fixed-function category

For the paged-attention example in this repo, you mostly care about the first two.

### Task State Machine

Read `:114-130`.

This is the runtime's main lifecycle:

```text
PENDING -> READY -> RUNNING -> COMPLETED -> CONSUMED
```

Interpretation:

- `PENDING`: some producer dependencies are still unfinished
- `READY`: dependency refcount caught up; task can enter a ready queue
- `RUNNING`: scheduler has dispatched active subtasks
- `COMPLETED`: all active subtasks finished
- `CONSUMED`: no remaining consumers or owning scopes hold the task alive

Why `CONSUMED` matters:

- task slots are reclaimed only after `CONSUMED`
- heap buffers are reclaimed only after the oldest finished task that owns them becomes reclaimable

So PTO2 is not "done when the kernel finishes". It is done when both execution and lifetime accounting are complete.

### Task Descriptor vs Slot State

The split between `PTO2TaskDescriptor`, `PTO2TaskPayload`, and `PTO2TaskSlotState` is one of the most important design decisions in PTO2.

Read:

- `pto_runtime2_types.h:284-294` for `PTO2TaskDescriptor`
- `pto_runtime2_types.h:306-315` for `PTO2TaskPayload`
- `pto_runtime2_types.h:329-353` for `PTO2TaskSlotState`

The logic is:

- descriptor: small shared-memory metadata needed for dispatch
- payload: cold data such as tensors, scalars, and fanin arrays
- slot state: hot scheduler-private state with atomics and refcounts

This split is why the scheduler hot path mostly touches a compact 64-byte slot state instead of walking a much larger descriptor blob.

### Fanout Lock

Read `pto_runtime2_types.h:420-463`.

This lock protects `fanout_head` and `fanout_count`.

Why it exists:

- the orchestrator may still be adding new consumers to a producer
- the scheduler may simultaneously traverse that producer's fanout list after completion

Without this lock, dependency edges could be lost or observed half-written.

## Ring Buffers: PTO2's Back-Pressure System

The file `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_ring_buffer.h` defines three allocation mechanisms:

- heap ring for output storage
- task ring for task IDs and slots
- dep pool ring for fanout linked-list nodes

This is the part of PTO2 that makes back-pressure explicit instead of hiding it behind dynamic allocation.

### Heap Ring

Read `pto_ring_buffer.h:62-234`.

The core routine is `pto2_heap_ring_alloc()` at `:80-168`.

Line-by-line reading guide:

- `:82` aligns the request for cache and DMA friendliness.
- `:95-114` loops until `try_alloc` succeeds.
- `:123-131` resets the spin counter when `heap_tail` advances, which means the scheduler reclaimed older packed output buffers.
- `:144-163` reports a fatal deadlock if nothing progresses for too long.

Then read `pto2_heap_ring_try_alloc()` at `:176-217`.

Its two cases are the usual circular-buffer cases:

- `top >= tail`: allocator is in the "tail behind me" region
- `top < tail`: allocator already wrapped and can only allocate within the gap

Important design rule:

- PTO2 never splits one output buffer across the wrap boundary

That makes buffer ownership simpler and avoids fragmented output records.

### Task Ring

Read `pto_ring_buffer.h:259-380`.

The main path is `pto2_task_ring_alloc()` at `:275-369`.

What to notice:

- `:287-305` tries allocation until it gets a new absolute `task_id`
- `:314-323` treats advance of `last_task_alive` as progress
- `:338-364` prints a deadlock diagnosis when the window stays full

Then read `pto2_task_ring_try_alloc()` at `:376-380` and beyond in the file.

The task ring is not allocating memory. It is allocating namespace. The runtime uses absolute `task_id` values, then maps them into slots by modulo.

The key invariant is:

```text
active task count = current_task_index - last_task_alive
```

If that reaches the window size, new tasks cannot be admitted.

### Dep Pool

The dep pool is less famous but just as important. It stores linked-list nodes that connect a producer to its consumers.

The orchestrator uses it while building dependencies; reclamation depends on old tasks retiring. That is why dep-pool deadlock messages often really mean "an older task never reached CONSUMED".

Read the reclaim helpers in `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp:179-246`.

The logic is:

- look at shared `last_task_alive`
- find the last consumed task's recorded dep-pool watermark
- advance the dep-pool tail to that watermark

So the dep pool is reclaimed in task order, not entry-by-entry.

## TensorMap: How PTO2 Discovers Dependencies Automatically

`src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_tensormap.h` and `.cpp` are the heart of PTO2's automatic dependency inference.

### The Big Idea

When the orchestrator submits a task:

- every input or inout tensor is looked up in TensorMap
- TensorMap returns the latest producer tasks that overlap that region
- those producer task IDs become fanin dependencies
- every output or inout tensor is inserted back into TensorMap as the new producer

So PTO2 infers edges from memory regions instead of asking the example author to manually describe a graph.

### Why Hash Only By Base Pointer

Read `pto_tensormap.h:20-29` and then `:219-274`.

This is the single most important TensorMap design choice:

- all views of the same raw tensor storage must land in the same bucket
- otherwise overlap detection would miss aliases that differ only by offset or shape

That is why the hash key is `tensor.buffer.addr`'s base storage address, not a full tensor descriptor hash.

### Entry Structure

Read `pto_tensormap.h:74-84`.

Fields to care about:

- `producer_task_id`: the task that most recently produced this region
- bucket links: for hash-chain traversal
- task links: for per-task cleanup
- `with_alloc`: distinguishes outputs from inouts in cleanup/overwrite logic
- `tensor`: full region descriptor used for overlap checks

### Allocation And Reuse

Read `pto_tensormap.h:136-174`.

`new_entry()` and `free_entry()` tell you a lot about PTO2 philosophy:

- prefer fixed pools
- reuse entries aggressively
- keep unlink operations O(1) by carrying both `prev` and `next`

### Lookup Path

Now read `pto_tensormap.h:219-274` slowly.

This is the line-by-line interpretation:

- `:220` hashes by base address
- `:221-223` start at the head of that bucket chain
- `:235-247` if the current entry is stale, truncate the rest of the chain
- `:253-264` if the base storage matches, run precise overlap detection
- `:257-263` push any overlapping producer into the result set

Why chain truncation is safe:

- bucket chains are newest-first
- if one entry is stale, everything behind it is older and therefore stale too

That is a neat optimization. The lookup path performs both query and opportunistic cleanup.

### Insert Path

Read `pto_tensormap.h:285-316`.

Insertion does two things at once:

1. inserts the entry at the head of the hash bucket
2. links the same entry into the producing task's task-local list

That second list is what makes later cleanup efficient.

### Validity And Cleanup

Read:

- `pto_tensormap.h:366-397`
- `pto_tensormap.cpp:212-227`

Key rule:

```text
entry is valid if producer_task_id >= last_task_alive
```

Meaning:

- once a producer task retires past the task window tail, its TensorMap entry is stale

`sync_tensormap()` does two jobs:

- refresh `last_task_alive` from shared memory
- trigger batch cleanup when free space is getting low or enough tasks retired

This is why the orchestrator starts each submission with TensorMap sync.

### Why INOUT Sometimes Removes Older Entries

Read `pto_orchestrator.cpp:479-487`.

If an `INOUT` region fully covers a previous region and that older entry was also an inout-style logical overwrite, PTO2 can remove the older entry to shorten future chains and promote chain-like dependencies.

This is subtle but important:

- it reduces redundant dependency fanin
- but it preserves the original allocating producer when needed for lifetime correctness

## Orchestrator Walkthrough: Submission From First Instruction To Ready Queue

The main orchestration logic is in `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp`.

### Initialization

Read `:94-149`.

The orchestrator owns:

- heap ring
- task ring
- private dep pool
- private TensorMap
- scope stack

That split matters:

- heap ring and task ring are tied to shared runtime state
- dep pool and TensorMap are orchestrator-side graph-building machinery

### Scope Tracking

Read `:253-293`.

`pto2_scope_begin()` records the current index into `scope_tasks`.

`pto2_scope_end()`:

- finds all tasks submitted since that begin offset
- passes them to the scheduler through `on_scope_end`
- rewinds the local vector

Interpret the scope as "an owning reference held by orchestration". Until the scope ends, each task keeps one extra `fanout_count` reference alive.

That is why large scopes can deadlock a small task window.

### The Submit Path

Read `pto2_submit_mixed_task()` from `:298-601`.

This function is the most important PTO2 function in the repository.

#### Step 0: Normalize Kernel Shape

Read `:302-317`.

PTO2 first derives an `active_mask` from `MixedKernels`. Then it normalizes one-AIV tasks so the runtime can always treat the first active vector subtask as `AIV0`.

That removes special cases later in dispatch.

#### Step 1: Sync And Pre-Check

Read `:319-362`.

What happens here:

- `:320` sync TensorMap validity
- `:323` reclaim dead dep-pool entries
- `:328` reject submission outside any open scope
- `:330-362` pre-check for scope deadlock

That scope deadlock check is worth understanding. Each task starts with `fanout_count = 1` because its enclosing scope owns it. If a single scope contains almost an entire task window, none of those tasks can fully retire until the scope ends, so new task slots cannot be reclaimed. PTO2 catches that before it creates a circular wait.

#### Step 2: Allocate A Task Slot

Read `:365-404`.

Key actions:

- allocate a new absolute `task_id`
- compute the wrapped slot index
- initialize `PTO2TaskDescriptor`
- initialize `PTO2TaskSlotState`
- set initial `fanout_count = 1` for scope ownership

This is the point where the task exists, but it has no edges yet.

#### Step 3: Copy Parameters

Read `:406-420`.

The payload stores a task-local copy of:

- tensor descriptors
- scalar values
- tensor/scalar tags

This is important because later dispatch should not depend on the caller's stack memory.

#### Step 4: Allocate Output Buffer Space

Read `:425-441`.

PTO2 scans outputs whose addresses are still zero and sums the aligned sizes into one packed allocation. If the task has multiple outputs, they live in one contiguous packed buffer from the heap ring.

That is why the task descriptor stores both `packed_buffer_base` and `packed_buffer_end`.

#### Step 5: First Tensor Pass, Build Fanin

Read `:448-508`.

This pass does different work by parameter type:

- `INPUT` and `INOUT`: query TensorMap for producers
- `OUTPUT`: assign a packed-buffer address if needed

The inner loop at `:460-488` is the real dependency builder:

- gather producer task IDs from lookup results
- deduplicate them
- optionally drop covered inout entries from TensorMap

At the end of this phase, the orchestrator knows which earlier tasks must finish first.

#### Step 6: Insert New Outputs Into TensorMap

Read `:511-518`.

Now this task becomes the latest producer for its outputs and inouts.

This ordering is intentional:

- first discover dependencies on older producers
- then publish yourself as the new producer

#### Step 7: Build Producer Fanout Lists And Possibly Make Task Ready

Read `:522-594`.

This block is where the orchestrator links graph structure into scheduler-visible metadata.

Important details:

- `:528-529` initialize task state and `fanout_refcount`
- `:531-537` ensure dep-pool space exists
- `:540` set `fanin_count = actual_fanin + 1`
- `:543` cache direct producer slot pointers for later release
- `:547-569` attach this consumer to each producer's `fanout_head`
- `:559-566` detect producers that already completed and count them as `early_finished`
- `:574-580` do one combined `fanin_refcount.fetch_add`

The `+1` redundancy is easy to miss and very important. PTO2 temporarily sets:

```text
fanin_count = actual_fanin + 1
```

Then it later adds `early_finished + 1` to the refcount in one atomic. That extra `+1` prevents the task from becoming ready too early while the orchestrator is still wiring the graph.

If the final refcount already reaches `fanin_count`, the task is pushed directly into the appropriate ready queue.

### What A Submission Really Means

After `pto2_submit_mixed_task()` returns, PTO2 has:

- allocated task ID and slot
- copied payload
- assigned output addresses
- inferred fanin
- registered itself as producer of outputs
- linked itself into producer fanout lists
- possibly queued itself as ready

That entire graph-building operation is triggered by a single high-level orchestration call such as `pto2_rt_submit_aic_task()`.

## Scheduler Walkthrough: When A Task Actually Runs

The scheduler interface is in `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_scheduler.h`, with setup code in `pto_scheduler.cpp`.

### Ready Queue Design

Read `pto_scheduler.h:36-252`.

The ready queue is a bounded lock-free MPMC queue using sequence counters. The reasons are visible in the implementation:

- producers only contend on `enqueue_pos`
- consumers only contend on `dequeue_pos`
- each slot's sequence number prevents ABA confusion

This is a queue designed for frequent cross-thread handoff under contention.

### Local Ready Buffers

Read `pto_scheduler.h:48-84`.

These are tiny per-thread staging buffers. They are an optimization, not the source of truth.

Idea:

- when a completion makes downstream work ready, the scheduler first tries to keep that work local to the current scheduling thread
- only overflow or unsuitable work goes back to the shared MPMC queue

This reduces global queue traffic.

### Scheduler Initialization

Read `pto_scheduler.cpp:105-158`.

The important decision here is allocating `slot_states` separately from shared memory. That means the fast scheduler state is local to AICPU scheduler threads and can be cache-tuned independently of the larger shared descriptor structures.

### Readiness Transition

Read `pto_scheduler.h:419-469`.

`release_fanin_and_check_ready()` is the PENDING to READY gate:

- increment `fanin_refcount`
- if it reaches `fanin_count`, the task can run
- try local buffer first
- otherwise push into the global ready queue for its resource shape

This is where dependency resolution turns into runnable work.

### Completion Is Two-Stage

Read:

- `pto_scheduler.h:530-536` for `on_subtask_complete`
- `pto_scheduler.h:549-617` for `on_mixed_task_complete`
- `pto_scheduler.h:625-660` for `on_task_release`

This split is essential because one logical PTO2 task may occupy multiple execution resources.

Example:

- AIC-only task: one subtask
- AIV_X2 task: two vector subtasks
- AIC_AIV_X1 task: one AIC subtask and one AIV subtask

So PTO2 first marks subtask bits done:

```text
subtask_done_mask |= done_bit
```

Only when the mask equals the task's `active_mask` does the mixed task become logically complete.

Then the scheduler:

1. marks the task `COMPLETED`
2. traverses its fanout list to wake consumers
3. later traverses its fanin list to release producer references
4. checks whether the task itself can become `CONSUMED`

That separation is why PTO2 can support mixed-resource tasks cleanly.

### Consumption And Ring Reclamation

Read `pto_scheduler.h:325-369`.

`check_and_handle_consumed()` checks:

```text
fanout_refcount == fanout_count
```

If true and the task is already `COMPLETED`, it CASes the task to `CONSUMED`.

Then `advance_ring_pointers()` at `:325-347` moves:

- `last_task_alive` forward across consecutive consumed tasks
- `heap_tail` forward to the newest consumed packed-buffer end

That is the bridge from logical lifetime to physical ring reclamation.

## AICPU Executor: The Real Scheduling Loop

The largest runtime file is `src/a2a3/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp`.

You should not read it top-to-bottom in one sitting. Use this order instead.

### Core Discovery And Ownership

Start with:

- `aicpu_executor.cpp:77-151` for core-tracking structs
- `:153-245` for `AicpuExecutor` state

The runtime groups one AIC core and two AIV cores into a logical cluster. That matches the resource shapes used by the scheduler:

```text
AIC_ONLY
AIV_X1
AIV_X2
AIC_AIV_X1
AIC_AIV_X2
```

A cluster is the minimum scheduling unit for mixed-resource tasks.

### Initialization

Read `aicpu_executor.cpp:756-835`.

Important actions:

- discover hardware cores through handshake
- assign clusters to scheduler threads
- inspect PTO2 shared memory to estimate total task count
- initialize profiling buffers if enabled

The executor also distinguishes:

- orchestration threads
- scheduler threads

For the paged-attention config in this repo:

- `aicpu_thread_num = 4`
- `orch_thread_num = 2`
- therefore `sched_thread_num = 2`

### One Scheduler Loop Iteration

Now read `resolve_and_dispatch_pto2()` from `aicpu_executor.cpp:860` onward, especially `:882-1212`.

This is the runtime loop that most users conceptually mean when they say "PTO2 scheduler".

#### Phase 0: One-Time Init

Read `:882-903`.

The first entering scheduler thread initializes profiling buffers and phase tracking. Other threads wait for that to finish.

#### Phase 1: Completion Processing

Read `:998-1055`.

Each thread:

- polls its running AIC cores
- polls its running AIV cores
- for each finished subtask:
  - marks the subtask complete
  - if this finishes the mixed task, runs `on_mixed_task_complete`
  - defers producer-release work into a small local array

Why defer producer release:

- it keeps the immediate completion hot path shorter
- it batches some of the slower refcount and reclamation work

#### Phase 2: Local Dispatch

Read `:1071-1138`.

The scheduler first drains `local_bufs`, which hold tasks that became ready due to nearby completions on this same thread.

For each task:

- derive its resource shape from `active_mask`
- find an idle cluster that can satisfy it
- dispatch subtasks directly without touching the global queue

This is a locality optimization and one of the more interesting PTO2 scheduling ideas.

#### Phase 3: Global Dispatch

Read `:1140-1212`.

Any local overflow is pushed back to the global ready queues. Then the scheduler probes global queues in a thread-dependent order:

- even threads are slightly AIC-first after the widest shapes
- odd threads are slightly AIV-first after the widest shapes

That reduces all scheduler threads hammering the same queue in the same order.

### Dispatch Mechanics

Read `dispatch_subtask_to_core()` at `aicpu_executor.cpp:474-504`.

The scheduler:

- builds a compact `PTO2DispatchPayload`
- records which subslot that core is executing
- writes the task ID into the core register
- moves the core from idle set to running set

Notice that the payload contains:

- function binary address
- tensor/scalar arguments

The AICore side does not rebuild scheduling metadata. It just executes what it is told.

## AICore Executor: Tiny But Important

Read `src/a2a3/runtime/tensormap_and_ringbuffer/aicore/aicore_executor.cpp`.

This file is short, but it defines the worker-side protocol exactly.

### The Dispatch Protocol

Read `aicore_executor.cpp:54-140`.

The AICore loop does:

1. clear stale dispatch register state from previous rounds
2. wait for AICPU-ready handshake
3. report physical core ID and core type
4. set itself idle
5. poll `DATA_MAIN_BASE`
6. if a new task arrives:
   - invalidate caches
   - read the payload pointer
   - write ACK
   - execute kernel
   - write FIN
7. exit only on `AICORE_EXIT_SIGNAL`

This is a very explicit register protocol:

```text
AICPU writes task_id+1  -> AICore observes new work
AICore writes ACK       -> AICPU knows the core accepted the task
AICore runs kernel
AICore writes FIN       -> AICPU completion phase can retire the subtask
```

This is why the scheduler completion loop polls registers rather than higher-level queues.

## Visualizing Synchronization And Lifetime

### End-To-End Task Lifecycle

```text
orchestrator submit
  |
  v
task slot allocated
  |
  v
fanin discovered from TensorMap
  |
  +--> not all producers done -> PENDING
  |
  +--> all producers already done -> READY
                                   |
                                   v
                           scheduler dispatches subtask(s)
                                   |
                                   v
                                RUNNING
                                   |
                                   v
                      all active subtasks finished?  no -> stay RUNNING
                                   |
                                  yes
                                   v
                               COMPLETED
                                   |
                     all consumer + scope refs released?
                          no                  yes
                          |                    |
                          v                    v
                    keep buffer alive      CONSUMED
                                                |
                                                v
                             task ring / heap ring tail may advance
```

### Why Scope References Exist

```text
Task T fanout_count starts at 1
  |
  +-- the "1" means: the owning orchestration scope still keeps T alive
  +-- each downstream consumer adds another fanout reference

When scope_end() runs:
  release_producer(T)

When each downstream consumer is done:
  release_producer(T)

Only when all such releases happen can T become CONSUMED.
```

This is the right mental model for debugging "scope deadlock" messages.

## Paged Attention Example: Map The Algorithm To PTO2

Now we can read the example with the runtime model in mind.

Files:

- `examples/a2a3/tensormap_and_ringbuffer/paged_attention/kernels/kernel_config.py`
- `examples/a2a3/tensormap_and_ringbuffer/paged_attention/kernels/orchestration/paged_attention_orch.cpp`
- `examples/a2a3/tensormap_and_ringbuffer/paged_attention/golden.py`

### Runtime Configuration

Read `kernel_config.py:41-46`.

This example chooses:

- runtime: `tensormap_and_ringbuffer`
- `aicpu_thread_num = 4`
- `orch_thread_num = 2`
- `block_dim = 24`

So two AICPU threads orchestrate and the remaining two schedule and dispatch.

### Kernel IDs

Read `kernel_config.py:29-37`.

The `func_id` mapping is:

- `0`: `QK`
- `1`: `SF`
- `2`: `PV`
- `3`: `UP`
- `4`: `AIC_HUB`
- `5`: `AIV_HUB`

For the main path in this example:

- `QK` runs on AIC
- `SF` runs on AIV
- `PV` runs on AIC
- `UP` runs on AIV
- `AIV_HUB` initializes the running accumulators

PTO2 can represent true mixed tasks that need AIC and AIV at the same time, but this example does not use that feature. `paged_attention` is written as an alternating chain of `AIC_ONLY` and `AIV_X1` tasks. That is a good teaching choice because the dependency graph is easier to see.

### Golden Case To Keep In Mind

Read `golden.py:14-24`.

`Case1` is:

- `batch = 1`
- `num_heads = 16`
- `head_dim = 16`
- `block_size = 16`
- `context_len = 33`

This is the easiest case to trace because:

- `q_tile = 16`, so `q_loop = 1`
- `context_len = 33`, so `bn_this_batch = ceil(33 / 16) = 3`

That means one query tile will be processed against three KV blocks.

### Orchestration Entry

Read `paged_attention_orch.cpp:45-60`.

Two exported functions matter:

- `aicpu_orchestration_config()`
- `aicpu_orchestration_entry()`

The config advertises `expected_arg_count = 10`. That tells the executor how many device arguments the orchestration entry expects.

### Argument Decode

Read `:63-93`.

This block unpacks:

- query pointer
- key cache pointer
- value cache pointer
- block table
- context lengths
- output pointer
- config pointer
- raw sizes

Then it derives:

- `batch`
- `num_heads`
- `head_dim`
- `block_size`
- `block_num`
- `scale_value`
- `q_tile = 16`
- `q_loop = ceil(num_heads / 16)`

### Batch Partition Across Orchestrators

Read `:96-103`.

This line is easy to skip:

```cpp
b_start = batch * orch_thread_index / orch_thread_num;
b_end   = batch * (orch_thread_index + 1) / orch_thread_num;
```

For `Case1` with `batch = 1` and `orch_thread_num = 2`:

- orchestrator thread 0 gets `[0, 0)`
- orchestrator thread 1 gets `[0, 1)`

So on this tiny case, only one orchestrator thread actually submits work. That is normal.

### Tensor Wrappers

Read `:104-117`.

The example wraps device buffers into PTO2 tensor descriptors:

- `query`, `key_cache`, `value_cache` use `make_tensor_external`
- `out` is also external because the final result should live in a caller-visible buffer

This is the point where the high-level example begins to hand PTO2 enough metadata for overlap-based dependency inference.

### The Actual Attention Loop

Now read `:119-205` as the algorithm.

#### Outer Loop Structure

The loop nest is:

```text
for each batch item b_idx assigned to this orch thread:
  compute cur_seq and number of blocks
  for each query-head tile q_idx:
    open a PTO2 scope
    initialize running accumulators
    for each KV block bn:
      QK
      SF
      PV
      UP
```

That is the cleanest way to see the example.

#### Scope Per Query Tile

Read `:122-123`.

Each `q_idx` tile is wrapped in `PTO2_SCOPE(rt)`.

This is a real lifetime boundary:

- all intermediates for one query tile are owned by this scope
- when the block ends, PTO2 releases the scope-held references

That prevents long-lived accumulation of temporary tensors across head tiles.

#### Initialize Running State

Read `:124-145`.

The example allocates:

- `oi`: running output accumulator
- `li_update`: running normalization accumulator
- `mi_update`: running max accumulator

Then it submits `AIV_HUB` with all three as outputs.

Interpretation:

- before processing any KV block, the example needs zeroed or initialized running state

#### Per-Block Task Chain

Read `:146-203`.

For each block:

1. `QK` at `:158-163`
   - inputs: query tile `qi`, key block `kj`
   - output: score tile `sij`

2. `SF` at `:165-177`
   - input: valid score slice `sij_valid`
   - scalar: scale
   - outputs: `pij_f16`, `mi`, `li`

3. `PV` at `:182-187`
   - inputs: `pij_f16`, value block `vj`
   - output: `oi_tmp`

4. `UP` at `:192-203`
   - inputs: `mi`, `li`, `oi_tmp`
   - inouts: running `mi_update`, `li_update`, `oi`
   - output: final `out_view`
   - scalars: `is_first`, `is_last`

The runtime builds the graph from tensor overlap. The example never manually says:

- "`SF` depends on `QK`"
- "`PV` depends on `SF`"
- "`UP` depends on `PV` and prior `UP` instances"

TensorMap infers all of that because later tasks read or update tensors written earlier.

### Case1 Task Count: Why It Is 13

For `Case1`:

- one batch item
- one query tile
- three KV blocks

Tasks submitted:

- `1` x `AIV_HUB`
- `3` x `QK`
- `3` x `SF`
- `3` x `PV`
- `3` x `UP`

Total:

```text
1 + 3 + 3 + 3 + 3 = 13 tasks
```

That is a very good sanity check when you read logs or instrument the example.

### The Dependency Story For One Block

For the first KV block, think of the graph like this:

```text
AIV_HUB -> QK -> SF -> PV -> UP
```

For later KV blocks, `UP` also depends on the previous block's running accumulators because `mi_update`, `li_update`, and `oi` are passed as `INOUT`.

So the chain is closer to:

```text
block 0: AIV_HUB -> QK0 -> SF0 -> PV0 -> UP0
block 1:            QK1 -> SF1 -> PV1 -> UP1
                                    ^      |
                                    |______|
                         depends on previous running state
```

That is exactly the pattern TensorMap is good at: repeated reuse of partially overlapping or fully overlapping logical tensors.

## How Scheduling Looks For Paged Attention

Now connect the example to the scheduler's resource model.

For this example:

- `QK` needs AIC only
- `SF` needs AIV only
- `PV` needs AIC only
- `UP` needs AIV only

So most tasks enter either `AIC_ONLY` or `AIV_X1` ready queues.

This makes paged attention a good runtime study case because it naturally alternates between:

- matrix-heavy work
- vector-heavy work

That means you can actually see the scheduler balancing two different resource classes instead of just one.

## The Five-Function Fast Path

If you only have time to read the absolute hot path, read these five functions in order:

1. `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_tensormap.h:219-274`
   - `PTO2TensorMap::lookup`
2. `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp:298-601`
   - `pto2_submit_mixed_task`
3. `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_scheduler.h:549-617`
   - `PTO2SchedulerState::on_mixed_task_complete`
4. `src/a2a3/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp:860-1212`
   - `AicpuExecutor::resolve_and_dispatch_pto2`
5. `src/a2a3/runtime/tensormap_and_ringbuffer/aicore/aicore_executor.cpp:54-140`
   - `aicore_execute`

That is the minimum complete story:

- find producers
- submit a task
- retire a completed mixed task
- dispatch work from AICPU
- execute work on AIC/AIV

## Profiling PTO2 The Right Way

PTO2 profiling has both compile-time and runtime parts.

### Compile-Time Profiling Levels

Read `src/a2a3/runtime/tensormap_and_ringbuffer/docs/profiling_levels.md`.

The high-level rule is:

- macros decide what profiling code is compiled in
- `enable_profiling` decides whether runtime timing data buffers are actively collected

So:

- `--enable-profiling` alone is not enough for detailed PTO2 breakdowns
- you also need the corresponding PTO2 profiling macros enabled at build time

The hierarchy starts in `pto_runtime2_types.h:30-56` and is documented in `profiling_levels.md`.

### Device Log Profiling

Read `src/a2a3/runtime/tensormap_and_ringbuffer/docs/device_log_profiling.md`.

This is the most practical profiling document for PTO2 because AICPU `DEV_ALWAYS` logs do not show up directly in the example runner's terminal output. They go into the Ascend device log directory.

What you get there:

- orchestrator profiling summary
- per-scheduler-thread phase breakdown

### What The Orchestrator Profiling Means

The counters are accumulated in `pto_orchestrator.cpp:47-74` and exported in `:649-687`.

Map them like this:

- `sync_tensormap`: pre-submit cleanup and validity sync
- `task_ring_alloc`: task slot allocation
- `param_copy`: payload copy
- `lookup+dep`: TensorMap lookup plus dependency edge construction
- `heap_alloc`: packed output allocation
- `tensormap_ins`: publish new producer entries
- `fanin+ready`: wire fanin/fanout and possibly queue the task
- `scope_end`: scope release overhead

If `lookup+dep` dominates, suspect:

- many aliased tensors
- long dependency chains
- too many overlapping inouts

If `heap_alloc` or `task_ring_alloc` shows high wait time, suspect ring pressure.

### What The Scheduler Profiling Means

The summary logic lives partly in `aicpu_executor.cpp` and partly in the scheduler counters declared in `pto_scheduler.cpp:20-53`.

The main scheduler phases are:

- complete
- dispatch
- scan
- idle

Interpret them like this:

- high `complete`: dependency notifications and release work dominate
- high `dispatch`: queue pop, payload setup, and register writes dominate
- high `idle`: the schedulers are starved, often because orchestration is too slow or not enough work is ready

### TensorMap Profiling

TensorMap counters are in:

- `pto_tensormap.h:48-55`
- `pto_tensormap.cpp:28-35`
- `pto_tensormap.cpp:233-250`

Use TensorMap profiling when you need to answer:

- how long bucket chains are becoming
- whether overlap checks are too frequent
- whether inserts greatly outnumber successful overlap hits

### Practical Debugging Checklist

If PTO2 is slow or stuck, inspect in this order:

1. Are task window and heap sizes too small?
   - check `PTO2_RING_TASK_WINDOW`
   - check `PTO2_RING_HEAP`
   - check `PTO2_RING_DEP_POOL`

2. Is the current scope too large?
   - scope deadlock errors in `pto_orchestrator.cpp:330-362`

3. Is the heap ring blocked?
   - heap deadlock logs in `pto_ring_buffer.h:144-163`

4. Is the task ring blocked?
   - flow-control deadlock logs in `pto_ring_buffer.h:338-364`

5. Is the dep pool blocked?
   - dep-pool deadlock logs in `pto_orchestrator.cpp:218-243`

6. Are TensorMap chains too long?
   - enable TensorMap profiling and inspect chain walk stats

## A Source-Guided Study Plan

If you want to do a real line-by-line reading in one afternoon, use this order:

1. Read `pto_runtime2_types.h:102-130`, `:284-353`, and `:420-463`.
   - understand workers, task states, slot state, and fanout lock

2. Read `pto_ring_buffer.h:80-217` and `:275-380`.
   - understand why PTO2 can block before scheduling even starts

3. Read `pto_tensormap.h:219-316` and `pto_tensormap.cpp:212-227`.
   - understand automatic dependency inference and cleanup

4. Read `pto_orchestrator.cpp:319-594`.
   - this is the graph builder

5. Read `pto_scheduler.h:419-660`.
   - this is the state-transition logic

6. Read `aicpu_executor.cpp:998-1212`.
   - this is the real dispatch loop

7. Read `aicore_executor.cpp:86-135`.
   - this is the worker-side protocol

8. Re-read `paged_attention_orch.cpp:119-205`.
   - now the example should look like a graph-construction program instead of opaque kernel glue

## What To Remember

If you forget most of the implementation details, keep these four ideas:

1. PTO2 builds dependencies from tensor overlap, not from explicit user-written graph edges.
2. PTO2 uses three ring-style resources, so back-pressure and lifetime are first-class runtime concepts.
3. Completion is not the same as consumption; reclamation waits for both execution and lifetime release.
4. `paged_attention` is a perfect study example because it alternates AIC and AIV work while repeatedly updating shared running tensors, which exercises nearly every interesting PTO2 mechanism.
