# Tensormap-Ringbuffer Runtime: Deep Dive

This document provides a detailed explanation of the **tensormap-ringbuffer runtime** and the **paged attention example**, combining architectural insights with real profiling data from the `outputs/` directory.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Core Data Structures](#3-core-data-structures)
4. [Execution Flow](#4-execution-flow)
5. [Paged Attention Example](#5-paged-attention-example)
6. [Performance Analysis](#6-performance-analysis)
7. [Profiling Output Reference](#7-profiling-output-reference)

---

## 1. Overview

The **tensormap-ringbuffer** runtime is a heterogeneous task graph executor optimized for Ascend NPU devices. It enables coordinated execution across three distinct processors:

| Processor | Role | Unit |
|-----------|------|------|
| **Host (CPU)** | Orchestration, data transfer, profiling | x86/ARM |
| **AICPU** | Task scheduling, dependency management | AI CPU cores |
| **AICore** | Kernel execution (matmul, vector ops) | CUBE (AIC) / VECTOR (AIV) |

### Key Design Goals

```
+---------------------------+------------------------------------------+
| Goal                      | Mechanism                                |
+---------------------------+------------------------------------------+
| Zero-allocation overhead  | Ring buffer based memory management      |
| Lazy invalidation         | TensorMap for dependency discovery       |
| Lock-free synchronization | Per-task spinlocks, atomic fanout updates|
| Scope-based lifecycle     | Buffer release tied to task scopes       |
+---------------------------+------------------------------------------+
```

### Three Independent Programs

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         HOST (.so)                                       │
│  - Python bindings, device management                                    │
│  - Performance collection (double-buffer polling)                        │
│  - Copy input/output data between host ↔ device                          │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ Device memory + Handshake
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        AICPU (.so)                                       │
│  - Load orchestration .so and execute task graph                         │
│  - TensorMap: track producer→consumer dependencies                       │
│  - Scheduler: dispatch ready tasks to AICore                             │
│  - Ring buffers: task slots, heap allocation, dep list pool              │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ Register-based dispatch
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        AICORE (.o)                                       │
│  - Poll DATA_MAIN_BASE register for new tasks                            │
│  - Execute kernel function (CUBE matmul / VECTOR ops)                    │
│  - Write performance timestamps to double-buffer                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Architecture

### 2.1 Communication Mechanisms

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DEVICE-SIDE SHARED MEMORY                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ Header (Flow Control Pointers)                                   │    │
│  │ ┌───────────────────┬───────────────────┬─────────────────────┐ │    │
│  │ │ current_task_index│ heap_top          │ orchestrator_done   │ │    │
│  │ │ (orch → sched)    │ (orch → sched)    │ (orch signal)       │ │    │
│  │ ├───────────────────┼───────────────────┼─────────────────────┤ │    │
│  │ │ last_task_alive   │ heap_tail         │                     │ │    │
│  │ │ (sched → orch)    │ (sched → orch)    │ (back-pressure)     │ │    │
│  │ └───────────────────┴───────────────────┴─────────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ TaskDescriptor[] Ring Buffer (task_window_size slots)            │    │
│  │ ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬───────────┐   │    │
│  │ │ T0  │ T1  │ T2  │ T3  │ ... │     │     │     │ (wrap)    │   │    │
│  │ └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴───────────┘   │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ DepListEntry[] Pool (Ring Buffer)                                │    │
│  │ ┌─────┬─────┬─────┬───────────────────────────────────────────┐ │    │
│  │ │ E0  │ E1  │ E2  │ ... singly-linked fanin/fanout entries    │ │    │
│  │ └─────┴─────┴─────┴───────────────────────────────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

Important clarification:
- The **HeapRing allocation region is NOT inside shared memory**. Shared memory only carries `heap_top` / `heap_tail` pointers for back-pressure.
- The actual heap storage is a separate GM buffer (`gm_heap`) owned by the runtime; see `src/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp` and `src/runtime/tensormap_and_ringbuffer/runtime/pto_ring_buffer.h`.

### 2.2 Handshake Buffer (Per-Core)

```cpp
struct Handshake {
    volatile uint32_t aicpu_ready;        // AICPU wakes the core (1=ready)
    volatile uint32_t aicore_done;        // AICore responds (physical_core_id+1)
    volatile uint64_t task;              // PTO2DispatchPayload* (set during handshake)
    volatile int32_t  task_status;        // 0=idle, 1=busy
    volatile int32_t  control;            // 0=execute, 1=quit
    volatile CoreType core_type;          // AIC / AIV
    volatile uint64_t perf_records_addr;  // PerfBuffer* (double-buffered)
    volatile uint32_t perf_buffer_status; // 0=not full, 1=full
    volatile uint32_t physical_core_id;   // Physical core id (device-specific)
};
```

### 2.3 Register-Based Dispatch

```
    AICPU (Scheduler)                              AICore (Worker)
    ─────────────────                              ────────────────
          │                                              │
          │  1. Pack PTO2DispatchPayload                 │
          │     - task_id, kernel_id                     │
          │     - function_bin_addr                      │
          │     - args[]                                 │
          │                                              │
          │  2. Write payload to handshake.task          │
          ├──────────────────────────────────────────────┤
          │                                              │
          │  3. Write (task_id + 1) to                   │
          │     DATA_MAIN_BASE register                  │
          │────────────────────────────────────────────►│
          │                                              │
          │                                              │ 4. Poll register
          │                                              │    until != 0
          │                                              │
          │                                              │ 5. dcci (invalidate)
          │                                              │
          │                                              │ 6. Read payload
          │                                              │    from handshake
          │                                              │
          │                                              │ 7. Execute kernel
          │                                              │
          │                                              │ 8. pipe_barrier
          │                                              │
          │  9. Poll COND register for completion       ◄│
          │◄───────────────────────────────────────────-│
          │                                              │
```

---

## 3. Core Data Structures

### 3.0 Tensor Descriptor (`Tensor`)

Most of the dependency tracking in this runtime is driven by the **`Tensor` descriptor** (not a raw buffer pointer). Kernels receive `Tensor*` arguments, and the orchestrator uses `Tensor` metadata to discover overlaps and build the DAG.

Key fields (see `src/runtime/tensormap_and_ringbuffer/runtime/tensor.h`):
- `buffer.addr` / `buffer.size`: underlying GM allocation (bytes)
- `start_offset`, `strides[]`, `repeats[]`, `ndims`: a strided access pattern **in elements**
- `dtype`: element type (overlap checks are done in bytes, so dtype matters)
- `version` / `overlap_type`: controls overlap semantics (`OverlapType::Fuzzy` forces conservative deps)

Overlap detection (see `src/runtime/tensormap_and_ringbuffer/runtime/tensor.cpp`):
1. Different `buffer.addr` ⇒ `NO_OVERLAP`.
2. Compute a fuzzy byte segment per tensor and early-reject if disjoint.
3. Fast exact-ish checks for common cases (1D contiguous, or matching dtype/ndims/strides).
4. Fallback `complex_overlap()` that iterates contiguous segments when layouts are complex.

Practical takeaway:
- Prefer simple `Tensor.view(...)` slices (like paged-attention does) to keep overlap checks cheap and precise.

### 3.1 TensorMap (Lazy Invalidation Hash Table)

The TensorMap tracks which task produced each tensor region, enabling automatic dependency discovery.

```
┌───────────────────────────────────────────────────────────────────────┐
│                         TENSORMAP STRUCTURE                            │
├───────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  Hash Buckets (65536 entries, power-of-2)                              │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │ bucket[0] ─────► Entry ─► Entry ─► NULL                        │   │
│  │ bucket[1] ─────► NULL                                          │   │
│  │ bucket[2] ─────► Entry ─► NULL                                 │   │
│  │ ...                                                             │   │
│  │ bucket[hash(base_ptr)] ─► Chain of entries with same base_ptr │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                                                                        │
│  Entry Pool (Bounded pool + lazy invalidation)                         │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │ struct PTO2TensorMapEntry {                                      │   │
│  │     Tensor  tensor;            // Full strided descriptor        │   │
│  │     int32_t producer_task_id;  // Producer task                  │   │
│  │     int32_t next_in_bucket;    // Bucket chain                   │   │
│  │     int32_t next_in_task;      // Per-task chain (cleanup)       │   │
│  │     bool    with_alloc;        // OUTPUT vs INOUT hint           │   │
│  │     ... (prev pointers, flags)                                   │   │
│  │ };                                                              │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                                                                        │
│  Validity Check: entry.valid = (producer_task_id >= last_task_alive)  │
│                                                                        │
│  Overlap Detection:                                                    │
│    - Hash ONLY by base_ptr (Tensor.buffer.addr)                        │
│    - For each entry in the bucket: Tensor::is_overlap(...)             │
│                                                                        │
└───────────────────────────────────────────────────────────────────────┘
```

Two subtle (but important) behaviors to know (from `src/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp` and `src/runtime/tensormap_and_ringbuffer/runtime/pto_tensormap.cpp`):
- **Chain truncation on staleness:** bucket chains are in descending `producer_task_id` order. When lookup hits the first stale entry (producer retired), it truncates the rest of the chain because all remaining entries must be older/stale too.
- **INOUT “COVERED” pruning:** when a new `INOUT` tensor fully covers a previous `INOUT` tensor region (`OverlapStatus::COVERED` and `with_alloc==false`), the orchestrator removes that older TensorMap entry. This intentionally turns “many overlapping versions” into a mostly-linear dependency chain and keeps lookup cost bounded. In paged-attention, the `UP` tasks repeatedly `INOUT` the same state tensors, so this pruning prevents TensorMap growth from exploding.

### 3.2 Task Descriptor

```cpp
struct PTO2TaskDescriptor {
    // Identity
    int32_t task_id;              // Unique identifier (0, 1, 2, ...)
    int32_t kernel_id;            // Function ID (opaque to runtime; defined by kernel_config)
    int32_t worker_type;          // CUBE/VECTOR/AI_CPU/ACCELERATOR

    // Fanin (read-only after submission)
    int32_t fanin_head;           // Head of dependency list
    int32_t fanin_count;          // Expected producer count

    // Fanout (protected by per-task spinlock)
    volatile int32_t fanout_lock;
    volatile int32_t fanout_head;
    volatile int32_t fanout_count; // 1 (scope owner) + consumers

    // Output buffer
    void* packed_buffer_base;
    void* packed_buffer_end;
    int32_t output_index[16];
    int32_t num_outputs;

    // Status & params
    bool is_active;
    PTOParam params[16];
    Tensor tensor_copies[16];  // Task-owned Tensor structs (params[i].tensor points here)
    int param_count;
};
```

For the paged-attention example, kernel IDs come from `tests/device_tests/tensormap_and_ringbuffer/paged_attention/kernels/kernel_config.py`:
- `0`: `QK` (AIC/CUBE)
- `1`: `SF` (AIV/VECTOR, softmax_prepare)
- `2`: `PV` (AIC/CUBE)
- `3`: `UP` (AIV/VECTOR, online_update + normalize)
- `4`: `AIC_HUB` (AIC/CUBE, no-op “hub”)
- `5`: `AIV_HUB` (AIV/VECTOR, no-op “hub”)

### 3.3 Task State Machine

```
                           ┌──────────────────────────────────────────────┐
                           │               TASK LIFECYCLE                  │
                           └──────────────────────────────────────────────┘

    ┌─────────┐                     ┌─────────┐                     ┌───────────┐
    │ PENDING │────────────────────►│  READY  │────────────────────►│  RUNNING  │
    └─────────┘  fanin_refcount=0   └─────────┘  dispatched to      └───────────┘
         │                               │       idle core               │
         │                               │                               │
         │ (waiting for                  │ (in ready queue)              │ (kernel executing)
         │  producer deps)               │                               │
         │                               │                               ▼
         │                               │                         ┌───────────┐
         │                               │                         │ COMPLETED │
         │                               │                         └───────────┘
         │                               │                               │
         │                               │                               │ all consumers
         │                               │                               │ discovered &
         │                               │                               │ completed
         │                               │                               ▼
         │                               │                         ┌───────────┐
         │                               │                         │ CONSUMED  │
         └───────────────────────────────┴─────────────────────────►   (freed) │
                                                                    └───────────┘
```

---

## 4. Execution Flow

### 4.1 Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      TENSORMAP-RINGBUFFER EXECUTION FLOW                     │
└─────────────────────────────────────────────────────────────────────────────┘

 PHASE 1: INITIALIZATION
 ═══════════════════════

 HOST                        AICPU                        AICORE (per core)
 ─────                       ─────                        ─────────────────
   │                           │                                 │
   │ Copy orchestration .so    │                                 │
   │ + kernel .o to device     │                                 │
   │──────────────────────────►│                                 │
   │                           │                                 │
   │ Launch AICPU entry        │                                 │
   │──────────────────────────►│ Load orchestration .so          │
   │                           │ Initialize shared memory        │
   │                           │ Set aicpu_ready = 1             │
   │                           │─────────────────────────────────►
   │                           │                                 │
   │                           │                                 │ Wait for aicpu_ready
   │                           │                                 │ Report physical_core_id
   │                           │◄────────────────────────────────│
   │                           │                                 │

 PHASE 2: ORCHESTRATION (Task Graph Construction)
 ════════════════════════════════════════════════

                          ORCHESTRATOR (runs on AICPU)
                          ────────────────────────────
                                     │
     For each task T:                │
     ┌─────────────────────────────────────────────────────────────────────┐
     │                                                                      │
     │  1. Allocate task slot from TaskRing                                │
     │     └─► Blocks if window full (back-pressure from scheduler)        │
     │                                                                      │
     │  2. Allocate output buffer from HeapRing                            │
     │     └─► Blocks if no space (back-pressure from scheduler)           │
     │                                                                      │
     │  3. Copy parameters & resolve output addresses                       │
     │                                                                      │
     │  4. TensorMap lookup for input dependencies                          │
     │     ┌───────────────────────────────────────────────────────────┐   │
     │     │ For each input tensor:                                     │   │
     │     │   hash = hash(base_ptr)                                    │   │
     │     │   For each entry in bucket[hash]:                          │   │
     │     │     if entry.valid && overlap(input, entry):               │   │
     │     │       Add entry.producer_task_id to T.fanin_list           │   │
     │     └───────────────────────────────────────────────────────────┘   │
     │                                                                      │
     │  5. Update producer fanout lists (with per-task spinlock)           │
     │     ┌───────────────────────────────────────────────────────────┐   │
     │     │ For each producer P in T.fanin_list:                       │   │
     │     │   spin_lock(P.fanout_lock)                                 │   │
     │     │   P.fanout_list.prepend(T)                                 │   │
     │     │   P.fanout_count++                                         │   │
     │     │   spin_unlock(P.fanout_lock)                               │   │
     │     └───────────────────────────────────────────────────────────┘   │
     │                                                                      │
     │  6. Register output tensors in TensorMap                            │
     │     └─► T becomes producer for these regions                        │
     │                                                                      │
     │  7. Publish current_task_index (release store)                      │
     │     └─► Scheduler threads discover roots (fanin_count==0) via SCAN  │
     │                                                                      │
     │  8. Early-ready fast path (AICPU parallel mode):                    │
     │     if a producer is already completed and this makes T ready,      │
     │     push T into orch_ready_queue (avoid O(N) scans)                 │
     │                                                                      │
     └─────────────────────────────────────────────────────────────────────┘

 PHASE 3: SCHEDULING & EXECUTION
 ════════════════════════════════

    SCHEDULER (AICPU threads)                      AICORE WORKERS
    ─────────────────────────                      ──────────────
              │                                          │
    ┌─────────────────────────────────────────────────────────────────────┐
    │ SCHEDULER LOOP (per thread)                                          │
    │                                                                       │
    │  1. SCAN: Check new tasks from orchestrator                          │
    │     └─► Atomically claim task indices from current_task_index         │
    │     └─► Add root tasks (fanin_count=0) to ready queue                │
    │                                                                       │
    │  2. ORCH_DRAIN: Drain orchestrator's ready queue                     │
    │     └─► Move early-ready tasks to scheduler's ready queue            │
    │                                                                       │
    │  3. COMPLETE: Poll cores for completion                               │
    │     ┌─────────────────────────────────────────────────────────────┐  │
    │     │ For each in-flight task T that completed:                    │  │
    │     │   mark T completed (per-slot state array)                    │  │
    │     │   traverse T.fanout_list: for each consumer C                │  │
    │     │     fanin_refcount[C]++                                      │  │
    │     │     if fanin_refcount[C] == C.fanin_count: enqueue C         │  │
    │     └─────────────────────────────────────────────────────────────┘  │
    │                                                                       │
    │  4. DISPATCH: Send ready tasks to idle cores                         │
    │     ┌─────────────────────────────────────────────────────────────┐  │
    │     │ task = ready_queue.dequeue()                                 │  │
    │     │ core = find_idle_core(task.worker_type)                      │  │
    │     │ payload = build_pto2_payload(task) into per-core buffer      │  │
    │     │ write_reg(COND, BUSY)                                        │  │
    │     │ write_reg(DATA_MAIN_BASE[core], task_id + 1) ────────────────│──►
    │     └─────────────────────────────────────────────────────────────┘  │
    │                                                                       │
    │  5. YIELD: If no progress, yield CPU                                 │
    │                                                                       │
    └─────────────────────────────────────────────────────────────────────┘
              │                                          │
              │                                          │ Poll register
              │                                          │ Read payload
              │                                          │ kernel(args)
              │                                          │ pipe_barrier
              │◄─────────────────────────────────────────│ Write COND reg
              │                                          │
```

---

## 5. Paged Attention Example

### 5.1 What is Paged Attention?

Paged attention addresses **KV cache fragmentation** in transformer inference:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TRADITIONAL vs PAGED ATTENTION                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  TRADITIONAL (Contiguous KV Cache):                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │ Seq 1: [████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  ││
│  │ Seq 2: [██████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  ││
│  │ Seq 3: [████████████████████████████████████████████████░░░░░░░░]  ││
│  │                                                                      ││
│  │ Problem: Variable lengths → external fragmentation (░ = wasted)     ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                                                          │
│  PAGED (Block-based KV Cache):                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │ Block Pool:  [B0][B1][B2][B3][B4][B5][B6][B7][B8][B9]...            ││
│  │                ▲   ▲   ▲   ▲   ▲   ▲   ▲   ▲   ▲   ▲               ││
│  │ Seq 1 → Block Table: [0, 3, 7, -, -, -]                             ││
│  │ Seq 2 → Block Table: [1, 5, -, -, -, -]                             ││
│  │ Seq 3 → Block Table: [2, 4, 6, 8, 9, -]                             ││
│  │                                                                      ││
│  │ Benefit: No fragmentation, blocks reusable across sequences         ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Online Softmax Algorithm

The implementation uses **online softmax** (flash attention style) for memory efficiency:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     ONLINE SOFTMAX ACCUMULATION                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  For each KV block j:                                                    │
│                                                                          │
│    Step 1: QK Matmul                                                     │
│    ┌──────────────────────────────────────────────────────────────────┐ │
│    │  s_ij = q_i @ k_j^T   (q_tile×head_dim @ block_size×head_dim^T)  │ │
│    │    Case1: (16×128) @ (128×128)^T → (16×128)                      │ │
│    │    Case2: (64×128) @ ( 64×128)^T → (64×64)                       │ │
│    └──────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│    Step 2: Softmax Prepare                                               │
│    ┌──────────────────────────────────────────────────────────────────┐ │
│    │  s_ij_masked = mask(s_ij, valid_len)  (pad invalid cols with -∞) │ │
│    │  s_ij_scaled = s_ij_masked * scale    (scale = 1/√d)             │ │
│    │  m_ij = rowmax(s_ij_scaled)           (q_tile×1)                 │ │
│    │  p_ij = exp(s_ij_scaled - m_ij)       (q_tile×block_size)        │ │
│    │  l_ij = rowsum(p_ij)                  (q_tile×1)                 │ │
│    └──────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│    Step 3: PV Matmul                                                     │
│    ┌──────────────────────────────────────────────────────────────────┐ │
│    │  o_ij = p_ij @ v_j   (q_tile×block_size @ block_size×head_dim)   │ │
│    │    Case1: (16×128) @ (128×128) → (16×128)                        │ │
│    │    Case2: (64× 64) @ ( 64×128) → (64×128)                        │ │
│    └──────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│    Step 4: Online Update                                                 │
│    ┌──────────────────────────────────────────────────────────────────┐ │
│    │  if first_block:                                                  │ │
│    │    m_i = m_ij, l_i = l_ij, o_i = o_ij                            │ │
│    │  else:                                                            │ │
│    │    m_i_new = max(m_i, m_ij)                                       │ │
│    │    alpha = exp(m_i - m_i_new)                                     │ │
│    │    beta = exp(m_ij - m_i_new)                                     │ │
│    │    l_i = alpha * l_i + beta * l_ij    (rescaled denominator)     │ │
│    │    o_i = alpha * o_i + beta * o_ij    (rescaled numerator)       │ │
│    │    m_i = m_i_new                                                  │ │
│    │                                                                   │ │
│    │  if last_block:                                                   │ │
│    │    output = o_i / l_i                 (final normalization)      │ │
│    └──────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Task DAG for One Query Tile

In `tests/device_tests/tensormap_and_ringbuffer/paged_attention/kernels/orchestration/paged_attention_orch.cpp`, each `(b_idx, q_idx)` iteration opens one `PTO2_SCOPE(rt) { ... }`.

Inside that scope:
- `AIV_HUB` is submitted once to allocate/register three persistent “state” tensors (`oi`, `li_update`, `mi_update`). In this example the hub kernel is a **no-op** (`kernel_entry` is empty), so it does *not* initialize values; it mainly exists to create buffers + TensorMap entries.
- For each KV block `bn`, four compute tasks are submitted: `QK → SF → PV → UP`.
- `UP` uses `INOUT` on the state tensors, which serializes the `UP` tasks: `UP_0 → UP_1 → … → UP_last`.

```mermaid
graph TD
  HUB[AIV_HUB (alloc/register state tensors; no-op)]

  HUB --> QK0[QK_0] --> SF0[SF_0] --> PV0[PV_0] --> UP0[UP_0 (is_first=1)]
  HUB --> UP0

  HUB --> QK1[QK_1] --> SF1[SF_1] --> PV1[PV_1] --> UP1[UP_1]
  UP0 --> UP1

  HUB --> QKn[QK_last] --> SFn[SF_last] --> PVn[PV_last] --> UPn[UP_last (is_last=1)]
  UP1 --> UPn

  UPn --> OUT[out_view written (q_tile×head_dim, fp32)]
```

What actually enforces ordering:
- Dependencies are discovered by TensorMap overlap on `Tensor` descriptors. Because `UP_bn` has `INOUT` params pointing at the same buffers as `UP_{bn-1}`, the runtime builds a true dependency edge `UP_{bn-1} → UP_bn`.
- `QK/SF/PV` for later `bn` can still get ahead (they don't touch the state tensors), but they will wait at `UP_bn` if the chain isn't ready yet.

### 5.4 Kernel Summary

| Kernel | Core | ID | Key inputs | Key outputs |
|--------|------|----|-----------|-------------|
| `QK` | AIC (CUBE) | 0 | `qi(q_tile, head_dim)` bf16, `kj(block_size, head_dim)` bf16 | `sij(q_tile, block_size)` fp32 |
| `SF` | AIV (VECTOR) | 1 | `sij` fp32 + `valid_len` encoded in `sij->repeats[1]`, `scale` scalar | `pij(q_tile, block_size)` bf16, `mi(q_tile)` fp32, `li(q_tile)` fp32 |
| `PV` | AIC (CUBE) | 2 | `pij(q_tile, block_size)` bf16, `vj(block_size, head_dim)` bf16 | `oi_tmp(q_tile, head_dim)` fp32 |
| `UP` | AIV (VECTOR) | 3 | `mi, li, oi_tmp` + `mi_update/li_update/oi` as `INOUT` + `is_first/is_last` | writes accumulators; on last block writes `out_view(q_tile, head_dim)` fp32 |
| `AIV_HUB` | AIV (VECTOR) | 5 | (none) | alloc/register `oi`, `li_update`, `mi_update` (no-op compute in this test) |

### 5.5 Test Configuration

```python
# Source of truth: tests/device_tests/tensormap_and_ringbuffer/paged_attention/golden.py
# Select via env var PA_CASE (defaults to "Case1").
#
# Case1:
#   batch=64, num_heads=16, kv_head_num=1, head_dim=128
#   block_size=128, context_len=8193, max_model_len=32768
#   max_num_blocks_per_req = 32768 / 128 = 256
#   bn_this_batch = ceil(8193 / 128) = 65
#   q_tile = min(num_heads, 128) = 16, q_loop = ceil(16/16) = 1
#   tasks_total = batch * q_loop * (1 + bn_this_batch * 4)
#               = 64 * 1 * (1 + 65*4) = 16,704  (matches profiling output)
#
# Case2:
#   batch=64, num_heads=64, kv_head_num=1, head_dim=128
#   block_size=64, context_len=8192, max_model_len=32768
#   max_num_blocks_per_req = 32768 / 64 = 512
#   bn_this_batch = 8192 / 64 = 128
#   q_tile = min(64, 128) = 64, q_loop = ceil(64/64) = 1
#   tasks_total = 64 * 1 * (1 + 128*4) = 32,832
```

---

## 6. Performance Analysis

### 6.1 Profiling Report Summary

Based on `outputs/pto2_schedule_report_20260226_104510.md`:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      PERFORMANCE OVERVIEW                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Task Count:             16,704 tasks                                    │
│  Global Span:            22,102.54 us (22.1 ms)                         │
│  Throughput:             ~756 tasks/ms                                   │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │           SCHEDULING WINDOW BREAKDOWN (finish - dispatch)         │   │
│  │                                                                   │   │
│  │   ████████████████████████████████████████████████░░░░░▓▓        │   │
│  │   ←───────────── tail (92.0%) ──────────────→ exec  head         │   │
│  │                                               (5.9%) (2.1%)       │   │
│  │                                                                   │   │
│  │   tail (end→finish):    452,557 us   92.0%  ← BOTTLENECK         │   │
│  │   exec (start→end):      28,849 us    5.9%                        │   │
│  │   head (dispatch→start): 10,288 us    2.1%                        │   │
│  │                                                                   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Per-Kernel Timing

```
┌───────────────────────────────────────────────────────────────────────────┐
│                         KERNEL EXECUTION TIMES                             │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  func_id │ name    │ count │ exec_mean │ sched_mean │ head_mean │ tail_mean│
│  ────────┼─────────┼───────┼───────────┼────────────┼───────────┼──────────│
│     0    │ QK      │ 4160  │  1.865 us │  28.116 us │  0.647 us │ 25.604 us│
│     1    │ SF      │ 4160  │  1.472 us │  32.432 us │  0.558 us │ 30.402 us│
│     2    │ PV      │ 4160  │  1.825 us │  25.653 us │  0.679 us │ 23.150 us│
│     3    │ UP      │ 4160  │  1.768 us │  31.613 us │  0.582 us │ 29.262 us│
│     5    │ AIV_HUB │   64  │  0.326 us │  24.851 us │  0.498 us │ 24.027 us│
│                                                                            │
│  Observations:                                                             │
│  - Execution time (exec_mean) is ~1.5-1.9 us for compute kernels          │
│  - Scheduling overhead (sched_mean) is ~25-32 us (15-20x exec time!)      │
│  - Tail dominates: completion detection latency is the bottleneck         │
│  - AIV_HUB is fast (0.326 us) but still incurs full scheduling overhead  │
│                                                                            │
│  Kernel Execution Time Distribution:                                       │
│                                                                            │
│  QK:      ██████████████████████████████████████  1.865 us                │
│  PV:      █████████████████████████████████████   1.825 us                │
│  UP:      ███████████████████████████████████     1.768 us                │
│  SF:      ██████████████████████████████          1.472 us                │
│  AIV_HUB: ██████                                  0.326 us                │
│                                                                            │
└───────────────────────────────────────────────────────────────────────────┘
```

### 6.3 Scheduler Loop Breakdown

```
┌───────────────────────────────────────────────────────────────────────────┐
│                    SCHEDULER LOOP PHASE ANALYSIS                           │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Scheduler Threads: 3                                                      │
│  Total Completed:   16,704 tasks                                          │
│  Avg Cost per Task: 5.700 us                                              │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │                    PHASE TIME DISTRIBUTION                            │ │
│  │                                                                       │ │
│  │  dispatch  ████████████████████████████████████████████████  52.3%   │ │
│  │  complete  ████████████████████████████████████████          38.9%   │ │
│  │  scan      █████                                               6.0%   │ │
│  │  yield     ███                                                 2.8%   │ │
│  │  orch_drain                                                    0.0%   │ │
│  │                                                                       │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
│  Phase Details:                                                            │
│  ┌─────────────┬───────────────┬────────────────────────────────────────┐ │
│  │ Phase       │ Time (us)     │ Description                            │ │
│  ├─────────────┼───────────────┼────────────────────────────────────────┤ │
│  │ dispatch    │ 49,813.36     │ Dequeue ready, build payload, write reg│ │
│  │ complete    │ 37,014.08     │ Poll cores, traverse fanout, update    │ │
│  │ scan        │  5,707.36     │ Scan new tasks, add roots to ready Q   │ │
│  │ yield       │  2,632.40     │ No progress, yield CPU                 │ │
│  │ orch_drain  │     44.02     │ Drain orchestrator ready queue         │ │
│  └─────────────┴───────────────┴────────────────────────────────────────┘ │
│                                                                            │
│  Per-Thread Statistics:                                                    │
│  ┌────────┬───────┬───────────┬──────────┬───────────┬───────────┐       │
│  │ Thread │ Cores │ Completed │ Total(us)│ Dispatch  │ Complete  │       │
│  ├────────┼───────┼───────────┼──────────┼───────────┼───────────┤       │
│  │   0    │  24   │   6,192   │ 31,738   │ 16,001 us │ 12,426 us │       │
│  │   1    │  24   │   5,968   │ 31,735   │ 16,513 us │ 12,203 us │       │
│  │   2    │  24   │   4,544   │ 31,737   │ 17,299 us │ 12,384 us │       │
│  └────────┴───────┴───────────┴──────────┴───────────┴───────────┘       │
│                                                                            │
└───────────────────────────────────────────────────────────────────────────┘
```

### 6.4 Scheduling Outliers

```
┌───────────────────────────────────────────────────────────────────────────┐
│                    TOP 10 SCHEDULING OUTLIERS                              │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  rank │ task_id │ func   │ core │ exec(us) │ head(us) │ tail(us) │ total │
│  ─────┼─────────┼────────┼──────┼──────────┼──────────┼──────────┼────────│
│    1  │  4862   │ UP(3)  │  70  │   1.620  │   0.760  │  68.860  │ 71.24 │
│    2  │ 13728   │ UP(3)  │  67  │   1.940  │   0.840  │  63.760  │ 66.54 │
│    3  │  5549   │ UP(3)  │  69  │   1.740  │   0.620  │  63.600  │ 65.96 │
│    4  │ 13499   │ UP(3)  │  66  │   1.700  │   0.600  │  62.720  │ 65.02 │
│    5  │  5091   │ UP(3)  │  68  │   1.780  │   0.480  │  61.740  │ 64.00 │
│    6  │  8713   │ UP(3)  │  69  │   2.140  │   0.660  │  60.380  │ 63.18 │
│    7  │ 14889   │ UP(3)  │  65  │   1.820  │   0.520  │  60.680  │ 63.02 │
│    8  │  4388   │ UP(3)  │  67  │   1.980  │   0.480  │  59.720  │ 62.18 │
│    9  │  8962   │ UP(3)  │  68  │   1.660  │   0.320  │  59.960  │ 61.94 │
│   10  │  4621   │ UP(3)  │  66  │   1.620  │   0.400  │  59.440  │ 61.46 │
│                                                                            │
│  Observations:                                                             │
│  - All outliers are UP (ONLINE_UPDATE) tasks                              │
│  - Execution time is normal (~1.6-2.1 us)                                 │
│  - Tail time dominates (59-69 us vs ~27 us average)                       │
│  - Suggests completion detection delays on specific cores                  │
│                                                                            │
└───────────────────────────────────────────────────────────────────────────┘
```

### 6.5 Optimization Recommendations

```
┌───────────────────────────────────────────────────────────────────────────┐
│                    OPTIMIZATION PRIORITY LIST                              │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Current bottleneck: TAIL (92.0% of scheduling window)                    │
│                                                                            │
│  Priority 1: Reduce Tail Latency (end → finish)                           │
│  ────────────────────────────────────────────────────────────────────────│
│  - Problem: AICPU polls COND register to detect AICore completion         │
│  - Solutions:                                                              │
│    a) Batch register reads (read multiple cores at once)                  │
│    b) Maintain busy_core_list (only poll cores with in-flight tasks)     │
│    c) Switch to shared memory completion queue                            │
│    d) Use completion bitmap (AICore writes, AICPU scans)                  │
│                                                                            │
│  Priority 2: Reduce Dispatch Overhead (52.3% of scheduler time)           │
│  ────────────────────────────────────────────────────────────────────────│
│  - Problem: Ready queue lock contention, payload building                 │
│  - Solutions:                                                              │
│    a) Per-thread ready queues (avoid lock contention)                     │
│    b) Pre-generate payload in orchestrator                                │
│    c) Cache common parameter layouts                                       │
│                                                                            │
│  Priority 3: Reduce Complete Phase (38.9% of scheduler time)              │
│  ────────────────────────────────────────────────────────────────────────│
│  - Problem: Fanout traversal, atomic updates                              │
│  - Solutions:                                                              │
│    a) Compress dependency lists                                            │
│    b) Batch atomic updates                                                 │
│    c) Improve cache locality of fanout lists                              │
│                                                                            │
└───────────────────────────────────────────────────────────────────────────┘
```

Concrete optimization checklist (code-linked):
- Eliminate no-op hub tasks (`AIV_HUB` / `AIC_HUB`): add a runtime API for “allocate/register tensor without dispatching a kernel”, then remove the hub submissions in `tests/device_tests/tensormap_and_ringbuffer/paged_attention/kernels/orchestration/paged_attention_orch.cpp`. Today these tasks still pay full schedule cost with near-zero exec time.
- Reduce completion tail latency: replace per-core `RegId::COND` polling with a shared-memory completion queue or bitmap written by AICore (hot path is in `src/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp`, COMPLETE phase).
- Reduce ready-queue contention: current ready queues are protected by shared `SpinLock`s across scheduler threads; consider per-thread queues + work stealing (hot path: `src/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp`, DISPATCH phase).
- Reduce payload build cost: `build_pto2_payload(...)` runs on every dispatch; consider pre-packing immutable parts at submit-time or caching per-kernel arg layouts (hot path: `src/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp`).
- Reduce fanout traversal cost: fanout lists are DepListPool linked lists; consider more cache-friendly layouts (e.g. contiguous fanout arrays per producer) or batching refcount updates (hot path: COMPLETE phase).
- Ensure long-run reclamation works on device: the device scheduler path does not currently advance `last_task_alive` / `heap_tail` via `pto_scheduler.cpp`, so TaskRing/HeapRing reuse is limited to “big enough pools”. If you push to very large graphs or smaller windows/heaps, integrating consumption + tail advancement becomes necessary (`src/runtime/tensormap_and_ringbuffer/runtime/pto_scheduler.cpp` vs device loop in `src/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp`).

---

## 7. Profiling Output Reference

### 7.0 Reproducing the Profile (paged-attention, real `a2a3`)

Run from repo root (adjust `-d` for your device id):

```bash
PA_CASE=Case1 \
python examples/scripts/run_example.py \
  -k tests/device_tests/tensormap_and_ringbuffer/paged_attention/kernels \
  -g tests/device_tests/tensormap_and_ringbuffer/paged_attention/golden.py \
  -p a2a3 -d 15 \
  --enable-profiling \
  --log-file outputs/pto2_profile_a2a3.log
```

Expected artifacts (written under `outputs/`):
- `perf_swimlane_*.json`, `merged_swimlane_*.json`
- `pto2_schedule_report_*.md`

### 7.1 Output Files

| File Pattern | Description |
|--------------|-------------|
| `perf_swimlane_*.json` | Chrome trace format (AICore execution) |
| `merged_swimlane_*.json` | Combined AICore + AICPU trace |
| `pto2_profile_*.log` | Host-side profiling log |
| `pto2_schedule_report_*.md` | Human-readable analysis report |

### 7.1.1 Scheduler Phase Export (works on real `a2a3`)

On real devices, AICPU logs may not be easily captured/parsed (e.g. dlog routing). The schedule phase breakdown used by `tools/pto2_schedule_report.py` is exported through perf shared memory:
- AICPU writes per-thread schedule phase cycle counters into `PerfDataHeader` near the end of scheduling (`src/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp`, `src/platform/include/common/perf_profiling.h`).
- The host perf exporter embeds these fields into `outputs/perf_swimlane_*.json` / `outputs/merged_swimlane_*.json` (`src/platform/src/host/performance_collector.cpp`).

### 7.2 Chrome Trace Format

The `perf_swimlane_*.json` and `merged_swimlane_*.json` files use Chrome's trace event format and can be viewed at `chrome://tracing`:

```json
{
  "traceEvents": [
    // Process/thread metadata
    {"cat": "__metadata", "name": "process_name", "ph": "M", "pid": 1,
     "args": {"name": "AICore View"}},
    {"cat": "__metadata", "name": "thread_name", "ph": "M", "pid": 1, "tid": 1000,
     "args": {"name": "AIC_0"}},

    // Task execution events
    {"name": "QK", "ph": "X", "pid": 1, "tid": 1000,
     "ts": 1000.5, "dur": 1.865,
     "args": {"task_id": 0, "func_id": 0, "core_id": 0}}
  ]
}
```

### 7.3 Performance Log Example

From `outputs/pto2_profile_a2a3_3.log`:

```
[INFO] ensure_device_set: DeviceRunner: device=15 set, streams created
[INFO] ensure_binaries_loaded: DeviceRunner: binaries loaded
[INFO] init_aicore_register_addresses: Retrieving and allocating AICore register addresses...
[INFO] get_aicore_reg_info: Register base: ptr=0xdfff8f7ff000, len=0xc800000
[INFO] get_aicore_regs: Retrieved 25 AIC and 50 AIV register addresses
[INFO] init_aicore_register_addresses: Successfully initialized register addresses: 75 addresses
[INFO] initialize: Initializing performance profiling
[INFO] poll_and_collect: Collecting performance data
[INFO] poll_and_collect: Waiting for AICPU to write total_tasks in PerfDataHeader...
[INFO] poll_and_collect: AICPU reported task count: 1
[INFO] poll_and_collect: Updated expected_tasks to 2 (orchestrator progress)
[INFO] poll_and_collect: Final expected_tasks: 16704 (orchestration complete)
[INFO] poll_and_collect: Total buffers processed: 72
[INFO] poll_and_collect: Total records collected: 16704
[INFO] export_swimlane_json: File: outputs/perf_swimlane_20260226_104508.json
[INFO] export_swimlane_json: Records: 16704
```

### 7.4 Viewing Traces

1. Open `chrome://tracing` in Chrome/Chromium
2. Click "Load" and select `merged_swimlane_*.json`
3. Use keyboard shortcuts:
   - `W`/`S`: Zoom in/out
   - `A`/`D`: Pan left/right
   - Click task to see details

---

## Appendix A: File Locations

```
src/runtime/tensormap_and_ringbuffer/
├── runtime/
│   ├── pto_runtime2.h/.cpp              # Main interface
│   ├── pto_runtime2_types.h             # Type definitions
│   ├── pto_orchestrator.h/.cpp          # Task submission
│   ├── pto_scheduler.h/.cpp             # Task dispatch
│   ├── pto_tensormap.h/.cpp             # Dependency tracking
│   ├── pto_ring_buffer.h/.cpp           # Memory management
│   ├── pto_shared_memory.h/.cpp         # Shared memory layout
│   └── pto2_dispatch_payload.h          # Dispatch packet
├── aicpu/
│   └── aicpu_executor.cpp               # AICPU main thread
├── aicore/
│   └── aicore_executor.cpp              # AICore worker loop
└── host/
    └── runtime_maker.cpp                # Host-side helpers

tests/device_tests/tensormap_and_ringbuffer/paged_attention/
├── golden.py                            # Reference implementation
├── kernels/
│   ├── kernel_config.py                 # Configuration
│   ├── orchestration/
│   │   └── paged_attention_orch.cpp     # Task DAG construction
│   ├── aic/
│   │   ├── aic_qk_matmul.cpp            # QK kernel
│   │   ├── aic_pv_matmul.cpp            # PV kernel
│   │   └── aic_hub.cpp                  # No-op hub kernel (allocation barrier)
│   └── aiv/
│       ├── aiv_softmax_prepare.cpp      # Softmax kernel
│       ├── aiv_online_update.cpp        # Online update + normalize
│       └── aiv_hub.cpp                  # No-op hub kernel (allocation barrier)

outputs/
├── perf_swimlane_*.json                 # AICore trace (Chrome format)
├── merged_swimlane_*.json               # Combined trace
├── pto2_profile_*.log                   # Host profiling log
└── pto2_schedule_report_*.md            # Analysis report
```

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **AIC** | AI Core - Matrix compute unit (CUBE) |
| **AIV** | AI Vector - Vector compute unit |
| **AICPU** | AI CPU - Task scheduler processor |
| **Fanin** | Dependencies (tasks this task waits for) |
| **Fanout** | Dependents (tasks waiting for this task) |
| **TensorMap** | Hash table tracking tensor producers |
| **HeapRing** | Ring buffer for output allocation |
| **TaskRing** | Ring buffer for task descriptors |
| **Scope** | Group of tasks with shared lifecycle |
| **Online Softmax** | Memory-efficient softmax with accumulation |
| **Paged Attention** | Block-based KV cache for variable sequences |

---

## Further Reading

- Deep code tour (architecture → functions): `docs/tensormap-ringbuffer-runtime-codewalk.md`
- Platform layer (a2a3 vs a2a3sim glue): `docs/platform-codewalk.md`
- Platform deep-dive (DeviceRunner + regs): `docs/annotated-platform-device-runner.md`
- Platform deep-dive (profiling subsystem): `docs/annotated-platform-profiling.md`
- Paged-attention example (task-by-task + kernel-by-kernel): `docs/paged-attention-example-codewalk.md`
- Line-by-line orchestrator walkthrough: `docs/annotated-pto-orchestrator.md`
- Line-numbered submit hot-path (`pto2_submit_task`): `docs/linebyline-pto2-submit-task.md`
- Line-by-line device scheduler walkthrough: `docs/annotated-aicpu-executor-scheduler.md`
- Line-numbered scheduler loop (`resolve_and_dispatch_pto2`): `docs/linebyline-aicpu-resolve-and-dispatch-pto2.md`
