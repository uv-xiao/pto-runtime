# Paged Attention Example - Concurrent Build/Schedule Analysis Report

This report documents the execution analysis of the `aicpu_build_graph/paged_attention` example
on Ascend NPU hardware (a2a3 platform), demonstrating concurrent build and schedule capabilities.

## Test Summary

| Runtime | Test Case | Blocks | Total Tasks | Platform | Result |
|---------|-----------|--------|-------------|----------|--------|
| aicpu_build_graph | Case1 | 1 | 4 | a2a3 (device 0) | **PASSED** |
| aicpu_build_graph | Case2 | 4 | 16 | a2a3 (device 0) | **PASSED** |
| host_build_graph | Case1 | 1 | 4 | a2a3 (device 0) | **PASSED** |
| host_build_graph | Case2 | 4 | 16 | a2a3 (device 0) | **PASSED** |

## Task Count Explanation

The task count formula is: **`batch × block_num × 4`**

Each block generates 4 tasks:
- **QK**: Q @ K^T matrix multiplication (AIC)
- **SF**: Softmax preparation (AIV)
- **PV**: P @ V matrix multiplication (AIC)
- **UP**: Online update (AIV)

### Why Not More Tasks?

The algorithm has three nested loops in the golden implementation:
```python
for b_idx in range(batch):           # batch dimension
    for h_idx in range(num_heads):   # head dimension (16 heads)
        for bn in range(block_num):  # block dimension
            # QK, SF, PV, UP operations
```

However, the **orchestration fuses the head dimension into each task** using 16×16 tile operations:
- Each task processes **all 16 heads simultaneously** as a single 16×16 matrix operation
- This is more efficient than creating 16× more tasks (one per head)
- The tile size (16×16) matches the head configuration (num_heads=16, head_dim=16)

### Task Count by Case

| Case | batch | num_heads | block_num | Tasks per Block | Total Tasks |
|------|-------|-----------|-----------|-----------------|-------------|
| Case1 | 1 | 16 (fused) | 1 | 4 | 1 × 1 × 4 = **4** |
| Case2 | 1 | 16 (fused) | 4 | 4 | 1 × 4 × 4 = **16** |

If heads were not fused, Case1 would have 1 × 16 × 1 × 4 = 64 tasks, and Case2 would have 256 tasks.

## Algorithm Overview

Paged Attention is an efficient attention mechanism that processes KV cache in fixed-size blocks,
enabling memory-efficient inference for long sequences. The implementation uses:

- **AIC kernels** (Cube unit): QK MatMul, PV MatMul
- **AIV kernels** (Vector unit): Softmax Prepare, Online Update
- **Online Softmax** algorithm for numerically stable incremental computation
- **16×16 tile operations** that process all heads in parallel

## Task Graph Structure

### Per-Block Pipeline

Each KV cache block generates a 4-task pipeline:

```
QK(bn) -> SF(bn) -> PV(bn) -> UP(bn)
```

### Multi-Block Dependencies (Case2: 4 blocks, 16 tasks)

```
Block 0: QK(0)  -> SF(1)  -> PV(2)  -> UP(3)  ----+
Block 1: QK(4)  -> SF(5)  -> PV(6)  -> UP(7)  ----+-- UP chain
Block 2: QK(8)  -> SF(9)  -> PV(10) -> UP(11) ----+   (serialized)
Block 3: QK(12) -> SF(13) -> PV(14) -> UP(15) ----+

Cross-block edges (Online Update serialization):
UP(3) -> UP(7) -> UP(11) -> UP(15)
```

**Key Design Point**: The Online Update tasks must be serialized because each iteration
accumulates into shared state (mi, li, oi). This creates cross-iteration dependencies
where UP(n) depends on UP(n-1).

## Runtime Comparison: host_build_graph vs aicpu_build_graph

### host_build_graph Runtime

The graph is built entirely on the **host** before execution starts.

**Execution Flow**:
1. Host builds complete task graph
2. Host copies graph to device
3. AICPU scheduler executes pre-built graph
4. No overlap between build and schedule

### aicpu_build_graph Runtime (Concurrent Build/Schedule)

The graph is built on **AICPU** concurrently with task execution.

**Execution Flow** (with `PTO_AICPU_BUILD_GRAPH_BUILD_MODE=1`):
1. Thread 0 (Builder): Runs orchestration plugin to create tasks
2. Threads 1-3 (Schedulers): Poll for ready tasks immediately
3. Tasks are dispatched as soon as they become ready
4. Build and schedule overlap in time

## Immediate Publish Pattern

The orchestration uses the **Immediate Publish Pattern** where each task is published
immediately after creation and edge setup:

```cpp
// Task 0: QK MatMul (no predecessors)
int t_qk = api.add_task(runtime, qk_args, 3, FUNC_QK_MATMUL, CoreType::AIC, 0);
api.publish_task(runtime, t_qk);  // Publish immediately - no predecessors

// Task 1: Softmax Prepare (depends on QK)
int t_sf = api.add_task(runtime, sf_args, 5, FUNC_SOFTMAX_PREPARE, CoreType::AIV, 0);
api.add_successor_conditional(runtime, t_qk, t_sf);  // Edge: QK -> SF
api.publish_task(runtime, t_sf);

// Task 2: PV MatMul (depends on SF)
int t_pv = api.add_task(runtime, pv_args, 3, FUNC_PV_MATMUL, CoreType::AIC, 0);
api.add_successor_conditional(runtime, t_sf, t_pv);  // Edge: SF -> PV
api.publish_task(runtime, t_pv);

// Task 3: Online Update (depends on PV and previous UP)
int t_up = api.add_task(runtime, up_args, 9, FUNC_ONLINE_UPDATE, CoreType::AIV, 0);
api.add_successor_conditional(runtime, t_pv, t_up);  // Edge: PV -> UP
if (t_up_prev >= 0) {
    api.add_successor_conditional(runtime, t_up_prev, t_up);  // Edge: UP(prev) -> UP
}
api.publish_task(runtime, t_up);
```

### Benefits of Immediate Publish

1. **Earlier task visibility**: First task (QK) becomes visible to schedulers immediately
2. **Better concurrency**: Schedulers can dispatch tasks while builder creates subsequent tasks
3. **Correct edge handling**: `add_successor_conditional` handles the case where predecessor
   may have already completed before the edge is added

## Concurrent Build/Schedule Device Log Analysis (Case2: 16 tasks)

The following analysis is based on actual device debug logs from a2a3 hardware execution,
captured via CANN's dlog API with `ASCEND_GLOBAL_LOG_LEVEL=0`.

### Thread Initialization

```
Thread 1: assigned 24 cores - AIC[0,1,2,3,4,5,6,7] AIV[24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]
Thread 2: assigned 24 cores - AIC[8,9,10,11,12,13,14,15] AIV[40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55]
Thread 3: assigned 24 cores - AIC[16,17,18,19,20,21,22,23] AIV[56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71]
```

### Concurrent Mode Activation

```
Thread 0: Builder starting (build_mode=1)
Thread 1: Concurrent mode: not waiting for builder barrier
Thread 2: Concurrent mode: not waiting for builder barrier
Thread 3: Concurrent mode: not waiting for builder barrier
```

### Multi-Block Interleaved Execution (Case2: 4 blocks, 16 tasks)

The following log excerpt shows concurrent build and schedule across multiple blocks:

```
# Schedulers start polling before any tasks exist
Thread 2: Observed published=0 (completed=0, build_done=0)
Thread 1: Observed published=0 (completed=0, build_done=0)
Thread 3: Observed published=0 (completed=0, build_done=0)

# ============ Block 0 (Tasks 0-3) ============
# Builder creates and publishes Task 0 (QK) immediately
Thread 0(builder): add_task -> task_id=0 fanin=0
Thread 0(builder): publish_task(0) done (total_published=1)

# Thread 1 dispatches Task 0 while builder creates Task 1
Thread 1: Dispatching AIC task 0 to core 3

# Builder creates Task 1 (SF), adds edge, publishes
Thread 0(builder): add_task -> task_id=1
Thread 0(builder): add_edge_conditional(0 -> 1)
Thread 0(builder): publish_task(1) done (total_published=2)

# Builder creates Tasks 2-3 while Task 0 executes
Thread 0(builder): add_task -> task_id=2
Thread 0(builder): add_edge_conditional(1 -> 2)
Thread 0(builder): publish_task(2) done (total_published=3)

# Task 0 completes, Task 1 becomes ready
Thread 1: Core 3 completed task 0
Thread 1: Task 1 becomes ready (published=1)

# Thread 3 dispatches Task 1 (SF)
Thread 3: Dispatching AIV task 1 to core 70

Thread 0(builder): add_task -> task_id=3
Thread 0(builder): add_edge_conditional(2 -> 3)
Thread 0(builder): publish_task(3) done (total_published=4)

# Task 1 completes, Task 2 becomes ready
Thread 3: Core 70 completed task 1
Thread 3: Task 2 becomes ready (published=1)

# ============ Block 1 (Tasks 4-7) starts while Block 0 executes ============
# Builder starts Block 1 while Block 0's Task 2 is dispatched
Thread 0(builder): add_task -> task_id=4 fanin=0
Thread 3: Dispatching AIC task 2 to core 16

Thread 0(builder): publish_task(4) done (total_published=5)

# Thread 3 dispatches Block 1's Task 4 (QK) in parallel with Block 0
Thread 3: Dispatching AIC task 4 to core 19

# Builder continues Block 1 while both blocks execute
Thread 0(builder): add_task -> task_id=5
Thread 0(builder): add_edge_conditional(4 -> 5)
Thread 0(builder): publish_task(5) done (total_published=6)

# ... Block 1 tasks 6-7 created ...

# Cross-block edge: UP(3) -> UP(7) for Online Update serialization
Thread 0(builder): add_edge_conditional(3 -> 7)
Thread 0(builder): publish_task(7) done (fanin=2 total_published=8)

# ============ Block 2 (Tasks 8-11) starts ============
Thread 0(builder): add_task -> task_id=8 fanin=0
Thread 0(builder): publish_task(8) done (total_published=9)

# Thread 1 dispatches Block 2's Task 8 while Block 0-1 tasks complete
Thread 1: Dispatching AIC task 8 to core 0

# ... Block 2 tasks 9-11 created ...

# Cross-block edge: UP(7) -> UP(11)
Thread 0(builder): add_edge_conditional(7 -> 11)
Thread 0(builder): publish_task(11) done (fanin=2 total_published=12)

# ============ Block 3 (Tasks 12-15) starts ============
Thread 0(builder): add_task -> task_id=12 fanin=0
Thread 0(builder): publish_task(12) done (total_published=13)

# Thread 3 dispatches Block 3's Task 12
Thread 3: Dispatching AIC task 12 to core 16

# ... Block 3 tasks 13-15 created ...

# Cross-block edge: UP(11) -> UP(15)
Thread 0(builder): add_edge_conditional(11 -> 15)
Thread 0(builder): publish_task(15) done (fanin=2 total_published=16)

# ============ Completion ============
# All 16 tasks complete before builder finishes
Thread 3: Core 66 completed task 15
Thread 0: Builder done (rc=0)

# Final task distribution
Thread 1: Execution complete, completed 5 tasks
Thread 2: Execution complete, completed 6 tasks
Thread 3: Execution complete, completed 5 tasks
```

### Key Observations: Cross-Block Parallelism

1. **Block 0 and Block 1 execute in parallel**: While Block 0's Task 2 (PV) runs on core 16,
   Block 1's Task 4 (QK) is dispatched to core 19 simultaneously.

2. **Online Update serialization**: The UP tasks (3, 7, 11, 15) are serialized via cross-block
   edges, but QK/SF/PV tasks from different blocks can run in parallel.

3. **Multi-thread task distribution**:
   - Thread 1: Tasks 0, 3, 8, 9, 10 (5 tasks)
   - Thread 2: Tasks 5, 6, 7, 11, 13, 14 (6 tasks)
   - Thread 3: Tasks 1, 2, 4, 12, 15 (5 tasks)

4. **Effective overlap**: 7 tasks completed while builder was still creating tasks
   (observed at `published=16, completed=7, build_done=0`)

### Timing Analysis (Case2: 16 tasks)

From the device log timestamps (microsecond precision):

| Event | Timestamp | Delta from Start |
|-------|-----------|------------------|
| Task 0 dispatched | 10:23:37.684.862 | 0 μs |
| Task 0 completed | 10:23:37.685.068 | 206 μs |
| Task 4 dispatched (Block 1) | 10:23:37.685.346 | 484 μs |
| Task 8 dispatched (Block 2) | 10:23:37.685.745 | 883 μs |
| Task 12 dispatched (Block 3) | 10:23:37.686.194 | 1,332 μs |
| Task 15 completed (final) | 10:23:37.686.826 | 1,964 μs |
| Builder done | 10:23:37.686.826 | 1,964 μs |

**Total execution time**: ~2 ms for 16 tasks across 4 blocks.

### Swimlane Visualization Data

A `swimlane.json` file has been generated for visualization. Key metrics:

| Metric | Value |
|--------|-------|
| Total tasks | 16 |
| Total edges | 15 (4 intra-block × 4 + 3 cross-block UP edges) |
| Cores used | 10 (5 AIC + 5 AIV) |
| Builder done time | 1,964 μs |

Task duration distribution:
- QK (AIC): 64-478 μs
- SF (AIV): 44-115 μs
- PV (AIC): 64-190 μs
- UP (AIV): 44-349 μs

## Cross-Block Dependency Analysis

The Online Update (UP) tasks form a serialized chain across blocks:

```
UP(3) ──────────────────────────────────────────────────────────────────┐
  │                                                                      │
  └──> UP(7) ──────────────────────────────────────────────────────────┐│
         │                                                              ││
         └──> UP(11) ─────────────────────────────────────────────────┐││
                │                                                      │││
                └──> UP(15)                                            │││
                                                                       │││
Timeline:                                                              │││
  Block 0: QK(0) -> SF(1) -> PV(2) -> UP(3) ────────────────────────────┘││
  Block 1: QK(4) -> SF(5) -> PV(6) -> UP(7) ─────────────────────────────┘│
  Block 2: QK(8) -> SF(9) -> PV(10) -> UP(11) ────────────────────────────┘
  Block 3: QK(12) -> SF(13) -> PV(14) -> UP(15)
```

**Why UP tasks are serialized**: The Online Softmax algorithm maintains running state
(mi, li, oi) that must be updated sequentially. Each UP task reads the previous state
and writes the updated state, requiring strict ordering.

**Parallelism opportunity**: While UP tasks are serialized, QK/SF/PV tasks from different
blocks can execute in parallel. For example:
- Block 1's QK(4) can start as soon as it's published, without waiting for Block 0's UP(3)
- Block 2's QK(8) can overlap with Block 1's SF(5) and PV(6)

## Verification Checklist

| Pattern | Evidence from Device Log |
|---------|--------------------------|
| Scheduler not waiting | `Thread N: Concurrent mode: not waiting for builder barrier` |
| Tasks observed while building | `Observed published=N (completed=M, build_done=0)` |
| Immediate publish | `publish_task(0) done` immediately after `add_task -> task_id=0` |
| Cross-block parallelism | Task 4 dispatched while Task 2 executing |
| UP serialization | `add_edge_conditional(3 -> 7)`, `add_edge_conditional(7 -> 11)` |
| Tasks completed before builder | 7 tasks completed at `build_done=0` |

## Benefits of Concurrent Build/Schedule

1. **Reduced latency**: Tasks start executing before graph is fully built
2. **Better resource utilization**: Builder and scheduler threads work in parallel
3. **Scalability**: For large graphs, execution can begin while later tasks are still being created
4. **Cross-block parallelism**: Independent tasks from different blocks execute concurrently

## Conclusion

The paged_attention example successfully demonstrates:

1. **Concurrent build and schedule** on Ascend NPU hardware (a2a3)
2. **Immediate Publish Pattern** for maximum concurrency
3. **Cross-block parallelism** with QK/SF/PV tasks from different blocks executing in parallel
4. **Correct Online Update serialization** via cross-block edges
5. **Efficient head fusion**: 16 heads processed per task using 16×16 tiles
6. **Multi-thread task distribution** across 3 scheduler threads

All 4 test cases pass on a2a3 hardware. The device logs clearly show that scheduler threads
do not wait for the builder to complete, and tasks from multiple blocks are dispatched and
executed concurrently while the graph is still being constructed.
