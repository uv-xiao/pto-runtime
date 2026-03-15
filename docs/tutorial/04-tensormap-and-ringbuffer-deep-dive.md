# Tensormap and Ringbuffer Deep Dive

This chapter will be the core technical tutorial. It will explain the PTO2 runtime in detail: shared memory, ring buffers, TensorMap dependency inference, orchestrator/scheduler split, ready queues, worker handshakes, completion handling, reclamation, and deadlock/back-pressure behavior.

## Files Covered

- `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_runtime2.h`
- `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_runtime2_types.h`
- `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_shared_memory.h`
- `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_ring_buffer.h`
- `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_scheduler.h`
- `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_scheduler.cpp`
- `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.h`
- `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp`
- `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_tensormap.h`
- `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_tensormap.cpp`
- `src/a2a3/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp`
- `src/a2a3/runtime/tensormap_and_ringbuffer/aicore/aicore_executor.cpp`
- `src/a2a3/runtime/tensormap_and_ringbuffer/host/runtime_maker.cpp`
- `examples/a2a3/tensormap_and_ringbuffer/paged_attention/`

## Reading Strategy

Read the runtime in three layers: data structures first, orchestration and scheduling second, and real example mapping third. Keep the shared-memory and state-transition diagrams open while reading the hot-path functions.

## Planned Diagrams

- shared-memory layout
- task state machine
- orchestrator/scheduler/AICore swimlane
- ring-buffer back-pressure and deadlock diagram

## Planned Code Walkthroughs

- task-slot and heap allocation
- TensorMap insert/lookup/cleanup
- fanin/fanout synchronization
- ready-queue push/pop and local-ready buffers
- mixed-task completion
- AICPU scheduler loop
- AICore execution loop

## Planned Verification Notes

- map `paged_attention` func IDs, worker types, and orchestration flow
- use stress tests with small ring sizes to explain back-pressure
- call out which behavior is visible in simulation versus hardware only
