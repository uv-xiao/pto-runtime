# Runtime Definition and Lifecycle

This chapter defines what a “runtime” means in PTO Runtime, how a runtime is selected and compiled, and how its lifecycle flows through the host C API. The critical idea is that a runtime is not just one class. In this repository, a runtime is a coordinated set of host, AICPU, AICore, and sometimes orchestration-side sources that together define graph construction, scheduling, and execution behavior.

## Files Covered

- `src/a2a3/runtime/host_build_graph/build_config.py`
- `src/a2a3/runtime/aicpu_build_graph/build_config.py`
- `src/a2a3/runtime/tensormap_and_ringbuffer/build_config.py`
- `src/a5/runtime/host_build_graph/build_config.py`
- `src/a5/runtime/tensormap_and_ringbuffer/build_config.py`
- `src/a2a3/runtime/*/host/runtime_maker.cpp`
- `src/a2a3/runtime/*/runtime/runtime.h`
- `src/a2a3/platform/onboard/host/pto_runtime_c_api.cpp`
- `src/a2a3/platform/include/host/pto_runtime_c_api.h`

## Reading Strategy

Read runtimes in four layers:

1. runtime selection from `kernel_config.py`
2. runtime composition from `build_config.py`
3. host-side integration in `runtime_maker.cpp`
4. lifecycle through `init_runtime`, `launch_runtime`, and `finalize_runtime`

Do not start with the runtime classes themselves. First understand how the build system decides what a runtime contains.

## 1. What a Runtime Means Here

In this repository, a runtime is a named implementation bundle that defines:

- how graph construction happens
- where orchestration runs
- how tasks are represented
- how scheduling sees those tasks
- what host-side initialization and cleanup are required

A runtime is therefore a collection of code grouped under:

```text
src/<arch>/runtime/<runtime_name>/
  build_config.py
  host/
  aicpu/
  aicore/
  runtime/
  orchestration/   (runtime-specific, only for some runtimes)
  docs/
```

The runtime name seen by users is the same name found in:

- `kernel_config.py -> RUNTIME_CONFIG["runtime"]`
- directory name under `src/<arch>/runtime/`

## 2. Runtime Selection From the User Side

The user or example author chooses a runtime in `kernel_config.py`.

Examples:

- `host_build_graph`
- `aicpu_build_graph`
- `tensormap_and_ringbuffer`

Python reads that string, then `RuntimeBuilder` verifies that the selected architecture root actually contains that runtime directory.

This gives a clean rule:

```text
runtime selected by example config
runtime availability determined by src/<arch>/runtime/<name>/build_config.py
```

## 3. `build_config.py`: the Actual Definition of a Runtime

Each runtime’s `build_config.py` tells the framework which source directories belong to which execution target.

### 3.1 `host_build_graph`

`src/a2a3/runtime/host_build_graph/build_config.py` says:

- AICore target gets `aicore` + `runtime`
- AICPU target gets `aicpu` + `runtime`
- Host target gets `host` + `runtime`

This is the simplest model:

- one runtime wrapper type
- one host initializer
- one AICPU executor
- one AICore executor

### 3.2 `aicpu_build_graph`

`src/a2a3/runtime/aicpu_build_graph/build_config.py` looks superficially similar, but the comments explain the important difference:

- graph build logic runs on AICPU
- orchestration is loaded on device as a plugin
- host-side init must marshal argument state and embed orchestration payloads for AICPU use

So two runtimes can have similar `build_config.py` shapes while still having meaningfully different runtime behavior.

### 3.3 `tensormap_and_ringbuffer`

`src/a2a3/runtime/tensormap_and_ringbuffer/build_config.py` adds an extra dimension:

- `orchestration` is its own section
- `orchestration` sources are compiled both into runtime targets and into the orchestration plugin

This is a strong signal that PTO2 is architecturally different from the older runtimes. It does not only swap executor logic. It changes the orchestration model itself.

## 4. Runtime Variant Comparison

The three runtime families should be understood as different answers to the same question: “Where is the graph built, and what data structure represents execution?”

| Runtime | Graph build location | Main task representation | Orchestration model | Scheduler model |
|---|---|---|---|---|
| `host_build_graph` | host CPU | fixed `Task[]` array | host loads orchestration `.so` and builds graph immediately | AICPU schedules a pre-built graph |
| `aicpu_build_graph` | AICPU | fixed `Task[]` array plus publish/build metadata | host embeds orchestration plugin for device-side graph build | AICPU can build and schedule on device |
| `tensormap_and_ringbuffer` | AICPU orchestrator threads | PTO2 shared-memory descriptors + ring buffers | orchestration uses PTO2 APIs and inferred dependencies | PTO2 scheduler works against shared memory, ready queues, and dispatch payloads |

The major transition is between the first two runtimes and PTO2:

- `host_build_graph` and `aicpu_build_graph` are fixed-task-array runtimes
- `tensormap_and_ringbuffer` is a fundamentally different runtime with shared memory, ring buffers, and TensorMap-driven dependency discovery

## 5. The Host C API: Where Every Runtime Must Plug In

The host platform expects every runtime to satisfy the same C API contract from `pto_runtime_c_api.h`.

The core lifecycle is:

```text
Python allocates opaque runtime buffer
  -> get_runtime_size()
  -> init_runtime(...)
  -> launch_runtime(...)
  -> finalize_runtime(...)
```

This API is intentionally runtime-agnostic. The platform host code does not need to know which runtime variant was chosen. It only needs a compiled host runtime library that exports the expected symbols.

## 6. Lifecycle Step 1: `get_runtime_size()`

`get_runtime_size()` returns `sizeof(Runtime)` for the runtime type compiled into the loaded host library.

This is why Python must load the host runtime binary before constructing a `Runtime` wrapper:

- `host_build_graph::Runtime` has one layout
- `aicpu_build_graph::Runtime` has another layout
- `tensormap_and_ringbuffer::Runtime` has yet another layout

So the runtime type is not abstract at the ABI boundary. The host `.so` determines the concrete `Runtime` layout.

## 7. Lifecycle Step 2: `init_runtime(...)`

`src/a2a3/platform/onboard/host/pto_runtime_c_api.cpp` shows the common host-side initialization path.

### 7.1 What the platform layer does

`init_runtime(...)` performs the runtime-independent front half of initialization:

1. placement-new a `Runtime` inside the caller-allocated buffer
2. populate `HostApi` callbacks:
   - `device_malloc`
   - `device_free`
   - `copy_to_device`
   - `copy_from_device`
   - `upload_kernel_binary`
   - `remove_kernel_binary`
3. dispatch to `init_runtime_impl(...)`

This means the runtime implementation does not need to know how to talk directly to CANN or simulation internals. It receives a callback table that already abstracts those platform operations.

### 7.2 Why this split matters

This design keeps responsibilities clean:

- platform code owns backend-specific memory and upload mechanics
- runtime code owns runtime-specific graph/orchestration setup

## 8. Lifecycle Step 3: `init_runtime_impl(...)`

This is where the runtime-specific lifecycle begins.

## 8.1 `host_build_graph`

`src/a2a3/runtime/host_build_graph/host/runtime_maker.cpp` is the simplest initializer.

Its source-order behavior is:

1. register kernel binaries through `host_api.upload_kernel_binary(...)`
2. copy executable addresses into `Runtime`
3. write orchestration `.so` bytes to a temp file
4. `dlopen(...)` the orchestration plugin on host
5. `dlsym(...)` the requested orchestration function
6. call it immediately to populate the runtime task graph
7. leave the graph fully built before launch

This is why the name `host_build_graph` is literal. The task graph exists before any AICPU scheduling starts.

## 8.2 `aicpu_build_graph`

`src/a2a3/runtime/aicpu_build_graph/host/runtime_maker.cpp` has different responsibilities:

1. register kernel binaries
2. auto-manage pointer arguments using `arg_types` and `arg_sizes`
3. allocate device memory for input/output tensors
4. copy host inputs to device
5. record copy-back tensors
6. marshal device pointers and scalars into `runtime->orch_args`
7. embed the AICPU orchestration plugin bytes inside `Runtime`
8. configure build mode for sequential vs concurrent build/schedule

The key difference is that host initialization prepares orchestration *state* but does **not** build the graph yet. That graph build happens later on AICPU.

## 8.3 `tensormap_and_ringbuffer`

`src/a2a3/runtime/tensormap_and_ringbuffer/host/runtime_maker.cpp` pushes that idea further.

Its initializer:

1. registers kernels
2. converts pointer arguments into device pointers
3. records output tensors for copy-back
4. copies orchestration `.so` bytes into device-managed runtime state
5. configures ready-queue sharding and PTO2 sizing from environment
6. allocates PTO2 shared memory and PTO2 heap on the device side
7. stores device pointers needed by the AICPU orchestrator and scheduler

The runtime wrapper here is mostly an integration shell. The real graph and runtime state will live in PTO2 structures, not in a traditional fixed task array.

## 9. Lifecycle Step 4: `launch_runtime(...)`

After initialization, `launch_runtime(...)` performs the shared platform-side handoff into execution.

The common logic in `pto_runtime_c_api.cpp` is:

1. convert AICPU and AICore binary bytes into vectors
2. write `orch_thread_num` into the runtime
3. call `DeviceRunner::run(...)`

This is the major transition point:

```text
runtime-specific host init is done
platform-specific backend launch begins
```

## 10. Lifecycle Step 5: `DeviceRunner::run(...)`

`DeviceRunner::run(...)` is backend-specific, but its conceptual job is always the same:

1. ensure backend initialization
2. ensure runtime binaries are loaded
3. move or expose `Runtime` state to the backend
4. start AICPU execution
5. start AICore execution
6. wait for completion
7. collect any runtime-side state needed for profiling or teardown

### 10.1 Simulation

In `src/a2a3/platform/sim/host/device_runner.cpp`, `run(...)`:

- validates launch shape
- loads AICPU and AICore executors through `dlopen`
- initializes handshake buffers in host memory
- launches host threads to emulate AICPU and AICore workers

### 10.2 Hardware

In `src/a2a3/platform/onboard/host/device_runner.cpp`, `run(...)`:

- sets the device and creates streams
- copies the AICPU `.so` to device memory
- copies the `Runtime` object to device memory
- registers AICore kernel binaries with the hardware runtime
- launches AICPU and AICore execution through CANN APIs

The abstraction here is semantic rather than mechanical: both backends “run the runtime,” but they do so in completely different ways.

## 11. Lifecycle Step 6: `finalize_runtime(...)`

`finalize_runtime(...)` is the mirror image of initialization:

1. call runtime-specific `validate_runtime_impl(...)`
2. run the `Runtime` destructor
3. return status to Python

Again, the front half is shared and the real cleanup semantics are runtime-specific.

## 12. What `validate_runtime_impl(...)` Does Per Runtime

### `host_build_graph`

The validator:

- copies recorded output tensors back to host
- frees device allocations
- unregisters uploaded kernels
- clears runtime bookkeeping

### `aicpu_build_graph`

Same high-level shape, but it also accounts for device allocations that were made to support AICPU-side graph construction and orchestration marshaling.

### `tensormap_and_ringbuffer`

The validator is responsible for copy-back and cleanup of the PTO2-oriented runtime shell:

- copy outputs back
- free orchestration and heap/shared-memory allocations
- remove uploaded kernel binaries

The graph itself is not “destroyed” like a fixed host task array would be. It is device-resident PTO2 state that gets reclaimed through the runtime’s host-owned cleanup path.

## 13. The Three Runtime Wrapper Types

At this point it is worth comparing the `Runtime` wrappers.

## 13.1 `host_build_graph::Runtime`

This runtime wrapper is close to the classical model:

- fixed `Task[]`
- fixed-size dependency arrays
- handshake buffers
- tensor-pair tracking
- function address table

It is a compact “host already knows the whole graph” runtime.

## 13.2 `aicpu_build_graph::Runtime`

This runtime wrapper extends the old model with AICPU-builder support:

- orchestration argument storage
- embedded orchestration plugin storage
- kernel address table for AICPU-created tasks
- publish/build metadata to support dynamic graph construction on device

It is still fundamentally a task-array runtime, but with a device-side build path.

## 13.3 `tensormap_and_ringbuffer::Runtime`

This wrapper is different in purpose. It does not own the primary graph representation.

Instead it owns:

- handshake buffers
- execution parameters
- tensor-pair tracking
- kernel address mapping
- pointers to PTO2 shared memory and heap state
- embedded device orchestration data

The actual runtime engine lives in PTO2 components such as:

- `PTO2Runtime`
- `PTO2SharedMemoryHandle`
- scheduler/orchestrator state
- dispatch payloads

So this `Runtime` is best thought of as the compatibility shell that satisfies platform expectations while pointing into the real PTO2 runtime.

## 14. Host-Built vs Device-Built Orchestration

This is the most important conceptual distinction across runtimes.

### 14.1 Host-built orchestration

```text
Python
  -> host runtime init
     -> host dlopen of orchestration plugin
     -> host builds graph now
  -> launch runtime
     -> AICPU schedules pre-built graph
```

### 14.2 Device-built orchestration

```text
Python
  -> host runtime init
     -> host prepares device pointers and embeds orchestration payload
  -> launch runtime
     -> AICPU loads or executes orchestration logic
     -> AICPU builds graph on device
     -> AICPU schedules resulting work
```

This distinction determines:

- orchestration compiler choice
- initialization responsibilities
- when the graph comes into existence
- how much host/device state must be marshaled before launch

## 15. Practical Reading Order for Later Chapters

Once you understand the lifecycle in this chapter, the rest of the repository becomes easier to read:

1. `host_build_graph` for the simplest end-to-end mental model
2. `aicpu_build_graph` for device-side graph construction without PTO2 complexity
3. `tensormap_and_ringbuffer` for the full production-style runtime

## 16. Key Takeaways

- A runtime is a buildable implementation bundle, not a single class.
- `build_config.py` is the real runtime-definition file.
- `init_runtime()` is shared platform logic; `init_runtime_impl()` is runtime-specific logic.
- `launch_runtime()` is the common handoff into backend execution.
- `finalize_runtime()` is the common handoff into runtime-specific cleanup.
- `host_build_graph` and `aicpu_build_graph` are fixed-task-array runtimes.
- `tensormap_and_ringbuffer` is a different architecture centered on PTO2 shared memory and runtime subsystems.
