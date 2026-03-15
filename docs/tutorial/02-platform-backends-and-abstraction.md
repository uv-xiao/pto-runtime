# Platform Backends and Abstraction

This chapter explains how PTO Runtime supports multiple chips and multiple execution backends without changing the Python user interface. The short version is: the build system presents a clean platform-runtime matrix, but the source tree is still mostly organized as two parallel architecture-specific copies, `a2a3` and `a5`.

## Files Covered

- `src/a2a3/platform/include/`
- `src/a2a3/platform/onboard/`
- `src/a2a3/platform/sim/`
- `src/a2a3/platform/src/`
- `src/a5/platform/include/`
- `src/a5/platform/onboard/`
- `src/a5/platform/sim/`
- `src/a5/platform/src/`
- `python/runtime_compiler.py`
- `python/kernel_compiler.py`

## Reading Strategy

Read platform code from the outside in:

1. start with the Python build layer that selects a backend
2. inspect the public platform headers in `include/`
3. compare `onboard` and `sim`
4. compare `a2a3` and `a5`
5. finish by identifying which parts are true abstractions and which are duplicated implementations

## 1. The Platform Matrix

The build system treats platform and runtime as orthogonal:

```text
platform choice -> architecture root -> backend root

a2a3    -> src/a2a3 -> platform/onboard
a2a3sim -> src/a2a3 -> platform/sim
a5      -> src/a5   -> platform/onboard
a5sim   -> src/a5   -> platform/sim
```

Each architecture has the same internal role split:

```text
src/<arch>/platform/
  include/   public headers and contracts
  onboard/   real hardware backend
  sim/       thread-based simulation backend
  src/       helper implementations shared inside that architecture
```

And each backend has the same execution-role split:

```text
host/    host runtime loader and memory API
aicpu/   device scheduler / control-side entrypoint
aicore/  compute-side entrypoint
```

That gives the full matrix:

| Chip | Backend | Host | AICPU | AICore |
|---|---|---|---|---|
| `a2a3` | `onboard` | real CANN host runtime | real device-side scheduler binary | real Ascend kernel object |
| `a2a3` | `sim` | host-thread runtime | host-thread AICPU emulator | host-thread AICore emulator |
| `a5` | `onboard` | real CANN host runtime | real device-side scheduler binary | real Ascend kernel object |
| `a5` | `sim` | host-thread runtime | host-thread AICPU emulator | host-thread AICore emulator |

## 2. What the Build System Actually Abstracts

The clean abstraction points are in Python:

- `python/runtime_builder.py`
- `python/runtime_compiler.py`
- `python/kernel_compiler.py`

These files abstract:

- architecture root selection
- backend root selection
- toolchain selection
- source injection into platform CMake targets

From the Python caller’s point of view, the API is compact:

```python
builder = RuntimeBuilder(platform="a2a3sim")
host_binary, aicpu_binary, aicore_binary = builder.build("host_build_graph")
```

But once control enters C++ sources, the codebase is not heavily deduplicated. Instead, it mostly mirrors the same shapes across:

- `src/a2a3/platform/...`
- `src/a5/platform/...`

That means the architecture is conceptually abstract, but the implementation is still largely copy-based.

## 3. Public Platform Contracts in `include/`

The most important headers are:

- `include/host/pto_runtime_c_api.h`
- `include/host/runtime_compile_info.h`
- `include/host/platform_compile_info.h`
- `include/host/memory_allocator.h`
- `include/common/core_type.h`
- `include/common/platform_config.h`
- `include/common/kernel_args.h`
- `include/common/perf_profiling.h`

These headers define the contracts that the rest of the repository depends on:

### 3.1 Host C API

`pto_runtime_c_api.h` is the stable Python-facing contract.

It defines:

- `get_runtime_size()`
- `init_runtime(...)`
- `launch_runtime(...)`
- `finalize_runtime(...)`
- `set_device(...)`
- device memory helper functions used by orchestration

This is the main reason the Python layer can stay backend-neutral. Python never talks directly to CANN or to simulation threads. It talks to this C API.

### 3.2 Compile strategy interface

`runtime_compile_info.h` and `platform_compile_info.h` let the C++ side tell Python which toolchain to use.

That split is subtle but important:

- `platform_compile_info.cpp` answers “which platform am I?”
- `runtime_compile_info.cpp` answers “which compiler should kernels or orchestration use for this runtime on this platform?”

This lets compile policy be runtime-dependent rather than hardcoded only in Python.

## 4. `onboard` vs `sim`: the Most Important Backend Split

The biggest real abstraction boundary is not `a2a3` vs `a5`. It is `onboard` vs `sim`.

## 4.1 Host backend comparison

### `onboard/host/device_runner.cpp`

This is the real hardware host launcher.

Its responsibilities include:

- `rtSetDevice(...)`
- stream creation
- device allocation and copies via CANN runtime APIs
- copying the AICPU `.so` into device memory
- copying the `Runtime` object into device memory
- registering the AICore binary with `rtRegisterAllKernel(...)`
- launching AICPU and AICore kernels through CANN

This is the file that “touches the metal” on the host side.

### `sim/host/device_runner.cpp`

This is the simulation host launcher.

Instead of CANN device APIs, it:

- writes the AICPU runtime bytes to a temp `.so`
- `dlopen`s the AICPU runtime
- resolves `aicpu_execute`
- writes the AICore runtime bytes to a temp `.so`
- `dlopen`s the AICore runtime
- resolves `aicore_execute_wrapper`
- uses host threads to emulate AICPU and AICore execution

So the simulation backend does not emulate the hardware driver interface. It emulates the execution roles.

## 4.2 AICPU backend comparison

### `onboard/aicpu/`

The AICPU backend is compiled as a real device-side shared library using the aarch64 cross-compiler. It is meant to run on the device control processors and participate in the real device handshake/register model.

### `sim/aicpu/`

The simulation AICPU backend is compiled as a host `.so` and loaded into the process. It still exports the same conceptual entrypoints, but those entrypoints run as normal host code.

The abstraction is therefore behavioral, not binary-level:

- same conceptual role
- very different loading and execution mechanism

## 4.3 AICore backend comparison

### `onboard/aicore/`

This path builds true hardware kernels:

- compile with Bisheng CCE
- split AIC and AIV variants
- relocatable-link intermediate objects
- emit one final `aicore_kernel.o`

### `sim/aicore/`

This path builds a host shared library:

- compile with host g++
- define `__CPU_SIM`
- export `aicore_execute_wrapper`
- load with `dlopen`

This is why the AICore artifact type changes across backends.

## 5. What Actually Stays the Same Across Backends

Even though the implementation changes substantially, these conceptual seams remain stable:

### 5.1 Same role decomposition

Every backend still thinks in three roles:

- host
- AICPU
- AICore

### 5.2 Same runtime handoff shape

The host backend still receives:

- a `Runtime`
- AICPU binary bytes
- AICore binary bytes
- launch parameters like `block_dim` and AICPU thread count

### 5.3 Same kernel registration intent

Both backends have a notion of “register a kernel binary and remember the execution address”:

- simulation stores a host function address obtained through `dlopen`/`dlsym`
- hardware stores a device address after uploading bytes into device-visible memory

Conceptually, both implement:

```text
func_id -> executable address
```

## 6. `a2a3` vs `a5`: abstraction vs reality

At the build-system level, `a2a3` and `a5` look symmetric. But the source tree reveals that A5 is still partly a validation copy of A2/A3.

The clearest evidence is `src/a5/runtime/README.md`, which states that A5 runtime support is currently a temporary validation layer.

Important current-state facts:

- `src/a2a3/runtime/` contains:
  - `host_build_graph`
  - `aicpu_build_graph`
  - `tensormap_and_ringbuffer`
- `src/a5/runtime/` contains:
  - `host_build_graph`
  - `tensormap_and_ringbuffer`
- there is no `aicpu_build_graph` implementation under `src/a5/runtime/`

So the abstraction is not “every chip supports every runtime.” The real rule is “available runtimes are discovered from the selected architecture tree.”

## 7. Duplicated Contracts vs Duplicated Implementations

It helps to separate two kinds of duplication.

### 7.1 Good duplication: contract mirroring

These pairs are meant to mirror each other:

- `src/a2a3/platform/include/...`
- `src/a5/platform/include/...`

They give both architectures the same conceptual API surface.

### 7.2 Costly duplication: implementation mirroring

These trees also mirror each other closely:

- `src/a2a3/platform/onboard/...`
- `src/a5/platform/onboard/...`
- `src/a2a3/platform/sim/...`
- `src/a5/platform/sim/...`

This is where the repository is less abstract than it first appears. Much of the logic is repeated as separate files rather than shared through a common implementation layer.

The same pattern exists in runtimes:

- `host_build_graph`
- `aicpu_build_graph`
- `tensormap_and_ringbuffer`

each carry their own runtime wrapper types and initialization logic rather than sharing one universal runtime class.

## 8. Where the Runtime Plugs Into the Platform

The handoff mechanism is worth making explicit.

```text
Python
  |
  v
RuntimeCompiler picks platform CMake root
  |
  v
platform CMakeLists.txt adds platform-local sources
  |
  +--> device_runner.cpp
  +--> pto_runtime_c_api.cpp
  +--> platform_compile_info.cpp
  +--> kernel.cpp
  |
  +--> append runtime CUSTOM_SOURCE_DIRS
         |
         +--> runtime_maker.cpp
         +--> runtime.cpp / runtime.h
         +--> runtime-specific executors
```

This is the key design idea:

- platform code owns backend plumbing
- runtime code injects backend-independent execution policy

It is not a perfect abstraction in implementation terms, but it is the right high-level mental model.

## 9. What Python Expects From Platform Code

Python assumes the following guarantees:

1. it can load a host runtime binary as a shared library
2. that library exports the C API from `pto_runtime_c_api.h`
3. `get_runtime_size()` matches the runtime type compiled into that library
4. `init_runtime(...)` can consume raw orchestration bytes and kernel bytes
5. `launch_runtime(...)` can consume runtime binaries as byte arrays
6. `finalize_runtime(...)` performs cleanup and result copy-back

As long as a backend satisfies those assumptions, the Python layer does not need to know how it launches kernels.

## 10. What Runtime Code Expects From Platform Code

Runtime code depends on platform code for:

- memory allocation
- memory copy
- kernel binary upload
- device or simulation launch
- handshake/register mechanics
- profiling buffer infrastructure

That dependency is visible in the `HostApi` callback tables populated by `pto_runtime_c_api.cpp`.

So the runtime does not call CANN or simulation internals directly. It asks the platform layer to perform the backend-specific operations on its behalf.

## 11. Hardware vs Simulation Summary

| Concern | Hardware `onboard` | Simulation `sim` |
|---|---|---|
| Host loader | CANN runtime APIs | host filesystem + `dlopen` |
| Device memory | real device memory | host heap / mapped memory |
| AICPU code | aarch64 device `.so` | host `.so` |
| AICore code | Ascend kernel object | host `.so` |
| Kernel registration | upload into GM and register | `dlopen` and cache symbol addresses |
| Execution engine | real Ascend launch path | host threads |
| Debuggability | more realistic, harder to inspect | easier to inspect, less faithful |

## 12. Key Takeaways

- The clean abstraction is strongest in the Python build layer.
- The strongest backend split is `onboard` vs `sim`, not `a2a3` vs `a5`.
- `a2a3` and `a5` mostly mirror each other as separate source trees.
- Platform code owns loading, memory, launch, and binary-format concerns.
- Runtime code plugs into platform CMake targets and host callbacks to supply scheduling/orchestration behavior.
- The public host C API is the most important stable seam in the entire backend story.
