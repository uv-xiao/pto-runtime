# Execution Model and Build Flow

This chapter explains the real execution path in the current repository, from the user-facing CLI to compiled binaries and finally to runtime launch. The most important correction up front is that the top-level `README.md` still describes an older layout; the actual code path today runs through `examples/scripts/`, `python/`, `src/a2a3/...`, and `src/a5/...`.

## Files Covered

- `examples/scripts/run_example.py`
- `examples/scripts/code_runner.py`
- `python/runtime_builder.py`
- `python/runtime_compiler.py`
- `python/kernel_compiler.py`
- `python/bindings.py`
- `python/elf_parser.py`
- representative `kernel_config.py` files under `examples/a2a3/`
- representative platform CMake files under `src/a2a3/platform/`

## Reading Strategy

Read this path in the same order the program executes:

1. `run_example.py` parses CLI options and sets process-level logging.
2. `CodeRunner` loads `kernel_config.py` and `golden.py`.
3. `RuntimeBuilder` discovers the selected runtime for the selected platform.
4. `RuntimeCompiler` builds host, AICPU, and AICore runtime artifacts.
5. `KernelCompiler` builds the orchestration plugin and every user kernel.
6. `bindings.py` loads the host runtime `.so` with `ctypes`.
7. The host runtime initializes `Runtime`, uploads kernels, launches AICPU and AICore binaries, and later finalizes the run.

Keep one concrete example open while reading. For this chapter, the simplest choices are:

- `examples/a2a3/host_build_graph/vector_example/`
- `examples/a2a3/tensormap_and_ringbuffer/vector_example/`

## 1. Big Picture

The build-and-run pipeline is intentionally split into two layers:

- Python decides *what* to build and run.
- C++ platform code decides *how* to load, launch, and synchronize the target backend.

The resulting control flow looks like this:

```text
user CLI
  |
  v
examples/scripts/run_example.py
  |
  v
examples/scripts/code_runner.py
  |
  +--> load golden.py
  +--> load kernel_config.py
  |      |
  |      +--> RUNTIME_CONFIG["runtime"]
  |      +--> ORCHESTRATION["source"], ["function_name"]
  |      +--> KERNELS[{func_id, source, core_type}]
  |
  +--> python/runtime_builder.py
  |      |
  |      +--> python/runtime_compiler.py
  |      |      +--> build host runtime binary
  |      |      +--> build AICPU runtime binary
  |      |      +--> build AICore runtime binary
  |      |
  |      +--> python/kernel_compiler.py
  |             +--> build orchestration plugin
  |             +--> build each incore kernel
  |
  +--> python/bindings.py
         |
         +--> load host runtime .so with ctypes
         +--> init_runtime(...)
         +--> launch_runtime(...)
         +--> finalize_runtime(...)
```

## 2. User-Facing Entry Point: `run_example.py`

`examples/scripts/run_example.py` is the real user entrypoint. CI scripts eventually funnel into it, so understanding this file gives you the true front door for both manual and automated execution.

Its responsibilities are intentionally narrow:

- parse `--kernels`, `--golden`, `--platform`, `--device`, and profiling options
- configure Python logging
- set `PTO_LOG_LEVEL` for the C++ side
- validate that the requested files exist
- instantiate `CodeRunner`
- call `CodeRunner.run()`

The important external selection point is:

```text
--platform a2a3 | a2a3sim | a5 | a5sim
```

This does **not** choose the runtime variant. It chooses the backend family. Runtime selection happens later from `kernel_config.py`.

## 3. Configuration Intake: `CodeRunner`

`examples/scripts/code_runner.py` is the real build-and-execute coordinator.

### 3.1 What it loads

`CodeRunner.__init__()` resolves and loads:

- `kernel_config.py`
- `golden.py`

From `kernel_config.py`, it extracts three contracts:

- `KERNELS`
- `ORCHESTRATION`
- `RUNTIME_CONFIG`

Those three variables are the user-visible knobs that shape the entire runtime session.

### 3.2 What `RUNTIME_CONFIG` controls

Representative examples:

`examples/a2a3/host_build_graph/vector_example/kernels/kernel_config.py`
```python
RUNTIME_CONFIG = {
    "runtime": "host_build_graph",
    "aicpu_thread_num": 3,
    "orch_thread_num": 0,
    "block_dim": 3,
}
```

`examples/a2a3/aicpu_build_graph/vector_example/kernels/kernel_config.py`
```python
RUNTIME_CONFIG = {
    "runtime": "aicpu_build_graph",
    "aicpu_thread_num": 4,
    "block_dim": 24,
}
```

`examples/a2a3/tensormap_and_ringbuffer/vector_example/kernels/kernel_config.py`
```python
RUNTIME_CONFIG = {
    "runtime": "tensormap_and_ringbuffer",
    "aicpu_thread_num": 4,
    "block_dim": 3,
    "rounds": 2,
}
```

This is the first runtime-selection point that matters. `CodeRunner` stores:

- `self.runtime_name`
- `self.aicpu_thread_num`
- `self.orch_thread_num`
- `self.block_dim`
- `self.repeat_rounds`

and later passes them into the build and launch pipeline.

### 3.3 What `CodeRunner.run()` actually does

The `run()` method is the clearest summary of the entire Python-side orchestration:

1. ensure `PTO_ISA_ROOT` is available, cloning `pto-isa` if necessary
2. create `RuntimeBuilder(platform=self.platform)`
3. validate that the requested runtime exists for this platform
4. compile in parallel:
   - runtime binaries
   - orchestration plugin
   - each kernel
5. load the host runtime binary with `bind_host_binary(...)`
6. call `set_device(device_id)`
7. for each test case:
   - generate tensors and scalars from `golden.py`
   - compute the reference output on host
   - call `runtime.initialize(...)`
   - call `launch_runtime(...)`
   - call `runtime.finalize()`
   - compare outputs against the golden result

That means `CodeRunner` is not just a wrapper around execution. It is the place where build, upload, launch, validation, and profiling opt-in all meet.

## 4. Runtime Discovery: `RuntimeBuilder`

`python/runtime_builder.py` is the top-level Python build coordinator.

### 4.1 Source-order walkthrough

#### `RuntimeBuilder.__init__(platform)`

This constructor does four important things in order:

1. store the user-selected platform string
2. map the platform into an architecture root
3. discover runtimes from `src/<arch>/runtime/*/build_config.py`
4. construct a platform-configured `RuntimeCompiler` and `KernelCompiler`

The key architectural mapping is:

```text
a2a3   -> src/a2a3/runtime
a2a3sim-> src/a2a3/runtime
a5     -> src/a5/runtime
a5sim  -> src/a5/runtime
```

This is the first place where the repository’s real structure becomes visible. The builder does **not** use `src/runtime/` as a shared implementation root. It chooses `src/a2a3/runtime/` or `src/a5/runtime/`.

#### Runtime discovery loop

The constructor walks the selected runtime directory and records every subdirectory that contains `build_config.py`. That means runtime availability is data-driven:

- if a runtime directory has `build_config.py`, it is discoverable
- if it does not, it does not exist from Python’s point of view

This is why `a2a3` exposes three runtimes while `a5` currently exposes only two.

#### `build(name, build_dir=None)`

This method is the handoff from “runtime selection” into “actual runtime binary production.”

It:

1. loads the runtime’s `build_config.py`
2. resolves include/source directories for:
   - `aicore`
   - `aicpu`
   - `host`
3. submits three parallel compile jobs through `RuntimeCompiler.compile(...)`
4. returns raw bytes:
   - `host_binary`
   - `aicpu_binary`
   - `aicore_binary`

Those three byte blobs are the central invariant of the framework. Regardless of backend, the runtime build stage always yields a host-side control binary, an AICPU-side control binary, and an AICore-side execution binary.

## 5. Backend Build Driver: `RuntimeCompiler`

`python/runtime_compiler.py` is the actual compile-and-link engine for runtime code.

### 5.1 What it selects

`RuntimeCompiler.__init__()` maps the user platform to a platform build root:

```text
a2a3    -> src/a2a3/platform/onboard
a2a3sim -> src/a2a3/platform/sim
a5      -> src/a5/platform/onboard
a5sim   -> src/a5/platform/sim
```

This is the decisive backend-selection point. After this point, all CMake builds use either the real hardware backend (`onboard`) or the simulation backend (`sim`).

### 5.2 Why the platform directories own CMake

The platform CMake trees own launch/build plumbing because they know:

- what the host runtime must link against
- how AICPU binaries should be produced
- how AICore binaries should be linked
- which libraries or emulation infrastructure a backend needs

The runtime contributes behavior by injecting `CUSTOM_INCLUDE_DIRS` and `CUSTOM_SOURCE_DIRS`.

This split is easiest to see in the A2/A3 platform CMake files:

- `src/a2a3/platform/onboard/host/CMakeLists.txt`
- `src/a2a3/platform/onboard/aicpu/CMakeLists.txt`
- `src/a2a3/platform/onboard/aicore/CMakeLists.txt`
- `src/a2a3/platform/sim/host/CMakeLists.txt`
- `src/a2a3/platform/sim/aicpu/CMakeLists.txt`
- `src/a2a3/platform/sim/aicore/CMakeLists.txt`

Each one starts from platform-local sources and then appends runtime-provided directories.

### 5.3 Artifact shape by backend

| Target | `onboard` backend | `sim` backend |
|---|---|---|
| Host runtime | shared library | shared library |
| AICPU runtime | shared library | shared library |
| AICore runtime | relocatable kernel object (`aicore_kernel.o`) | host-loadable shared library |

The biggest difference is AICore:

- hardware AICore code is compiled with Bisheng CCE and linked into a relocatable kernel object
- simulation AICore code is compiled as a host `.so` and `dlopen`ed

The hardware AICore CMake file makes this explicit:

1. compile each source for AIC
2. compile each source for AIV
3. combine AIC objects with `ld -r`
4. combine AIV objects with `ld -r`
5. link both combined objects into one final relocatable kernel object

## 6. User-Kernel and Orchestration Build Driver: `KernelCompiler`

`python/kernel_compiler.py` handles the pieces owned by the example rather than the runtime:

- orchestration plugin
- individual user kernels

### 6.1 Why this compiler is separate

Runtime binaries are large, stable framework artifacts.

User kernels and orchestration are small, frequently changing example-specific artifacts. Keeping them in a separate compiler lets the framework:

- rebuild only the changed example pieces
- choose toolchains based on runtime policy
- inject runtime include paths into kernel compilation

### 6.2 Toolchain selection is not purely Python-driven

One subtle but important design choice is that `KernelCompiler` can query the already-built host runtime for compile policy via:

- `get_incore_compiler()`
- `get_orchestration_compiler()`

Those functions are implemented per runtime in:

- `src/a2a3/runtime/host_build_graph/host/runtime_compile_info.cpp`
- `src/a2a3/runtime/aicpu_build_graph/host/runtime_compile_info.cpp`
- `src/a2a3/runtime/tensormap_and_ringbuffer/host/runtime_compile_info.cpp`

That means compile policy is partly a runtime decision:

- `host_build_graph` orchestration always uses host g++
- `aicpu_build_graph` orchestration uses aarch64 g++ on real hardware
- `tensormap_and_ringbuffer` orchestration also uses aarch64 g++ on real hardware

This is an important design point: the runtime is allowed to say “my orchestration code lives on host” or “my orchestration code runs on AICPU.”

### 6.3 Special case: extracting AICore `.text`

After `compile_incore(...)`, `CodeRunner` does backend-specific post-processing:

- on simulation, it keeps the full `.so`
- on hardware, it passes the produced object through `python/elf_parser.py::extract_text_section(...)`

That means hardware kernel upload sends only executable kernel text, not the full host-style object packaging used by simulation.

## 7. Load Boundary: `bindings.py`

`python/bindings.py` is the Python-to-C boundary.

### 7.1 What `bind_host_binary()` does

It writes the host runtime bytes to a temporary file and loads them with:

- `ctypes.CDLL(..., RTLD_GLOBAL)`

This gives Python access to the platform C API declared in `pto_runtime_c_api.h`.

### 7.2 What `Runtime.initialize()` does conceptually

It prepares the arguments for `init_runtime(...)`:

- orchestration binary bytes
- orchestration symbol name
- function arguments
- argument kinds (`ARG_SCALAR`, `ARG_INPUT_PTR`, `ARG_OUTPUT_PTR`, `ARG_INOUT_PTR`)
- kernel binaries as `(func_id, bytes)` pairs

This is the moment where Python’s high-level view becomes the runtime’s internal view.

### 7.3 Launch and finalize

`launch_runtime(...)` hands over:

- `aicpu_thread_num`
- `block_dim`
- `device_id`
- AICPU runtime bytes
- AICore runtime bytes
- `orch_thread_num`

`finalize_runtime(...)` triggers copy-back and cleanup.

So Python does not manually call platform memory APIs during execution. It front-loads the state into the C API and lets the runtime/platform layers own the actual launch lifecycle.

## 8. Runtime Build, Kernel Build, and Launch as One Sequence

The most important integrated view is this one:

```text
run_example.py
  -> CodeRunner.run()
      -> RuntimeBuilder.list_runtimes()
      -> RuntimeBuilder.build(runtime_name)
          -> RuntimeCompiler.compile("host")
          -> RuntimeCompiler.compile("aicpu")
          -> RuntimeCompiler.compile("aicore")
      -> KernelCompiler.compile_orchestration(...)
      -> KernelCompiler.compile_incore(...) for each kernel
      -> bind_host_binary(host_binary)
      -> set_device(device_id)
      -> runtime.initialize(...)
      -> launch_runtime(...)
      -> runtime.finalize()
      -> compare against golden
```

This sequence is the right mental model for the whole repository. Everything else in later chapters is a more detailed explanation of one stage inside this sequence.

## 9. What Changes Across Backends

Two things stay invariant:

- the Python control flow
- the host C API surface

Two things change:

- which platform build tree is used
- what binary format gets produced and loaded

That leads to this summary:

| Concern | Stable across backends | Backend-specific |
|---|---|---|
| CLI entrypoint | yes | no |
| `kernel_config.py` contract | yes | no |
| runtime discovery | yes | architecture root changes |
| runtime C API | yes | implementation changes |
| host runtime load | yes | linked libraries differ |
| AICPU launch | no | hardware uses CANN, sim uses `dlopen` + threads |
| AICore launch | no | hardware registers kernel objects, sim loads `.so` |

## 10. Concrete Commands to Run While Reading

### 10.1 Minimal host-built graph run

```bash
python examples/scripts/run_example.py \
  -k examples/a2a3/host_build_graph/vector_example/kernels \
  -g examples/a2a3/host_build_graph/vector_example/golden.py \
  -p a2a3sim
```

What this proves:

- the example config loads correctly
- `host_build_graph` is discoverable on `a2a3sim`
- runtime, orchestration, and kernels all compile
- the simulation host/AICPU/AICore backend loads and executes

### 10.2 Minimal PTO2 run

```bash
python examples/scripts/run_example.py \
  -k examples/a2a3/tensormap_and_ringbuffer/vector_example/kernels \
  -g examples/a2a3/tensormap_and_ringbuffer/vector_example/golden.py \
  -p a2a3sim
```

What this proves:

- `tensormap_and_ringbuffer` is discoverable and buildable on the selected platform
- the runtime-specific orchestration compile policy works
- the PTO2 runtime wrapper, not just the older fixed-graph runtimes, can be initialized and executed

## 11. Key Takeaways

- `run_example.py` is the CLI shell; `CodeRunner` is the real controller.
- runtime selection comes from `kernel_config.py`, not the CLI.
- backend selection comes from `--platform`, then becomes `src/<arch>/platform/onboard` or `src/<arch>/platform/sim`.
- runtime implementations live under `src/a2a3/runtime/` and `src/a5/runtime/`, not under the stale paths described in the root README.
- the framework always builds three runtime artifacts: host, AICPU, and AICore.
- the host runtime C API is the central seam between Python orchestration and backend-specific execution.
