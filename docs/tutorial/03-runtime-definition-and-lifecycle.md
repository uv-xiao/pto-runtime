# Runtime Definition and Lifecycle

This chapter will define what a “runtime” means in PTO Runtime, how a runtime is declared by `build_config.py`, and how the host initializes, launches, and finalizes it. It will compare the runtime variants and explain the common lifecycle that the host platform expects each runtime to satisfy.

## Files Covered

- `src/a2a3/runtime/*/build_config.py`
- `src/a5/runtime/*/build_config.py`
- `src/a2a3/runtime/*/host/runtime_maker.cpp`
- `src/a2a3/runtime/*/runtime/runtime.h`
- `src/a2a3/platform/onboard/host/pto_runtime_c_api.cpp`
- runtime compile-info files for runtime-specific toolchain policy

## Reading Strategy

Start with `build_config.py`, then read the host-side `runtime_maker.cpp` glue, then inspect the runtime wrapper class that the platform sees. Use that to compare `host_build_graph`, `aicpu_build_graph`, and `tensormap_and_ringbuffer`.

## Planned Diagrams

- runtime selection path from `kernel_config.py` to `RuntimeBuilder.build()`
- lifecycle diagram for `init_runtime` -> `launch_runtime` -> `finalize_runtime`
- host-built vs device-built orchestration comparison

## Planned Code Walkthroughs

- placement-new runtime construction
- kernel binary registration
- orchestration loading
- `DeviceRunner::run` handoff

## Planned Verification Notes

- example runtime configurations to inspect
- commands to run at least one `host_build_graph` example and one `tensormap_and_ringbuffer` example
- notes on hardware-only behavior when simulation cannot prove a path
