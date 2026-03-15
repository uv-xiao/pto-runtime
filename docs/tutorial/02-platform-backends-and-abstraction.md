# Platform Backends and Abstraction

This chapter will explain how PTO Runtime supports different chips and backend modes. It will compare `src/a2a3/platform/` and `src/a5/platform/`, show how `onboard` and `sim` are selected, and clarify which interfaces are stable versus which implementations are duplicated.

## Files Covered

- `src/a2a3/platform/include/`
- `src/a2a3/platform/onboard/`
- `src/a2a3/platform/sim/`
- `src/a2a3/platform/src/`
- `src/a5/platform/include/`
- `src/a5/platform/onboard/`
- `src/a5/platform/sim/`
- `src/a5/platform/src/`

## Reading Strategy

Start with the shared public headers, then compare host/AICPU/AICore implementations across `onboard` and `sim`. Finish by mapping how the Python build layer chooses each platform tree and how the runtime code plugs into the platform CMake targets.

## Planned Diagrams

- platform matrix: chip x backend x execution role
- interface boundary diagram for Python, host C API, platform code, and runtime code
- hardware-vs-simulation comparison table

## Planned Code Walkthroughs

- host `device_runner.cpp`
- host `pto_runtime_c_api.cpp`
- `platform_compile_info.cpp`
- AICPU and AICore kernel entry files

## Planned Verification Notes

- where platform selection happens in Python
- which binary/toolchain combinations are produced for `a2a3`, `a2a3sim`, `a5`, and `a5sim`
- which abstractions are real contracts and which are mirrored copies
