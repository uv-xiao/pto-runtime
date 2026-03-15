# Execution Model and Build Flow

This chapter will explain the full execution path from `examples/scripts/run_example.py` to compiled binaries and runtime launch. The emphasis is on the real control flow in this repository today, including where platform and runtime choices are made and how Python hands control to the host C API.

## Files Covered

- `examples/scripts/run_example.py`
- `examples/scripts/code_runner.py`
- `python/runtime_builder.py`
- `python/runtime_compiler.py`
- `python/kernel_compiler.py`
- `python/bindings.py`
- representative `kernel_config.py` files under `examples/a2a3/` and `examples/a5/`

## Reading Strategy

Read top-down from the CLI entrypoint, then follow the build pipeline in the same order the code runs: argument parsing, config loading, runtime selection, kernel compilation, runtime compilation, ctypes binding, runtime initialization, and launch.

## Planned Diagrams

- end-to-end build and launch swimlane
- compilation matrix for host/AICPU/AICore targets
- control-transfer diagram from Python into the host runtime

## Planned Code Walkthroughs

- `RuntimeBuilder`
- `RuntimeCompiler`
- `KernelCompiler`
- `bindings.py` runtime lifecycle

## Planned Verification Notes

- simulation example command for `host_build_graph`
- simulation example command for `tensormap_and_ringbuffer`
- explanation of what each successful run proves
