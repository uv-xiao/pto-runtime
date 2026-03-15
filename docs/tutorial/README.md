# PTO Runtime Tutorial Roadmap

This directory will hold an ultra-detailed tutorial series for PTO Runtime. The target audience is an engineer who can read C++ and Python but has zero prior context on this repository's runtime model, backend layout, or execution path.

## Scope

The tutorial series must cover:
- the overall architecture and execution flow from `run_example.py` through runtime launch
- how different backends are supported and abstracted, with focus on `src/a2a3/platform/` and `src/a5/platform/`
- how runtimes are defined, selected, and invoked
- a deep dive into `tensormap_and_ringbuffer`, including scheduling, synchronization, and profiling
- runnable examples and debugging/profiling workflows

## Planned Chapters

1. `01-execution-model-and-build-flow.md`
2. `02-platform-backends-and-abstraction.md`
3. `03-runtime-definition-and-lifecycle.md`
4. `04-tensormap-and-ringbuffer-deep-dive.md`
5. `05-profiling-examples-and-debugging.md`

## Source Map

The tutorial series is anchored to the real code paths in this repository:
- Python build and launch flow: `examples/scripts/`, `python/`
- A2/A3 platform code: `src/a2a3/platform/`
- A5 platform code: `src/a5/platform/`
- A2/A3 runtimes: `src/a2a3/runtime/`
- A5 runtimes: `src/a5/runtime/`
- Examples and tests: `examples/`, `tests/device_tests/`

## Important Note About Paths

The root `README.md` still describes older `src/platform/...` and `src/runtime/...` paths in several places. The tutorial should follow the current repository layout under `src/a2a3/...` and `src/a5/...`, and explicitly call out stale references when they could confuse a reader.
