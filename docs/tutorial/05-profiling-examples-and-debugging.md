# Profiling, Examples, and Debugging

This chapter will explain how to learn the runtime empirically: which examples to run first, how profiling is enabled, where the host and device profiling outputs go, and how to debug common issues such as ring exhaustion, stale path assumptions, or scheduler starvation.

## Files Covered

- `.claude/commands/profile.md`
- `tools/README.md`
- `tools/swimlane_converter.py`
- `tools/sched_overhead_analysis.py`
- `tools/benchmark_rounds.sh`
- `src/a2a3/runtime/tensormap_and_ringbuffer/docs/RUNTIME_LOGIC.md`
- `src/a2a3/runtime/tensormap_and_ringbuffer/docs/device_log_profiling.md`
- `src/a2a3/runtime/tensormap_and_ringbuffer/docs/profiling_levels.md`
- selected examples under `examples/a2a3/tensormap_and_ringbuffer/`
- selected tests under `tests/device_tests/a2a3/tensormap_and_ringbuffer/`

## Reading Strategy

Start with the smallest examples, then move to realistic pipelines, then connect the observed outputs back to the runtime internals from the earlier chapters. Treat the profiling tools and device logs as a second reading of the runtime, not a separate topic.

## Planned Diagrams

- example progression map
- profiling artifact flow from runtime buffers to `swimlane.json`
- debugging decision tree for common runtime failures

## Planned Code Walkthroughs

- `--enable-profiling` path in `run_example.py`
- `swimlane_converter.py` output expectations
- scheduler-overhead analysis workflow

## Planned Verification Notes

- canonical profiling command
- where device logs are written
- how to extract orchestrator and scheduler summaries
- how compile-time profiling macros differ from runtime `enable_profiling`
