# PTO Runtime Deep Tutorial Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Author an ultra-detailed tutorial series under `docs/tutorial/` that teaches PTO Runtime architecture, execution flow, backend abstraction, runtime lifecycle, and the `tensormap_and_ringbuffer` runtime in enough depth for a new engineer to trace the code confidently.

**Architecture:** The tutorial should mirror the real repository layering: Python build/launch entrypoints, chip-specific platform trees (`src/a2a3/platform`, `src/a5/platform`), chip-specific runtime implementations (`src/a2a3/runtime`, `src/a5/runtime`), and the host/AICPU/AICore control plane. The tutorial should be split into a small set of focused Markdown files, each grounded in concrete code paths, diagrams, and verified example/profiling commands.

**Tech Stack:** Markdown, ASCII/Mermaid diagrams, Python build scripts, C++ runtime/platform code, GitHub PR workflow, example runners in `examples/scripts/`

---

### Task 1: Create the Tutorial Skeleton

**Files:**
- Create: `docs/tutorial/README.md`
- Create: `docs/tutorial/01-execution-model-and-build-flow.md`
- Create: `docs/tutorial/02-platform-backends-and-abstraction.md`
- Create: `docs/tutorial/03-runtime-definition-and-lifecycle.md`
- Create: `docs/tutorial/04-tensormap-and-ringbuffer-deep-dive.md`
- Create: `docs/tutorial/05-profiling-examples-and-debugging.md`

**Step 1: Create the directory and index file**

Add a `docs/tutorial/README.md` file that:
- states the tutorial audience and learning goals
- lists the five tutorial chapters in reading order
- names the exact source directories that each chapter studies
- warns that `README.md` at repo root contains stale paths and that the tutorial follows the real `src/a2a3/...` and `src/a5/...` layout

**Step 2: Add chapter stubs with required section headings**

Each chapter file should begin with:
- a one-paragraph purpose statement
- a `## Files Covered` section
- a `## Reading Strategy` section
- placeholder headings for diagrams, code walkthroughs, and verification notes

**Step 3: Verify the file set exists**

Run:
```bash
find docs/tutorial -maxdepth 1 -type f | sort
```

Expected:
- `README.md`
- `01-execution-model-and-build-flow.md`
- `02-platform-backends-and-abstraction.md`
- `03-runtime-definition-and-lifecycle.md`
- `04-tensormap-and-ringbuffer-deep-dive.md`
- `05-profiling-examples-and-debugging.md`

**Step 4: Commit**

```bash
git add docs/tutorial
git commit -m "Add: scaffold PTO runtime tutorial docs"
```

### Task 2: Document the End-to-End Build and Execution Flow

**Files:**
- Modify: `docs/tutorial/01-execution-model-and-build-flow.md`
- Reference: `examples/scripts/run_example.py`
- Reference: `examples/scripts/code_runner.py`
- Reference: `python/runtime_builder.py`
- Reference: `python/runtime_compiler.py`
- Reference: `python/kernel_compiler.py`
- Reference: `python/bindings.py`

**Step 1: Trace the Python entrypoint path**

Explain, in order:
- argument parsing in `examples/scripts/run_example.py`
- how `golden.py` and `kernel_config.py` are loaded
- where platform and runtime selections are read
- how build artifacts are produced and passed into `bindings.py`

Include a diagram showing:
- `golden.py`
- `kernel_config.py`
- kernel/orchestration compilation
- runtime compilation
- host runtime initialization
- launch of AICPU/AICore binaries

**Step 2: Document platform/runtime selection points**

Explain the exact control points:
- `RUNTIME_CONFIG["runtime"]` in example `kernel_config.py`
- `RuntimeBuilder(platform=...)`
- `RuntimeCompiler(platform=...)`
- `KernelCompiler(platform=...)`
- where `a2a3` and `a2a3sim`, `a5` and `a5sim` diverge

**Step 3: Add a line-by-line reading guide for the most important files**

For each of these files, add a code-reading subsection that walks the important methods in source order:
- `python/runtime_builder.py`
- `python/runtime_compiler.py`
- `python/kernel_compiler.py`
- `python/bindings.py`

The explanation should call out why the method exists, what state it mutates, and what the next stage consumes.

**Step 4: Add runnable verification commands**

Include at least these commands:
```bash
python examples/scripts/run_example.py \
  -k examples/a2a3/host_build_graph/vector_example/kernels \
  -g examples/a2a3/host_build_graph/vector_example/golden.py \
  -p a2a3sim

python examples/scripts/run_example.py \
  -k examples/a2a3/tensormap_and_ringbuffer/paged_attention/kernels \
  -g examples/a2a3/tensormap_and_ringbuffer/paged_attention/golden.py \
  -p a2a3sim
```

Document what each command proves.

**Step 5: Commit**

```bash
git add docs/tutorial/01-execution-model-and-build-flow.md
git commit -m "Add: document PTO build and execution flow"
```

### Task 3: Document Backend Support and Abstraction Layers

**Files:**
- Modify: `docs/tutorial/02-platform-backends-and-abstraction.md`
- Reference: `src/a2a3/platform/include/`
- Reference: `src/a2a3/platform/onboard/`
- Reference: `src/a2a3/platform/sim/`
- Reference: `src/a2a3/platform/src/`
- Reference: `src/a5/platform/include/`
- Reference: `src/a5/platform/onboard/`
- Reference: `src/a5/platform/sim/`
- Reference: `src/a5/platform/src/`

**Step 1: Write the platform taxonomy**

Explain the real structure:
- chip split: `a2a3` vs `a5`
- backend split inside each chip: `onboard` vs `sim`
- role split inside each backend: `host`, `aicpu`, `aicore`
- shared helper code under `platform/src`

**Step 2: Compare real hardware and simulation backends**

Create a comparison table covering:
- compiler/toolchain differences
- binary formats produced for host/AICPU/AICore
- how kernels are loaded
- how host memory vs device memory behaves
- how simulation emulates AICPU/AICore execution

**Step 3: Add focused code walkthroughs**

Walk through the important parts of:
- `device_runner.cpp`
- `pto_runtime_c_api.cpp`
- `platform_compile_info.cpp`
- `kernel.cpp` for both AICPU and AICore

Explain what is common across chips and what is duplicated or specialized.

**Step 4: Explain abstraction boundaries**

Explicitly answer:
- what Python expects from the host runtime C API
- what a runtime expects from platform code
- what platform code expects from runtime code
- how function pointer upload and dispatch bridge host/runtime/platform layers

**Step 5: Commit**

```bash
git add docs/tutorial/02-platform-backends-and-abstraction.md
git commit -m "Add: explain platform backend abstraction layers"
```

### Task 4: Document How Runtime Is Defined and Used

**Files:**
- Modify: `docs/tutorial/03-runtime-definition-and-lifecycle.md`
- Reference: `src/a2a3/runtime/host_build_graph/build_config.py`
- Reference: `src/a2a3/runtime/aicpu_build_graph/build_config.py`
- Reference: `src/a2a3/runtime/tensormap_and_ringbuffer/build_config.py`
- Reference: `src/a5/runtime/host_build_graph/build_config.py`
- Reference: `src/a5/runtime/tensormap_and_ringbuffer/build_config.py`
- Reference: `src/a2a3/runtime/*/host/runtime_maker.cpp`
- Reference: `src/a2a3/runtime/*/runtime/runtime.h`
- Reference: `src/a2a3/platform/onboard/host/pto_runtime_c_api.cpp`

**Step 1: Define what a runtime means in this repository**

Explain:
- why a runtime is more than one C++ file
- how `build_config.py` defines the host/AICPU/AICore/orchestration source sets
- how the runtime name in `kernel_config.py` selects the implementation

**Step 2: Compare the runtime variants**

Create a comparison table for:
- `host_build_graph`
- `aicpu_build_graph`
- `tensormap_and_ringbuffer`

Cover:
- where graph construction happens
- where scheduling happens
- what data structure stores tasks
- what degree of host/device orchestration coupling exists

**Step 3: Walk the runtime lifecycle**

Document the lifecycle in order:
- `get_runtime_size`
- placement-new in `init_runtime`
- `init_runtime_impl`
- kernel registration
- orchestration loading
- `launch_runtime`
- `DeviceRunner::run`
- `finalize_runtime`
- result copy-back and cleanup

Explain which pieces are runtime-specific and which are platform-specific.

**Step 4: Add diagrams**

Add one lifecycle diagram for:
- host-built orchestration
- device-built orchestration

The diagrams should show where control transfers between host, AICPU, AICore, and shared memory.

**Step 5: Commit**

```bash
git add docs/tutorial/03-runtime-definition-and-lifecycle.md
git commit -m "Add: explain runtime definition and lifecycle"
```

### Task 5: Deep Dive Into `tensormap_and_ringbuffer`

**Files:**
- Modify: `docs/tutorial/04-tensormap-and-ringbuffer-deep-dive.md`
- Reference: `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_runtime2.h`
- Reference: `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_shared_memory.h`
- Reference: `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_ring_buffer.h`
- Reference: `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_scheduler.h`
- Reference: `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_scheduler.cpp`
- Reference: `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.h`
- Reference: `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp`
- Reference: `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_tensormap.h`
- Reference: `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_tensormap.cpp`
- Reference: `src/a2a3/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp`
- Reference: `src/a2a3/runtime/tensormap_and_ringbuffer/aicore/aicore_executor.cpp`
- Reference: `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/runtime.h`
- Reference: `src/a2a3/runtime/tensormap_and_ringbuffer/host/runtime_maker.cpp`

**Step 1: Write the runtime overview**

Explain the core objects and their ownership:
- `Runtime`
- `PTO2Runtime`
- `PTO2SharedMemoryHandle`
- orchestrator state
- scheduler state
- dispatch payloads
- handshake buffers

**Step 2: Add line-by-line walkthroughs for the critical headers and source files**

For each important file, explain the important declarations or functions in source order:
- shared memory layout
- task ring / heap ring allocation and reclamation
- ready queue design
- TensorMap lookup/insert/cleanup
- scope stack behavior
- mixed-task completion logic
- AICPU scheduler loop
- AICore worker execution loop

Do not summarize only at the class level; explain the hot-path functions and the rationale for their atomics, state transitions, and memory-order choices.

**Step 3: Add visual explanations for scheduling and synchronization**

Include diagrams for:
- orchestrator thread vs scheduler threads
- scheduler threads vs AIC/AIV workers
- task submission, dependency discovery, ready-queue insertion, dispatch, completion, reclamation
- deadlock/back-pressure case when the ring is too small

At minimum include:
- one swimlane diagram
- one shared-memory layout diagram
- one state-machine diagram for task state transitions

**Step 4: Explain why the mechanisms exist**

For each mechanism, include a short “why this exists” subsection:
- task ring back-pressure
- heap ring reclamation
- TensorMap stale-entry cleanup
- per-task fanout lock
- local-ready-buffer optimization
- sharded ready queues
- orchestrator/scheduler split

**Step 5: Tie the runtime to a real example**

Walk through `examples/a2a3/tensormap_and_ringbuffer/paged_attention` and map:
- each kernel `func_id`
- orchestration entry
- worker types
- `aicpu_thread_num`, `orch_thread_num`, and `block_dim`
- how the mixed AIC/AIV graph becomes scheduler-ready tasks

**Step 6: Commit**

```bash
git add docs/tutorial/04-tensormap-and-ringbuffer-deep-dive.md
git commit -m "Add: deep dive into tensormap and ringbuffer runtime"
```

### Task 6: Add Profiling, Examples, and Debugging Guidance

**Files:**
- Modify: `docs/tutorial/05-profiling-examples-and-debugging.md`
- Reference: `.claude/commands/profile.md`
- Reference: `src/a2a3/runtime/tensormap_and_ringbuffer/docs/RUNTIME_LOGIC.md`
- Reference: `src/a2a3/runtime/tensormap_and_ringbuffer/docs/device_log_profiling.md`
- Reference: `src/a2a3/runtime/tensormap_and_ringbuffer/docs/profiling_levels.md`
- Reference: `examples/a2a3/tensormap_and_ringbuffer/paged_attention/`
- Reference: `examples/a2a3/tensormap_and_ringbuffer/vector_example/`
- Reference: `examples/a2a3/tensormap_and_ringbuffer/batch_paged_attention/`
- Reference: `tests/device_tests/a2a3/tensormap_and_ringbuffer/`

**Step 1: Document the example progression**

Recommend a reading/running order:
1. `vector_example`
2. `mixed_example`
3. `paged_attention`
4. `batch_paged_attention`

For each example, explain what it teaches about the runtime.

**Step 2: Document profiling workflows**

Include:
- `--enable-profiling`
- where `swimlane.json` is emitted
- where device logs are written
- how to extract orchestrator and scheduler summaries
- how compile-time profiling levels differ from runtime `enable_profiling`

**Step 3: Add a debugging checklist**

Cover:
- stale path confusion between old docs and real source layout
- wrong platform/runtime pairing
- missing kernel registration
- ring-size deadlocks
- scheduler starvation
- device-log inspection

**Step 4: Add final verification instructions**

Document which commands should be run before merging the tutorial PR:
```bash
git diff --check

python examples/scripts/run_example.py \
  -k examples/a2a3/host_build_graph/vector_example/kernels \
  -g examples/a2a3/host_build_graph/vector_example/golden.py \
  -p a2a3sim
```

If simulation for `tensormap_and_ringbuffer` is not currently supported or does not pass, explicitly document that limit instead of guessing.

**Step 5: Commit**

```bash
git add docs/tutorial/05-profiling-examples-and-debugging.md
git commit -m "Add: document profiling and debugging workflows"
```

### Task 7: Final Editorial Pass and PR Update

**Files:**
- Modify: `docs/tutorial/README.md`
- Modify: `docs/tutorial/01-execution-model-and-build-flow.md`
- Modify: `docs/tutorial/02-platform-backends-and-abstraction.md`
- Modify: `docs/tutorial/03-runtime-definition-and-lifecycle.md`
- Modify: `docs/tutorial/04-tensormap-and-ringbuffer-deep-dive.md`
- Modify: `docs/tutorial/05-profiling-examples-and-debugging.md`

**Step 1: Check coverage against the requested scope**

Verify the tutorial explicitly covers:
- overall architecture and execution flow
- backend support/abstraction in `src/a2a3/platform` and `src/a5/platform`
- runtime definition and usage
- `tensormap_and_ringbuffer` internals
- scheduling and synchronization diagrams
- profiling and examples

**Step 2: Run documentation verification**

Run:
```bash
git diff --check
rg -n "^## " docs/tutorial
```

Expected:
- no whitespace errors
- all chapters contain their required major sections

**Step 3: Review for stale paths and unsupported claims**

Manually check that:
- paths refer to `src/a2a3/...` and `src/a5/...` where appropriate
- claims about profiling or example support are backed by code or successful commands
- unsupported or hardware-only steps are labeled clearly

**Step 4: Squash docs into a clean PR-ready commit**

```bash
git add docs/tutorial
git commit -m "Add: write deep PTO runtime tutorial"
```

**Step 5: Push and update PR**

```bash
git push origin tutorial
gh pr create --base main --head tutorial
```

Use a PR body that highlights:
- tutorial scope
- source directories covered
- which walkthroughs are line-by-line
- what commands were run for verification

Plan complete and saved to `docs/plans/2026-03-16-pto-runtime-deep-tutorial.md`. Two execution options:

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?
