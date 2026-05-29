# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for CallConfig and ChipWorker state machine."""

import subprocess
import textwrap

import pytest
from _task_interface import CallConfig, _ChipWorker  # pyright: ignore[reportMissingImports]

# ============================================================================
# CallConfig tests
# ============================================================================


class TestCallConfig:
    def test_defaults(self):
        config = CallConfig()
        assert config.block_dim == 24
        assert config.aicpu_thread_num == 3
        assert config.enable_l2_swimlane == 0
        assert config.enable_dump_tensor is False
        assert config.enable_pmu == 0
        assert config.enable_dep_gen is False

    def test_setters(self):
        # enable_l2_swimlane accepts both an int perf_level (0-4) and a Python
        # bool. `True` maps to level 4 (preserves the pre-perf_level "fully on"
        # semantics for legacy callers); explicit ints select a specific level.
        config = CallConfig()
        config.block_dim = 32
        config.aicpu_thread_num = 4
        config.enable_l2_swimlane = True
        assert config.block_dim == 32
        assert config.aicpu_thread_num == 4
        assert config.enable_l2_swimlane == 4
        config.enable_l2_swimlane = 2
        assert config.enable_l2_swimlane == 2
        config.enable_l2_swimlane = False
        assert config.enable_l2_swimlane == 0

    def test_diagnostics_subfeatures_are_parallel(self):
        # Guard against drift: the four diagnostics sub-features under the
        # profiling umbrella must all round-trip through the nanobind surface.
        config = CallConfig()
        config.enable_l2_swimlane = True
        config.enable_dump_tensor = True
        config.enable_pmu = 2
        config.enable_dep_gen = True
        assert config.enable_l2_swimlane == 4
        assert config.enable_dump_tensor is True
        assert config.enable_pmu == 2
        assert config.enable_dep_gen is True
        r = repr(config)
        assert "enable_l2_swimlane=4" in r
        assert "enable_dump_tensor=True" in r
        assert "enable_pmu=2" in r
        assert "enable_dep_gen=True" in r

    def test_repr(self):
        config = CallConfig()
        r = repr(config)
        assert "block_dim=24" in r
        assert "enable_l2_swimlane=0" in r


# ============================================================================
# ChipWorker state machine tests
# ============================================================================


class TestChipWorkerStateMachine:
    def test_initial_state(self):
        worker = _ChipWorker()
        assert worker.initialized is False
        assert worker.device_id == -1

    def test_finalize_idempotent(self):
        worker = _ChipWorker()
        worker.finalize()
        worker.finalize()
        assert worker.initialized is False

    def test_init_after_finalize_raises(self):
        worker = _ChipWorker()
        worker.finalize()
        with pytest.raises(RuntimeError, match="finalized"):
            worker.init("/nonexistent/libfoo.so", "/dev/null", "/dev/null", device_id=0)

    def test_init_with_nonexistent_lib_raises(self):
        worker = _ChipWorker()
        with pytest.raises(RuntimeError, match="dlopen"):
            worker.init("/nonexistent/libfoo.so", "/dev/null", "/dev/null", device_id=0)

    def test_init_with_negative_device_id_raises(self):
        worker = _ChipWorker()
        with pytest.raises(RuntimeError, match="device_id"):
            worker.init("/nonexistent/libfoo.so", "/dev/null", "/dev/null", -1)

    def test_prepare_callable_before_init_raises(self):
        from _task_interface import ChipCallable  # noqa: PLC0415

        worker = _ChipWorker()
        callable_obj = ChipCallable.build(signature=[], func_name="test", binary=b"\x00", children=[])
        with pytest.raises(RuntimeError, match="not initialized"):
            worker.prepare_callable(0, callable_obj)

    def test_prepare_callable_from_blob_before_init_raises(self):
        # The from_blob overload shares the underlying ChipWorker::prepare_callable
        # entrypoint with the typed overload, so it must enforce the same
        # initialization guard. This protects the dynamic-register IPC handler
        # (which is the sole caller) from silently no-op'ing on a stale worker.
        from _task_interface import ChipCallable  # noqa: PLC0415

        worker = _ChipWorker()
        callable_obj = ChipCallable.build(signature=[], func_name="test", binary=b"\x00", children=[])
        with pytest.raises(RuntimeError, match="not initialized"):
            worker.prepare_callable_from_blob(0, callable_obj.buffer_ptr())

    def test_run_before_init_raises(self):
        from _task_interface import ChipStorageTaskArgs  # noqa: PLC0415

        worker = _ChipWorker()
        config = CallConfig()
        args = ChipStorageTaskArgs()
        with pytest.raises(RuntimeError, match="not initialized"):
            worker.run(0, args, config)

    def test_unregister_callable_before_init_raises(self):
        worker = _ChipWorker()
        with pytest.raises(RuntimeError, match="not initialized"):
            worker.unregister_callable(0)


# ============================================================================
# Python-level ChipWorker wrapper tests
# ============================================================================


class TestChipWorkerPython:
    def test_import(self):
        from simpler.task_interface import (  # noqa: PLC0415
            CallConfig as PyCallConfig,  # pyright: ignore[reportAttributeAccessIssue]
        )
        from simpler.task_interface import ChipWorker  # noqa: PLC0415  # pyright: ignore[reportAttributeAccessIssue]

        worker = ChipWorker()
        assert worker.initialized is False
        assert isinstance(PyCallConfig(), CallConfig)

    def test_cpp_chip_worker_exposes_role_keyed_init(self):
        worker = _ChipWorker()
        assert hasattr(worker, "init_roles")

    def test_cpp_chip_worker_prefers_role_keyed_runtime_init(self, tmp_path, monkeypatch):
        source = tmp_path / "fake_role_runtime.c"
        marker = tmp_path / "role-init.txt"
        device_bin = tmp_path / "device.bin"
        scheduler_bin = tmp_path / "scheduler.bin"
        host_lib = tmp_path / "libfake_role_runtime.so"
        device_bin.write_bytes(b"device-image")
        scheduler_bin.write_bytes(b"scheduler-image")
        source.write_text(
            textwrap.dedent(
                """\
                #include <stdint.h>
                #include <stdio.h>
                #include <stdlib.h>
                #include <string.h>

                typedef void *DeviceContextHandle;
                typedef void *RuntimeHandle;

                typedef struct PtoRunTiming {
                    uint64_t host_wall_ns;
                    uint64_t device_wall_ns;
                } PtoRunTiming;

                typedef struct PtoRuntimeBinaryRole {
                    const char *role;
                    const uint8_t *binary;
                    size_t size;
                } PtoRuntimeBinaryRole;

                typedef struct PtoRuntimeBinaryMap {
                    const PtoRuntimeBinaryRole *entries;
                    size_t count;
                } PtoRuntimeBinaryMap;

                DeviceContextHandle create_device_context(void) { return malloc(1); }
                void destroy_device_context(DeviceContextHandle ctx) { free(ctx); }
                size_t get_runtime_size(void) { return 1; }
                void *device_malloc_ctx(DeviceContextHandle ctx, size_t size) { (void)ctx; return malloc(size); }
                void device_free_ctx(DeviceContextHandle ctx, void *dev_ptr) { (void)ctx; free(dev_ptr); }
                int copy_to_device_ctx(DeviceContextHandle ctx, void *dev_ptr, const void *host_ptr, size_t size) {
                    (void)ctx;
                    memcpy(dev_ptr, host_ptr, size);
                    return 0;
                }
                int copy_from_device_ctx(DeviceContextHandle ctx, void *host_ptr, const void *dev_ptr, size_t size) {
                    (void)ctx;
                    memcpy(host_ptr, dev_ptr, size);
                    return 0;
                }
                int simpler_init(
                    DeviceContextHandle ctx, int device_id, const uint8_t *aicpu_binary, size_t aicpu_size,
                    const uint8_t *aicore_binary, size_t aicore_size
                ) {
                    (void)ctx;
                    (void)device_id;
                    (void)aicpu_binary;
                    (void)aicpu_size;
                    (void)aicore_binary;
                    (void)aicore_size;
                    return -77;
                }
                int simpler_init_roles(DeviceContextHandle ctx, int device_id, const PtoRuntimeBinaryMap *binaries) {
                    (void)ctx;
                    int saw_device = 0;
                    int saw_scheduler = 0;
                    int saw_host = 0;
                    size_t device_size = 0;
                    size_t scheduler_size = 0;
                    for (size_t i = 0; binaries != NULL && i < binaries->count; ++i) {
                        const PtoRuntimeBinaryRole *entry = &binaries->entries[i];
                        if (strcmp(entry->role, "device") == 0) {
                            saw_device = entry->binary != NULL;
                            device_size = entry->size;
                        } else if (strcmp(entry->role, "scheduler") == 0) {
                            saw_scheduler = entry->binary != NULL;
                            scheduler_size = entry->size;
                        } else if (strcmp(entry->role, "host") == 0) {
                            saw_host = 1;
                        }
                    }
                    const char *marker = getenv("PTO_FAKE_ROLE_INIT_MARKER");
                    if (marker != NULL) {
                        FILE *f = fopen(marker, "w");
                        if (f != NULL) {
                            fprintf(
                                f,
                                "device_id=%d,count=%zu,device=%d:%zu,scheduler=%d:%zu,host=%d\\n",
                                device_id,
                                binaries == NULL ? 0 : binaries->count,
                                saw_device,
                                device_size,
                                saw_scheduler,
                                scheduler_size,
                                saw_host
                            );
                            fclose(f);
                        }
                    }
                    return saw_device && saw_scheduler && !saw_host ? 0 : -1;
                }
                int finalize_device(DeviceContextHandle ctx) { (void)ctx; return 0; }
                int prepare_callable(DeviceContextHandle ctx, int32_t callable_id, const void *callable) {
                    (void)ctx; (void)callable_id; (void)callable; return 0;
                }
                int run_prepared(
                    DeviceContextHandle ctx, RuntimeHandle runtime, int32_t callable_id, const void *args,
                    int block_dim, int aicpu_thread_num, int enable_l2_swimlane, int enable_dump_tensor,
                    int enable_pmu, int enable_dep_gen, const char *output_prefix, PtoRunTiming *out_timing
                ) {
                    (void)ctx; (void)runtime; (void)callable_id; (void)args; (void)block_dim;
                    (void)aicpu_thread_num; (void)enable_l2_swimlane; (void)enable_dump_tensor;
                    (void)enable_pmu; (void)enable_dep_gen; (void)output_prefix; (void)out_timing; return 0;
                }
                int unregister_callable(DeviceContextHandle ctx, int32_t callable_id) {
                    (void)ctx; (void)callable_id; return 0;
                }
                size_t get_aicpu_dlopen_count(DeviceContextHandle ctx) { (void)ctx; return 0; }
                size_t get_host_dlopen_count(DeviceContextHandle ctx) { (void)ctx; return 0; }
                int ensure_acl_ready_ctx(DeviceContextHandle ctx, int device_id) {
                    (void)ctx; (void)device_id; return 0;
                }
                void *create_comm_stream_ctx(DeviceContextHandle ctx) { (void)ctx; return NULL; }
                int destroy_comm_stream_ctx(DeviceContextHandle ctx, void *stream) {
                    (void)ctx; (void)stream; return 0;
                }
                void *comm_init(int rank, int nranks, void *stream, const char *rootinfo_path) {
                    (void)rank; (void)nranks; (void)stream; (void)rootinfo_path; return NULL;
                }
                int comm_alloc_windows(void *handle, size_t win_size, uint64_t *device_ctx) {
                    (void)handle; (void)win_size; if (device_ctx != NULL) *device_ctx = 0; return 0;
                }
                int comm_get_local_window_base(void *handle, uint64_t *base) {
                    (void)handle; if (base != NULL) *base = 0; return 0;
                }
                int comm_get_window_size(void *handle, size_t *win_size) {
                    (void)handle; if (win_size != NULL) *win_size = 0; return 0;
                }
                int comm_derive_context(
                    void *handle, const uint32_t *rank_ids, size_t rank_count, uint32_t domain_rank,
                    size_t window_offset, size_t window_size, uint64_t *device_ctx
                ) {
                    (void)handle; (void)rank_ids; (void)rank_count; (void)domain_rank;
                    (void)window_offset; (void)window_size; if (device_ctx != NULL) *device_ctx = 0; return 0;
                }
                int comm_alloc_domain_windows(
                    void *handle, uint64_t allocation_id, const uint32_t *rank_ids, size_t rank_count,
                    uint32_t domain_rank, size_t window_size, uint64_t *device_ctx, uint64_t *local_window_base
                ) {
                    (void)handle; (void)allocation_id; (void)rank_ids; (void)rank_count;
                    (void)domain_rank; (void)window_size;
                    if (device_ctx != NULL) *device_ctx = 0;
                    if (local_window_base != NULL) *local_window_base = 0;
                    return 0;
                }
                int comm_release_domain_windows(
                    void *handle, uint64_t allocation_id, size_t rank_count, uint32_t domain_rank
                ) {
                    (void)handle; (void)allocation_id; (void)rank_count; (void)domain_rank; return 0;
                }
                int comm_barrier(void *handle) { (void)handle; return 0; }
                int comm_destroy(void *handle) { (void)handle; return 0; }
                """
            )
        )

        result = subprocess.run(
            ["cc", "-shared", "-fPIC", "-o", str(host_lib), str(source)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"

        monkeypatch.setenv("PTO_FAKE_ROLE_INIT_MARKER", str(marker))

        worker = _ChipWorker()
        worker.init_roles({"host": str(host_lib), "device": str(device_bin), "scheduler": str(scheduler_bin)}, 7)
        worker.finalize()

        assert marker.read_text() == (
            f"device_id=7,count=2,device=1:{device_bin.stat().st_size},"
            f"scheduler=1:{scheduler_bin.stat().st_size},host=0\n"
        )

    def test_init_accepts_cuda_role_only_runtime_binaries(self, monkeypatch, tmp_path):
        from simpler import task_interface as task_interface_module  # noqa: PLC0415
        from simpler.task_interface import ChipWorker  # noqa: PLC0415  # pyright: ignore[reportAttributeAccessIssue]

        class FakeLogInit:
            def __init__(self):
                self.calls = []

            def __call__(self, log_level, log_info_v):
                self.calls.append((log_level, log_info_v))
                return 0

        class FakeLogHandle:
            def __init__(self):
                self.simpler_log_init = FakeLogInit()

        class FakeImpl:
            def __init__(self):
                self.init_args = None
                self.init_roles_args = None

            def init(self, *args):
                self.init_args = args

            def init_roles(self, *args):
                self.init_roles_args = args

        class RoleOnlyBins:
            simpler_log_path = tmp_path / "libsimpler_log.so"
            sim_context_path = None

            def __init__(self):
                self.role_paths = {
                    "host": tmp_path / "libhost_runtime.so",
                    "device": tmp_path / "libcuda_device_runtime.so",
                }

            def path_for_role(self, role):
                return self.role_paths[role]

        fake_log_handle = FakeLogHandle()
        monkeypatch.setattr(task_interface_module, "_preload_global", lambda path: fake_log_handle)

        worker = ChipWorker()
        fake_impl = FakeImpl()
        worker._impl = fake_impl

        worker.init(0, RoleOnlyBins(), log_level=1, log_info_v=2)

        device_path = str(tmp_path / "libcuda_device_runtime.so")
        assert fake_log_handle.simpler_log_init.calls == [(1, 2)]
        assert fake_impl.init_args is None
        assert fake_impl.init_roles_args == (
            {
                "host": str(tmp_path / "libhost_runtime.so"),
                "device": device_path,
            },
            0,
        )
