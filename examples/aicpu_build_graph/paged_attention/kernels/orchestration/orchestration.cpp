/**
 * Paged Attention orchestration for the aicpu_build_graph runtime.
 *
 * This function runs on AICPU (device-side) and builds the task graph using the
 * AICPU build API table: `runtime->aicpu_build_api`.
 *
 * The host runtime (`init_runtime_impl`) auto-manages I/O tensor device memory
 * and populates `runtime->orch_args[]` using the golden's `TENSOR_ORDER`:
 *
 *   orch_args[0..6]   = device pointers for: query, key_cache, value_cache,
 *                       block_table, context_lens, out, config
 *   orch_args[7..13]  = byte sizes for each tensor (as scalars)
 *   orch_args[14]     = element count of the first tensor (query), as scalar
 *
 * The orchestration allocates intermediate buffers in HBM via
 * `api.device_malloc()` and builds the dependency graph:
 *
 *   per block bn:
 *     QK(bn) -> SF(bn) -> PV(bn) -> UP(bn)
 *   plus serialized online-update chain:
 *     UP(0) -> UP(1) -> ... -> UP(bn_this_batch-1)
 *
 * Kernel binding:
 *   Pass `function_bin_addr=0` to `api.add_task(...)` to use the runtime's
 *   `func_id -> runtime->kernel_addrs[]` mapping.
 */

#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "runtime.h"

namespace {
constexpr int FUNC_QK_MATMUL = 0;
constexpr int FUNC_SOFTMAX_PREPARE = 1;
constexpr int FUNC_PV_MATMUL = 2;
constexpr int FUNC_ONLINE_UPDATE = 3;

constexpr int kTileSize = 16;
constexpr int kTileElements = kTileSize * kTileSize;  // 256

constexpr size_t kTileBytes = static_cast<size_t>(kTileElements) * sizeof(float);  // 1024
constexpr size_t kScalarBytes = static_cast<size_t>(kTileSize) * sizeof(float);    // 64

constexpr int ARG_QUERY = 0;
constexpr int ARG_KEY_CACHE = 1;
constexpr int ARG_VALUE_CACHE = 2;
constexpr int ARG_BLOCK_TABLE = 3;
constexpr int ARG_CONTEXT_LENS = 4;
constexpr int ARG_OUT = 5;
constexpr int ARG_CONFIG = 6;

constexpr int ORCH_ARGC_EXPECTED = 15;

static inline size_t align_up(size_t v, size_t a) { return (v + (a - 1)) & ~(a - 1); }
}  // namespace

extern "C" int orchestration(Runtime* runtime) {
    if (runtime == nullptr) {
        return -1;
    }

    if (runtime->orch_argc < ORCH_ARGC_EXPECTED) {
        std::fprintf(
            stderr, "paged_attention: expected orch_argc >= %d, got %d\n", ORCH_ARGC_EXPECTED, runtime->orch_argc);
        return -1;
    }

    const uint64_t dev_query_u64 = runtime->orch_args[ARG_QUERY];
    const uint64_t dev_key_u64 = runtime->orch_args[ARG_KEY_CACHE];
    const uint64_t dev_val_u64 = runtime->orch_args[ARG_VALUE_CACHE];
    const uint64_t dev_block_table_u64 = runtime->orch_args[ARG_BLOCK_TABLE];
    const uint64_t dev_context_lens_u64 = runtime->orch_args[ARG_CONTEXT_LENS];
    const uint64_t dev_out_u64 = runtime->orch_args[ARG_OUT];
    const uint64_t dev_config_u64 = runtime->orch_args[ARG_CONFIG];

    if (dev_query_u64 == 0 || dev_key_u64 == 0 || dev_val_u64 == 0 || dev_block_table_u64 == 0 ||
        dev_context_lens_u64 == 0 || dev_out_u64 == 0 || dev_config_u64 == 0) {
        std::fprintf(stderr, "paged_attention: got null device pointer in orch_args\n");
        return -1;
    }

    const auto* block_table = reinterpret_cast<const int32_t*>(dev_block_table_u64);
    const auto* context_lens = reinterpret_cast<const int32_t*>(dev_context_lens_u64);
    const auto* config = reinterpret_cast<const int64_t*>(dev_config_u64);

    const int batch = static_cast<int>(config[0]);
    const int num_heads = static_cast<int>(config[1]);
    const int head_dim = static_cast<int>(config[3]);
    const int block_size = static_cast<int>(config[4]);
    const int block_num = static_cast<int>(config[5]);
    const uint64_t scale_value_bits = static_cast<uint64_t>(config[6]);

    // This example is fixed to the 16x16 framework-kernel variant.
    if (num_heads != kTileSize || head_dim != kTileSize || block_size != kTileSize) {
        std::fprintf(stderr,
            "paged_attention: unsupported config (num_heads=%d, head_dim=%d, block_size=%d), expected 16x16\n",
            num_heads,
            head_dim,
            block_size);
        return -1;
    }
    if (batch <= 0 || block_num <= 0) {
        std::fprintf(stderr, "paged_attention: invalid config (batch=%d, block_num=%d)\n", batch, block_num);
        return -1;
    }

    const AicpuBuildApi& api = runtime->aicpu_build_api;
    if (api.add_task == nullptr || api.add_successor_conditional == nullptr || api.publish_task == nullptr ||
        api.device_malloc == nullptr) {
        std::fprintf(stderr, "paged_attention: missing aicpu_build_api entry points\n");
        return -1;
    }

    // Allocate intermediates as contiguous HBM slabs for simple indexing.
    const int total_buffers = batch * block_num;

    const size_t tile_stride = align_up(kTileBytes, 32);
    const size_t scalar_stride = align_up(kScalarBytes, 32);

    uint8_t* sij_base = static_cast<uint8_t*>(api.device_malloc(tile_stride * static_cast<size_t>(total_buffers)));
    uint8_t* pij_base = static_cast<uint8_t*>(api.device_malloc(tile_stride * static_cast<size_t>(total_buffers)));
    uint8_t* oi_new_base = static_cast<uint8_t*>(api.device_malloc(tile_stride * static_cast<size_t>(total_buffers)));
    uint8_t* mij_base = static_cast<uint8_t*>(api.device_malloc(scalar_stride * static_cast<size_t>(total_buffers)));
    uint8_t* lij_base = static_cast<uint8_t*>(api.device_malloc(scalar_stride * static_cast<size_t>(total_buffers)));

    uint8_t* mi_base = static_cast<uint8_t*>(api.device_malloc(scalar_stride * static_cast<size_t>(batch)));
    uint8_t* li_base = static_cast<uint8_t*>(api.device_malloc(scalar_stride * static_cast<size_t>(batch)));
    uint8_t* oi_base = static_cast<uint8_t*>(api.device_malloc(tile_stride * static_cast<size_t>(batch)));

    if (sij_base == nullptr || pij_base == nullptr || oi_new_base == nullptr || mij_base == nullptr ||
        lij_base == nullptr || mi_base == nullptr || li_base == nullptr || oi_base == nullptr) {
        std::fprintf(stderr, "paged_attention: device_malloc failed for intermediates\n");
        return -1;
    }

    auto buf_tile_ptr = [&](uint8_t* base, int buf_idx) -> uint64_t {
        return reinterpret_cast<uint64_t>(base + tile_stride * static_cast<size_t>(buf_idx));
    };
    auto buf_scalar_ptr = [&](uint8_t* base, int buf_idx) -> uint64_t {
        return reinterpret_cast<uint64_t>(base + scalar_stride * static_cast<size_t>(buf_idx));
    };
    auto batch_tile_ptr = [&](uint8_t* base, int b) -> uint64_t {
        return reinterpret_cast<uint64_t>(base + tile_stride * static_cast<size_t>(b));
    };
    auto batch_scalar_ptr = [&](uint8_t* base, int b) -> uint64_t {
        return reinterpret_cast<uint64_t>(base + scalar_stride * static_cast<size_t>(b));
    };

    // Base tensors.
    const uint64_t dev_query = dev_query_u64;
    const uint64_t dev_key_cache = dev_key_u64;
    const uint64_t dev_value_cache = dev_val_u64;
    const uint64_t dev_out = dev_out_u64;

    for (int b_idx = 0; b_idx < batch; ++b_idx) {
        const int cur_seq = context_lens[b_idx];
        int bn_this_batch = (cur_seq + block_size - 1) / block_size;
        if (bn_this_batch < 0) bn_this_batch = 0;
        if (bn_this_batch > block_num) bn_this_batch = block_num;

        const uint64_t qi_ptr = dev_query + static_cast<uint64_t>(b_idx) * kTileBytes;
        const uint64_t out_ptr = dev_out + static_cast<uint64_t>(b_idx) * kTileBytes;

        const uint64_t dev_mi = batch_scalar_ptr(mi_base, b_idx);
        const uint64_t dev_li = batch_scalar_ptr(li_base, b_idx);
        const uint64_t dev_oi = batch_tile_ptr(oi_base, b_idx);

        int t_up_prev = -1;

        for (int bn = 0; bn < bn_this_batch; ++bn) {
            const int cur_block_idx = block_table[b_idx * block_num + bn];
            const uint64_t kj_ptr = dev_key_cache + static_cast<uint64_t>(cur_block_idx) * kTileBytes;
            const uint64_t vj_ptr = dev_value_cache + static_cast<uint64_t>(cur_block_idx) * kTileBytes;

            const int buf_idx = b_idx * block_num + bn;
            const uint64_t dev_sij = buf_tile_ptr(sij_base, buf_idx);
            const uint64_t dev_pij = buf_tile_ptr(pij_base, buf_idx);
            const uint64_t dev_oi_new = buf_tile_ptr(oi_new_base, buf_idx);
            const uint64_t dev_mij = buf_scalar_ptr(mij_base, buf_idx);
            const uint64_t dev_lij = buf_scalar_ptr(lij_base, buf_idx);

            // Create tasks.
            uint64_t qk_args[3] = {qi_ptr, kj_ptr, dev_sij};
            int t_qk = api.add_task(runtime, qk_args, 3, FUNC_QK_MATMUL, CoreType::AIC, /*function_bin_addr=*/0);
            if (t_qk < 0) return -1;

            uint64_t sf_args[5] = {dev_sij, scale_value_bits, dev_pij, dev_mij, dev_lij};
            int t_sf = api.add_task(runtime, sf_args, 5, FUNC_SOFTMAX_PREPARE, CoreType::AIV, /*function_bin_addr=*/0);
            if (t_sf < 0) return -1;

            uint64_t pv_args[3] = {dev_pij, vj_ptr, dev_oi_new};
            int t_pv = api.add_task(runtime, pv_args, 3, FUNC_PV_MATMUL, CoreType::AIC, /*function_bin_addr=*/0);
            if (t_pv < 0) return -1;

            const int is_first = (bn == 0) ? 1 : 0;
            const int is_last = (bn == bn_this_batch - 1) ? 1 : 0;

            uint64_t up_args[9] = {dev_mij,
                dev_lij,
                dev_oi_new,
                dev_mi,
                dev_li,
                dev_oi,
                static_cast<uint64_t>(is_first),
                static_cast<uint64_t>(is_last),
                out_ptr};
            int t_up = api.add_task(runtime, up_args, 9, FUNC_ONLINE_UPDATE, CoreType::AIV, /*function_bin_addr=*/0);
            if (t_up < 0) return -1;

            // Add edges (safe for concurrent build||schedule).
            api.add_successor_conditional(runtime, t_qk, t_sf);
            api.add_successor_conditional(runtime, t_sf, t_pv);
            api.add_successor_conditional(runtime, t_pv, t_up);
            if (t_up_prev >= 0) {
                api.add_successor_conditional(runtime, t_up_prev, t_up);
            }

            // Publish tasks (builder order: create -> edges -> publish).
            api.publish_task(runtime, t_qk);
            api.publish_task(runtime, t_sf);
            api.publish_task(runtime, t_pv);
            api.publish_task(runtime, t_up);

            t_up_prev = t_up;
        }
    }

    return 0;
}
