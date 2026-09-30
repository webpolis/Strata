// include/strata/core/gpu_tier.hpp - extra GPUs as more expert tiers.
//
// The main card's cache holds the most-routed experts; each extra card holds the ones ranked next, which the CPU
// would otherwise compute.  Per layer of a verify window the host writes the window's activations into a pinned
// block the card reads directly (zero-copy; STRATA_TIER_ZEROCOPY=0 copies them instead), the card computes them
// with the main card's own kernels (quantize_q8_1_rows + native_expert_grouped) and writes the rows into a pinned
// block the host reads back into the CPU's mapped rows before the layer's flag is raised - so the main card's graph
// is unchanged.  Native packs only.
#pragma once

#include "strata/core/expert_cache.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {

class ExpertSource;

class GpuTier {
public:
    GpuTier() = default;
    ~GpuTier();
    GpuTier(const GpuTier&) = delete;
    GpuTier& operator=(const GpuTier&) = delete;

    /// Streams and buffers on CUDA device `device` for up to `max_tok` tokens and `cap` routed entries per layer.
    /// `home` is the engine's device; it is current again whenever a call returns.
    bool open(int device, int home, int64_t n_layers, int64_t n_expert, int64_t n_embd, int64_t max_tok, int64_t cap,
              std::string& err);
    /// Slots for the ranked pairs from `ranked[first]` on, sized per layer, as many as fit in the card's free VRAM
    /// minus `reserve_mib`, filled from `src`.
    bool fill(const std::vector<std::pair<int32_t, int32_t>>& ranked, size_t first, ExpertSource& src, int reserve_mib,
              std::string& err);
    void close();

    int device() const { return dev_; }
    int64_t slots() const { return cache_.slots(); }
    int64_t bytes() const { return cache_.bytes(); }
    bool zero_copy() const { return zero_copy_; }
    int32_t slot_of(int64_t layer, int32_t e) const { return res_[(size_t) (layer * n_expert_ + e)]; }
    /// Resident here, or on its way in (an adaptive swap still copying).
    bool holds(int64_t layer, int32_t e) const { return holds_at((size_t) (layer * n_expert_ + e)); }
    bool holds_at(size_t i) const { return res_[i] >= 0 || incoming_[i]; }
    /// The expert at residency index `i` leaves this card (the main card's cache took it): its slot is refilled by
    /// the next `adapt`.  Between verify windows only.
    void release(size_t i);

    // ---- one layer of a verify window, driven by the pool dispatch
    void begin(int64_t layer);
    void add_group(int32_t expert);
    void add_entry(int32_t routed, int32_t tok);
    /// Publishes the plan and `n_tok` activation rows (host) to the card and starts its groups.  Returns at once;
    /// does nothing when the layer routed nothing here.
    bool launch(const float* x, int64_t n_tok, std::string& err);
    /// Waits for the card and writes each entry's row into `out` at its routed index.
    bool finish(float* out, std::string& err);

    // ---- the adaptive tier: the main card's rule, over the experts no card holds
    /// Fills this card's freed slots and swaps its least-routed experts for the most-routed ones not in `held`
    /// (resident or arriving on any card), up to `max_swaps` copies and `max_bytes` in all, and marks the new ones
    /// in `held`.  Copies start on the card's own stream; the evicted experts leave at once, the new ones arrive in
    /// `apply`.  Returns the bytes the copies move, -1 on failure.
    int64_t adapt(const float* usage, std::vector<uint8_t>& held, ExpertSource& src, int max_swaps, int64_t max_bytes,
                  std::string& err);
    /// Admits the swapped-in experts once their copies have landed (`wait`: block until they have).
    bool apply(bool wait, std::string& err);

    int64_t entries = 0;        ///< routed entries this card computed
    int64_t launches = 0;       ///< layers this card took part in
    double ms_launch = 0;       ///< host time spent staging and starting the card's work
    double ms_wait = 0;         ///< host time spent waiting for the card after the CPU's share
    uint64_t refill_bytes = 0;  ///< bytes the adaptive swaps copied into this card

private:
    int dev_ = -1, home_ = 0;
    int64_t n_layers_ = 0, n_expert_ = 0, n_embd_ = 0, max_tok_ = 0, cap_ = 0;
    ExpertCache cache_;
    std::vector<int32_t> res_;
    std::vector<uint8_t> incoming_;
    std::vector<std::vector<int32_t>> free_;             ///< per layer: released slots (sized for that layer)
    std::vector<std::pair<int32_t, int32_t>> pending_;   ///< (residency index, slot) copying in
    void* stream_ = nullptr;
    void* adapt_stream_ = nullptr;
    void* done_ = nullptr;
    void* adapt_ev_ = nullptr;
    // The plan and the activations, one block on the host (pinned, mapped) and its mirror on the card:
    // counts(4) | start(cap+1) | dst(cap) | tok(cap) | pad | ptr(cap u64) | x(max_tok * n_embd f32)
    uint8_t* h_in_ = nullptr;
    uint8_t* d_in_ = nullptr;
    uint8_t* z_in_ = nullptr;       ///< h_in_ as the card sees it (zero-copy: the activations are read in place)
    size_t off_start_ = 0, off_dst_ = 0, off_tok_ = 0, off_ptr_ = 0, off_x_ = 0;
    uint8_t* d_xq_ = nullptr;
    void* d_scratch_ = nullptr;
    float* d_out_ = nullptr;
    float* h_out_ = nullptr;
    float* z_out_ = nullptr;        ///< h_out_ as the card sees it (zero-copy: the rows are written in place)
    bool zero_copy_ = false;
    std::vector<int32_t> row_of_;   ///< compact row -> routed index
    int64_t layer_ = 0;
    int ng_ = 0, ne_ = 0;
    bool launched_ = false;
};

/// The first tier holding `(layer, e)` resident, or -1.
int tier_of(const std::vector<GpuTier*>& tiers, int64_t layer, int32_t e);
/// Resident or arriving on any tier (the main card's adaptive candidates skip these).
bool held_by_tier(const std::vector<GpuTier*>& tiers, int64_t layer, int32_t e);
/// Every tier's adaptive swaps for one round, after the main card's (`main_res`, `main_incoming`): no expert ends
/// up cached twice, and all the tiers' copies together stay within `budget_bytes`.  False on a failed copy.
bool adapt_tiers(const std::vector<GpuTier*>& tiers, const float* usage, const std::vector<int32_t>& main_res,
                 const std::vector<std::pair<int32_t, int32_t>>& main_incoming, ExpertSource& src, int max_swaps,
                 int64_t budget_bytes, std::string& err);

}  // namespace strata::core
