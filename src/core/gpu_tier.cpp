// src/core/gpu_tier.cpp - extra GPUs as more expert tiers.  See the header.
#include "strata/core/gpu_tier.hpp"

#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstring>

namespace strata::core {
namespace {

// Makes `dev` current for a scope and puts `home` back, so every entry point leaves the engine's device current.
struct OnDevice {
    int home;
    cudaError_t status;
    OnDevice(int dev, int home_) : home(home_), status(cudaSetDevice(dev)) {}
    ~OnDevice() { if (status == cudaSuccess) cudaSetDevice(home); }
};

size_t align(size_t v, size_t a) { return (v + a - 1) / a * a; }

}  // namespace

GpuTier::~GpuTier() { close(); }

bool GpuTier::open(int device, int home, int64_t n_layers, int64_t n_expert, int64_t n_embd, int64_t max_tok,
                      int64_t cap, std::string& err) {
    close();
    if (dev_ >= 0) { err = "the previous GPU tier could not be closed"; return false; }
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (!lay.native) { err = "the second GPU tier needs a native pack (the GGUF's own expert formats)"; return false; }
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || device < 0 || device >= count || device == home) {
        err = "--extra-gpus " + std::to_string(device) + ": no such CUDA device besides the main one (" +
              std::to_string(count) + " visible)";
        return false;
    }
    dev_ = device;
    home_ = home;
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    n_embd_ = n_embd;
    max_tok_ = max_tok;
    cap_ = cap;
    int64_t n_ff = 0;
    for (const auto& f : lay.fmt) n_ff = std::max<int64_t>(n_ff, f.n_ff);
    off_start_ = 4 * sizeof(int32_t);
    off_dst_ = off_start_ + (size_t) (cap + 1) * sizeof(int32_t);
    off_tok_ = off_dst_ + (size_t) cap * sizeof(int32_t);
    off_ptr_ = align(off_tok_ + (size_t) cap * sizeof(int32_t), 8);
    off_x_ = align(off_ptr_ + (size_t) cap * sizeof(unsigned long long), 256);
    const size_t in_bytes = off_x_ + (size_t) (max_tok * n_embd) * sizeof(float);
    const size_t out_bytes = (size_t) (cap * n_embd) * sizeof(float);
    OnDevice on(dev_, home_);
    if (on.status != cudaSuccess) {
        err = "cannot select extra GPU " + std::to_string(dev_) + ": " + cudaGetErrorString(on.status);
        return false;
    }
    cudaStream_t s = nullptr, as = nullptr;
    cudaEvent_t d = nullptr, ae = nullptr;
    const bool ok = cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) == cudaSuccess &&
                    cudaStreamCreateWithFlags(&as, cudaStreamNonBlocking) == cudaSuccess &&
                    cudaEventCreateWithFlags(&d, cudaEventDisableTiming) == cudaSuccess &&
                    cudaEventCreateWithFlags(&ae, cudaEventDisableTiming) == cudaSuccess &&
                    cudaHostAlloc((void**) &h_in_, in_bytes, cudaHostAllocPortable) == cudaSuccess &&
                    cudaHostAlloc((void**) &h_out_, out_bytes, cudaHostAllocPortable) == cudaSuccess &&
                    cudaMalloc((void**) &d_in_, in_bytes) == cudaSuccess &&
                    cudaMalloc((void**) &d_out_, out_bytes) == cudaSuccess &&
                    cudaMalloc((void**) &d_xq_, (size_t) max_tok * (size_t) (n_embd / 32) * 36) == cudaSuccess &&
                    cudaMalloc(&d_scratch_, strata::kernels::native_expert_scratch_bytes(cap, n_ff)) == cudaSuccess;
    stream_ = s;
    adapt_stream_ = as;
    done_ = d;
    adapt_ev_ = ae;
    if (!ok) {
        err = std::string("the second GPU's buffers: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    std::memset(h_in_, 0, in_bytes);
    res_.assign((size_t) (n_layers * n_expert), kNotResident);
    incoming_.assign(res_.size(), 0);
    row_of_.assign((size_t) cap, 0);
    return true;
}

bool GpuTier::fill(const std::vector<std::pair<int32_t, int32_t>>& ranked, size_t first, ExpertSource& src,
                      int reserve_mib, std::string& err) {
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (first >= ranked.size()) { err = "every ranked expert is already cached"; return false; }
    OnDevice on(dev_, home_);
    if (on.status != cudaSuccess) {
        err = "cannot select extra GPU " + std::to_string(dev_) + ": " + cudaGetErrorString(on.status);
        return false;
    }
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) { err = "cudaMemGetInfo failed on the second GPU"; return false; }
    const uint64_t room = free_b > ((size_t) reserve_mib << 20) ? free_b - ((size_t) reserve_mib << 20) : 0;
    std::vector<int64_t> sizes;
    uint64_t used = 0;
    for (size_t i = first; i < ranked.size(); ++i) {
        const uint64_t b = align(lay.blob_bytes(ranked[i].first), 256);
        if (used + b > room) break;
        used += b;
        sizes.push_back((int64_t) lay.blob_bytes(ranked[i].first));
    }
    if (sizes.empty()) { err = "no room for experts"; return false; }
    if (!cache_.open_sized(sizes, n_layers_, n_expert_, err)) return false;
    for (size_t j = 0; j < sizes.size(); ++j) {
        const auto [l, e] = ranked[first + j];
        const uint8_t* b = src.blob(l, e);
        if (b == nullptr || !cache_.fill_slot((int32_t) j, b, stream_, err, sizes[j])) {
            if (err.empty()) err = "the expert source could not produce a blob for the second GPU";
            return false;
        }
        res_[(size_t) (l * n_expert_ + e)] = (int32_t) j;
    }
    if (cudaStreamSynchronize((cudaStream_t) stream_) != cudaSuccess) {
        err = std::string("filling the slots: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    const auto [l0, e0] = ranked[first];
    return cache_.verify_slot(0, src.blob(l0, e0), err, sizes[0]);
}

void GpuTier::close() {
    if (dev_ < 0) return;
    {
        OnDevice on(dev_, home_);
        if (on.status != cudaSuccess) return;
        cudaDeviceSynchronize();
        cache_.close();
        if (d_in_) cudaFree(d_in_);
        if (d_out_) cudaFree(d_out_);
        if (d_xq_) cudaFree(d_xq_);
        if (d_scratch_) cudaFree(d_scratch_);
        if (h_in_) cudaFreeHost(h_in_);
        if (h_out_) cudaFreeHost(h_out_);
        if (done_) cudaEventDestroy((cudaEvent_t) done_);
        if (adapt_ev_) cudaEventDestroy((cudaEvent_t) adapt_ev_);
        if (stream_) cudaStreamDestroy((cudaStream_t) stream_);
        if (adapt_stream_) cudaStreamDestroy((cudaStream_t) adapt_stream_);
    }
    d_in_ = h_in_ = d_xq_ = nullptr;
    d_out_ = h_out_ = nullptr;
    d_scratch_ = stream_ = adapt_stream_ = done_ = adapt_ev_ = nullptr;
    res_.clear();
    incoming_.clear();
    pending_.clear();
    dev_ = -1;
}

void GpuTier::begin(int64_t layer) {
    layer_ = layer;
    ng_ = 0;
    ne_ = 0;
}

void GpuTier::add_group(int32_t expert) {
    auto* start = (int32_t*) (h_in_ + off_start_);
    auto* ptr = (unsigned long long*) (h_in_ + off_ptr_);
    start[ng_] = ne_;
    ptr[ng_] = (unsigned long long) cache_.device_slot(res_[(size_t) (layer_ * n_expert_ + expert)]);
    ++ng_;
}

void GpuTier::add_entry(int32_t routed, int32_t tok) {
    ((int32_t*) (h_in_ + off_dst_))[ne_] = ne_;   // compact: the card's rows come back in entry order
    ((int32_t*) (h_in_ + off_tok_))[ne_] = tok;
    row_of_[(size_t) ne_] = routed;
    ++ne_;
}

bool GpuTier::launch(const float* x, int64_t n_tok, std::string& err) {
    if (ng_ == 0) return true;
    if (n_tok > max_tok_ || ne_ > cap_) { err = "a verify window is larger than the extra GPU's buffers"; return false; }
    auto* counts = (int32_t*) h_in_;
    counts[0] = ng_;
    counts[1] = ne_;
    ((int32_t*) (h_in_ + off_start_))[ng_] = ne_;
    const size_t xb = (size_t) (n_tok * n_embd_) * sizeof(float);
    std::memcpy(h_in_ + off_x_, x, xb);
    const auto& f = strata::kernels::cpu::expert_layout().fmt[(size_t) layer_];
    const strata::kernels::NativeExpertLayout L = strata::kernels::native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff);
    OnDevice on(dev_, home_);
    if (on.status != cudaSuccess) {
        err = "cannot select extra GPU " + std::to_string(dev_) + ": " + cudaGetErrorString(on.status);
        return false;
    }
    cudaStream_t s = (cudaStream_t) stream_;
    if (const cudaError_t e = cudaMemcpyAsync(d_in_, h_in_, off_x_ + xb, cudaMemcpyHostToDevice, s); e != cudaSuccess) {
        err = std::string("copying to the extra GPU: ") + cudaGetErrorString(e);
        return false;
    }
    strata::kernels::quantize_q8_1_rows((const float*) (d_in_ + off_x_), n_tok, n_embd_, d_xq_, s);
    strata::kernels::native_expert_grouped(L, (const unsigned long long*) (d_in_ + off_ptr_),
                                           (const int32_t*) (d_in_ + off_start_), (const int32_t*) d_in_,
                                           (const int32_t*) (d_in_ + off_dst_), (const int32_t*) (d_in_ + off_tok_), ng_,
                                           ne_, d_xq_, d_scratch_, d_out_, s);
    if (const cudaError_t e = cudaMemcpyAsync(h_out_, d_out_, (size_t) ne_ * (size_t) n_embd_ * sizeof(float),
                                              cudaMemcpyDeviceToHost, s); e != cudaSuccess) {
        err = std::string("copying from the extra GPU: ") + cudaGetErrorString(e);
        return false;
    }
    const cudaError_t e = cudaEventRecord((cudaEvent_t) done_, s);
    if (e != cudaSuccess) { err = std::string("the extra GPU: ") + cudaGetErrorString(e); return false; }
    launched_ = true;
    return true;
}

bool GpuTier::finish(float* out, std::string& err) {
    if (!launched_) return true;
    launched_ = false;
    const auto t0 = std::chrono::steady_clock::now();
    cudaError_t e;
    {
        OnDevice on(dev_, home_);
        if (on.status != cudaSuccess) {
            err = "cannot select extra GPU " + std::to_string(dev_) + ": " + cudaGetErrorString(on.status);
            return false;
        }
        while ((e = cudaEventQuery((cudaEvent_t) done_)) == cudaErrorNotReady) {}
    }
    ms_wait += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (e != cudaSuccess) { err = std::string("the extra GPU: ") + cudaGetErrorString(e); return false; }
    const size_t row = (size_t) n_embd_ * sizeof(float);
    for (int j = 0; j < ne_; ++j) std::memcpy(out + (size_t) row_of_[(size_t) j] * (size_t) n_embd_, h_out_ + (size_t) j * n_embd_, row);
    entries += ne_;
    return true;
}

int GpuTier::adapt(const float* usage, std::vector<uint8_t>& held, ExpertSource& src, int max_swaps, std::string& err) {
    if (!pending_.empty()) return 0;   // the previous swaps are still copying
    struct Swap { float gain; int32_t layer, in, out; };
    std::vector<Swap> swaps;
    std::vector<std::pair<float, int32_t>> cand, vict;
    for (int64_t l = 0; l < n_layers_; ++l) {
        cand.clear();
        vict.clear();
        const size_t b = (size_t) (l * n_expert_);
        for (int32_t e = 0; e < (int32_t) n_expert_; ++e) {
            const size_t i = b + (size_t) e;
            if (res_[i] >= 0) vict.emplace_back(usage[i], e);
            else if (!held[i] && usage[i] >= 2.0f) cand.emplace_back(usage[i], e);
        }
        if (cand.empty() || vict.empty()) continue;
        std::sort(cand.begin(), cand.end(), [](auto& x, auto& y) { return x.first > y.first; });
        const size_t nc = std::min(cand.size(), vict.size());
        std::partial_sort(vict.begin(), vict.begin() + (ptrdiff_t) nc, vict.end(),
                          [](auto& x, auto& y) { return x.first < y.first; });
        for (size_t i = 0; i < nc; ++i) {
            if (cand[i].first < vict[i].first + 1.5f) break;
            swaps.push_back({cand[i].first - vict[i].first, (int32_t) l, cand[i].second, vict[i].second});
        }
    }
    std::sort(swaps.begin(), swaps.end(), [](const Swap& x, const Swap& y) { return x.gain > y.gain; });
    if ((int) swaps.size() > max_swaps) swaps.resize((size_t) max_swaps);
    if (swaps.empty()) return 0;
    const auto& lay = strata::kernels::cpu::expert_layout();
    OnDevice on(dev_, home_);
    if (on.status != cudaSuccess) {
        err = "cannot select extra GPU " + std::to_string(dev_) + ": " + cudaGetErrorString(on.status);
        return -1;
    }
    for (const Swap& s : swaps) {
        const size_t in = (size_t) s.layer * n_expert_ + s.in, out = (size_t) s.layer * n_expert_ + s.out;
        const int32_t slot = res_[out];
        const uint8_t* blob = src.blob(s.layer, s.in);
        if (blob == nullptr || cudaMemcpyAsync(cache_.device_slot(slot), blob, (size_t) lay.blob_bytes(s.layer),
                                               cudaMemcpyHostToDevice, (cudaStream_t) adapt_stream_) != cudaSuccess) {
            err = "an adaptive refill of an extra GPU failed";
            return -1;
        }
        res_[out] = kNotResident;   // evicted now: the CPU computes it meanwhile
        incoming_[in] = 1;
        held[in] = 1;
        pending_.emplace_back((int32_t) in, slot);
    }
    if (const cudaError_t e = cudaEventRecord((cudaEvent_t) adapt_ev_, (cudaStream_t) adapt_stream_); e != cudaSuccess) {
        err = std::string("recording the extra GPU refill: ") + cudaGetErrorString(e);
        return -1;
    }
    return (int) swaps.size();
}

bool GpuTier::apply(bool wait, std::string& err) {
    if (pending_.empty()) return true;
    {
        OnDevice on(dev_, home_);
        if (on.status != cudaSuccess) {
            err = "cannot select extra GPU " + std::to_string(dev_) + ": " + cudaGetErrorString(on.status);
            return false;
        }
        const cudaError_t e = wait ? cudaEventSynchronize((cudaEvent_t) adapt_ev_)
                                   : cudaEventQuery((cudaEvent_t) adapt_ev_);
        if (e == cudaErrorNotReady) return true;
        if (e != cudaSuccess) {
            err = std::string("finishing the extra GPU refill: ") + cudaGetErrorString(e);
            return false;
        }
    }
    for (const auto& [i, slot] : pending_) {
        res_[(size_t) i] = slot;
        incoming_[(size_t) i] = 0;
    }
    pending_.clear();
    return true;
}

int tier_of(const std::vector<GpuTier*>& tiers, int64_t layer, int32_t e) {
    for (size_t t = 0; t < tiers.size(); ++t)
        if (tiers[t]->slot_of(layer, e) >= 0) return (int) t;
    return -1;
}

bool held_by_tier(const std::vector<GpuTier*>& tiers, int64_t layer, int32_t e) {
    for (const GpuTier* t : tiers)
        if (t->holds(layer, e)) return true;
    return false;
}

bool adapt_tiers(const std::vector<GpuTier*>& tiers, const float* usage, const std::vector<int32_t>& main_res,
                 const std::vector<std::pair<int32_t, int32_t>>& main_incoming, ExpertSource& src, int max_swaps,
                 std::string& err) {
    if (tiers.empty()) return true;
    std::vector<uint8_t> held(main_res.size(), 0);
    for (size_t i = 0; i < main_res.size(); ++i) held[i] = main_res[i] >= 0;
    for (const auto& pr : main_incoming) held[(size_t) pr.first] = 1;
    for (const GpuTier* t : tiers)
        for (size_t i = 0; i < held.size(); ++i)
            if (t->holds_at(i)) held[i] = 1;
    for (GpuTier* t : tiers)
        if (t->adapt(usage, held, src, max_swaps, err) < 0) return false;
    return true;
}

}  // namespace strata::core
