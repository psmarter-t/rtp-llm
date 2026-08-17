// Microbenchmark + correctness harness for the radix topk paths behind
// invokeTopkLastDim (multi-block vs one-block, selected via force_path).
//
// Usage:
//   topk_bench                                  run matrix + semantic checks
//   topk_bench BATCH LEN K SORTED PATH [REPS]   time one combo
//     SORTED: 0/1 (1 = trailing by-value sort, production stage-2 behavior)
//     PATH:   0 = auto, 1 = force multi-block, 2 = force one-block
//
// Part A (routing matrix): shapes x ks x sorted variants x {multi, one-block},
//   prints CSV rows and checks both paths are bitwise identical per combo.
// Part B (semantic checks): compares invokeTopkLastDim output against a host
//   reference (value desc, index asc) — covers the ROCm k<=256 WarpSort gate
//   and the radix path directly.
// Matrix mode exits non-zero if any comparison fails (single-combo mode always
// returns 0).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <random>
#include <vector>
#include <algorithm>
#include <numeric>

#include "3rdparty/trt_beam_search/common.h"
#include "3rdparty/trt_beam_search/topkLastDim.h"

namespace
{

using tensorrt_llm::kernels::invokeComputeTopkLastDimWorkspaceSize;
using tensorrt_llm::kernels::invokeTopkLastDim;

// Logprob-like values in [-20, 0]; every 7919th element pinned to create
// exact-equal ties so the tie-break path is exercised.
void fillInput(std::vector<float>& host, int batch, int len)
{
    host.resize(static_cast<size_t>(batch) * len);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-20.f, 0.f);
    for (auto& v : host)
    {
        v = dist(rng);
    }
    for (size_t i = 0; i < host.size(); i += 7919)
    {
        host[i] = -1.5f;
    }
}

float* toDevice(std::vector<float> const& host)
{
    float* dev = nullptr;
    cudaMalloc(&dev, host.size() * sizeof(float));
    check_cuda_error();
    cudaMemcpy(dev, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice);
    check_cuda_error();
    return dev;
}

// Returns microseconds per call. dOutVal/dOutIdx receive the last call's output.
double timePath(int batch, int len, int k, bool sorted, int forcePath, float const* dIn, float* dOutVal,
    int* dOutIdx, void* workspace, int reps, cudaStream_t stream)
{
    std::optional<float> mask(-std::numeric_limits<float>::infinity());  // same as beam search
    for (int i = 0; i < 3; ++i)
    {
        invokeTopkLastDim<float>(batch, len, k, true, mask, dIn, dOutVal, dOutIdx, workspace, stream, sorted, forcePath);
    }
    cudaStreamSynchronize(stream);
    check_cuda_error();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);
    for (int i = 0; i < reps; ++i)
    {
        invokeTopkLastDim<float>(batch, len, k, true, mask, dIn, dOutVal, dOutIdx, workspace, stream, sorted, forcePath);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    check_cuda_error();
    return static_cast<double>(ms) * 1000.0 / reps;
}

struct PathOutput
{
    double us;
    std::vector<float> vals;
    std::vector<int> idxs;
};

PathOutput runPath(int batch, int len, int k, bool sorted, int forcePath, float const* dIn, int reps, cudaStream_t stream)
{
    size_t const wsBytes = invokeComputeTopkLastDimWorkspaceSize<float>(batch, len, k, true, forcePath);
    void* workspace = nullptr;
    cudaMalloc(&workspace, wsBytes);
    float* dOutVal = nullptr;
    int* dOutIdx = nullptr;
    cudaMalloc(&dOutVal, sizeof(float) * k * batch);
    cudaMalloc(&dOutIdx, sizeof(int) * k * batch);
    check_cuda_error();

    PathOutput out;
    out.us = timePath(batch, len, k, sorted, forcePath, dIn, dOutVal, dOutIdx, workspace, reps, stream);
    out.vals.resize(static_cast<size_t>(k) * batch);
    out.idxs.resize(static_cast<size_t>(k) * batch);
    cudaMemcpy(out.vals.data(), dOutVal, out.vals.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out.idxs.data(), dOutIdx, out.idxs.size() * sizeof(int), cudaMemcpyDeviceToHost);
    check_cuda_error();

    cudaFree(workspace);
    cudaFree(dOutVal);
    cudaFree(dOutIdx);
    return out;
}

bool sameOutput(PathOutput const& a, PathOutput const& b)
{
    return a.vals.size() == b.vals.size() && 0 == std::memcmp(a.vals.data(), b.vals.data(), a.vals.size() * sizeof(float))
        && 0 == std::memcmp(a.idxs.data(), b.idxs.data(), a.idxs.size() * sizeof(int));
}

// Host reference: per row, top-k by (value desc, index asc). indexAscending=true
// returns the same set ordered by index (the sorted=false production layout).
void hostTopk(std::vector<float> const& in, int batch, int len, int k, bool indexAscending,
    std::vector<float>& refVals, std::vector<int>& refIdxs)
{
    refVals.assign(static_cast<size_t>(k) * batch, 0.f);
    refIdxs.assign(static_cast<size_t>(k) * batch, 0);
    std::vector<int> ord(len);
    for (int b = 0; b < batch; ++b)
    {
        std::iota(ord.begin(), ord.end(), 0);
        float const* row = in.data() + static_cast<size_t>(b) * len;
        std::partial_sort(ord.begin(), ord.begin() + k, ord.end(),
            [&](int i, int j) { return row[i] > row[j] || (row[i] == row[j] && i < j); });
        if (indexAscending)
        {
            std::sort(ord.begin(), ord.begin() + k);
        }
        for (int i = 0; i < k; ++i)
        {
            refIdxs[static_cast<size_t>(b) * k + i] = ord[i];
            refVals[static_cast<size_t>(b) * k + i] = row[ord[i]];
        }
    }
}

// Part B: one semantic check of invokeTopkLastDim against the host reference.
// strictOrder=true: exact match on values and index order (radix path semantics).
// strictOrder=false: exact match on values + per-row index multiset equality
// (WarpSort semantics: exact top-k set, unspecified order among equal values).
// Returns true on match under the selected mode; tie-order divergence is
// reported as informational only.
bool checkVsHost(int batch, int len, int k, bool sorted, std::vector<float> const& hostIn, float const* dIn,
    cudaStream_t stream, char const* label, bool strictOrder)
{
    PathOutput dev = runPath(batch, len, k, sorted, /*forcePath=*/0, dIn, /*reps=*/3, stream);
    std::vector<float> refVals;
    std::vector<int> refIdxs;
    hostTopk(hostIn, batch, len, k, /*indexAscending=*/!sorted, refVals, refIdxs);
    bool const valuesOk
        = 0 == std::memcmp(dev.vals.data(), refVals.data(), refVals.size() * sizeof(float));
    bool ok;
    if (strictOrder)
    {
        ok = valuesOk
            && 0 == std::memcmp(dev.idxs.data(), refIdxs.data(), refIdxs.size() * sizeof(int));
    }
    else
    {
        ok = valuesOk;
        int tieRows = 0;
        for (int b = 0; b < batch && ok; ++b)
        {
            std::vector<int> devRow(dev.idxs.begin() + static_cast<size_t>(b) * k,
                dev.idxs.begin() + static_cast<size_t>(b + 1) * k);
            std::vector<int> refRow(refIdxs.begin() + static_cast<size_t>(b) * k,
                refIdxs.begin() + static_cast<size_t>(b + 1) * k);
            if (devRow != refRow)
            {
                ++tieRows;
            }
            std::sort(devRow.begin(), devRow.end());
            std::sort(refRow.begin(), refRow.end());
            if (devRow != refRow)
            {
                ok = false;
            }
        }
        std::printf("check[%s]: batch=%d len=%d k=%d sorted=%d -> %s (values %s, rows with tie-order diff %d)\n",
            label, batch, len, k, sorted ? 1 : 0, ok ? "PASS" : "FAIL", valuesOk ? "exact" : "MISMATCH", tieRows);
        return ok;
    }
    std::printf("check[%s]: batch=%d len=%d k=%d sorted=%d -> %s\n", label, batch, len, k, sorted ? 1 : 0,
        ok ? "PASS" : "FAIL");
    if (!ok)
    {
        for (size_t i = 0; i < refIdxs.size(); ++i)
        {
            if (dev.idxs[i] != refIdxs[i] || dev.vals[i] != refVals[i])
            {
                std::printf("  first mismatch at i=%zu: dev(idx=%d val=%f) ref(idx=%d val=%f)\n", i, dev.idxs[i],
                    dev.vals[i], refIdxs[i], refVals[i]);
                break;
            }
        }
    }
    return ok;
}

struct Shape
{
    // cls: 'A' production stage-1, 'C' production stage-2, 'X' crossover probe,
    //      'B' beam-4096/8192 spot check.
    char cls;
    int batch;
    int len;
    char const* note;
    int k = 0;  // 0 = use the class default k list
};

int runMatrix(int reps)
{
    // (batch, len) shapes. grid_dim is a function of (batch, len) only; the x-*
    // probes land in the grid_dim 10..36 crossover region (approximate values
    // computed for gfx942 fp32, active_blocks=240).
    Shape const shapes[] = {
        // production shapes
        {'A', 50, 65660, "A-step2"},
        {'A', 1500, 65660, "A-steady"},
        {'C', 1, 150000, "C-2k"},
        {'C', 1, 75000, "C-1k"},
        {'C', 1, 16777216, "C-steady-4096"},
        // vocab variants
        {'A', 50, 217303, "A-fullvocab"},
        {'A', 50, 262144, "A-bigvocab"},
        {'A', 50, 32000, "A-smallvocab"},
        // crossover probes
        {'X', 16, 65660, "x-grid11"},
        {'X', 16, 32000, "x-grid15"},
        {'X', 8, 32000, "x-grid16"},
        {'X', 10, 217303, "x-grid22"},
        {'X', 8, 217303, "x-grid27"},
        {'X', 6, 131072, "x-grid32"},
        {'X', 10, 65660, "x-grid33"},
        {'X', 5, 217303, "x-grid36"},
        // beam-4096 (k) and its 2x (8192) spot checks on the production A shapes
        {'B', 50, 65660, "A-step2-beam4096", 4096},
        {'B', 1500, 65660, "A-steady-beam4096", 4096},
        {'B', 50, 65660, "A-step2-beam8192", 8192},
        {'B', 1500, 65660, "A-steady-beam8192", 8192},
    };
    auto ksFor = [](Shape const& s) -> std::vector<int> {
        if (s.k != 0)
        {
            return {s.k};
        }
        switch (s.cls)
        {
        case 'C': return s.len > 1000000 ? std::vector<int>{4096} : std::vector<int>{1500, 3000};
        case 'X': return {1500};
        default: return {1500, 3000};
        }
    };
    auto sortedFor = [](Shape const& s) -> std::vector<bool> {
        return s.cls == 'C' ? std::vector<bool>{true} : std::vector<bool>{true, false};
    };

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    std::printf("batch,len,k,sorted,note,multi_us,oneblock_us,speedup,identical\n");
    int failures = 0;

    auto runCombo = [&](Shape const& s, int k, bool sorted) {
        std::vector<float> hostIn;
        fillInput(hostIn, s.batch, s.len);
        float* dIn = toDevice(hostIn);
        PathOutput multi = runPath(s.batch, s.len, k, sorted, 1, dIn, reps, stream);
        PathOutput oneblk = runPath(s.batch, s.len, k, sorted, 2, dIn, reps, stream);
        bool const identical = sameOutput(multi, oneblk);
        if (!identical)
        {
            ++failures;
        }
        std::printf("%d,%d,%d,%d,%s,%.1f,%.1f,%.2fx,%s\n", s.batch, s.len, k, sorted ? 1 : 0, s.note, multi.us,
            oneblk.us, multi.us / oneblk.us, identical ? "yes" : "NO");
        cudaFree(dIn);
    };

    for (auto const& s : shapes)
    {
        for (int k : ksFor(s))
        {
            for (bool sorted : sortedFor(s))
            {
                runCombo(s, k, sorted);
            }
        }
    }

    // Part B: semantic checks vs host reference.
    std::vector<float> hostIn;
    float* dIn = nullptr;
    auto prep = [&](int batch, int len) {
        if (dIn)
        {
            cudaFree(dIn);
        }
        fillInput(hostIn, batch, len);
        dIn = toDevice(hostIn);
    };
    // ROCm k<=256 gate (WarpSort path, pre-existing behavior for beams <=256).
    prep(50, 65660);
    failures += checkVsHost(50, 65660, 200, true, hostIn, dIn, stream, "warpsort-A", false) ? 0 : 1;
    prep(1, 75000);
    failures += checkVsHost(1, 75000, 200, true, hostIn, dIn, stream, "warpsort-C", false) ? 0 : 1;
    // k=400: beams 257..512 stay on the radix path after the gate narrowed to 256.
    prep(50, 217303);
    failures += checkVsHost(50, 217303, 400, true, hostIn, dIn, stream, "radix-k400-fullvocab", true) ? 0 : 1;
    // Radix path vs reference (k>512): sorted=true layout and sorted=false layout.
    prep(50, 65660);
    failures += checkVsHost(50, 65660, 1500, true, hostIn, dIn, stream, "radix-sorted", true) ? 0 : 1;
    failures += checkVsHost(50, 65660, 1500, false, hostIn, dIn, stream, "radix-unsorted", true) ? 0 : 1;
    if (dIn)
    {
        cudaFree(dIn);
    }

    cudaStreamDestroy(stream);
    return failures == 0 ? 0 : 1;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc == 1)
    {
        return runMatrix(20);
    }
    if (argc < 6 || argc > 7)
    {
        std::fprintf(stderr, "usage: %s [BATCH LEN K SORTED PATH [REPS]]\n", argv[0]);
        return 2;
    }
    int const batch = std::atoi(argv[1]);
    int const len = std::atoi(argv[2]);
    int const k = std::atoi(argv[3]);
    bool const sorted = std::atoi(argv[4]) != 0;
    int const path = std::atoi(argv[5]);
    int const reps = argc > 6 ? std::atoi(argv[6]) : 20;

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    std::vector<float> hostIn;
    fillInput(hostIn, batch, len);
    float* dIn = toDevice(hostIn);
    PathOutput out = runPath(batch, len, k, sorted, path, dIn, reps, stream);
    std::printf("batch=%d len=%d k=%d sorted=%d path=%d: %.1f us/call\n", batch, len, k, sorted ? 1 : 0, path, out.us);
    cudaFree(dIn);
    cudaStreamDestroy(stream);
    return 0;
}
