#pragma once

#include <memory>
#include <torch/all.h>
#include <utility>
#include <vector>
#include "absl/status/status.h"
#include "autil/LockFreeThreadPool.h"
#include "rtp_llm/cpp/engine_base/stream/StreamGroups.h"
#include "rtp_llm/cpp/models/SampleInfos.h"

namespace rtp_llm {

class NormalOutputDispatcher {
public:
    explicit NormalOutputDispatcher(std::shared_ptr<autil::LockFreeThreadPool> thread_pool      = nullptr,
                                    std::vector<int64_t>                       output_vocab_ids = {}):
        thread_pool_(std::move(thread_pool)), output_vocab_ids_(std::move(output_vocab_ids)) {}

    absl::Status dispatch(const StreamGroups& stream_groups, const MergedOutput& merge_outputs) const;

private:
    torch::Tensor calculateSelectedTokenProbs(const torch::Tensor& logits,
                                              const torch::Tensor& token_ids,
                                              const torch::Tensor& src_batch_indices) const;

    bool restoreCurrentTokenIds(const GenerateStreamPtr& stream,
                                torch::Tensor&           batch_token_ids,
                                torch::Tensor&           current_token_ids,
                                size_t                   token_position) const;

    void dispatchSingleStream(GenerateStreamPtr    stream,
                              const MergedOutput&  merge_outputs,
                              int                  batch_idx_in,
                              int                  batch_idx_out,
                              int                  token_offset,
                              bool                 return_all_probs,
                              const torch::Tensor& new_tokens_all,
                              const torch::Tensor& token_ids_cpu,
                              const torch::Tensor& success_cpu) const;

    std::shared_ptr<autil::LockFreeThreadPool> thread_pool_;
    std::vector<int64_t>                       output_vocab_ids_;
};

}  // namespace rtp_llm
