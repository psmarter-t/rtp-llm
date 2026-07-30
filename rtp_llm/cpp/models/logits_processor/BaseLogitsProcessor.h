#pragma once

#include <optional>
#include <utility>

#include "rtp_llm/cpp/config/OutputVocabMapping.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"

namespace rtp_llm {

class BaseLogitsProcessor {
public:
    BaseLogitsProcessor() = default;
    virtual ~BaseLogitsProcessor() {}
    static const float neg_inf;

public:
    virtual void process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) = 0;
    virtual void updateMultiSeqStatus(const std::vector<int>& src_batch_indices)           = 0;
    virtual void updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens)     = 0;
    void         setOutputVocabMapping(OutputVocabMappingPtr output_vocab_mapping) {
        output_vocab_mapping_ = std::move(output_vocab_mapping);
    }
    void          memFill(const torch::Tensor& new_tokens_logits, size_t vocab_size, size_t index);
    void          maskLogits(torch::Tensor& new_token_logits, const torch::Tensor& vocab_mask);
    torch::Tensor generateVocabMask(size_t                                  batch_size,
                                    size_t                                  vocab_size,
                                    const std::vector<std::vector<size_t>>& batch_candidate_token_ids);

protected:
    std::optional<int32_t> toLogitsTokenId(int32_t full_token_id) const {
        if (!output_vocab_mapping_) {
            return full_token_id >= 0 ? std::optional<int32_t>(full_token_id) : std::nullopt;
        }
        return output_vocab_mapping_->toLocal(full_token_id);
    }

private:
    OutputVocabMappingPtr output_vocab_mapping_;
};

typedef std::shared_ptr<BaseLogitsProcessor> BaseLogitsProcessorPtr;

}  // namespace rtp_llm
