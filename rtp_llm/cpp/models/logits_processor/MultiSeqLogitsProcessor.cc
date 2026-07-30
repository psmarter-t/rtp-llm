#include "rtp_llm/cpp/models/logits_processor/MultiSeqLogitsProcessor.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

std::shared_ptr<MultiSeqLogitsProcessor>
MultiSeqLogitsProcessor::fromGenerateInput(std::shared_ptr<GenerateInput> generate_input, int64_t eos_token_id) {

    if (generate_input->generate_config->num_return_sequences <= 1 && !generate_input->generate_config->hasNumBeams()) {
        return nullptr;
    }

    RTP_LLM_CHECK_WITH_INFO(eos_token_id >= 0,
                            "beam search and multiple return sequences require a valid EOS token id");

    auto processor_ptr           = std::make_shared<MultiSeqLogitsProcessor>();
    processor_ptr->eos_token_id_ = eos_token_id;

    return processor_ptr;
}

void MultiSeqLogitsProcessor::process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) {
    size_t batch_size = finish_idx - start_idx;
    size_t vocab_size = inputs.logits.size(1);

    RTP_LLM_CHECK_WITH_INFO(eos_token_id_ >= 0 && eos_token_id_ < static_cast<int64_t>(vocab_size),
                            "multi-sequence EOS token id [%ld] is outside logits vocabulary [0, %lu)",
                            eos_token_id_,
                            vocab_size);

    auto logits            = inputs.logits.narrow(0, start_idx, batch_size);
    auto finished_mask_ptr = reinterpret_cast<bool*>(inputs.finished_mask.data_ptr()) + start_idx;

    // return early when no sequence needs processing
    if (!std::any_of(finished_mask_ptr, finished_mask_ptr + batch_size, [](bool v) { return v; })) {
        return;
    }

    // mask all logits of the finished sequences except the eos token
    auto logit_mask_host_tensor = torch::zeros({(int64_t)batch_size, (int64_t)vocab_size}, torch::kUInt8);
    auto logit_mask_host_ptr    = logit_mask_host_tensor.data_ptr<uint8_t>();

    for (size_t idx = 0; idx < batch_size; ++idx) {
        if (finished_mask_ptr[idx]) {
            auto cur_logit_mask_host_ptr = logit_mask_host_ptr + idx * vocab_size;
            memset(cur_logit_mask_host_ptr, 1, vocab_size * sizeof(uint8_t));
            cur_logit_mask_host_ptr[eos_token_id_] = 0;
            if (inputs.no_repeat_ngram_size.defined()) {
                inputs.no_repeat_ngram_size.data_ptr<int32_t>()[start_idx + idx] = 0;
            }
        }
    }

    auto logit_mask = logit_mask_host_tensor.to(torch::kCUDA);

    maskLogits(logits, logit_mask);
    // MultiSeq is the final processor and owns the finished-row contract. Restore
    // EOS in case an earlier Think/Recommendation processor masked it.
    auto finished_mask = inputs.finished_mask.narrow(0, start_idx, batch_size).to(torch::kCUDA);
    logits.select(1, eos_token_id_).masked_fill_(finished_mask, 0.0f);
}

void MultiSeqLogitsProcessor::updateMultiSeqStatus(const std::vector<int>& src_batch_indices) {
    // do nothing
}

void MultiSeqLogitsProcessor::updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) {
    // do nothing
}

}  // namespace rtp_llm
