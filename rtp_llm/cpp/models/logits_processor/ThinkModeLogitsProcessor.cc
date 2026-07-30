#include "rtp_llm/cpp/models/logits_processor/ThinkModeLogitsProcessor.h"

using namespace std;

namespace rtp_llm {

ThinkModeLogitsProcessor::ThinkModeLogitsProcessor(std::vector<StreamThinkInfo> think_infos):
    think_infos_(think_infos) {};

void ThinkModeLogitsProcessor::process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) {
    RTP_LLM_CHECK(size() == finish_idx - start_idx);

    for (size_t i = 0; i < size(); ++i) {
        auto& info = think_infos_[i];
        if (!info.in_think_mode)
            continue;

        int*       input_lengths    = inputs.input_lengths.data_ptr<int32_t>();
        int*       sequence_lengths = inputs.sequence_lengths.data_ptr<int32_t>();
        int        num_new_tokens   = 1;
        bool       enforce          = (sequence_lengths[i + start_idx] + num_new_tokens
                        >= info.max_thinking_tokens + input_lengths[i + start_idx]);
        const bool hard_controlled  = !info.dfa_ptr->isFinished() && enforce;
        setVocabMask(info.dfa_ptr,
                     inputs.logits[i + start_idx],
                     num_new_tokens,
                     info.end_think_token_ids,
                     inputs.vocab_size,
                     enforce);
        if (hard_controlled && inputs.no_repeat_ngram_size.defined()) {
            inputs.no_repeat_ngram_size.data_ptr<int32_t>()[i + start_idx] = 0;
        }
    }
}

void ThinkModeLogitsProcessor::setVocabMask(std::shared_ptr<StringContainDFA<size_t, int>> dfa_ptr,
                                            const torch::Tensor&                           new_tokens_logits,
                                            int                                            num_new_tokens,
                                            std::vector<int>                               template_token_ids,
                                            size_t                                         vocab_size,
                                            bool                                           enforce) {
    if (!dfa_ptr->isFinished() && enforce) {
        RTP_LLM_LOG_INFO("sampler enforce transfer status");
        const auto full_token_id   = template_token_ids[dfa_ptr->status()];
        const auto logits_token_id = toLogitsTokenId(full_token_id);
        RTP_LLM_CHECK_WITH_INFO(logits_token_id.has_value(),
                                "think-mode token id [%d] is not present in the output vocabulary",
                                full_token_id);
        memFill(new_tokens_logits, vocab_size, static_cast<size_t>(logits_token_id.value()));
    }
}

void ThinkModeLogitsProcessor::updateMultiSeqStatus(const std::vector<int>& src_batch_indices) {
    std::vector<StreamThinkInfo> new_think_infos;
    for (auto src_batch_idx : src_batch_indices) {
        new_think_infos.push_back(think_infos_[src_batch_idx].copy());
    }
    think_infos_ = new_think_infos;
}

void ThinkModeLogitsProcessor::updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) {
    RTP_LLM_CHECK(2 == new_tokens.dim());
    RTP_LLM_CHECK(new_tokens.scalar_type() == torch::kInt32);
    RTP_LLM_CHECK(new_tokens.device().is_cpu());
    RTP_LLM_CHECK(new_tokens.is_contiguous());
    RTP_LLM_CHECK(num_new_tokens >= 0);
    RTP_LLM_CHECK(size() == (size_t)new_tokens.size(0));

    const int64_t stride = new_tokens.size(1);
    RTP_LLM_CHECK_WITH_INFO(stride >= num_new_tokens,
                            "think updateStatus token shape mismatch: num_new_tokens=%d, new_tokens.size(1)=%ld",
                            num_new_tokens,
                            stride);

    for (size_t i = 0; i < size(); i++) {
        auto& info = think_infos_[i];
        if (!info.in_think_mode)
            continue;

        // GenerateStream may use incremental [N, num_new_tokens] rows before a
        // dynamic-beam expansion and full-position rows after expansion. The
        // current tensor shape, rather than the request's static beam config,
        // defines which layout this update carries.
        const bool    use_token_offset = stride > num_new_tokens;
        const int64_t offset           = use_token_offset ? info.current_output_length + info.input_length : 0;
        if (use_token_offset) {
            RTP_LLM_CHECK_WITH_INFO(
                offset + num_new_tokens <= stride,
                "think updateStatus full-position token offset out of range: offset=%ld, num_new_tokens=%d, "
                "new_tokens.size(1)=%ld, input_length=%d, current_output_length=%d",
                offset,
                num_new_tokens,
                stride,
                info.input_length,
                info.current_output_length);
        }

        for (int32_t j = 0; j < num_new_tokens; ++j) {
            const auto current_token_id = new_tokens.data_ptr<int32_t>()[i * stride + j + offset];
            info.dfa_ptr->next(current_token_id);
        }

        info.current_output_length += num_new_tokens;
    }
}

ThinkModeLogitsProcessorPtr ThinkModeLogitsProcessor::fromGenerateInput(std::shared_ptr<GenerateInput> generate_input,
                                                                        int32_t                        num) {
    if (!generate_input->generate_config->in_think_mode || generate_input->generate_config->max_thinking_tokens == 0) {
        return nullptr;
    }

    auto processor_ptr = std::make_shared<ThinkModeLogitsProcessor>();
    for (size_t i = 0; i < num; i++) {
        StreamThinkInfo think_info(
            generate_input->generate_config->in_think_mode,
            generate_input->generate_config->max_thinking_tokens,
            generate_input->generate_config->end_think_token_ids,
            generate_input->inputLength(),
            0,
            std::make_shared<StringContainDFA<size_t, int>>(generate_input->generate_config->end_think_token_ids));
        std::vector<StreamThinkInfo> think_infos = {think_info};
        auto                         ptr         = std::make_shared<ThinkModeLogitsProcessor>(think_infos);

        processor_ptr->insert(ptr, 1);
    }
    return processor_ptr;
}

std::vector<size_t> ThinkModeLogitsProcessor::thinkEndTokensStatus() {
    std::vector<size_t> status;
    for (auto think_info : think_infos_) {
        auto dfa = think_info.dfa_ptr;
        status.push_back(dfa->status());
    }
    return status;
}

}  // namespace rtp_llm
