#include "rtp_llm/cpp/normal_engine/NormalOutputDispatcher.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/TensorDebugUtils.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include <limits>
#include <vector>
#if USING_CUDA
#include "rtp_llm/models_py/bindings/cuda/ops/StandaloneOps.h"
#include "ATen/cuda/CUDAContext.h"
#endif

namespace rtp_llm {

namespace {

torch::Tensor localToFullIndices(const OutputVocabMappingPtr& mapping, const torch::Device& device) {
    std::vector<int64_t> indices(mapping->localToFull().begin(), mapping->localToFull().end());
    return torch::tensor(indices, torch::TensorOptions().dtype(torch::kInt64)).to(device);
}

torch::Tensor
expandLocalVocab(const torch::Tensor& local_values, const OutputVocabMappingPtr& mapping, double fill_value) {
    RTP_LLM_CHECK(local_values.defined() && local_values.dim() >= 1);
    RTP_LLM_CHECK(local_values.size(-1) == static_cast<int64_t>(mapping->size()));
    auto full_shape   = local_values.sizes().vec();
    full_shape.back() = mapping->fullVocabSize();
    auto full_values  = torch::full(full_shape, fill_value, local_values.options());
    full_values.index_copy_(-1, localToFullIndices(mapping, local_values.device()), local_values);
    return full_values;
}

struct LocalTokenIds {
    torch::Tensor ids;
    torch::Tensor valid;
};

LocalTokenIds
mapFullTokenIds(const int* full_ids, size_t count, const OutputVocabMappingPtr& mapping, const torch::Device& device) {
    auto  local_ids_cpu = torch::zeros({static_cast<int64_t>(count)}, torch::kInt64);
    auto  valid_cpu     = torch::zeros({static_cast<int64_t>(count)}, torch::kBool);
    auto* local_ids     = local_ids_cpu.data_ptr<int64_t>();
    auto* valid         = valid_cpu.data_ptr<bool>();
    for (size_t i = 0; i < count; ++i) {
        auto local_id = mapping->toLocal(full_ids[i]);
        if (local_id.has_value()) {
            local_ids[i] = local_id.value();
            valid[i]     = true;
        }
    }
    return {local_ids_cpu.to(device), valid_cpu.to(device)};
}

}  // namespace

absl::Status NormalOutputDispatcher::dispatch(const StreamGroups& stream_groups,
                                              const MergedOutput& merge_outputs) const {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    const auto&  sampler_output       = merge_outputs.sampler_output;
    const size_t total_batch_size_out = stream_groups.totalSamplerBatchSizeOut();
    RTP_LLM_CHECK(total_batch_size_out == (size_t)sampler_output.token_ids.size(0));
    // token_ids and success may be CUDA tensors (Sampler keeps them on GPU to avoid D2H sync during sampling).
    // Move to CPU once here so dispatchSingleStream can use data_ptr safely.
    const torch::Tensor token_ids_cpu =
        sampler_output.token_ids.defined() ? sampler_output.token_ids.cpu() : torch::Tensor();
    RTP_LLM_LOG_DEBUG("new_all_token_ids = [%s]", tensorDebugStringWithData<int32_t>(token_ids_cpu).c_str());
    const torch::Tensor success_cpu = sampler_output.success.defined() ? sampler_output.success.cpu() : torch::Tensor();
    int                 batch_idx_in     = 0;
    int                 batch_idx_out    = 0;
    int                 token_offset     = 0;
    bool                return_all_probs = stream_groups.needReturnAllProbs() != ReturnAllProbsMode::NONE;
    auto                new_tokens_all   = torch::empty({(int64_t)total_batch_size_out, 1}, torch::kInt32);

    for (auto& stream : stream_groups.allStreams()) {
        auto cur_batch_size  = stream->currentBatchSize();
        auto next_batch_size = stream->nextBatchSize();
        auto token_size      = stream->currentExecuteTokenSize();

        dispatchSingleStream(stream,
                             merge_outputs,
                             batch_idx_in,
                             batch_idx_out,
                             token_offset,
                             return_all_probs,
                             new_tokens_all,
                             token_ids_cpu,
                             success_cpu);

        batch_idx_in += cur_batch_size;
        batch_idx_out += next_batch_size;
        token_offset += token_size;
    }

    RTP_LLM_LOG_DEBUG("dispatch done");
    return absl::OkStatus();
}

void NormalOutputDispatcher::dispatchSingleStream(GenerateStreamPtr    stream,
                                                  const MergedOutput&  merge_outputs,
                                                  int                  batch_idx_in,
                                                  int                  batch_idx_out,
                                                  int                  token_offset,
                                                  bool                 return_all_probs,
                                                  const torch::Tensor& new_tokens_all,
                                                  const torch::Tensor& token_ids_cpu,
                                                  const torch::Tensor& success_cpu) const {

    const auto&  model_output      = merge_outputs.model_output;
    const auto&  sampler_output    = merge_outputs.sampler_output;
    const auto&  new_all_token_ids = token_ids_cpu;
    const size_t token_stride      = new_all_token_ids.size(1);

    auto       cur_batch_size         = stream->currentBatchSize();
    auto       next_batch_size        = stream->nextBatchSize();
    auto       token_size             = stream->currentExecuteTokenSize();
    const bool uses_beam_token_layout = stream->usesBeamSearchTokenLayoutForCurrentStep();

    for (int i = 0; i < cur_batch_size; ++i) {
        if (success_cpu.defined() && !(success_cpu.data_ptr<bool>()[batch_idx_in + i])) {
            stream->reportError(ErrorCode::UNKNOWN_ERROR, "sampler has no valid token candidate");
            return;
        }
    }

    // Greedy/Sampling writes the selected token into the shared last column of
    // the batched sampler buffer. Beam Search writes it at the sequence-specific
    // position reported to the Beam kernel.
    auto new_token_position = uses_beam_token_layout ? stream->seqLength() : token_stride - 1;
    RTP_LLM_CHECK(new_token_position < token_stride);

    auto batch_new_all_token_ids = new_all_token_ids.narrow(0, batch_idx_out, next_batch_size);

    const bool has_var_batch = cur_batch_size != next_batch_size;

    // construct mapping from output batches to input batches
    torch::Tensor src_batch_indices;
    if (uses_beam_token_layout) {
        // beam search
        src_batch_indices = sampler_output.beam_index.narrow(0, batch_idx_out, next_batch_size);
    } else if (has_var_batch) {
        // from context stream to decode straem, there might be other cases in future
        src_batch_indices = torch::zeros({(int64_t)next_batch_size}, torch::kInt32);
    }
    const auto get_src_idx = [&](int32_t dst_idx) {
        return src_batch_indices.defined() ? src_batch_indices.data_ptr<int32_t>()[dst_idx] : dst_idx;
    };
    const auto reorder_rows = [&](const torch::Tensor& tensor) {
        if (!tensor.defined() || !src_batch_indices.defined()) {
            return tensor;
        }
        auto indices = src_batch_indices.to(tensor.device(), torch::kLong);
        return tensor.index_select(0, indices);
    };

    // construct update info
    torch::Tensor batch_hidden_states;
    if (stream->generateConfig()->return_hidden_states) {
        batch_hidden_states = reorder_rows(model_output.hidden_states.narrow(0, batch_idx_in, cur_batch_size));
    }

    torch::Tensor batch_logits;
    if (stream->returnLogits() || stream->calculateSoftmaxProbs() || uses_beam_token_layout) {
        batch_logits = model_output.logits.narrow(0, batch_idx_in, cur_batch_size);
    }
    torch::Tensor output_logits;
    if (stream->returnLogits()) {
        output_logits = reorder_rows(batch_logits);
        if (output_vocab_mapping_) {
            output_logits =
                expandLocalVocab(output_logits, output_vocab_mapping_, -std::numeric_limits<float>::infinity());
        }
    }

    torch::Tensor all_probs;
    if (return_all_probs) {
        all_probs = sampler_output.all_probs.narrow(0, batch_idx_out, next_batch_size);
        if (output_vocab_mapping_) {
            all_probs = expandLocalVocab(all_probs, output_vocab_mapping_, 0.0);
        }
    };

    torch::Tensor batch_cum_log_probs;
    if (sampler_output.cum_log_probs.defined()) {
        batch_cum_log_probs = sampler_output.cum_log_probs.narrow(0, batch_idx_out, next_batch_size);
    }

    torch::Tensor loss;
    if (stream->calculateLoss()) {
        auto all_logits_tensor = model_output.all_logits.narrow(0, token_offset, token_size - 1);
        auto tokens            = stream->currentExecuteTokens(0);
        if (output_vocab_mapping_) {
            auto labels = mapFullTokenIds(
                tokens.data() + 1, tokens.size() - 1, output_vocab_mapping_, all_logits_tensor.device());
            auto negative_log_probs =
                -torch::log_softmax(all_logits_tensor, -1).gather(1, labels.ids.unsqueeze(1)).squeeze(1);
            loss = torch::where(labels.valid,
                                negative_log_probs,
                                torch::full_like(negative_log_probs, std::numeric_limits<float>::infinity()))
                       .to(torch::kFloat32);
        } else {
            auto label_tensor =
                torch::from_blob(const_cast<int*>(tokens.data() + 1), {(int64_t)(tokens.size() - 1)}, torch::kInt32)
                    .to(torch::kCUDA);
            auto labels_int64 = label_tensor.toType(torch::kInt64);
            loss = torch::cross_entropy_loss(all_logits_tensor, labels_int64, torch::nullopt, at::Reduction::None)
                       .to(torch::kFloat32);
        }
    }

    // Prompt scoring: guarded by all_logits.defined() which is only produced during prefill
    // (NormalModelInputGatherer sets need_all_logits only in processContextStreams).
    std::optional<PromptLogitsOutput> prompt_logits_output;
    if (stream->returnPromptLogits() && !model_output.all_logits.defined()) {
        RTP_LLM_LOG_WARNING("stream [%ld] prompt_logits requested but all_logits not produced", stream->streamId());
    }
    if (stream->returnPromptLogits() && model_output.all_logits.defined()) {
        auto config    = stream->generateConfig();
        int  ts        = (int)token_size;
        int  start_pos = std::clamp(config->prompt_logits_start >= 0 ? config->prompt_logits_start : 0, 0, ts);
        int  end_pos   = std::clamp(config->prompt_logits_end >= 0 ? config->prompt_logits_end : ts, start_pos, ts);
        int  slice_len = end_pos - start_pos;
        if (slice_len > 0) {
            int top_k = std::min(config->prompt_logits_top_k, (int)model_output.all_logits.size(1));

            auto sliced_logits =
                model_output.all_logits.narrow(0, token_offset + start_pos, slice_len).to(torch::kFloat32);

            // topk on raw logits (monotonicity of softmax preserves ranking)
            auto [topk_values_raw, topk_indices] = sliced_logits.topk(top_k, -1);
            if (output_vocab_mapping_) {
                auto indices = localToFullIndices(output_vocab_mapping_, topk_indices.device());
                topk_indices = indices.index_select(0, topk_indices.reshape({-1})).reshape_as(topk_indices);
            }

            // single reduce for log-normalizer, avoids materializing [slice_len, vocab_size]
            auto log_sum_exp   = sliced_logits.logsumexp(-1, /*keepdim=*/true);
            auto topk_logprobs = topk_values_raw - log_sum_exp;

            // target_logprobs[i] = logprob of token[start_pos+i+1] at position start_pos+i.
            // Length = min(slice_len, tokens.size() - start_pos - 1): equals slice_len when
            // end_pos < tokens.size(), or slice_len-1 when end_pos == tokens.size() (last
            // position has no next token as label).
            torch::Tensor target_logprobs;
            if (config->return_target_logprob) {
                auto tokens      = stream->currentExecuteTokens(0);
                int  label_start = start_pos + 1;
                int  label_end   = std::min(end_pos + 1, (int)tokens.size());
                int  logprob_len = label_end - label_start;
                if (logprob_len > 0) {
                    if (output_vocab_mapping_) {
                        auto labels = mapFullTokenIds(
                            tokens.data() + label_start, logprob_len, output_vocab_mapping_, sliced_logits.device());
                        auto target_raw =
                            sliced_logits.narrow(0, 0, logprob_len).gather(1, labels.ids.unsqueeze(1)).squeeze(1);
                        auto mapped_logprobs = target_raw - log_sum_exp.narrow(0, 0, logprob_len).squeeze(1);
                        target_logprobs =
                            torch::where(labels.valid,
                                         mapped_logprobs,
                                         torch::full_like(mapped_logprobs, -std::numeric_limits<float>::infinity()))
                                .cpu();
                    } else {
                        // from_blob + to(kCUDA) is a synchronous copy; token buffer is stable during prefill.
                        auto label_tensor = torch::from_blob(const_cast<int*>(tokens.data() + label_start),
                                                             {(int64_t)logprob_len},
                                                             torch::kInt32)
                                                .to(torch::kCUDA)
                                                .toType(torch::kInt64)
                                                .unsqueeze(1);
                        auto target_raw = sliced_logits.narrow(0, 0, logprob_len).gather(1, label_tensor).squeeze(1);
                        target_logprobs = (target_raw - log_sum_exp.narrow(0, 0, logprob_len).squeeze(1)).cpu();
                    }
                }
            }

            prompt_logits_output = PromptLogitsOutput{
                topk_logprobs.cpu(), topk_indices.to(torch::kInt32).cpu(), target_logprobs, start_pos, end_pos};
        }
    }

    torch::Tensor all_hidden_states;
    if (stream->needReturnHiddenStates()) {
        all_hidden_states = model_output.all_hidden_states.narrow(0, token_offset, token_size);
    }

    auto new_tokens = new_tokens_all.narrow(0, batch_idx_out, next_batch_size);
    for (size_t i = 0; i < next_batch_size; ++i) {
        new_tokens.data_ptr<int32_t>()[i] =
            new_all_token_ids.data_ptr<int32_t>()[(batch_idx_out + i) * token_stride + new_token_position];
    }

    torch::Tensor current_softmax_result;
    if (stream->calculateSoftmaxProbs()) {
        // Softmax is in-place on CUDA; keep raw logits intact when both outputs are requested.
        auto batch_softmax_input = batch_logits.to(torch::kFloat32).contiguous().clone();
#if USING_CUDA
        cudaSoftmaxInplace(batch_softmax_input, at::cuda::getCurrentCUDAStream().stream());
#else
        batch_softmax_input = torch::softmax(batch_softmax_input, -1);
#endif
        auto batch_softmax_tensor = batch_softmax_input.cpu();
        current_softmax_result    = torch::empty({(int64_t)next_batch_size, 1}, torch::kFloat32);
        for (int i = 0; i < next_batch_size; ++i) {
            current_softmax_result[i][0] = batch_softmax_tensor[get_src_idx(i)][new_tokens.data_ptr<int32_t>()[i]];
        }
    }

    if (output_vocab_mapping_) {
        if (uses_beam_token_layout) {
            auto* token_ids = batch_new_all_token_ids.data_ptr<int32_t>();
            for (size_t i = 0; i < next_batch_size; ++i) {
                const auto token_index = i * token_stride + new_token_position;
                token_ids[token_index] = output_vocab_mapping_->toFull(token_ids[token_index]);
            }
        } else {
            auto* token_ids = new_tokens.data_ptr<int32_t>();
            for (size_t i = 0; i < next_batch_size; ++i) {
                token_ids[i] = output_vocab_mapping_->toFull(token_ids[i]);
            }
        }
    }

    RTP_LLM_LOG_DEBUG("stream [%ld], new_tokens size = [%ld]", stream->streamId(), new_tokens.numel());

    stream->update({uses_beam_token_layout ? batch_new_all_token_ids : new_tokens,
                    1,
                    batch_hidden_states,
                    output_logits,
                    current_softmax_result,
                    batch_cum_log_probs,
                    all_probs,
                    loss,
                    src_batch_indices,
                    all_hidden_states,
                    true,
                    false,
                    prompt_logits_output});
}

}  // namespace rtp_llm
