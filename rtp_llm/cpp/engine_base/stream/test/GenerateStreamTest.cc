
#include "gtest/gtest.h"

#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

using namespace std;

namespace rtp_llm {

class GenerateStreamBuilder {
public:
    GenerateStreamBuilder() {
        model_config_.max_seq_len = 2048;
        model_config_.vocab_size  = 2048;
    }

    CacheConfig init_config() {
        return test::makeSimpleMhaCacheConfig(
            /*layer_num=*/3, /*block_num=*/9, /*tokens_per_block=*/2, rtp_llm::DataType::TYPE_INT8);
    }

    GenerateStreamPtr createContextStream(std::vector<int> input_ids) {
        std::shared_ptr<GenerateInput>  generate_input(new GenerateInput());
        std::shared_ptr<GenerateConfig> generate_config(new GenerateConfig());
        ResourceContext                 resource_context;
        generate_input->generate_config = generate_config;
        generate_input->input_ids =
            torch::tensor(std::vector<int32_t>(input_ids.begin(), input_ids.end()), torch::kInt32);
        return std::make_shared<NormalGenerateStream>(
            generate_input, model_config_, runtime_config_, resource_context, nullptr);
    };

    GenerateStreamPtr createComplexContextStream(std::vector<int> input_ids) {
        autil::EnvGuard perf_scope("PERF_TEST", "1");

        auto cache_config  = init_config();
        auto cache_manager = std::make_shared<KVCacheManager>(cache_config);
        cache_manager->init();
        ResourceContext resource_context;
        resource_context.cache_manager = cache_manager;
        resource_context.reuse_cache   = true;

        std::shared_ptr<GenerateInput>  generate_input(new GenerateInput());
        std::shared_ptr<GenerateConfig> generate_config(new GenerateConfig());
        generate_config->num_return_sequences = 2;
        generate_input->input_ids =
            torch::tensor(std::vector<int32_t>(input_ids.begin(), input_ids.end()), torch::kInt32);
        generate_input->generate_config = generate_config;
        ModelConfig   model_config;
        RuntimeConfig runtime_config;
        model_config.max_seq_len = 2048;
        auto stream              = std::make_shared<NormalGenerateStream>(
            generate_input, model_config, runtime_config, resource_context, nullptr);

        return stream;
    }

    GenerateStreamPtr createDecoderStream(std::vector<int> input_ids, std::vector<int> new_token_ids) {
        std::shared_ptr<GenerateInput>  generate_input(new GenerateInput());
        std::shared_ptr<GenerateConfig> generate_config(new GenerateConfig());
        ResourceContext                 resource_context;
        generate_input->generate_config = generate_config;
        generate_input->input_ids =
            torch::tensor(std::vector<int32_t>(input_ids.begin(), input_ids.end()), torch::kInt32);
        auto stream_ptr = std::make_shared<NormalGenerateStream>(
            generate_input, model_config_, runtime_config_, resource_context, nullptr);
        stream_ptr->setIsContextStream(false);
        auto complete_ids = stream_ptr->completeTokenIds();
        std::memcpy(complete_ids.data_ptr<int32_t>() + stream_ptr->seqLength(),
                    new_token_ids.data(),
                    new_token_ids.size() * sizeof(int));
        stream_ptr->setSeqLength(stream_ptr->seqLength() + new_token_ids.size());
        return stream_ptr;
    };

private:
    ModelConfig   model_config_;
    RuntimeConfig runtime_config_;
};

class GenerateStreamTest: public DeviceTestBase {
protected:
};

TEST_F(GenerateStreamTest, testConstruct) {
    auto builder = GenerateStreamBuilder();
    auto stream1 = builder.createContextStream({{1, 2, 3, 4, 5}, {}});
    auto stream2 = builder.createDecoderStream({1, 2, 3, 4, 5}, {1, 2, 3});
}

TEST_F(GenerateStreamTest, testBatchSizeWithNumReturnSequences) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 2048;
    RuntimeConfig runtime_config;

    auto generate_input                                   = std::make_shared<GenerateInput>();
    generate_input->generate_config                       = std::make_shared<GenerateConfig>();
    generate_input->generate_config->num_return_sequences = 3;
    generate_input->input_ids                             = torch::tensor({1, 2, 3}, torch::kInt32);

    auto stream =
        std::make_shared<NormalGenerateStream>(generate_input, model_config, runtime_config, resource_context, nullptr);

    EXPECT_EQ(1, stream->batchSize(0));
    EXPECT_EQ(3, stream->batchSize(1));
    EXPECT_EQ(3, stream->batchSize(5));
    EXPECT_EQ(3, stream->maxBatchSize());
    EXPECT_TRUE(stream->needTilingForSampling());
}

TEST_F(GenerateStreamTest, testBatchSizeWithBeamSearch) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 2048;
    RuntimeConfig runtime_config;

    auto generate_input                        = std::make_shared<GenerateInput>();
    generate_input->generate_config            = std::make_shared<GenerateConfig>();
    generate_input->generate_config->num_beams = 4;
    generate_input->input_ids                  = torch::tensor({1, 2, 3}, torch::kInt32);

    auto stream =
        std::make_shared<NormalGenerateStream>(generate_input, model_config, runtime_config, resource_context, nullptr);

    EXPECT_EQ(1, stream->batchSize(0));
    EXPECT_EQ(4, stream->batchSize(1));
    EXPECT_EQ(4, stream->maxBatchSize());
    EXPECT_FALSE(stream->needTilingForSampling());
}

TEST_F(GenerateStreamTest, testCompleteTokenIdsUsesRequestBoundAndInitializesAllRows) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 128;
    RuntimeConfig runtime_config;

    auto generate_input                             = std::make_shared<GenerateInput>();
    generate_input->generate_config                 = std::make_shared<GenerateConfig>();
    generate_input->generate_config->num_beams      = 2;
    generate_input->generate_config->max_new_tokens = 4;
    generate_input->input_ids                       = torch::tensor({7, 8, 9}, torch::kInt32);

    auto stream =
        std::make_shared<NormalGenerateStream>(generate_input, model_config, runtime_config, resource_context, nullptr);

    auto token_ids = stream->completeTokenIds();
    ASSERT_EQ(2, token_ids.size(0));
    ASSERT_EQ(7, token_ids.size(1));
    EXPECT_TRUE(torch::equal(token_ids[0].narrow(0, 0, 3), generate_input->input_ids));
    EXPECT_TRUE(torch::equal(token_ids[1].narrow(0, 0, 3), generate_input->input_ids));
}

TEST_F(GenerateStreamTest, testGenerateStreamReuseCacheMethod) {
    auto builder = GenerateStreamBuilder();
    auto stream  = builder.createContextStream({1, 2, 3, 4, 5, 6});

    // default true
    ASSERT_TRUE(stream->reuseCache());

    // flip to false and verify
    stream->generate_input_->generate_config->reuse_cache = false;
    ASSERT_FALSE(stream->reuseCache());

    // flip back to true and verify
    stream->generate_input_->generate_config->reuse_cache = true;
    ASSERT_TRUE(stream->reuseCache());
}

TEST_F(GenerateStreamTest, testNonStreamingFinalOutputReturnsCachedAllHiddenStates) {
    auto builder                                                       = GenerateStreamBuilder();
    auto stream                                                        = builder.createContextStream({1, 2});
    stream->generate_input_->generate_config->max_new_tokens           = 2;
    stream->generate_input_->generate_config->return_all_hidden_states = true;
    stream->generate_input_->generate_config->return_incremental       = true;
    stream->generate_input_->generate_config->is_streaming             = false;

    auto first_all_hidden_states = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f}).reshape({2, 2});
    stream->step();
    stream->update(StreamUpdateInfo{torch::tensor({10}, torch::kInt32).reshape({1, 1}),
                                    1,
                                    torch::Tensor(),
                                    torch::Tensor(),
                                    torch::Tensor(),
                                    torch::Tensor(),
                                    torch::Tensor(),
                                    torch::Tensor(),
                                    torch::Tensor(),
                                    first_all_hidden_states,
                                    false});
    ASSERT_FALSE(stream->hasOutput());

    stream->step();
    stream->update(StreamUpdateInfo{torch::tensor({11}, torch::kInt32).reshape({1, 1}),
                                    1,
                                    torch::Tensor(),
                                    torch::Tensor(),
                                    torch::Tensor(),
                                    torch::Tensor(),
                                    torch::Tensor(),
                                    torch::Tensor(),
                                    torch::Tensor(),
                                    torch::Tensor(),
                                    false});

    ASSERT_TRUE(stream->hasOutput());
    auto output = stream->nextOutput();
    ASSERT_TRUE(output.ok());
    ASSERT_EQ(output.value().generate_outputs.size(), 1);
    const auto& generate_output = output.value().generate_outputs[0];
    ASSERT_TRUE(generate_output.finished);
    ASSERT_TRUE(generate_output.all_hidden_states.has_value());
    ASSERT_TRUE(torch::equal(generate_output.all_hidden_states.value(), first_all_hidden_states));
}

TEST_F(GenerateStreamTest, testAllHiddenStatesCopiedToCpuOnceForMultipleOutputs) {
    auto builder = GenerateStreamBuilder();
    auto stream  = std::dynamic_pointer_cast<NormalGenerateStream>(builder.createComplexContextStream({1, 2}));
    ASSERT_NE(stream, nullptr);
    stream->generate_input_->generate_config->return_all_hidden_states = true;
    stream->iter_count_                                                = 1;

    auto all_hidden_states =
        torch::tensor({1.0f, 2.0f, 3.0f, 4.0f}, torch::TensorOptions().device(torch::kCUDA)).reshape({2, 2});
    StreamUpdateInfo update_info{torch::Tensor(),
                                 0,
                                 torch::Tensor(),
                                 torch::Tensor(),
                                 torch::Tensor(),
                                 torch::Tensor(),
                                 torch::Tensor(),
                                 torch::Tensor(),
                                 torch::Tensor(),
                                 all_hidden_states,
                                 false};

    auto outputs = stream->prepareGenerateOutput(update_info);

    ASSERT_EQ(outputs.generate_outputs.size(), 2);
    const auto& first  = outputs.generate_outputs[0].all_hidden_states;
    const auto& second = outputs.generate_outputs[1].all_hidden_states;
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());
    ASSERT_FALSE(first->is_cuda());
    ASSERT_EQ(first->data_ptr(), second->data_ptr());
    ASSERT_TRUE(torch::equal(first.value(), all_hidden_states.cpu()));
}

TEST_F(GenerateStreamTest, testInputEmbeddingsDisableTokenOnlyReuseCache) {
    auto builder                                   = GenerateStreamBuilder();
    auto stream                                    = builder.createContextStream({1, 2, 3, 4, 5, 6});
    stream->generate_input_->input_embeddings      = std::vector<torch::Tensor>{torch::rand({1, 8}, torch::kFloat32)};
    stream->generate_input_->input_embeddings_locs = std::vector<int32_t>{2};

    ASSERT_TRUE(stream->hasInputEmbeddings());
    ASSERT_FALSE(stream->reuseCache());
    ASSERT_FALSE(stream->enableDeviceCache());
    ASSERT_FALSE(stream->enableMemoryCache());
    ASSERT_FALSE(stream->enableRemoteCache());
}

TEST_F(GenerateStreamTest, testDynamicBeamLayoutDependsOnCurrentTransition) {
    ModelConfig model_config;
    model_config.max_seq_len = 8;
    model_config.vocab_size  = 10;
    RuntimeConfig   runtime_config;
    ResourceContext resource_context;

    auto input                                 = std::make_shared<GenerateInput>();
    input->input_ids                           = torch::tensor({2}, torch::kInt32);
    input->generate_config                     = std::make_shared<GenerateConfig>();
    input->generate_config->variable_num_beams = {1, 2, 1};
    auto stream =
        std::make_shared<NormalGenerateStream>(input, model_config, runtime_config, resource_context, nullptr);

    EXPECT_FALSE(stream->usesBeamSearchTokenLayoutForCurrentStep());
    stream->setSeqLength(2);
    EXPECT_TRUE(stream->usesBeamSearchTokenLayoutForCurrentStep());
    stream->setSeqLength(3);
    EXPECT_TRUE(stream->usesBeamSearchTokenLayoutForCurrentStep());
    stream->setSeqLength(4);
    EXPECT_FALSE(stream->usesBeamSearchTokenLayoutForCurrentStep());
}

TEST_F(GenerateStreamTest, testDynamicBeamOutputUsesUpdatedCurrentBatchSize) {
    ModelConfig model_config;
    model_config.max_seq_len = 8;
    model_config.vocab_size  = 10;
    RuntimeConfig   runtime_config;
    ResourceContext resource_context;

    auto input                                 = std::make_shared<GenerateInput>();
    input->input_ids                           = torch::tensor({2}, torch::kInt32);
    input->generate_config                     = std::make_shared<GenerateConfig>();
    input->generate_config->variable_num_beams = {2, 3};
    auto stream =
        std::make_shared<NormalGenerateStream>(input, model_config, runtime_config, resource_context, nullptr);

    auto beam_tokens    = torch::tensor({2, 4, 2, 7}, torch::kInt32).reshape({2, 2});
    int  error_token_id = 0;
    ASSERT_TRUE(
        stream->complete_token_ids_->update(beam_tokens, 0, 1, 1, 8, 10, true, stream->streamId(), error_token_id));
    stream->resizeSubGenerateStatus(2);
    ASSERT_EQ(stream->currentBatchSize(), 2);
    ASSERT_EQ(stream->nextBatchSize(), 3);

    auto outputs = stream->prepareGenerateOutput({beam_tokens,
                                                  1,
                                                  torch::Tensor(),
                                                  torch::Tensor(),
                                                  torch::Tensor(),
                                                  torch::Tensor(),
                                                  torch::Tensor(),
                                                  torch::Tensor(),
                                                  torch::Tensor(),
                                                  torch::Tensor()});
    ASSERT_EQ(outputs.generate_outputs.size(), 2);
    EXPECT_EQ(outputs.generate_outputs[0].output_ids[0][0].item<int32_t>(), 4);
    EXPECT_EQ(outputs.generate_outputs[1].output_ids[0][0].item<int32_t>(), 7);
}

TEST_F(GenerateStreamTest, testDynamicBeamSoftmaxHistoryFollowsParentRows) {
    ModelConfig model_config;
    model_config.max_seq_len = 8;
    model_config.vocab_size  = 10;
    RuntimeConfig   runtime_config;
    ResourceContext resource_context;

    auto input                                   = std::make_shared<GenerateInput>();
    input->input_ids                             = torch::tensor({2}, torch::kInt32);
    input->generate_config                       = std::make_shared<GenerateConfig>();
    input->generate_config->variable_num_beams   = {2, 3};
    input->generate_config->max_new_tokens       = 3;
    input->generate_config->return_softmax_probs = true;
    auto stream =
        std::make_shared<NormalGenerateStream>(input, model_config, runtime_config, resource_context, nullptr);

    int  error_token_id = 0;
    auto first_tokens   = torch::tensor({2, 4, 2, 7}, torch::kInt32).reshape({2, 2});
    ASSERT_TRUE(
        stream->complete_token_ids_->update(first_tokens, 0, 1, 1, 8, 10, true, stream->streamId(), error_token_id));
    stream->setSoftmaxProbs(torch::tensor({0.1f, 0.2f}).reshape({2, 1}), 1, torch::tensor({0, 0}, torch::kInt32));

    auto second_tokens = torch::tensor({2, 7, 1, 2, 7, 3, 2, 4, 5}, torch::kInt32).reshape({3, 3});
    ASSERT_TRUE(
        stream->complete_token_ids_->update(second_tokens, 0, 1, 1, 8, 10, true, stream->streamId(), error_token_id));
    stream->setSoftmaxProbs(
        torch::tensor({0.3f, 0.4f, 0.5f}).reshape({3, 1}), 2, torch::tensor({1, 1, 0}, torch::kInt32));

    auto probabilities = stream->getSoftmaxProbs();
    EXPECT_FLOAT_EQ(probabilities[0][0].item<float>(), 0.2f);
    EXPECT_FLOAT_EQ(probabilities[1][0].item<float>(), 0.2f);
    EXPECT_FLOAT_EQ(probabilities[2][0].item<float>(), 0.1f);
    EXPECT_FLOAT_EQ(probabilities[0][1].item<float>(), 0.3f);
    EXPECT_FLOAT_EQ(probabilities[1][1].item<float>(), 0.4f);
    EXPECT_FLOAT_EQ(probabilities[2][1].item<float>(), 0.5f);

    stream->setSoftmaxProbs(torch::tensor({0.6f, 0.7f, 0.8f}).reshape({3, 1}), 3, torch::Tensor());
    EXPECT_EQ(probabilities.size(0), 3);
    EXPECT_EQ(probabilities.size(1), 3);
    EXPECT_FLOAT_EQ(probabilities[0][2].item<float>(), 0.6f);
    EXPECT_FLOAT_EQ(probabilities[1][2].item<float>(), 0.7f);
    EXPECT_FLOAT_EQ(probabilities[2][2].item<float>(), 0.8f);
}

}  // namespace rtp_llm
