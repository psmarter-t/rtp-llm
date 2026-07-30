#include <cmath>
#include <memory>
#include <numeric>
#include "torch/all.h"
#include "gtest/gtest.h"

#define private public
#include "rtp_llm/cpp/normal_engine/NormalBatchStreamProcessor.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/models/Sampler.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.h"
#include "rtp_llm/cpp/models/logits_processor/MultiSeqLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/RecommendationLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/ThinkModeLogitsProcessor.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/GenerationLimits.h"
#include "rtp_llm/cpp/config/OutputVocabMapping.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"

using namespace std;

namespace rtp_llm {

template<typename T>
std::vector<T> toVec(const torch::Tensor& t) {
    auto c = t.contiguous();
    return std::vector<T>(c.data_ptr<T>(), c.data_ptr<T>() + c.numel());
}

static torch::Tensor hostIntBuffer(std::vector<int32_t> data) {
    return torch::tensor(data, torch::kInt32);
}

static SamplerInputs makeValidatedBeamInputs(const torch::Tensor& logits,
                                             int64_t              logical_batch_size,
                                             int64_t              num_beams_in,
                                             int64_t              num_beams_out,
                                             const torch::Tensor& cum_log_probs = torch::Tensor()) {
    const int64_t batch_size_in = logical_batch_size * num_beams_in;
    SamplerInputs inputs;
    inputs.logits           = logits.to(torch::kCUDA);
    inputs.token_ids        = torch::zeros({batch_size_in, 3}, torch::kInt32);
    inputs.input_lengths    = torch::ones({batch_size_in}, torch::kInt32);
    inputs.sequence_lengths = torch::ones({batch_size_in}, torch::kInt32);
    inputs.cum_log_probs    = cum_log_probs.defined() ? cum_log_probs.to(torch::kFloat32).cpu().contiguous() :
                                                        torch::zeros({batch_size_in}, torch::kFloat32);
    RTP_LLM_CHECK(inputs.cum_log_probs.numel() == batch_size_in);
    inputs.num_beams_in               = torch::full({batch_size_in}, num_beams_in, torch::kLong);
    inputs.num_beams_out              = torch::full({batch_size_in}, num_beams_out, torch::kLong);
    inputs.step                       = 1;
    inputs.batch_size                 = batch_size_in;
    inputs.batch_size_out             = logical_batch_size * num_beams_out;
    inputs.vocab_size                 = logits.size(1);
    inputs.validate_logits_candidates = true;
    return inputs;
}

class ForceFullTokenProcessor: public BaseLogitsProcessor {
public:
    explicit ForceFullTokenProcessor(int32_t full_token_id): full_token_id_(full_token_id) {}

    void process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) override {
        observed_vocab_size_       = inputs.logits.size(1);
        const auto logits_token_id = toLogitsTokenId(full_token_id_);
        ASSERT_TRUE(logits_token_id.has_value());
        ASSERT_GT(inputs.logits.size(1), logits_token_id.value());
        auto logits = inputs.logits.narrow(0, start_idx, finish_idx - start_idx);
        logits.fill_(neg_inf);
        logits.index_put_({torch::indexing::Slice(), logits_token_id.value()}, 1.0f);
    }

    void updateMultiSeqStatus(const std::vector<int>&) override {}
    void updateStatus(const torch::Tensor&, int32_t) override {}

    int64_t observedVocabSize() const {
        return observed_vocab_size_;
    }

private:
    int32_t full_token_id_;
    int64_t observed_vocab_size_ = -1;
};

class LocalEosOnlyProcessor: public MultiSeqLogitsProcessor {
public:
    explicit LocalEosOnlyProcessor(int64_t local_eos_token_id) {
        eos_token_id_ = local_eos_token_id;
    }
};

class MaskAllCandidatesProcessor: public BaseLogitsProcessor {
public:
    void process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) override {
        inputs.logits.narrow(0, start_idx, finish_idx - start_idx).fill_(neg_inf);
    }

    void updateMultiSeqStatus(const std::vector<int>&) override {}
    void updateStatus(const torch::Tensor&, int32_t) override {}
};

static void initFullCacheConfig(CacheConfig& cache_config, int layer_num) {
    auto spec = std::make_shared<MHAKVCacheSpec>();
    spec->tag = "default";
    std::vector<int> layer_ids(static_cast<size_t>(layer_num));
    std::iota(layer_ids.begin(), layer_ids.end(), 0);
    cache_config.layer_num     = static_cast<uint32_t>(layer_num);
    cache_config.layer_all_num = static_cast<uint32_t>(layer_num);
    cache_config.fromGroupedSpecs({spec}, {layer_ids}, {CacheGroupType::FULL}, {"default"});
}

class NormalBatchStreamProcessorTest: public DeviceTestBase {
protected:
    static ModelConfig makeOutputVocabTestModelConfig() {
        ModelConfig model_config;
        model_config.max_seq_len = 8;
        model_config.vocab_size  = 10;
        model_config.num_layers  = 1;
        return model_config;
    }

    static std::shared_ptr<NormalGenerateStream> makeDynamicBeamStream(const std::vector<int32_t>& input_ids,
                                                                       const std::vector<int>&     variable_num_beams,
                                                                       int                         max_new_tokens,
                                                                       const ModelConfig&          model_config,
                                                                       const RuntimeConfig&        runtime_config,
                                                                       const ResourceContext&      resource_context) {
        auto query                                 = make_shared<GenerateInput>();
        query->input_ids                           = hostIntBuffer(input_ids);
        query->generate_config                     = make_shared<GenerateConfig>();
        query->generate_config->variable_num_beams = variable_num_beams;
        query->generate_config->max_new_tokens     = max_new_tokens;
        query->generate_config->do_sample          = false;
        auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        stream->generate_status_->status = StreamState::RUNNING;
        return stream;
    }
};

TEST_F(NormalBatchStreamProcessorTest, testWarmUpWithoutCacheManager) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len      = 2048;
    model_config.vocab_size       = 2048;
    model_config.input_vocab_size = 2048;
    model_config.num_layers       = 1;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    RuntimeConfig               runtime_config;

    auto query             = make_shared<GenerateInput>();
    query->input_ids       = hostIntBuffer({1, 2, 3});
    query->generate_config = make_shared<GenerateConfig>();
    GenerateStreamPtr stream =
        make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    stream->generate_status_->status = StreamState::RUNNING;
    StreamGroups stream_groups({stream});

    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, true);

    EXPECT_EQ(processor.model_input_gatherer_config_.kv_cache_group_nums, 0);
    EXPECT_TRUE(processor.model_input_gatherer_config_.kv_cache_group_types.empty());
    ASSERT_EQ(stream->kvCache().groupNums(), 1);
    EXPECT_EQ(stream->kvCache().cacheResource().soleGroupTagForLayer(0), "__warmup__");
    auto model_input = processor.gatherModelInput(stream_groups);
    ASSERT_TRUE(model_input.ok());
    EXPECT_FALSE(model_input->kv_cache_block_id.defined());
    EXPECT_FALSE(model_input->kv_cache_kernel_block_id.defined());
}

TEST_F(NormalBatchStreamProcessorTest, testCacheKeyWidthIndependentOfBlockTable) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 2048;
    model_config.vocab_size  = 2048;
    model_config.num_layers  = 1;

    PDSepConfig pd_sep_config;
    pd_sep_config.role_type = RoleType::PREFILL;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    initFullCacheConfig(cache_config, model_config.num_layers);
    RuntimeConfig runtime_config;

    auto query                                   = make_shared<GenerateInput>();
    query->input_ids                             = hostIntBuffer({1, 2, 3});
    query->generate_config                       = make_shared<GenerateConfig>();
    query->generate_config->num_return_sequences = 2;
    GenerateStreamPtr stream =
        make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);

    BatchKVCacheResource resource;
    resource.resetBatchSize(2);
    resource.initGroups(cache_config.topologyPtr());
    resource.setBatchBlocks(0, 0, {1, 2});
    resource.setBatchBlocks(1, 0, {3, 4});
    resource.setBatchCacheKeys(0, CacheKeysType{101, 102, 103});
    resource.setBatchCacheKeys(1, CacheKeysType{201, 202, 203, 204, 205});
    stream->setKVCache(resource);
    stream->generate_status_->status = StreamState::RUNNING;

    StreamGroups stream_groups({stream});
    EXPECT_EQ(stream_groups.curBlocksNum(), 2);
    EXPECT_EQ(stream_groups.maxCacheKeysNum(), 5);

    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);
    auto merge_input_status = processor.gatherModelInput(stream_groups);
    ASSERT_TRUE(merge_input_status.ok());
    EXPECT_TRUE(merge_input_status.value().pd_separation);
    const auto& cache_keys = merge_input_status.value().cache_keys;
    ASSERT_TRUE(cache_keys.defined());
    EXPECT_EQ(cache_keys.size(0), 2);
    EXPECT_EQ(cache_keys.size(1), 5);
    EXPECT_EQ(toVec<int64_t>(cache_keys), (std::vector<int64_t>{101, 102, 103, 0, 0, 201, 202, 203, 204, 205}));
}

TEST_F(NormalBatchStreamProcessorTest, testSimpleAssemble) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len                = 2048;
    model_config.vocab_size                 = 2048;
    model_config.num_layers                 = 2;
    model_config.attn_config.kv_cache_dtype = KvCacheDataType::FP8;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    initFullCacheConfig(cache_config, model_config.num_layers);
    cache_config.kv_block_stride_bytes = 4096;
    cache_config.kv_scale_stride_bytes = 256;

    RuntimeConfig              runtime_config;
    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);

    std::shared_ptr<GenerateInput> query1 = make_shared<GenerateInput>();
    query1->input_ids                     = hostIntBuffer({1, 2});
    query1->generate_config               = make_shared<GenerateConfig>();
    GenerateStreamPtr stream1 =
        make_shared<NormalGenerateStream>(query1, model_config, runtime_config, resource_context, nullptr);
    query1->input_ids = hostIntBuffer({1});
    BatchKVCacheResource addr1;
    addr1.resetBatchSize(1);
    addr1.initGroups(cache_config.topologyPtr());
    addr1.setBatchBlocks(0, 0, {1, 2, 3, 4});
    stream1->setKVCache(addr1);
    stream1->setIsContextStream(false);

    std::shared_ptr<GenerateInput> query2 = make_shared<GenerateInput>();
    query2->input_ids                     = hostIntBuffer({1, 2, 3});
    query2->generate_config               = make_shared<GenerateConfig>();
    GenerateStreamPtr stream2 =
        make_shared<NormalGenerateStream>(query2, model_config, runtime_config, resource_context, nullptr);
    query2->input_ids = hostIntBuffer({1, 2});
    BatchKVCacheResource addr2;
    addr2.resetBatchSize(1);
    addr2.initGroups(cache_config.topologyPtr());
    addr2.setBatchBlocks(0, 0, {5, 6, 7, 8});
    stream2->setKVCache(addr2);
    stream2->setIsContextStream(false);

    std::shared_ptr<GenerateInput> query3 = make_shared<GenerateInput>();
    query3->input_ids                     = hostIntBuffer({1, 2, 3});
    query3->generate_config               = make_shared<GenerateConfig>();
    GenerateStreamPtr stream3 =
        make_shared<NormalGenerateStream>(query3, model_config, runtime_config, resource_context, nullptr);
    BatchKVCacheResource addr3;
    addr3.resetBatchSize(1);
    addr3.initGroups(cache_config.topologyPtr());
    addr3.setBatchBlocks(0, 0, {9, 10});
    stream3->setKVCache(addr3);

    std::shared_ptr<GenerateInput> query4 = make_shared<GenerateInput>();
    query4->input_ids                     = hostIntBuffer({1, 2, 3, 4});
    query4->generate_config               = make_shared<GenerateConfig>();
    GenerateStreamPtr stream4 =
        make_shared<NormalGenerateStream>(query4, model_config, runtime_config, resource_context, nullptr);
    BatchKVCacheResource addr4;
    addr4.resetBatchSize(1);
    addr4.initGroups(cache_config.topologyPtr());
    addr4.setBatchBlocks(0, 0, {11, 12, 13, 14});
    stream4->setKVCache(addr4);
    stream4->setReuseLength(1);

    std::list<GenerateStreamPtr> streams;
    streams.emplace_back(stream1);
    streams.emplace_back(stream2);
    streams.emplace_back(stream3);
    streams.emplace_back(stream4);

    for (const auto& stream : streams) {
        stream->generate_status_->status = StreamState::RUNNING;
    }

    {
        StreamGroups stream_groups(streams);

        auto merge_input_status = processor.gatherModelInput(stream_groups);

        EXPECT_TRUE(merge_input_status.ok());
        auto&       model_input       = merge_input_status.value();
        vector<int> combo_tokens      = {2, 3, 1, 2, 3, 2, 3, 4};
        vector<int> input_lengths     = {1, 2, 3, 3};
        vector<int> sequence_lengths  = {1, 2};
        vector<int> prefix_lengths    = {0, 1};
        vector<int> kv_cache_block_id = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0, 11, 12, 13, 14};
        EXPECT_EQ(combo_tokens, toVec<int>(model_input.combo_tokens));
        EXPECT_EQ(input_lengths, toVec<int>(model_input.input_lengths));
        EXPECT_EQ(sequence_lengths, toVec<int>(model_input.sequence_lengths));
        EXPECT_EQ(prefix_lengths, toVec<int>(model_input.prefix_lengths));
        EXPECT_EQ(kv_cache_block_id, toVec<int>(model_input.kv_cache_block_id));
        EXPECT_EQ(model_input.kv_block_stride_bytes, cache_config.kv_block_stride_bytes);
        EXPECT_EQ(model_input.kv_scale_stride_bytes, cache_config.kv_scale_stride_bytes);
    }
    {
        MMModelConfig mm_model_config;
        model_config.mm_model_config = mm_model_config;
        NormalBatchStreamProcessor processor(
            model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);

        StreamGroups stream_groups(streams);
        auto         merge_input_status = processor.gatherModelInput(stream_groups);
        EXPECT_TRUE(merge_input_status.ok());
        auto& model_input = merge_input_status.value();
        EXPECT_FALSE(model_input.attention_mask.defined());
    }
}

TEST_F(NormalBatchStreamProcessorTest, testSoftmaxProbs) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 2048;
    model_config.vocab_size  = 2;
    model_config.num_layers  = 2;

    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    initFullCacheConfig(cache_config, model_config.num_layers);
    RuntimeConfig                  runtime_config;
    std::shared_ptr<GenerateInput> query1         = make_shared<GenerateInput>();
    query1->input_ids                             = hostIntBuffer({1});
    query1->generate_config                       = make_shared<GenerateConfig>();
    query1->generate_config->return_softmax_probs = true;
    GenerateStreamPtr stream1 =
        make_shared<NormalGenerateStream>(query1, model_config, runtime_config, resource_context, nullptr);
    BatchKVCacheResource addr1;
    addr1.resetBatchSize(1);
    addr1.initGroups(cache_config.topologyPtr());
    addr1.setBatchBlocks(0, 0, {1});
    stream1->setKVCache(addr1);

    std::list<GenerateStreamPtr> streams;
    streams.emplace_back(stream1);

    for (const auto& stream : streams) {
        stream->generate_status_->status = StreamState::RUNNING;
    }
    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);

    StreamGroups stream_groups(streams);
    auto         merge_input_status = processor.gatherModelInput(stream_groups);
    EXPECT_TRUE(merge_input_status.ok());

    SamplerInputs sampler_inputs;
    MergedOutput  merge_outputs;
    auto          hidden_tensor                = torch::tensor({1.0f, 2.0f}).reshape({1, 2}).to(torch::kCUDA);
    auto          logits_tensor                = torch::tensor({1.0f, 2.0f}).reshape({1, 2}).to(torch::kCUDA);
    merge_outputs.model_output.hidden_states   = hidden_tensor;
    merge_outputs.model_output.logits          = logits_tensor;
    merge_outputs.sampler_output.token_ids     = torch::tensor({0, 1}, torch::kInt32).reshape({1, 2});
    merge_outputs.sampler_output.cum_log_probs = torch::tensor({1.0f}).to(torch::kCUDA);
    auto status                                = processor.dispatch(stream_groups, merge_outputs);
    EXPECT_TRUE(status.ok());

    auto softmax_probs = stream1->getSoftmaxProbs();
    EXPECT_TRUE(softmax_probs.defined());
    EXPECT_EQ(2048, softmax_probs.numel());
    EXPECT_NEAR(0.731058, softmax_probs.data_ptr<float>()[1], 0.0001);
}

TEST_F(NormalBatchStreamProcessorTest, testLoss) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 2048;
    model_config.vocab_size  = 2048;
    model_config.num_layers  = 2;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    initFullCacheConfig(cache_config, model_config.num_layers);
    RuntimeConfig                  runtime_config;
    std::shared_ptr<GenerateInput> query1   = make_shared<GenerateInput>();
    query1->input_ids                       = hostIntBuffer({1});
    query1->generate_config                 = make_shared<GenerateConfig>();
    query1->generate_config->calculate_loss = 1;
    GenerateStreamPtr stream1 =
        make_shared<NormalGenerateStream>(query1, model_config, runtime_config, resource_context, nullptr);
    BatchKVCacheResource addr1;
    addr1.resetBatchSize(1);
    addr1.initGroups(cache_config.topologyPtr());
    addr1.setBatchBlocks(0, 0, {1});
    stream1->setKVCache(addr1);

    std::shared_ptr<GenerateInput> query3   = make_shared<GenerateInput>();
    query3->input_ids                       = hostIntBuffer({0, 1});
    query3->generate_config                 = make_shared<GenerateConfig>();
    query3->generate_config->calculate_loss = 2;
    GenerateStreamPtr stream3 =
        make_shared<NormalGenerateStream>(query3, model_config, runtime_config, resource_context, nullptr);
    BatchKVCacheResource addr3;
    addr3.resetBatchSize(1);
    addr3.initGroups(cache_config.topologyPtr());
    addr3.setBatchBlocks(0, 0, {9});
    stream3->setKVCache(addr3);

    std::shared_ptr<GenerateInput> query4   = make_shared<GenerateInput>();
    query4->input_ids                       = hostIntBuffer({0, 1, 0});
    query4->generate_config                 = make_shared<GenerateConfig>();
    query4->generate_config->calculate_loss = 1;
    GenerateStreamPtr stream4 =
        make_shared<NormalGenerateStream>(query4, model_config, runtime_config, resource_context, nullptr);
    BatchKVCacheResource addr4;
    addr4.resetBatchSize(1);
    addr4.initGroups(cache_config.topologyPtr());
    addr4.setBatchBlocks(0, 0, {11, 12});
    stream4->setKVCache(addr4);

    std::list<GenerateStreamPtr> streams;
    streams.emplace_back(stream1);
    streams.emplace_back(stream3);
    streams.emplace_back(stream4);

    for (const auto& stream : streams) {
        stream->generate_status_->status = StreamState::RUNNING;
    }
    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);

    StreamGroups stream_groups(streams);
    auto         merge_input_status = processor.gatherModelInput(stream_groups);
    EXPECT_TRUE(merge_input_status.ok());
    EXPECT_TRUE(merge_input_status.value().need_all_logits);

    SamplerInputs sampler_inputs;
    MergedOutput  merge_outputs;
    auto loss_hidden_tensor = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}).reshape({3, 2}).to(torch::kCUDA);
    auto loss_logits_tensor = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}).reshape({3, 2}).to(torch::kCUDA);
    auto loss_all_logits_tensor =
        torch::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f})
            .reshape({6, 2})
            .to(torch::kCUDA);
    merge_outputs.model_output.hidden_states = loss_hidden_tensor;
    merge_outputs.model_output.logits        = loss_logits_tensor;
    merge_outputs.model_output.all_logits    = loss_all_logits_tensor;
    merge_outputs.sampler_output.token_ids =
        torch::tensor({0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1}, torch::kInt32).reshape({3, 4});
    merge_outputs.sampler_output.cum_log_probs = torch::tensor({1.0f, 2.0f, 3.0f}).to(torch::kCUDA);
    auto status                                = processor.dispatch(stream_groups, merge_outputs);
    EXPECT_TRUE(status.ok());
    EXPECT_FALSE(stream1->getLoss().defined());
    EXPECT_TRUE(stream3->getLoss().defined());
    auto loss3 = stream3->getLoss();
    EXPECT_EQ(1, loss3.numel());
    EXPECT_NEAR(0.31326, loss3.data_ptr<float>()[0], 0.0001);
    EXPECT_TRUE(stream4->getLoss().defined());
    auto loss4 = stream4->getLoss();
    EXPECT_EQ(2, loss4.numel());
    EXPECT_NEAR(2.25525, *(torch::mean(loss4).exp().data_ptr<float>()), 0.0001);
}

TEST_F(NormalBatchStreamProcessorTest, testMultimodalGatherBatch) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len                   = 2048;
    model_config.vocab_size                    = 2048;
    model_config.num_layers                    = 2;
    model_config.attn_config.kv_cache_dtype    = KvCacheDataType::FP8;
    model_config.mm_model_config.is_multimodal = true;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    initFullCacheConfig(cache_config, model_config.num_layers);
    RuntimeConfig              runtime_config;
    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);

    std::shared_ptr<GenerateInput> query1 = make_shared<GenerateInput>();
    query1->input_ids                     = hostIntBuffer({1, -1, -1, -1, 2});
    query1->generate_config               = make_shared<GenerateConfig>();
    query1->mm_locs                       = torch::tensor({1}, torch::kInt32);
    query1->text_tokens_mask              = torch::tensor({1, 0, 0, 0, 1}, torch::kInt32);
    query1->multimodal_features           = {torch::rand({3, 10}, torch::kFloat16)};
    GenerateStreamPtr stream1 =
        make_shared<NormalGenerateStream>(query1, model_config, runtime_config, resource_context, nullptr);
    stream1->setIsContextStream(true);

    std::shared_ptr<GenerateInput> query2 = make_shared<GenerateInput>();
    query2->input_ids                     = hostIntBuffer({3, 4, 5});
    query2->generate_config               = make_shared<GenerateConfig>();
    GenerateStreamPtr stream2 =
        make_shared<NormalGenerateStream>(query2, model_config, runtime_config, resource_context, nullptr);
    stream2->setIsContextStream(true);

    std::shared_ptr<GenerateInput> query3 = make_shared<GenerateInput>();
    query3->input_ids                     = hostIntBuffer({6, 7, -1, -1, 8});
    query3->generate_config               = make_shared<GenerateConfig>();
    query3->mm_locs                       = torch::tensor({2}, torch::kInt32);
    query3->text_tokens_mask              = torch::tensor({1, 1, 0, 0, 1}, torch::kInt32);
    query3->multimodal_features           = {torch::rand({2, 10}, torch::kFloat16)};
    GenerateStreamPtr stream3 =
        make_shared<NormalGenerateStream>(query3, model_config, runtime_config, resource_context, nullptr);
    stream3->setIsContextStream(true);

    std::list<GenerateStreamPtr> streams;
    streams.emplace_back(stream1);
    streams.emplace_back(stream2);
    streams.emplace_back(stream3);

    for (const auto& stream : streams) {
        stream->generate_status_->status = StreamState::RUNNING;
    }

    {
        StreamGroups stream_groups(streams);

        auto merge_input_status = processor.gatherModelInput(stream_groups);
        EXPECT_TRUE(merge_input_status.ok());

        auto&       model_input      = merge_input_status.value();
        vector<int> combo_tokens     = {1, -1, -1, -1, 2, 3, 4, 5, 6, 7, -1, -1, 8};
        vector<int> input_lengths    = {5, 3, 5};
        vector<int> text_tokens_mask = {1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1};
        vector<int> mm_features_locs = {1, 10};

        EXPECT_EQ(combo_tokens, toVec<int>(model_input.combo_tokens));
        EXPECT_EQ(input_lengths, toVec<int>(model_input.input_lengths));
        EXPECT_EQ(text_tokens_mask, toVec<int>(model_input.text_tokens_mask));
        EXPECT_EQ(mm_features_locs, toVec<int>(model_input.mm_features_locs));

        EXPECT_EQ(model_input.multimodal_features.value().size(), 2);
        EXPECT_EQ(model_input.multimodal_features.value()[0].numel(), 3 * 10);
        EXPECT_EQ(model_input.multimodal_features.value()[1].numel(), 2 * 10);
    }
}

TEST_F(NormalBatchStreamProcessorTest, testOutputVocabMapsGreedyTokenBeforeStreamUpdate) {
    auto mapping = std::make_shared<OutputVocabMapping>(10, std::vector<int32_t>{0, 2, 7}, "test-model", "digest");

    ResourceContext resource_context;
    resource_context.output_vocab_mapping = mapping;
    ModelConfig model_config;
    model_config.max_seq_len = 8;
    model_config.vocab_size  = 10;
    model_config.num_layers  = 1;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    RuntimeConfig               runtime_config;

    auto query             = make_shared<GenerateInput>();
    query->input_ids       = hostIntBuffer({2});
    query->generate_config = make_shared<GenerateConfig>();
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    stream->generate_status_->status = StreamState::RUNNING;

    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false, mapping);
    StreamGroups    stream_groups({stream});
    GptModelOutputs model_outputs;
    model_outputs.logits = torch::tensor({0.0f, 1.0f, 9.0f}, torch::kFloat32).reshape({1, 3}).to(torch::kCUDA);
    auto sampler_inputs  = processor.gatherSamplerInput(stream_groups, GptModelInputs(), model_outputs);

    ASSERT_TRUE(sampler_inputs.ok());
    Sampler      sampler(SamplerInitParams{});
    MergedOutput merge_outputs;
    merge_outputs.model_output   = model_outputs;
    merge_outputs.sampler_output = sampler.forward(*sampler_inputs);

    ASSERT_TRUE(processor.dispatch(stream_groups, merge_outputs).ok());
    EXPECT_EQ(stream->completeTokenIdsVec(0), (std::vector<int>{2, 7}));
}

TEST_F(NormalBatchStreamProcessorTest, testOutputVocabSamplerFailureIsolatedBeforeTokenMapping) {
    auto mapping = std::make_shared<OutputVocabMapping>(10, std::vector<int32_t>{0, 2, 7}, "test-model", "digest");

    ResourceContext resource_context;
    resource_context.output_vocab_mapping = mapping;
    auto          model_config            = makeOutputVocabTestModelConfig();
    RuntimeConfig runtime_config;

    auto failed_query             = make_shared<GenerateInput>();
    failed_query->input_ids       = hostIntBuffer({2});
    failed_query->generate_config = make_shared<GenerateConfig>();
    auto failed_stream =
        make_shared<NormalGenerateStream>(failed_query, model_config, runtime_config, resource_context, nullptr);
    failed_stream->generate_status_->status = StreamState::RUNNING;

    auto healthy_query             = make_shared<GenerateInput>();
    healthy_query->input_ids       = hostIntBuffer({2});
    healthy_query->generate_config = make_shared<GenerateConfig>();
    auto healthy_stream =
        make_shared<NormalGenerateStream>(healthy_query, model_config, runtime_config, resource_context, nullptr);
    healthy_stream->generate_status_->status = StreamState::RUNNING;

    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    NormalBatchStreamProcessor  processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false, mapping);
    StreamGroups stream_groups({failed_stream, healthy_stream});

    MergedOutput merge_outputs;
    // Row 0 deliberately contains an invalid local id. Its failed status must
    // be observed before local-to-full mapping or stream update.
    merge_outputs.sampler_output.token_ids = torch::tensor({2, 99, 2, 2}, torch::kInt32).reshape({2, 2});
    merge_outputs.sampler_output.success   = torch::tensor({false, true}, torch::kBool);

    ASSERT_TRUE(processor.dispatch(stream_groups, merge_outputs).ok());
    EXPECT_TRUE(failed_stream->hasError());
    EXPECT_EQ(failed_stream->completeTokenIdsVec(0), (std::vector<int>{2}));
    EXPECT_FALSE(healthy_stream->hasError());
    EXPECT_EQ(healthy_stream->completeTokenIdsVec(0), (std::vector<int>{2, 7}));
}

TEST_F(NormalBatchStreamProcessorTest, testOutputVocabGreedyRejectsFullyMaskedCandidates) {
    auto mapping = std::make_shared<OutputVocabMapping>(10, std::vector<int32_t>{0, 2, 7}, "test-model", "digest");

    ResourceContext resource_context;
    resource_context.output_vocab_mapping = mapping;
    auto          model_config            = makeOutputVocabTestModelConfig();
    RuntimeConfig runtime_config;

    auto query                        = make_shared<GenerateInput>();
    query->input_ids                  = hostIntBuffer({2});
    query->generate_config            = make_shared<GenerateConfig>();
    query->generate_config->do_sample = false;
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    stream->logits_processor_list_.insert(stream->logits_processor_list_.begin(),
                                          std::make_shared<MaskAllCandidatesProcessor>());
    stream->generate_status_->status = StreamState::RUNNING;

    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    NormalBatchStreamProcessor  processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false, mapping);
    StreamGroups    stream_groups({stream});
    GptModelOutputs model_outputs;
    model_outputs.logits = torch::zeros({1, 3}, torch::kFloat32).to(torch::kCUDA);

    auto sampler_inputs = processor.gatherSamplerInput(stream_groups, GptModelInputs(), model_outputs);
    ASSERT_TRUE(sampler_inputs.ok());
    EXPECT_TRUE(sampler_inputs->validate_logits_candidates);
    Sampler sampler(SamplerInitParams{});
    auto    sampler_output = sampler.forward(*sampler_inputs);

    ASSERT_TRUE(sampler_output.success.defined());
    EXPECT_FALSE(sampler_output.success.cpu().item<bool>());
}

TEST_F(NormalBatchStreamProcessorTest, testOutputVocabSamplingRejectsNoRepeatWhenKIsOne) {
    auto mapping = std::make_shared<OutputVocabMapping>(10, std::vector<int32_t>{2}, "test-model", "digest");

    ResourceContext resource_context;
    resource_context.output_vocab_mapping = mapping;
    auto          model_config            = makeOutputVocabTestModelConfig();
    RuntimeConfig runtime_config;

    auto query                                   = make_shared<GenerateInput>();
    query->input_ids                             = hostIntBuffer({2});
    query->generate_config                       = make_shared<GenerateConfig>();
    query->generate_config->do_sample            = true;
    query->generate_config->top_k                = 0;
    query->generate_config->no_repeat_ngram_size = 1;
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    stream->generate_status_->status = StreamState::RUNNING;

    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    NormalBatchStreamProcessor  processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false, mapping);
    StreamGroups    stream_groups({stream});
    GptModelOutputs model_outputs;
    model_outputs.logits = torch::zeros({1, 1}, torch::kFloat32).to(torch::kCUDA);

    auto sampler_inputs = processor.gatherSamplerInput(stream_groups, GptModelInputs(), model_outputs);
    ASSERT_TRUE(sampler_inputs.ok());
    ASSERT_TRUE(sampler_inputs->penalty_token_ids.defined());
    Sampler sampler(SamplerInitParams{});
    auto    sampler_output = sampler.forward(*sampler_inputs);

    ASSERT_TRUE(sampler_output.success.defined());
    EXPECT_FALSE(sampler_output.success.cpu().item<bool>());
}

TEST_F(NormalBatchStreamProcessorTest, testOutputVocabThinkMaskSurvivesTemperatureBeforeNoRepeat) {
    auto mapping = std::make_shared<OutputVocabMapping>(10, std::vector<int32_t>{0, 2, 7}, "test-model", "digest");

    ResourceContext resource_context;
    resource_context.output_vocab_mapping = mapping;
    auto          model_config            = makeOutputVocabTestModelConfig();
    RuntimeConfig runtime_config;

    auto query                                   = make_shared<GenerateInput>();
    query->input_ids                             = hostIntBuffer({7});
    query->generate_config                       = make_shared<GenerateConfig>();
    query->generate_config->do_sample            = true;
    query->generate_config->top_k                = 0;
    query->generate_config->temperature          = 2.0f;
    query->generate_config->no_repeat_ngram_size = 1;
    query->generate_config->in_think_mode        = true;
    query->generate_config->max_thinking_tokens  = 1;
    query->generate_config->end_think_token_ids  = {7};
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    stream->generate_status_->status = StreamState::RUNNING;

    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    NormalBatchStreamProcessor  processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false, mapping);
    StreamGroups    stream_groups({stream});
    GptModelOutputs model_outputs;
    model_outputs.logits = torch::zeros({1, 3}, torch::kFloat32).to(torch::kCUDA);

    auto sampler_inputs = processor.gatherSamplerInput(stream_groups, GptModelInputs(), model_outputs);
    ASSERT_TRUE(sampler_inputs.ok());
    ASSERT_TRUE(sampler_inputs->penalty_token_ids.defined());
    Sampler sampler(SamplerInitParams{});
    auto    sampler_output = sampler.forward(*sampler_inputs);

    ASSERT_TRUE(sampler_output.success.defined());
    EXPECT_TRUE(sampler_output.success.cpu().item<bool>());
    EXPECT_EQ(sampler_inputs->no_repeat_ngram_size.data_ptr<int32_t>()[0], 0);
    auto sampled_ids = sampler_output.token_ids.cpu();
    EXPECT_EQ(sampled_ids[0][sampled_ids.size(1) - 1].item<int32_t>(), 2);
}

TEST_F(NormalBatchStreamProcessorTest, testOutputVocabBeamRejectsFullyMaskedCandidates) {
    auto mapping =
        std::make_shared<OutputVocabMapping>(10, std::vector<int32_t>{0, 2, 4, 7, 9}, "test-model", "digest");

    ResourceContext resource_context;
    resource_context.output_vocab_mapping = mapping;
    auto          model_config            = makeOutputVocabTestModelConfig();
    RuntimeConfig runtime_config;

    auto query                        = make_shared<GenerateInput>();
    query->input_ids                  = hostIntBuffer({2});
    query->generate_config            = make_shared<GenerateConfig>();
    query->generate_config->num_beams = 2;
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    stream->logits_processor_list_.insert(stream->logits_processor_list_.begin(),
                                          std::make_shared<MaskAllCandidatesProcessor>());
    stream->generate_status_->status = StreamState::RUNNING;

    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    NormalBatchStreamProcessor  processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false, mapping);
    StreamGroups    stream_groups({stream});
    GptModelOutputs model_outputs;
    model_outputs.logits = torch::zeros({1, 5}, torch::kFloat32).to(torch::kCUDA);

    auto sampler_inputs = processor.gatherSamplerInput(stream_groups, GptModelInputs(), model_outputs);
    ASSERT_TRUE(sampler_inputs.ok());
    Sampler sampler(SamplerInitParams{});
    auto    sampler_output = sampler.forward(*sampler_inputs);

    ASSERT_TRUE(sampler_output.success.defined());
    EXPECT_FALSE(sampler_output.success.cpu().item<bool>());
}

TEST_F(NormalBatchStreamProcessorTest, testOutputVocabBeamRejectsInsufficientGroupCandidates) {
    const float neg_inf = -std::numeric_limits<float>::infinity();

    {
        auto inputs = makeValidatedBeamInputs(
            torch::tensor({{1.0f, neg_inf, neg_inf, neg_inf, neg_inf, neg_inf, neg_inf, neg_inf, neg_inf}},
                          torch::kFloat32),
            1,
            1,
            4);
        Sampler sampler(SamplerInitParams{});
        auto    output = sampler.forward(inputs);
        EXPECT_EQ(toVec<bool>(output.success.cpu()), (std::vector<bool>{false}));
    }

    {
        auto inputs = makeValidatedBeamInputs(
            torch::tensor({{1.0f, 2.0f, neg_inf, neg_inf, neg_inf, neg_inf, neg_inf, neg_inf, neg_inf},
                           {3.0f, neg_inf, neg_inf, neg_inf, neg_inf, neg_inf, neg_inf, neg_inf, neg_inf}},
                          torch::kFloat32),
            1,
            2,
            4);
        Sampler sampler(SamplerInitParams{});
        auto    output = sampler.forward(inputs);
        EXPECT_EQ(toVec<bool>(output.success.cpu()), (std::vector<bool>{false, false}));
    }
}

TEST_F(NormalBatchStreamProcessorTest, testOutputVocabBeamUsesOtherParentsWhenOneIsEmpty) {
    const float neg_inf = -std::numeric_limits<float>::infinity();
    auto        inputs  = makeValidatedBeamInputs(
        torch::tensor({{neg_inf, neg_inf, neg_inf, neg_inf, neg_inf, neg_inf, neg_inf, neg_inf, neg_inf},
                               {4.0f, 3.0f, neg_inf, 2.0f, neg_inf, 1.0f, neg_inf, neg_inf, neg_inf}},
                      torch::kFloat32),
        1,
        2,
        4);

    Sampler sampler(SamplerInitParams{});
    auto    output = sampler.forward(inputs);
    EXPECT_EQ(toVec<bool>(output.success.cpu()), (std::vector<bool>{true, true}));
    EXPECT_EQ(toVec<int32_t>(output.beam_index.cpu()), (std::vector<int32_t>{1, 1, 1, 1}));

    auto token_ids = output.token_ids.cpu().contiguous();
    for (int64_t child = 0; child < token_ids.size(0); ++child) {
        const int32_t token = token_ids[child][1].item<int32_t>();
        EXPECT_TRUE(token == 0 || token == 1 || token == 3 || token == 5);
    }
}

TEST_F(NormalBatchStreamProcessorTest, testOutputVocabBeamCandidateFailuresAreIsolatedByGroup) {
    const float neg_inf = -std::numeric_limits<float>::infinity();
    auto        inputs  = makeValidatedBeamInputs(
        torch::tensor({{4.0f, 3.0f, 2.0f, 1.0f, neg_inf, neg_inf, neg_inf, neg_inf, neg_inf},
                               {4.0f, neg_inf, neg_inf, neg_inf, neg_inf, neg_inf, neg_inf, neg_inf, neg_inf}},
                      torch::kFloat32),
        2,
        1,
        4);

    Sampler sampler(SamplerInitParams{});
    auto    output = sampler.forward(inputs);
    EXPECT_EQ(toVec<bool>(output.success.cpu()), (std::vector<bool>{true, false}));
}

TEST_F(NormalBatchStreamProcessorTest, testOutputVocabBeamRejectsMixedNanAndPositiveInfinityByGroup) {
    const float neg_inf = -std::numeric_limits<float>::infinity();
    for (const float unsafe_value : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
        SCOPED_TRACE(std::isnan(unsafe_value) ? "NaN" : "+inf");
        auto inputs = makeValidatedBeamInputs(
            torch::tensor({{4.0f, 3.0f, 2.0f, 1.0f, unsafe_value, neg_inf, neg_inf, neg_inf, neg_inf},
                           {4.0f, 3.0f, 2.0f, 1.0f, neg_inf, neg_inf, neg_inf, neg_inf, neg_inf}},
                          torch::kFloat32),
            2,
            1,
            4);
        inputs.all_probs = torch::zeros({8, 9}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

        Sampler sampler(SamplerInitParams{});
        auto    output = sampler.forward(inputs);

        EXPECT_EQ(toVec<bool>(output.success.cpu()), (std::vector<bool>{false, true}));
        EXPECT_FALSE(torch::isnan(inputs.logits).any().item<bool>());
        EXPECT_FALSE(torch::isnan(output.all_probs).any().item<bool>());
    }
}

TEST_F(NormalBatchStreamProcessorTest, testOutputVocabBeamExcludesNonFiniteParentScoresFromCapacity) {
    const float neg_inf = -std::numeric_limits<float>::infinity();
    for (const float invalid_score :
         {neg_inf, std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
        SCOPED_TRACE(std::to_string(invalid_score));
        auto enough_inputs = makeValidatedBeamInputs(
            torch::tensor({{9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f},
                           {4.0f, 3.0f, 2.0f, 1.0f, neg_inf, neg_inf, neg_inf, neg_inf, neg_inf}},
                          torch::kFloat32),
            1,
            2,
            4,
            torch::tensor({invalid_score, 0.0f}, torch::kFloat32));
        Sampler enough_sampler(SamplerInitParams{});
        auto    enough_output = enough_sampler.forward(enough_inputs);
        EXPECT_EQ(toVec<bool>(enough_output.success.cpu()), (std::vector<bool>{true, true}));
        EXPECT_EQ(toVec<int32_t>(enough_output.beam_index.cpu()), (std::vector<int32_t>{1, 1, 1, 1}));

        auto insufficient_inputs = makeValidatedBeamInputs(
            torch::tensor({{9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f},
                           {4.0f, 3.0f, 2.0f, neg_inf, neg_inf, neg_inf, neg_inf, neg_inf, neg_inf}},
                          torch::kFloat32),
            1,
            2,
            4,
            torch::tensor({invalid_score, 0.0f}, torch::kFloat32));
        Sampler insufficient_sampler(SamplerInitParams{});
        auto    insufficient_output = insufficient_sampler.forward(insufficient_inputs);
        EXPECT_EQ(toVec<bool>(insufficient_output.success.cpu()), (std::vector<bool>{false, false}));
    }
}

TEST_F(NormalBatchStreamProcessorTest, testFinishedEosOverridesThinkHardTerminatorBeforeBeamSampling) {
    const float neg_inf         = -std::numeric_limits<float>::infinity();
    auto        inputs          = makeValidatedBeamInputs(torch::zeros({2, 5}, torch::kFloat32), 1, 2, 2);
    inputs.finished_mask        = torch::tensor({true, false}, torch::kBool);
    inputs.no_repeat_ngram_size = torch::ones({2}, torch::kInt32);

    std::vector<StreamThinkInfo> think_infos;
    for (int i = 0; i < 2; ++i) {
        think_infos.emplace_back(
            true, 1, std::vector<int>{1}, 1, 0, std::make_shared<StringContainDFA<size_t, int>>(std::vector<int>{1}));
    }
    auto states = std::make_shared<LogitsProcessorStates>();
    states->insert(std::make_shared<ThinkModeLogitsProcessor>(std::move(think_infos)), 0, 2);
    states->insert(std::make_shared<LocalEosOnlyProcessor>(2), 0, 2);
    inputs.logits_processor_states_ptr = states;

    Sampler sampler(SamplerInitParams{});
    auto    output = sampler.forward(inputs);

    EXPECT_EQ(toVec<bool>(output.success.cpu()), (std::vector<bool>{true, true}));
    auto processed_logits = inputs.logits.cpu();
    EXPECT_FLOAT_EQ(processed_logits[0][2].item<float>(), 0.0f);
    EXPECT_EQ(processed_logits[0][1].item<float>(), neg_inf);
    EXPECT_FLOAT_EQ(processed_logits[1][1].item<float>(), 0.0f);
    EXPECT_EQ(processed_logits[1][2].item<float>(), neg_inf);
    EXPECT_EQ(toVec<int32_t>(inputs.no_repeat_ngram_size), (std::vector<int32_t>{0, 0}));
}

TEST_F(NormalBatchStreamProcessorTest, testOutputVocabMapsMixedLengthGreedyBatchFromLastColumn) {
    auto mapping = std::make_shared<OutputVocabMapping>(10, std::vector<int32_t>{0, 2, 7}, "test-model", "digest");

    ResourceContext resource_context;
    resource_context.output_vocab_mapping = mapping;
    ModelConfig model_config;
    model_config.max_seq_len = 8;
    model_config.vocab_size  = 10;
    model_config.num_layers  = 1;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    RuntimeConfig               runtime_config;

    auto short_query             = make_shared<GenerateInput>();
    short_query->input_ids       = hostIntBuffer({2});
    short_query->generate_config = make_shared<GenerateConfig>();
    auto short_stream =
        make_shared<NormalGenerateStream>(short_query, model_config, runtime_config, resource_context, nullptr);
    short_stream->generate_status_->status = StreamState::RUNNING;

    auto long_query             = make_shared<GenerateInput>();
    long_query->input_ids       = hostIntBuffer({2, 7, 2});
    long_query->generate_config = make_shared<GenerateConfig>();
    auto long_stream =
        make_shared<NormalGenerateStream>(long_query, model_config, runtime_config, resource_context, nullptr);
    long_stream->generate_status_->status = StreamState::RUNNING;

    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false, mapping);
    StreamGroups stream_groups({short_stream, long_stream});
    MergedOutput merge_outputs;
    merge_outputs.sampler_output.token_ids = torch::tensor({2, 99, 99, 2, 2, 7, 2, 1}, torch::kInt32).reshape({2, 4});

    ASSERT_TRUE(processor.dispatch(stream_groups, merge_outputs).ok());
    EXPECT_EQ(short_stream->completeTokenIdsVec(0), (std::vector<int>{2, 7}));
    EXPECT_EQ(long_stream->completeTokenIdsVec(0), (std::vector<int>{2, 7, 2, 2}));
}

TEST_F(NormalBatchStreamProcessorTest, testOutputVocabMapsOnlyLatestBeamTokens) {
    auto mapping =
        std::make_shared<OutputVocabMapping>(10, std::vector<int32_t>{0, 2, 4, 7, 9}, "test-model", "digest");

    auto cache_config = test::makeSimpleMhaCacheConfig(
        /*layer_num=*/1, /*block_num=*/8, /*tokens_per_block=*/2, rtp_llm::TYPE_INT8);
    ResourceContext resource_context;
    resource_context.output_vocab_mapping = mapping;
    resource_context.cache_manager        = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(resource_context.cache_manager->init());
    ModelConfig model_config;
    model_config.max_seq_len = 8;
    model_config.vocab_size  = 10;
    model_config.num_layers  = 1;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    RuntimeConfig               runtime_config;

    auto query                        = make_shared<GenerateInput>();
    query->input_ids                  = hostIntBuffer({2});
    query->generate_config            = make_shared<GenerateConfig>();
    query->generate_config->num_beams = 2;
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    ASSERT_TRUE(stream->initKVBlock().ok());
    stream->generate_status_->status = StreamState::RUNNING;

    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false, mapping);
    StreamGroups    stream_groups({stream});
    GptModelOutputs model_outputs;
    model_outputs.logits =
        torch::tensor({0.0f, 1.0f, 2.0f, 10.0f, 9.0f}, torch::kFloat32).reshape({1, 5}).to(torch::kCUDA);
    auto sampler_inputs = processor.gatherSamplerInput(stream_groups, GptModelInputs(), model_outputs);

    ASSERT_TRUE(sampler_inputs.ok());
    EXPECT_EQ(sampler_inputs->batch_size, 1);
    EXPECT_EQ(sampler_inputs->batch_size_out, 2);
    Sampler      sampler(SamplerInitParams{});
    MergedOutput merge_outputs;
    merge_outputs.model_output   = model_outputs;
    merge_outputs.sampler_output = sampler.forward(*sampler_inputs);

    ASSERT_TRUE(processor.dispatch(stream_groups, merge_outputs).ok());
    EXPECT_EQ(stream->completeTokenIdsVec(0), (std::vector<int>{2, 7}));
    EXPECT_EQ(stream->completeTokenIdsVec(1), (std::vector<int>{2, 9}));
}

TEST_F(NormalBatchStreamProcessorTest, testOutputVocabSupportsSmallSamplingVocabAndClampsTopK) {
    auto mapping = std::make_shared<OutputVocabMapping>(10, std::vector<int32_t>{0, 7}, "test-model", "digest");

    ResourceContext resource_context;
    resource_context.output_vocab_mapping = mapping;
    ModelConfig model_config;
    model_config.max_seq_len = 8;
    model_config.vocab_size  = 10;
    model_config.num_layers  = 1;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    RuntimeConfig               runtime_config;

    auto query                        = make_shared<GenerateInput>();
    query->input_ids                  = hostIntBuffer({2});
    query->generate_config            = make_shared<GenerateConfig>();
    query->generate_config->top_k     = 8;
    query->generate_config->top_p     = 0.8f;
    query->generate_config->do_sample = true;
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    stream->generate_status_->status = StreamState::RUNNING;

    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false, mapping);
    StreamGroups    stream_groups({stream});
    GptModelOutputs model_outputs;
    model_outputs.logits = torch::zeros({1, 2}, torch::kFloat32).to(torch::kCUDA);
    auto sampler_inputs  = processor.gatherSamplerInput(stream_groups, GptModelInputs(), model_outputs);

    ASSERT_TRUE(sampler_inputs.ok());
    EXPECT_EQ(sampler_inputs->top_k.data_ptr<int32_t>()[0], 2);
    EXPECT_TRUE(sampler_inputs->do_sample.data_ptr<bool>()[0]);

    MergedOutput merge_outputs;
    merge_outputs.model_output.logits      = model_outputs.logits;
    merge_outputs.sampler_output.token_ids = torch::tensor({2, 1}, torch::kInt32).reshape({1, 2});
    ASSERT_TRUE(processor.dispatch(stream_groups, merge_outputs).ok());
    EXPECT_EQ(stream->completeTokenIdsVec(0), (std::vector<int>{2, 7}));
}

TEST_F(NormalBatchStreamProcessorTest, testOutputVocabMapsDynamicBeamAcrossExpansionSteps) {
    auto mapping =
        std::make_shared<OutputVocabMapping>(10, std::vector<int32_t>{0, 1, 2, 4, 5, 7, 9}, "test-model", "digest");

    auto cache_config = test::makeSimpleMhaCacheConfig(
        /*layer_num=*/1, /*block_num=*/16, /*tokens_per_block=*/2, rtp_llm::TYPE_INT8);
    ResourceContext resource_context;
    resource_context.output_vocab_mapping = mapping;
    resource_context.cache_manager        = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(resource_context.cache_manager->init());
    ModelConfig model_config;
    model_config.max_seq_len = 8;
    model_config.vocab_size  = 10;
    model_config.num_layers  = 1;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    RuntimeConfig               runtime_config;

    auto query                                 = make_shared<GenerateInput>();
    query->input_ids                           = hostIntBuffer({2});
    query->generate_config                     = make_shared<GenerateConfig>();
    query->generate_config->variable_num_beams = {2, 3};
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    ASSERT_TRUE(stream->initKVBlock().ok());
    stream->generate_status_->status = StreamState::RUNNING;

    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false, mapping);

    {
        StreamGroups    stream_groups({stream});
        GptModelOutputs model_outputs;
        model_outputs.logits = torch::zeros({1, 7}, torch::kFloat32).to(torch::kCUDA);
        auto sampler_inputs  = processor.gatherSamplerInput(stream_groups, GptModelInputs(), model_outputs);
        ASSERT_TRUE(sampler_inputs.ok());
        EXPECT_EQ(sampler_inputs->num_beams_in.data_ptr<int64_t>()[0], 1);
        EXPECT_EQ(sampler_inputs->num_beams_out.data_ptr<int64_t>()[0], 2);

        MergedOutput merge_outputs;
        merge_outputs.model_output.logits       = model_outputs.logits;
        merge_outputs.sampler_output.token_ids  = torch::tensor({2, 3, 2, 5}, torch::kInt32).reshape({2, 2});
        merge_outputs.sampler_output.beam_index = torch::zeros({2}, torch::kInt32);

        ASSERT_TRUE(processor.dispatch(stream_groups, merge_outputs).ok());
        EXPECT_EQ(stream->completeTokenIdsVec(0), (std::vector<int>{2, 4}));
        EXPECT_EQ(stream->completeTokenIdsVec(1), (std::vector<int>{2, 7}));
    }

    {
        StreamGroups    stream_groups({stream});
        GptModelOutputs model_outputs;
        model_outputs.logits = torch::zeros({2, 7}, torch::kFloat32).to(torch::kCUDA);
        auto sampler_inputs  = processor.gatherSamplerInput(stream_groups, GptModelInputs(), model_outputs);
        ASSERT_TRUE(sampler_inputs.ok());
        EXPECT_EQ(sampler_inputs->num_beams_in.data_ptr<int64_t>()[0], 2);
        EXPECT_EQ(sampler_inputs->num_beams_out.data_ptr<int64_t>()[0], 3);

        MergedOutput merge_outputs;
        merge_outputs.model_output.logits = model_outputs.logits;
        merge_outputs.sampler_output.token_ids =
            torch::tensor({2, 4, 1, 2, 7, 3, 2, 7, 6}, torch::kInt32).reshape({3, 3});
        merge_outputs.sampler_output.beam_index = torch::tensor({0, 1, 1}, torch::kInt32);

        ASSERT_TRUE(processor.dispatch(stream_groups, merge_outputs).ok());
        EXPECT_EQ(stream->completeTokenIdsVec(0), (std::vector<int>{2, 4, 1}));
        EXPECT_EQ(stream->completeTokenIdsVec(1), (std::vector<int>{2, 7, 4}));
        EXPECT_EQ(stream->completeTokenIdsVec(2), (std::vector<int>{2, 7, 9}));
    }
}

TEST_F(NormalBatchStreamProcessorTest, testOutputVocabDynamicBeamOneToOneThenExpandInMixedSamplerBatch) {
    auto mapping =
        std::make_shared<OutputVocabMapping>(10, std::vector<int32_t>{0, 1, 2, 4, 5, 7, 9}, "test-model", "digest");
    auto cache_config = test::makeSimpleMhaCacheConfig(
        /*layer_num=*/1, /*block_num=*/32, /*tokens_per_block=*/2, rtp_llm::TYPE_INT8);
    ResourceContext resource_context;
    resource_context.output_vocab_mapping = mapping;
    resource_context.cache_manager        = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(resource_context.cache_manager->init());

    auto                        model_config = makeOutputVocabTestModelConfig();
    RuntimeConfig               runtime_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    auto incremental_stream = makeDynamicBeamStream({2}, {1, 2}, 4, model_config, runtime_config, resource_context);
    auto expanding_stream   = makeDynamicBeamStream({2}, {2}, 4, model_config, runtime_config, resource_context);
    ASSERT_TRUE(incremental_stream->initKVBlock().ok());
    ASSERT_TRUE(expanding_stream->initKVBlock().ok());

    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false, mapping);
    EXPECT_FALSE(incremental_stream->usesBeamSearchTokenLayoutForCurrentStep());
    EXPECT_TRUE(expanding_stream->usesBeamSearchTokenLayoutForCurrentStep());

    {
        StreamGroups    stream_groups({incremental_stream, expanding_stream});
        GptModelOutputs model_outputs;
        model_outputs.logits =
            torch::tensor({0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 10.0f, 0.0f, 0.0f, 0.0f, 0.0f, 10.0f, 9.0f, 0.0f, 0.0f},
                          torch::kFloat32)
                .reshape({2, 7})
                .to(torch::kCUDA);
        auto sampler_inputs = processor.gatherSamplerInput(stream_groups, GptModelInputs(), model_outputs);
        ASSERT_TRUE(sampler_inputs.ok());
        EXPECT_EQ(sampler_inputs->batch_size, 2);
        EXPECT_EQ(sampler_inputs->batch_size_out, 3);
        EXPECT_EQ(toVec<int64_t>(sampler_inputs->num_beams_in), (std::vector<int64_t>{1, 1}));
        EXPECT_EQ(toVec<int64_t>(sampler_inputs->num_beams_out), (std::vector<int64_t>{1, 2}));

        Sampler      sampler(SamplerInitParams{});
        MergedOutput merge_outputs;
        merge_outputs.model_output   = model_outputs;
        merge_outputs.sampler_output = sampler.forward(*sampler_inputs);
        ASSERT_EQ(merge_outputs.sampler_output.token_ids.size(0), 3);
        ASSERT_EQ(merge_outputs.sampler_output.token_ids.size(1), 2);
        ASSERT_TRUE(processor.dispatch(stream_groups, merge_outputs).ok());
    }

    EXPECT_EQ(incremental_stream->completeTokenIdsVec(0), (std::vector<int>{2, 7}));
    EXPECT_EQ(incremental_stream->kvCache().batchSize(), 1);
    EXPECT_TRUE(incremental_stream->streamCacheResource().getKVBlockUpdateMapping().empty());
    EXPECT_EQ(expanding_stream->completeTokenIdsVec(0), (std::vector<int>{2, 4}));
    EXPECT_EQ(expanding_stream->completeTokenIdsVec(1), (std::vector<int>{2, 5}));
    EXPECT_EQ(expanding_stream->kvCache().batchSize(), 2);
    EXPECT_TRUE(incremental_stream->usesBeamSearchTokenLayoutForCurrentStep());

    {
        StreamGroups stream_groups({incremental_stream});
        MergedOutput merge_outputs;
        merge_outputs.model_output.logits       = torch::zeros({1, 7}, torch::kFloat32).to(torch::kCUDA);
        merge_outputs.sampler_output.token_ids  = torch::tensor({2, 7, 3, 2, 7, 6}, torch::kInt32).reshape({2, 3});
        merge_outputs.sampler_output.beam_index = torch::zeros({2}, torch::kInt32);
        ASSERT_TRUE(processor.dispatch(stream_groups, merge_outputs).ok());
    }

    EXPECT_EQ(incremental_stream->completeTokenIdsVec(0), (std::vector<int>{2, 7, 4}));
    EXPECT_EQ(incremental_stream->completeTokenIdsVec(1), (std::vector<int>{2, 7, 9}));
    EXPECT_EQ(incremental_stream->kvCache().batchSize(), 2);
}

TEST_F(NormalBatchStreamProcessorTest, testOutputVocabMixedExpansionAndShrinkUseSeparateSamplerOutputs) {
    auto mapping =
        std::make_shared<OutputVocabMapping>(10, std::vector<int32_t>{0, 1, 2, 4, 5, 7, 9}, "test-model", "digest");
    auto cache_config = test::makeSimpleMhaCacheConfig(
        /*layer_num=*/1, /*block_num=*/32, /*tokens_per_block=*/2, rtp_llm::TYPE_INT8);
    ResourceContext resource_context;
    resource_context.output_vocab_mapping = mapping;
    resource_context.cache_manager        = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(resource_context.cache_manager->init());

    auto                        model_config = makeOutputVocabTestModelConfig();
    RuntimeConfig               runtime_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    auto expanding_stream = makeDynamicBeamStream({2}, {2}, 3, model_config, runtime_config, resource_context);
    auto shrinking_stream = makeDynamicBeamStream({2}, {2, 1}, 4, model_config, runtime_config, resource_context);
    ASSERT_TRUE(expanding_stream->initKVBlock().ok());
    ASSERT_TRUE(shrinking_stream->initKVBlock().ok());
    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false, mapping);

    // Advance one stream to 2 -> 1 and give its two parents distinct histories
    // and cumulative scores before the balanced mixed batch.
    {
        StreamGroups stream_groups({shrinking_stream});
        MergedOutput merge_outputs;
        merge_outputs.model_output.logits          = torch::zeros({1, 7}, torch::kFloat32).to(torch::kCUDA);
        merge_outputs.sampler_output.token_ids     = torch::tensor({2, 5, 2, 6}, torch::kInt32).reshape({2, 2});
        merge_outputs.sampler_output.beam_index    = torch::zeros({2}, torch::kInt32);
        merge_outputs.sampler_output.cum_log_probs = torch::tensor({-0.2f, -5.0f}, torch::kFloat32);
        ASSERT_TRUE(processor.dispatch(stream_groups, merge_outputs).ok());
    }
    EXPECT_EQ(shrinking_stream->completeTokenIdsVec(0), (std::vector<int>{2, 7}));
    EXPECT_EQ(shrinking_stream->completeTokenIdsVec(1), (std::vector<int>{2, 9}));
    EXPECT_TRUE(shrinking_stream->usesBeamSearchTokenLayoutForCurrentStep());

    // Keep both streams in the decode list so the 1 -> 2 group is processed
    // before the 2 -> 1 group. Total sampler size remains 3 -> 3.
    expanding_stream->setIsContextStream(false);
    StreamGroups stream_groups({expanding_stream, shrinking_stream});
    EXPECT_EQ(stream_groups.totalSamplerBatchSizeIn(), 3);
    EXPECT_EQ(stream_groups.totalSamplerBatchSizeOut(), 3);

    GptModelOutputs model_outputs;
    model_outputs.logits = torch::tensor({0.0f, 0.0f, 0.0f,   10.0f, 9.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                          0.0f, 0.0f, 100.0f, 0.0f,  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
                                         torch::kFloat32)
                               .reshape({3, 7})
                               .to(torch::kCUDA);
    auto reference_log_probs = torch::log_softmax(model_outputs.logits.cpu(), -1).contiguous();
    auto sampler_inputs      = processor.gatherSamplerInput(stream_groups, GptModelInputs(), model_outputs);
    ASSERT_TRUE(sampler_inputs.ok());
    EXPECT_EQ(toVec<int64_t>(sampler_inputs->num_beams_in), (std::vector<int64_t>{1, 2, 2}));
    EXPECT_EQ(toVec<int64_t>(sampler_inputs->num_beams_out), (std::vector<int64_t>{2, 1, 1}));
    auto input_cum_log_probs = toVec<float>(sampler_inputs->cum_log_probs);
    ASSERT_EQ(input_cum_log_probs.size(), 3);
    EXPECT_FLOAT_EQ(input_cum_log_probs[0], 0.0f);
    EXPECT_FLOAT_EQ(input_cum_log_probs[1], -0.2f);
    EXPECT_FLOAT_EQ(input_cum_log_probs[2], -5.0f);

    Sampler      sampler(SamplerInitParams{});
    MergedOutput merge_outputs;
    merge_outputs.model_output   = model_outputs;
    merge_outputs.sampler_output = sampler.forward(*sampler_inputs);
    ASSERT_EQ(merge_outputs.sampler_output.token_ids.size(0), 3);
    ASSERT_EQ(merge_outputs.sampler_output.token_ids.size(1), 3);
    ASSERT_EQ(merge_outputs.sampler_output.cum_log_probs.numel(), 3);
    ASSERT_EQ(merge_outputs.sampler_output.beam_index.numel(), 3);

    auto        output_token_ids     = merge_outputs.sampler_output.token_ids.cpu().contiguous();
    auto        output_cum_log_probs = merge_outputs.sampler_output.cum_log_probs.cpu().contiguous();
    auto        output_beam_indices  = merge_outputs.sampler_output.beam_index.cpu().contiguous();
    const auto  token_stride         = output_token_ids.size(1);
    const auto* token_ptr            = output_token_ids.data_ptr<int32_t>();
    const auto* cum_ptr              = output_cum_log_probs.data_ptr<float>();
    const auto* beam_ptr             = output_beam_indices.data_ptr<int32_t>();
    const auto* ref_ptr              = reference_log_probs.data_ptr<float>();
    const int   expanding_token_0    = token_ptr[1];
    const int   expanding_token_1    = token_ptr[token_stride + 1];
    const int   shrinking_parent     = beam_ptr[2];
    const int   shrinking_token      = token_ptr[2 * token_stride + 2];
    ASSERT_EQ(shrinking_parent, 0);
    EXPECT_NEAR(cum_ptr[0], ref_ptr[expanding_token_0], 1e-5);
    EXPECT_NEAR(cum_ptr[1], ref_ptr[expanding_token_1], 1e-5);
    EXPECT_NEAR(cum_ptr[2], -0.2f + ref_ptr[7 + shrinking_token], 1e-5);

    ASSERT_TRUE(processor.dispatch(stream_groups, merge_outputs).ok());
    EXPECT_EQ(expanding_stream->completeTokenIdsVec(0), (std::vector<int>{2, mapping->toFull(expanding_token_0)}));
    EXPECT_EQ(expanding_stream->completeTokenIdsVec(1), (std::vector<int>{2, mapping->toFull(expanding_token_1)}));
    EXPECT_EQ(shrinking_stream->completeTokenIdsVec(0), (std::vector<int>{2, 7, mapping->toFull(shrinking_token)}));
    EXPECT_EQ(expanding_stream->kvCache().batchSize(), 2);
    EXPECT_EQ(shrinking_stream->kvCache().batchSize(), 1);

    auto expanding_cum_log_probs = toVec<float>(expanding_stream->cumLogProbs());
    auto shrinking_cum_log_probs = toVec<float>(shrinking_stream->cumLogProbs());
    ASSERT_EQ(expanding_cum_log_probs.size(), 2);
    ASSERT_EQ(shrinking_cum_log_probs.size(), 1);
    EXPECT_NEAR(expanding_cum_log_probs[0], cum_ptr[0], 1e-5);
    EXPECT_NEAR(expanding_cum_log_probs[1], cum_ptr[1], 1e-5);
    EXPECT_NEAR(shrinking_cum_log_probs[0], cum_ptr[2], 1e-5);
}

TEST_F(NormalBatchStreamProcessorTest, testOutputVocabDynamicBeamTwoToOneToOneUsesStepLayouts) {
    auto mapping =
        std::make_shared<OutputVocabMapping>(10, std::vector<int32_t>{0, 1, 2, 4, 5, 7, 9}, "test-model", "digest");
    auto cache_config = test::makeSimpleMhaCacheConfig(
        /*layer_num=*/1, /*block_num=*/24, /*tokens_per_block=*/2, rtp_llm::TYPE_INT8);
    ResourceContext resource_context;
    resource_context.output_vocab_mapping = mapping;
    resource_context.cache_manager        = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(resource_context.cache_manager->init());

    auto                        model_config = makeOutputVocabTestModelConfig();
    RuntimeConfig               runtime_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    auto stream = makeDynamicBeamStream({2}, {2, 1}, 4, model_config, runtime_config, resource_context);
    ASSERT_TRUE(stream->initKVBlock().ok());
    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false, mapping);

    {
        EXPECT_TRUE(stream->usesBeamSearchTokenLayoutForCurrentStep());
        StreamGroups stream_groups({stream});
        MergedOutput merge_outputs;
        merge_outputs.model_output.logits       = torch::zeros({1, 7}, torch::kFloat32).to(torch::kCUDA);
        merge_outputs.sampler_output.token_ids  = torch::tensor({2, 3, 2, 5}, torch::kInt32).reshape({2, 2});
        merge_outputs.sampler_output.beam_index = torch::zeros({2}, torch::kInt32);
        ASSERT_TRUE(processor.dispatch(stream_groups, merge_outputs).ok());
    }
    EXPECT_EQ(stream->completeTokenIdsVec(0), (std::vector<int>{2, 4}));
    EXPECT_EQ(stream->completeTokenIdsVec(1), (std::vector<int>{2, 7}));
    EXPECT_EQ(stream->kvCache().batchSize(), 2);

    {
        EXPECT_TRUE(stream->usesBeamSearchTokenLayoutForCurrentStep());
        StreamGroups stream_groups({stream});
        MergedOutput merge_outputs;
        merge_outputs.model_output.logits       = torch::zeros({2, 7}, torch::kFloat32).to(torch::kCUDA);
        merge_outputs.sampler_output.token_ids  = torch::tensor({2, 7, 6}, torch::kInt32).reshape({1, 3});
        merge_outputs.sampler_output.beam_index = torch::tensor({1}, torch::kInt32);
        ASSERT_TRUE(processor.dispatch(stream_groups, merge_outputs).ok());
    }
    EXPECT_EQ(stream->completeTokenIdsVec(0), (std::vector<int>{2, 7, 9}));
    EXPECT_EQ(stream->kvCache().batchSize(), 1);
    EXPECT_FALSE(stream->usesBeamSearchTokenLayoutForCurrentStep());

    {
        StreamGroups stream_groups({stream});
        MergedOutput merge_outputs;
        merge_outputs.sampler_output.token_ids = torch::tensor({2, 7, 9, 3}, torch::kInt32).reshape({1, 4});
        ASSERT_TRUE(processor.dispatch(stream_groups, merge_outputs).ok());
    }
    EXPECT_EQ(stream->completeTokenIdsVec(0), (std::vector<int>{2, 7, 9, 4}));
    EXPECT_EQ(stream->kvCache().batchSize(), 1);
    EXPECT_TRUE(stream->streamCacheResource().getKVBlockUpdateMapping().empty());
}

TEST_F(NormalBatchStreamProcessorTest, testDynamicBeamOneToOneThenExpandWithoutOutputVocabPruning) {
    auto cache_config = test::makeSimpleMhaCacheConfig(
        /*layer_num=*/1, /*block_num=*/16, /*tokens_per_block=*/2, rtp_llm::TYPE_INT8);
    ResourceContext resource_context;
    resource_context.cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(resource_context.cache_manager->init());

    auto                        model_config = makeOutputVocabTestModelConfig();
    RuntimeConfig               runtime_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    auto stream = makeDynamicBeamStream({2}, {1, 2}, 3, model_config, runtime_config, resource_context);
    ASSERT_TRUE(stream->initKVBlock().ok());
    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);

    {
        EXPECT_FALSE(stream->usesBeamSearchTokenLayoutForCurrentStep());
        StreamGroups stream_groups({stream});
        MergedOutput merge_outputs;
        merge_outputs.sampler_output.token_ids = torch::tensor({2, 9}, torch::kInt32).reshape({1, 2});
        ASSERT_TRUE(processor.dispatch(stream_groups, merge_outputs).ok());
    }
    EXPECT_EQ(stream->completeTokenIdsVec(0), (std::vector<int>{2, 9}));
    EXPECT_EQ(stream->kvCache().batchSize(), 1);

    {
        EXPECT_TRUE(stream->usesBeamSearchTokenLayoutForCurrentStep());
        StreamGroups stream_groups({stream});
        MergedOutput merge_outputs;
        merge_outputs.model_output.logits       = torch::zeros({1, 10}, torch::kFloat32).to(torch::kCUDA);
        merge_outputs.sampler_output.token_ids  = torch::tensor({2, 9, 4, 2, 9, 7}, torch::kInt32).reshape({2, 3});
        merge_outputs.sampler_output.beam_index = torch::zeros({2}, torch::kInt32);
        ASSERT_TRUE(processor.dispatch(stream_groups, merge_outputs).ok());
    }
    EXPECT_EQ(stream->completeTokenIdsVec(0), (std::vector<int>{2, 9, 4}));
    EXPECT_EQ(stream->completeTokenIdsVec(1), (std::vector<int>{2, 9, 7}));
    EXPECT_EQ(stream->kvCache().batchSize(), 2);
}

TEST_F(NormalBatchStreamProcessorTest, testThinkTracksDynamicBeamIncrementalThenFullHistoryLayouts) {
    auto cache_config = test::makeSimpleMhaCacheConfig(
        /*layer_num=*/1, /*block_num=*/16, /*tokens_per_block=*/2, rtp_llm::TYPE_INT8);
    ResourceContext resource_context;
    resource_context.cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(resource_context.cache_manager->init());

    auto                        model_config = makeOutputVocabTestModelConfig();
    RuntimeConfig               runtime_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    auto                        query           = make_shared<GenerateInput>();
    query->input_ids                            = hostIntBuffer({2});
    query->generate_config                      = make_shared<GenerateConfig>();
    query->generate_config->variable_num_beams  = {1, 2};
    query->generate_config->max_new_tokens      = 3;
    query->generate_config->in_think_mode       = true;
    query->generate_config->max_thinking_tokens = 10;
    query->generate_config->end_think_token_ids = {9, 7};
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    ASSERT_FALSE(stream->hasError());
    ASSERT_TRUE(stream->initKVBlock().ok());
    stream->generate_status_->status = StreamState::RUNNING;

    auto processors = stream->getAllLogitsProcessorPtr();
    auto think_it   = std::find_if(processors.begin(), processors.end(), [](const auto& processor) {
        return std::dynamic_pointer_cast<ThinkModeLogitsProcessor>(processor) != nullptr;
    });
    ASSERT_NE(think_it, processors.end());
    auto think_processor = std::dynamic_pointer_cast<ThinkModeLogitsProcessor>(*think_it);

    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);
    {
        ASSERT_FALSE(stream->usesBeamSearchTokenLayoutForCurrentStep());
        StreamGroups stream_groups({stream});
        MergedOutput merge_outputs;
        merge_outputs.sampler_output.token_ids = torch::tensor({2, 9}, torch::kInt32).reshape({1, 2});
        ASSERT_TRUE(processor.dispatch(stream_groups, merge_outputs).ok());
        EXPECT_EQ(think_processor->thinkEndTokensStatus(), (std::vector<size_t>{1}));
    }

    {
        ASSERT_TRUE(stream->usesBeamSearchTokenLayoutForCurrentStep());
        StreamGroups stream_groups({stream});
        MergedOutput merge_outputs;
        merge_outputs.model_output.logits       = torch::zeros({1, 10}, torch::kFloat32).to(torch::kCUDA);
        merge_outputs.sampler_output.token_ids  = torch::tensor({2, 9, 7, 2, 9, 4}, torch::kInt32).reshape({2, 3});
        merge_outputs.sampler_output.beam_index = torch::zeros({2}, torch::kInt32);
        ASSERT_TRUE(processor.dispatch(stream_groups, merge_outputs).ok());
        EXPECT_EQ(think_processor->thinkEndTokensStatus(), (std::vector<size_t>{2, 0}));
    }
}

TEST_F(NormalBatchStreamProcessorTest, testOutputVocabSkipsPenaltyHistoryWhenPenaltiesAreDisabled) {
    auto mapping = std::make_shared<OutputVocabMapping>(10, std::vector<int32_t>{0, 2, 7}, "test-model", "digest");

    ResourceContext resource_context;
    resource_context.output_vocab_mapping    = mapping;
    auto                        model_config = makeOutputVocabTestModelConfig();
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    RuntimeConfig               runtime_config;

    auto query             = make_shared<GenerateInput>();
    query->input_ids       = hostIntBuffer({5, 5, 2});
    query->generate_config = make_shared<GenerateConfig>();
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    stream->generate_status_->status = StreamState::RUNNING;

    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false, mapping);
    StreamGroups    stream_groups({stream});
    GptModelOutputs model_outputs;
    model_outputs.logits = torch::zeros({1, 3}, torch::kFloat32).to(torch::kCUDA);

    auto sampler_inputs = processor.gatherSamplerInput(stream_groups, GptModelInputs(), model_outputs);
    ASSERT_TRUE(sampler_inputs.ok());
    EXPECT_FALSE(sampler_inputs->penalty_token_ids.defined());
}

TEST_F(NormalBatchStreamProcessorTest, testOutputVocabMapsHistoryForPenalty) {
    auto mapping = std::make_shared<OutputVocabMapping>(10, std::vector<int32_t>{0, 2, 7}, "test-model", "digest");

    ResourceContext resource_context;
    resource_context.output_vocab_mapping = mapping;
    ModelConfig model_config;
    model_config.max_seq_len = 8;
    model_config.vocab_size  = 10;
    model_config.num_layers  = 1;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    RuntimeConfig               runtime_config;

    auto query                                 = make_shared<GenerateInput>();
    query->input_ids                           = hostIntBuffer({5, 5, 2});
    query->generate_config                     = make_shared<GenerateConfig>();
    query->generate_config->repetition_penalty = 2.0f;
    query->generate_config->top_k              = 1;
    query->generate_config->do_sample          = true;
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    stream->generate_status_->status = StreamState::RUNNING;

    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false, mapping);
    StreamGroups    stream_groups({stream});
    GptModelOutputs model_outputs;
    model_outputs.logits = torch::tensor({0.0f, 10.0f, 9.0f}, torch::kFloat32).reshape({1, 3}).to(torch::kCUDA);

    auto sampler_inputs = processor.gatherSamplerInput(stream_groups, GptModelInputs(), model_outputs);
    ASSERT_TRUE(sampler_inputs.ok());
    EXPECT_EQ(toVec<int32_t>(sampler_inputs->penalty_token_ids), (std::vector<int32_t>{-6, -6, 1, -1}));

    Sampler      sampler(SamplerInitParams{});
    MergedOutput merge_outputs;
    merge_outputs.model_output   = model_outputs;
    merge_outputs.sampler_output = sampler.forward(*sampler_inputs);
    ASSERT_TRUE(processor.dispatch(stream_groups, merge_outputs).ok());
    EXPECT_EQ(stream->completeTokenIdsVec(0), (std::vector<int>{5, 5, 2, 7}));
}

TEST_F(NormalBatchStreamProcessorTest, testOutputVocabNoRepeatNgramIgnoresOutOfSetBannedToken) {
    auto mapping = std::make_shared<OutputVocabMapping>(10, std::vector<int32_t>{0, 2, 7}, "test-model", "digest");

    ResourceContext resource_context;
    resource_context.output_vocab_mapping    = mapping;
    auto                        model_config = makeOutputVocabTestModelConfig();
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    RuntimeConfig               runtime_config;

    auto query                                   = make_shared<GenerateInput>();
    query->input_ids                             = hostIntBuffer({2, 5, 2});
    query->generate_config                       = make_shared<GenerateConfig>();
    query->generate_config->no_repeat_ngram_size = 2;
    query->generate_config->top_k                = 1;
    query->generate_config->do_sample            = true;
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    stream->generate_status_->status = StreamState::RUNNING;

    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false, mapping);
    StreamGroups    stream_groups({stream});
    GptModelOutputs model_outputs;
    model_outputs.logits = torch::tensor({0.0f, 1.0f, 9.0f}, torch::kFloat32).reshape({1, 3}).to(torch::kCUDA);

    auto sampler_inputs = processor.gatherSamplerInput(stream_groups, GptModelInputs(), model_outputs);
    ASSERT_TRUE(sampler_inputs.ok());
    EXPECT_EQ(toVec<int32_t>(sampler_inputs->penalty_token_ids), (std::vector<int32_t>{1, -6, 1, -1}));

    Sampler      sampler(SamplerInitParams{});
    MergedOutput merge_outputs;
    merge_outputs.model_output   = model_outputs;
    merge_outputs.sampler_output = sampler.forward(*sampler_inputs);
    ASSERT_TRUE(processor.dispatch(stream_groups, merge_outputs).ok());
    EXPECT_EQ(stream->completeTokenIdsVec(0), (std::vector<int>{2, 5, 2, 7}));
}

TEST_F(NormalBatchStreamProcessorTest, testOutputVocabMapsFullIdLogitsProcessorOnDemand) {
    auto mapping = std::make_shared<OutputVocabMapping>(10, std::vector<int32_t>{0, 2, 7}, "test-model", "digest");

    ResourceContext resource_context;
    resource_context.output_vocab_mapping    = mapping;
    auto                        model_config = makeOutputVocabTestModelConfig();
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    RuntimeConfig               runtime_config;

    auto query                        = make_shared<GenerateInput>();
    query->input_ids                  = hostIntBuffer({2});
    query->generate_config            = make_shared<GenerateConfig>();
    query->generate_config->do_sample = false;
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    auto force_processor = std::make_shared<ForceFullTokenProcessor>(7);
    stream->logits_processor_list_.insert(stream->logits_processor_list_.begin(), force_processor);
    stream->generate_status_->status = StreamState::RUNNING;

    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false, mapping);
    StreamGroups    stream_groups({stream});
    GptModelOutputs model_outputs;
    model_outputs.logits = torch::zeros({1, 3}, torch::kFloat32).to(torch::kCUDA);

    auto sampler_inputs = processor.gatherSamplerInput(stream_groups, GptModelInputs(), model_outputs);
    ASSERT_TRUE(sampler_inputs.ok());
    Sampler      sampler(SamplerInitParams{});
    MergedOutput merge_outputs;
    merge_outputs.model_output   = model_outputs;
    merge_outputs.sampler_output = sampler.forward(*sampler_inputs);
    EXPECT_EQ(force_processor->observedVocabSize(), 3);
    ASSERT_TRUE(processor.dispatch(stream_groups, merge_outputs).ok());
    EXPECT_EQ(stream->completeTokenIdsVec(0), (std::vector<int>{2, 7}));
}

TEST_F(NormalBatchStreamProcessorTest, testOutputVocabPreservesLogitsProcessorOrder) {
    auto mapping = std::make_shared<OutputVocabMapping>(10, std::vector<int32_t>{0, 2, 7}, "test-model", "digest");

    LogitsProcessorStates states(mapping);
    states.insert(std::make_shared<LocalEosOnlyProcessor>(1), 0, 1);
    states.insert(std::make_shared<ForceFullTokenProcessor>(7), 0, 1);

    SamplerInputs inputs;
    inputs.logits        = torch::zeros({1, 3}, torch::kFloat32).to(torch::kCUDA);
    inputs.finished_mask = torch::ones({1}, torch::kBool);
    inputs.vocab_size    = 3;
    states.batchProcess(inputs);

    EXPECT_FLOAT_EQ(inputs.logits[0][0].item<float>(), BaseLogitsProcessor::neg_inf);
    EXPECT_FLOAT_EQ(inputs.logits[0][1].item<float>(), BaseLogitsProcessor::neg_inf);
    EXPECT_FLOAT_EQ(inputs.logits[0][2].item<float>(), 1.0f);
}

TEST_F(NormalBatchStreamProcessorTest, testOutputVocabReturnsFullIdLogitsShape) {
    auto mapping = std::make_shared<OutputVocabMapping>(10, std::vector<int32_t>{0, 2, 7}, "test-model", "digest");

    ResourceContext resource_context;
    resource_context.output_vocab_mapping    = mapping;
    auto                        model_config = makeOutputVocabTestModelConfig();
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    RuntimeConfig               runtime_config;

    auto query                                   = make_shared<GenerateInput>();
    query->input_ids                             = hostIntBuffer({2});
    query->generate_config                       = make_shared<GenerateConfig>();
    query->generate_config->do_sample            = false;
    query->generate_config->max_new_tokens       = 1;
    query->generate_config->return_logits        = true;
    query->generate_config->return_softmax_probs = true;
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    stream->generate_status_->status = StreamState::RUNNING;

    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false, mapping);
    StreamGroups    stream_groups({stream});
    GptModelOutputs model_outputs;
    model_outputs.logits = torch::tensor({0.0f, 1.0f, 9.0f}, torch::kFloat32).reshape({1, 3}).to(torch::kCUDA);

    auto sampler_inputs = processor.gatherSamplerInput(stream_groups, GptModelInputs(), model_outputs);
    ASSERT_TRUE(sampler_inputs.ok());
    Sampler      sampler(SamplerInitParams{});
    MergedOutput merge_outputs;
    merge_outputs.model_output   = model_outputs;
    merge_outputs.sampler_output = sampler.forward(*sampler_inputs);
    ASSERT_TRUE(processor.dispatch(stream_groups, merge_outputs).ok());

    auto output = stream->nextOutput();
    ASSERT_TRUE(output.ok());
    ASSERT_EQ(output.value().generate_outputs.size(), 1);
    ASSERT_TRUE(output.value().generate_outputs[0].logits.has_value());
    auto logits = output.value().generate_outputs[0].logits.value();
    ASSERT_EQ(logits.sizes().vec(), (std::vector<int64_t>{1, 10}));
    EXPECT_FLOAT_EQ(logits[0][0].item<float>(), 0.0f);
    EXPECT_FLOAT_EQ(logits[0][2].item<float>(), 1.0f);
    EXPECT_FLOAT_EQ(logits[0][7].item<float>(), 9.0f);
    EXPECT_TRUE(std::isinf(logits[0][1].item<float>()));

    const auto expected_prob = torch::softmax(torch::tensor({0.0f, 1.0f, 9.0f}), -1)[2].item<float>();
    EXPECT_NEAR(stream->getSoftmaxProbs()[0][1].item<float>(), expected_prob, 1e-6);
}

TEST_F(NormalBatchStreamProcessorTest, testOutputVocabSoftmaxUsesLocalIdBeforeRestore) {
    auto mapping = std::make_shared<OutputVocabMapping>(10, std::vector<int32_t>{0, 2, 7}, "test-model", "digest");

    ResourceContext resource_context;
    resource_context.output_vocab_mapping    = mapping;
    auto                        model_config = makeOutputVocabTestModelConfig();
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    RuntimeConfig               runtime_config;

    auto query                                   = make_shared<GenerateInput>();
    query->input_ids                             = hostIntBuffer({2});
    query->generate_config                       = make_shared<GenerateConfig>();
    query->generate_config->do_sample            = false;
    query->generate_config->max_new_tokens       = 1;
    query->generate_config->return_softmax_probs = true;
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    stream->generate_status_->status = StreamState::RUNNING;

    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false, mapping);
    StreamGroups    stream_groups({stream});
    GptModelOutputs model_outputs;
    model_outputs.logits = torch::tensor({0.0f, 1.0f, 9.0f}, torch::kFloat32).reshape({1, 3}).to(torch::kCUDA);

    auto sampler_inputs = processor.gatherSamplerInput(stream_groups, GptModelInputs(), model_outputs);
    ASSERT_TRUE(sampler_inputs.ok());
    Sampler      sampler(SamplerInitParams{});
    MergedOutput merge_outputs;
    merge_outputs.model_output   = model_outputs;
    merge_outputs.sampler_output = sampler.forward(*sampler_inputs);
    ASSERT_TRUE(processor.dispatch(stream_groups, merge_outputs).ok());

    EXPECT_EQ(stream->completeTokenIdsVec(0), (std::vector<int>{2, 7}));
    auto softmax_probs = stream->getSoftmaxProbs();
    ASSERT_TRUE(softmax_probs.defined());
    auto expected = torch::softmax(torch::tensor({0.0f, 1.0f, 9.0f}), -1)[2].item<float>();
    EXPECT_NEAR(softmax_probs.data_ptr<float>()[1], expected, 1e-6);
}

TEST_F(NormalBatchStreamProcessorTest, testOutputVocabReturnsFullIdProbabilityShape) {
    auto mapping = std::make_shared<OutputVocabMapping>(10, std::vector<int32_t>{0, 2, 7}, "test-model", "digest");

    ResourceContext resource_context;
    resource_context.output_vocab_mapping    = mapping;
    auto                        model_config = makeOutputVocabTestModelConfig();
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    RuntimeConfig               runtime_config;

    auto query                               = make_shared<GenerateInput>();
    query->input_ids                         = hostIntBuffer({2});
    query->generate_config                   = make_shared<GenerateConfig>();
    query->generate_config->max_new_tokens   = 1;
    query->generate_config->return_all_probs = ReturnAllProbsMode::DEFAULT;
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    stream->generate_status_->status = StreamState::RUNNING;

    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false, mapping);
    StreamGroups stream_groups({stream});
    MergedOutput merge_outputs;
    merge_outputs.sampler_output.token_ids = torch::tensor({2, 2}, torch::kInt32).reshape({1, 2});
    merge_outputs.sampler_output.all_probs =
        torch::tensor({0.1f, 0.2f, 0.7f}, torch::kFloat32).reshape({1, 3}).to(torch::kCUDA);

    ASSERT_TRUE(processor.dispatch(stream_groups, merge_outputs).ok());
    auto output = stream->nextOutput();
    ASSERT_TRUE(output.ok());
    const auto& probs = output.value().generate_outputs[0].aux_info.all_probs;
    ASSERT_TRUE(probs.has_value());
    ASSERT_EQ(probs.value().sizes().vec(), (std::vector<int64_t>{1, 10}));
    EXPECT_FLOAT_EQ(probs.value()[0][0].item<float>(), 0.1f);
    EXPECT_FLOAT_EQ(probs.value()[0][2].item<float>(), 0.2f);
    EXPECT_FLOAT_EQ(probs.value()[0][7].item<float>(), 0.7f);
    EXPECT_FLOAT_EQ(probs.value()[0][1].item<float>(), 0.0f);
}

TEST_F(NormalBatchStreamProcessorTest, testOutputVocabMapsLossLabels) {
    auto mapping = std::make_shared<OutputVocabMapping>(10, std::vector<int32_t>{0, 2, 7}, "test-model", "digest");

    ResourceContext resource_context;
    resource_context.output_vocab_mapping    = mapping;
    auto                        model_config = makeOutputVocabTestModelConfig();
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    RuntimeConfig               runtime_config;

    auto query                             = make_shared<GenerateInput>();
    query->input_ids                       = hostIntBuffer({0, 2, 5});
    query->generate_config                 = make_shared<GenerateConfig>();
    query->generate_config->calculate_loss = 2;
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    stream->generate_status_->status = StreamState::RUNNING;

    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false, mapping);
    StreamGroups stream_groups({stream});
    MergedOutput merge_outputs;
    merge_outputs.model_output.all_logits =
        torch::tensor({0.0f, 2.0f, 1.0f, 1.0f, 0.0f, 2.0f}, torch::kFloat32).reshape({2, 3}).to(torch::kCUDA);
    merge_outputs.sampler_output.token_ids = torch::tensor({0, 2, 5, 1}, torch::kInt32).reshape({1, 4});

    ASSERT_TRUE(processor.dispatch(stream_groups, merge_outputs).ok());
    auto loss = stream->getLoss();
    ASSERT_TRUE(loss.defined());
    ASSERT_EQ(loss.numel(), 2);
    EXPECT_TRUE(std::isfinite(loss[0].item<float>()));
    EXPECT_TRUE(std::isinf(loss[1].item<float>()));
    EXPECT_GT(loss[1].item<float>(), 0.0f);
}

TEST_F(NormalBatchStreamProcessorTest, testOutputVocabMapsPromptLogitsIdsAndTargets) {
    auto mapping = std::make_shared<OutputVocabMapping>(10, std::vector<int32_t>{0, 2, 7}, "test-model", "digest");

    ResourceContext resource_context;
    resource_context.output_vocab_mapping    = mapping;
    auto                        model_config = makeOutputVocabTestModelConfig();
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    RuntimeConfig               runtime_config;

    auto query                                    = make_shared<GenerateInput>();
    query->input_ids                              = hostIntBuffer({0, 2, 5});
    query->generate_config                        = make_shared<GenerateConfig>();
    query->generate_config->max_new_tokens        = 1;
    query->generate_config->return_prompt_logits  = true;
    query->generate_config->prompt_logits_top_k   = 2;
    query->generate_config->return_target_logprob = true;
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    stream->generate_status_->status = StreamState::RUNNING;

    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false, mapping);
    StreamGroups stream_groups({stream});
    MergedOutput merge_outputs;
    merge_outputs.model_output.all_logits =
        torch::tensor({0.0f, 3.0f, 2.0f, 0.0f, 3.0f, 2.0f, 0.0f, 3.0f, 2.0f}, torch::kFloat32)
            .reshape({3, 3})
            .to(torch::kCUDA);
    merge_outputs.sampler_output.token_ids = torch::tensor({0, 2, 5, 1}, torch::kInt32).reshape({1, 4});

    ASSERT_TRUE(processor.dispatch(stream_groups, merge_outputs).ok());
    auto output = stream->nextOutput();
    ASSERT_TRUE(output.ok());
    ASSERT_TRUE(output.value().generate_outputs[0].prompt_logits.has_value());
    const auto& prompt_logits = output.value().generate_outputs[0].prompt_logits.value();
    ASSERT_EQ(prompt_logits.topk_token_ids.sizes().vec(), (std::vector<int64_t>{3, 2}));
    EXPECT_EQ(prompt_logits.topk_token_ids[0][0].item<int32_t>(), 2);
    EXPECT_EQ(prompt_logits.topk_token_ids[0][1].item<int32_t>(), 7);
    ASSERT_EQ(prompt_logits.target_logprobs.numel(), 2);
    EXPECT_TRUE(std::isfinite(prompt_logits.target_logprobs[0].item<float>()));
    EXPECT_TRUE(std::isinf(prompt_logits.target_logprobs[1].item<float>()));
    EXPECT_LT(prompt_logits.target_logprobs[1].item<float>(), 0.0f);
}

TEST_F(NormalBatchStreamProcessorTest, testOutputVocabMapsEveryReturnedSamplingSequence) {
    auto mapping = std::make_shared<OutputVocabMapping>(10, std::vector<int32_t>{0, 2, 7, 9}, "test-model", "digest");

    ResourceContext resource_context;
    resource_context.output_vocab_mapping    = mapping;
    auto                        model_config = makeOutputVocabTestModelConfig();
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    RuntimeConfig               runtime_config;

    auto query                                   = make_shared<GenerateInput>();
    query->input_ids                             = hostIntBuffer({2});
    query->generate_config                       = make_shared<GenerateConfig>();
    query->generate_config->num_return_sequences = 4;
    query->generate_config->max_new_tokens       = 1;
    query->generate_config->do_sample            = true;
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    stream->generate_status_->status = StreamState::RUNNING;

    ASSERT_EQ(stream->currentBatchSize(), 4);
    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false, mapping);
    StreamGroups stream_groups({stream});
    MergedOutput merge_outputs;
    merge_outputs.sampler_output.token_ids = torch::tensor({0, 1, 2, 3}, torch::kInt32).reshape({4, 1});

    ASSERT_TRUE(processor.dispatch(stream_groups, merge_outputs).ok());
    const std::vector<int32_t> expected_full_ids{0, 2, 7, 9};
    for (size_t i = 0; i < expected_full_ids.size(); ++i) {
        EXPECT_EQ(stream->completeTokenIdsVec(i), (std::vector<int>{2, expected_full_ids[i]}));
    }

    auto output = stream->nextOutput();
    ASSERT_TRUE(output.ok());
    ASSERT_EQ(output.value().generate_outputs.size(), expected_full_ids.size());
    for (size_t i = 0; i < expected_full_ids.size(); ++i) {
        const auto& generate_output = output.value().generate_outputs[i];
        ASSERT_EQ(generate_output.output_ids.sizes().vec(), (std::vector<int64_t>{1, 1}));
        EXPECT_EQ(generate_output.output_ids[0][0].item<int32_t>(), expected_full_ids[i]);
        EXPECT_TRUE(generate_output.finished);
    }
}

TEST_F(NormalBatchStreamProcessorTest, testOutputVocabUsesLocalEosAndFinishesWithFullEos) {
    auto mapping = std::make_shared<OutputVocabMapping>(10, std::vector<int32_t>{0, 2, 7}, "test-model", "digest");

    ResourceContext resource_context;
    resource_context.output_vocab_mapping    = mapping;
    auto model_config                        = makeOutputVocabTestModelConfig();
    model_config.special_tokens.eos_token_id = 7;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    RuntimeConfig               runtime_config;

    auto query                                   = make_shared<GenerateInput>();
    query->input_ids                             = hostIntBuffer({2});
    query->generate_config                       = make_shared<GenerateConfig>();
    query->generate_config->num_return_sequences = 2;
    query->generate_config->max_new_tokens       = 4;
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    stream->generate_status_->status = StreamState::RUNNING;

    ASSERT_EQ(stream->logits_processor_list_.size(), 1);
    SamplerInputs processor_inputs;
    processor_inputs.logits =
        torch::tensor({{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, -std::numeric_limits<float>::infinity()}}, torch::kFloat32)
            .to(torch::kCUDA);
    processor_inputs.finished_mask        = torch::tensor({false, true}, torch::kBool);
    processor_inputs.no_repeat_ngram_size = torch::tensor({1, 1}, torch::kInt32);
    stream->logits_processor_list_[0]->process(processor_inputs, 0, 2);
    EXPECT_FLOAT_EQ(processor_inputs.logits[1][2].item<float>(), 0.0f);
    EXPECT_FLOAT_EQ(processor_inputs.logits[1][0].item<float>(), BaseLogitsProcessor::neg_inf);
    EXPECT_FLOAT_EQ(processor_inputs.logits[1][1].item<float>(), BaseLogitsProcessor::neg_inf);
    EXPECT_EQ(toVec<int32_t>(processor_inputs.no_repeat_ngram_size), (std::vector<int32_t>{1, 0}));

    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false, mapping);
    StreamGroups stream_groups({stream});
    MergedOutput merge_outputs;
    merge_outputs.sampler_output.token_ids = torch::tensor({2, 2}, torch::kInt32).reshape({2, 1});

    ASSERT_TRUE(processor.dispatch(stream_groups, merge_outputs).ok());
    EXPECT_EQ(stream->completeTokenIdsVec(0), (std::vector<int>{2, 7}));
    EXPECT_EQ(stream->completeTokenIdsVec(1), (std::vector<int>{2, 7}));

    auto output = stream->nextOutput();
    ASSERT_TRUE(output.ok());
    ASSERT_EQ(output.value().generate_outputs.size(), 2);
    EXPECT_TRUE(output.value().generate_outputs[0].finished);
    EXPECT_TRUE(output.value().generate_outputs[1].finished);
    EXPECT_EQ(output.value().generate_outputs[0].output_ids[0][0].item<int32_t>(), 7);
    EXPECT_EQ(output.value().generate_outputs[1].output_ids[0][0].item<int32_t>(), 7);
}

TEST_F(NormalBatchStreamProcessorTest, testOutputVocabBeamAllProbsFollowSelectedParents) {
    auto mapping =
        std::make_shared<OutputVocabMapping>(10, std::vector<int32_t>{0, 1, 2, 4, 5, 7, 9}, "test-model", "digest");
    auto cache_config = test::makeSimpleMhaCacheConfig(
        /*layer_num=*/1, /*block_num=*/64, /*tokens_per_block=*/2, rtp_llm::TYPE_INT8);

    ResourceContext resource_context;
    resource_context.output_vocab_mapping = mapping;
    resource_context.cache_manager        = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(resource_context.cache_manager->init());

    auto model_config                        = makeOutputVocabTestModelConfig();
    model_config.special_tokens.eos_token_id = 7;
    RuntimeConfig               runtime_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;

    auto query                               = make_shared<GenerateInput>();
    query->input_ids                         = hostIntBuffer({2});
    query->generate_config                   = make_shared<GenerateConfig>();
    query->generate_config->num_beams        = 3;
    query->generate_config->max_new_tokens   = 2;
    query->generate_config->return_all_probs = ReturnAllProbsMode::DEFAULT;
    query->generate_config->aux_info         = true;
    query->generate_config->ignore_eos       = true;
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    ASSERT_FALSE(stream->hasError());
    ASSERT_TRUE(stream->initKVBlock().ok());
    stream->generate_status_->status = StreamState::RUNNING;

    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false, mapping);
    Sampler sampler(SamplerInitParams{});

    {
        StreamGroups    stream_groups({stream});
        GptModelOutputs model_outputs;
        model_outputs.logits = torch::zeros({1, 7}, torch::kFloat32).to(torch::kCUDA);
        auto expected_probs  = torch::softmax(model_outputs.logits.cpu(), -1);
        auto sampler_inputs  = processor.gatherSamplerInput(stream_groups, GptModelInputs(), model_outputs);

        ASSERT_TRUE(sampler_inputs.ok());
        ASSERT_EQ(sampler_inputs->all_probs.sizes().vec(), (std::vector<int64_t>{3, 7}));
        MergedOutput merge_outputs;
        merge_outputs.model_output   = model_outputs;
        merge_outputs.sampler_output = sampler.forward(*sampler_inputs);

        auto actual_probs = merge_outputs.sampler_output.all_probs.cpu();
        ASSERT_EQ(actual_probs.sizes().vec(), (std::vector<int64_t>{3, 7}));
        EXPECT_TRUE(torch::allclose(actual_probs, expected_probs.repeat({3, 1}), 1e-6, 1e-6));
        EXPECT_GT(actual_probs.min().item<float>(), 0.0f);
        ASSERT_TRUE(processor.dispatch(stream_groups, merge_outputs).ok());
        ASSERT_EQ(stream->currentBatchSize(), 3);
    }

    {
        StreamGroups    stream_groups({stream});
        GptModelOutputs model_outputs;
        model_outputs.logits = torch::tensor({0.0f, 0.0f, 0.0f, 0.0f, 0.0f,  0.0f, 0.0f, 12.0f, 0.0f, 0.0f, 0.0f,
                                              0.0f, 0.0f, 0.0f, 0.0f, 12.0f, 0.0f, 0.0f, 0.0f,  0.0f, 0.0f},
                                             torch::kFloat32)
                                   .reshape({3, 7})
                                   .to(torch::kCUDA);
        auto expected_probs = torch::softmax(model_outputs.logits.cpu(), -1);
        auto sampler_inputs = processor.gatherSamplerInput(stream_groups, GptModelInputs(), model_outputs);

        ASSERT_TRUE(sampler_inputs.ok());
        MergedOutput merge_outputs;
        merge_outputs.model_output   = model_outputs;
        merge_outputs.sampler_output = sampler.forward(*sampler_inputs);

        auto actual_probs   = merge_outputs.sampler_output.all_probs.cpu();
        auto parent_indices = merge_outputs.sampler_output.beam_index.cpu();
        ASSERT_EQ(actual_probs.sizes().vec(), (std::vector<int64_t>{3, 7}));
        ASSERT_EQ(parent_indices.numel(), 3);
        bool saw_nonzero_parent = false;
        for (int64_t i = 0; i < parent_indices.numel(); ++i) {
            const auto parent = parent_indices[i].item<int32_t>();
            ASSERT_GE(parent, 0);
            ASSERT_LT(parent, 3);
            saw_nonzero_parent |= parent != 0;
            EXPECT_TRUE(torch::allclose(actual_probs[i], expected_probs[parent], 1e-6, 1e-6));
        }
        EXPECT_TRUE(saw_nonzero_parent);

        ASSERT_TRUE(processor.dispatch(stream_groups, merge_outputs).ok());
    }

    auto output = stream->nextOutput();
    ASSERT_TRUE(output.ok());
    ASSERT_EQ(output.value().generate_outputs.size(), 3);
    for (const auto& generate_output : output.value().generate_outputs) {
        ASSERT_TRUE(generate_output.aux_info.all_probs.has_value());
        const auto& probs = generate_output.aux_info.all_probs.value();
        ASSERT_EQ(probs.sizes().vec(), (std::vector<int64_t>{1, 10}));
        EXPECT_NEAR(probs.sum().item<float>(), 1.0f, 1e-5);
    }
}

TEST_F(NormalBatchStreamProcessorTest, testOutputVocabBeamReordersOutputsAndSoftmaxHistory) {
    auto mapping =
        std::make_shared<OutputVocabMapping>(10, std::vector<int32_t>{0, 1, 2, 4, 5, 7, 9}, "test-model", "digest");
    auto cache_config = test::makeSimpleMhaCacheConfig(
        /*layer_num=*/1, /*block_num=*/64, /*tokens_per_block=*/2, rtp_llm::TYPE_INT8);

    ResourceContext resource_context;
    resource_context.output_vocab_mapping = mapping;
    resource_context.cache_manager        = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(resource_context.cache_manager->init());

    auto model_config                        = makeOutputVocabTestModelConfig();
    model_config.special_tokens.eos_token_id = 7;
    RuntimeConfig               runtime_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;

    auto query                                   = make_shared<GenerateInput>();
    query->input_ids                             = hostIntBuffer({2});
    query->generate_config                       = make_shared<GenerateConfig>();
    query->generate_config->variable_num_beams   = {2, 3};
    query->generate_config->max_new_tokens       = 2;
    query->generate_config->return_logits        = true;
    query->generate_config->return_hidden_states = true;
    query->generate_config->return_softmax_probs = true;
    query->generate_config->ignore_eos           = true;
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    ASSERT_FALSE(stream->hasError());
    ASSERT_TRUE(stream->initKVBlock().ok());
    stream->generate_status_->status = StreamState::RUNNING;

    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false, mapping);

    auto first_logits = torch::tensor({0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, torch::kFloat32).reshape({1, 7});
    auto first_probs  = torch::softmax(first_logits, -1);
    {
        StreamGroups stream_groups({stream});
        MergedOutput merge_outputs;
        merge_outputs.model_output.logits = first_logits.to(torch::kCUDA);
        merge_outputs.model_output.hidden_states =
            torch::tensor({10.0f, 11.0f}, torch::kFloat32).reshape({1, 2}).to(torch::kCUDA);
        merge_outputs.sampler_output.token_ids  = torch::tensor({2, 1, 2, 5}, torch::kInt32).reshape({2, 2});
        merge_outputs.sampler_output.beam_index = torch::zeros({2}, torch::kInt32);
        ASSERT_TRUE(processor.dispatch(stream_groups, merge_outputs).ok());
    }
    ASSERT_EQ(stream->currentBatchSize(), 2);

    auto second_logits =
        torch::tensor({0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f},
                      torch::kFloat32)
            .reshape({2, 7});
    auto second_probs = torch::softmax(second_logits, -1);
    {
        StreamGroups stream_groups({stream});
        MergedOutput merge_outputs;
        merge_outputs.model_output.logits = second_logits.to(torch::kCUDA);
        merge_outputs.model_output.hidden_states =
            torch::tensor({10.0f, 11.0f, 20.0f, 21.0f}, torch::kFloat32).reshape({2, 2}).to(torch::kCUDA);
        merge_outputs.sampler_output.token_ids =
            torch::tensor({2, 7, 2, 2, 1, 4, 2, 7, 6}, torch::kInt32).reshape({3, 3});
        merge_outputs.sampler_output.beam_index = torch::tensor({1, 0, 1}, torch::kInt32);
        ASSERT_TRUE(processor.dispatch(stream_groups, merge_outputs).ok());
    }

    ASSERT_EQ(stream->currentBatchSize(), 3);
    auto softmax_history = stream->getSoftmaxProbs();
    ASSERT_EQ(softmax_history.size(0), 3);
    EXPECT_NEAR(softmax_history[0][1].item<float>(), first_probs[0][5].item<float>(), 1e-6);
    EXPECT_NEAR(softmax_history[1][1].item<float>(), first_probs[0][1].item<float>(), 1e-6);
    EXPECT_NEAR(softmax_history[2][1].item<float>(), first_probs[0][5].item<float>(), 1e-6);
    EXPECT_NEAR(softmax_history[0][2].item<float>(), second_probs[1][2].item<float>(), 1e-6);
    EXPECT_NEAR(softmax_history[1][2].item<float>(), second_probs[0][4].item<float>(), 1e-6);
    EXPECT_NEAR(softmax_history[2][2].item<float>(), second_probs[1][6].item<float>(), 1e-6);

    auto output = stream->nextOutput();
    ASSERT_TRUE(output.ok());
    ASSERT_EQ(output.value().generate_outputs.size(), 3);
    const std::vector<float> expected_parent_logit{6.0f, 0.0f, 6.0f};
    const std::vector<float> expected_parent_hidden{20.0f, 10.0f, 20.0f};
    for (size_t i = 0; i < output.value().generate_outputs.size(); ++i) {
        const auto& generate_output = output.value().generate_outputs[i];
        ASSERT_TRUE(generate_output.logits.has_value());
        ASSERT_TRUE(generate_output.hidden_states.has_value());
        EXPECT_FLOAT_EQ(generate_output.logits.value()[0][0].item<float>(), expected_parent_logit[i]);
        EXPECT_FLOAT_EQ(generate_output.hidden_states.value()[0][0].item<float>(), expected_parent_hidden[i]);
    }
}

TEST_F(NormalBatchStreamProcessorTest, testLensRecallSizedOutputVocabSupportsBeam8_16_32) {
    constexpr int32_t    full_vocab_size    = 217303;
    constexpr int32_t    output_vocab_start = 151643;
    constexpr int32_t    output_vocab_size  = 65660;
    std::vector<int32_t> local_to_full(output_vocab_size);
    std::iota(local_to_full.begin(), local_to_full.end(), output_vocab_start);
    auto mapping = std::make_shared<OutputVocabMapping>(
        full_vocab_size, std::move(local_to_full), "LensRecall_nd_pg_attn@test", "digest");

    for (const int beam_width : {8, 16, 32}) {
        SCOPED_TRACE("beam_width=" + std::to_string(beam_width));
        auto cache_config = test::makeSimpleMhaCacheConfig(
            /*layer_num=*/1, /*block_num=*/256, /*tokens_per_block=*/2, rtp_llm::TYPE_INT8);
        ResourceContext resource_context;
        resource_context.output_vocab_mapping = mapping;
        resource_context.cache_manager        = std::make_shared<KVCacheManager>(cache_config);
        ASSERT_TRUE(resource_context.cache_manager->init());

        ModelConfig model_config;
        model_config.max_seq_len                 = 8;
        model_config.vocab_size                  = full_vocab_size;
        model_config.num_layers                  = 1;
        model_config.special_tokens.eos_token_id = 151645;
        PDSepConfig                 pd_sep_config;
        ProfilingDebugLoggingConfig profiling_debug_logging_config;
        RuntimeConfig               runtime_config;

        auto query                             = make_shared<GenerateInput>();
        query->input_ids                       = hostIntBuffer({151666});
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->num_beams      = beam_width;
        query->generate_config->max_new_tokens = 2;
        query->generate_config->do_sample      = false;
        auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        ASSERT_FALSE(stream->hasError());
        ASSERT_TRUE(stream->initKVBlock().ok());
        stream->generate_status_->status = StreamState::RUNNING;

        NormalBatchStreamProcessor processor(
            model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false, mapping);
        Sampler sampler(SamplerInitParams{});

        for (int step = 0; step < 2; ++step) {
            StreamGroups    stream_groups({stream});
            GptModelOutputs model_outputs;
            auto            logits_row =
                torch::arange(output_vocab_size, torch::kFloat32).div(static_cast<float>(output_vocab_size));
            model_outputs.logits = logits_row.repeat({stream->currentBatchSize(), 1}).to(torch::kCUDA);
            auto sampler_inputs  = processor.gatherSamplerInput(stream_groups, GptModelInputs(), model_outputs);

            ASSERT_TRUE(sampler_inputs.ok());
            EXPECT_EQ(sampler_inputs->vocab_size, output_vocab_size);
            EXPECT_EQ(sampler_inputs->batch_size_out, beam_width);

            MergedOutput merge_outputs;
            merge_outputs.model_output   = model_outputs;
            merge_outputs.sampler_output = sampler.forward(*sampler_inputs);
            ASSERT_TRUE(processor.dispatch(stream_groups, merge_outputs).ok());
            ASSERT_EQ(stream->currentBatchSize(), beam_width);

            for (int beam_idx = 0; beam_idx < beam_width; ++beam_idx) {
                const auto tokens = stream->completeTokenIdsVec(beam_idx);
                ASSERT_EQ(tokens.size(), static_cast<size_t>(step + 2));
                EXPECT_EQ(tokens.front(), 151666);
                EXPECT_GE(tokens.back(), output_vocab_start);
                EXPECT_LT(tokens.back(), full_vocab_size);
            }
        }
    }
}

TEST_F(NormalBatchStreamProcessorTest, testOutputVocabInvalidBeamWidthIsAStreamError) {
    auto mapping = std::make_shared<OutputVocabMapping>(10, std::vector<int32_t>{0, 2, 7}, "test-model", "digest");

    ResourceContext resource_context;
    resource_context.output_vocab_mapping = mapping;
    auto          model_config            = makeOutputVocabTestModelConfig();
    RuntimeConfig runtime_config;

    auto query                        = make_shared<GenerateInput>();
    query->input_ids                  = hostIntBuffer({2});
    query->generate_config            = make_shared<GenerateConfig>();
    query->generate_config->num_beams = 2;
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);

    EXPECT_TRUE(stream->hasError());
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_NE(stream->statusInfo().ToString().find("twice the maximum beam width"), std::string::npos);
}

TEST_F(NormalBatchStreamProcessorTest, testNoRepeatNgramSizeKernelBoundary) {
    ResourceContext resource_context;
    auto            model_config = makeOutputVocabTestModelConfig();
    RuntimeConfig   runtime_config;

    auto make_stream = [&](int no_repeat_ngram_size) {
        auto query                                   = make_shared<GenerateInput>();
        query->input_ids                             = hostIntBuffer({2});
        query->generate_config                       = make_shared<GenerateConfig>();
        query->generate_config->no_repeat_ngram_size = no_repeat_ngram_size;
        return make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    };

    auto valid_stream = make_stream(kMaxNoRepeatNgramSize);
    EXPECT_FALSE(valid_stream->hasError());

    auto invalid_stream = make_stream(kMaxNoRepeatNgramSize + 1);
    EXPECT_TRUE(invalid_stream->hasError());
    EXPECT_EQ(invalid_stream->statusInfo().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_NE(invalid_stream->statusInfo().ToString().find("no_repeat_ngram_size must be in"), std::string::npos);
}

TEST_F(NormalBatchStreamProcessorTest, testMultiSequenceRejectsInvalidEosButSingleSequenceAllowsIt) {
    ResourceContext resource_context;
    auto            model_config             = makeOutputVocabTestModelConfig();
    model_config.special_tokens.eos_token_id = -1;
    RuntimeConfig runtime_config;

    auto make_stream = [&](int num_beams, int num_return_sequences) {
        auto query                                   = make_shared<GenerateInput>();
        query->input_ids                             = hostIntBuffer({2});
        query->generate_config                       = make_shared<GenerateConfig>();
        query->generate_config->num_beams            = num_beams;
        query->generate_config->num_return_sequences = num_return_sequences;
        return make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    };

    auto greedy_stream = make_stream(1, 1);
    EXPECT_FALSE(greedy_stream->hasError());
    EXPECT_TRUE(greedy_stream->logits_processor_list_.empty());

    for (const auto [num_beams, num_return_sequences] : {std::pair<int, int>{2, 1}, std::pair<int, int>{1, 2}}) {
        auto stream = make_stream(num_beams, num_return_sequences);
        EXPECT_TRUE(stream->hasError());
        EXPECT_EQ(stream->statusInfo().code(), ErrorCode::INVALID_PARAMS);
        EXPECT_NE(stream->statusInfo().ToString().find("require an EOS token id"), std::string::npos);
    }
}

TEST_F(NormalBatchStreamProcessorTest, testMultiSeqProcessorDefensivelyRejectsOutOfRangeEos) {
    SamplerInputs inputs;
    inputs.logits        = torch::zeros({1, 3}, torch::kFloat32).to(torch::kCUDA);
    inputs.finished_mask = torch::tensor({true}, torch::kBool);

    LocalEosOnlyProcessor negative_eos(-1);
    EXPECT_ANY_THROW(negative_eos.process(inputs, 0, 1));

    LocalEosOnlyProcessor oversized_eos(3);
    EXPECT_ANY_THROW(oversized_eos.process(inputs, 0, 1));
}

TEST_F(NormalBatchStreamProcessorTest, testThinkHardTerminatorRejectsFixedBeamExpansion) {
    std::vector<int32_t> retained_tokens(9);
    std::iota(retained_tokens.begin(), retained_tokens.end(), 0);
    auto mapping = std::make_shared<OutputVocabMapping>(10, retained_tokens, "test-model", "digest");

    ResourceContext resource_context;
    resource_context.output_vocab_mapping    = mapping;
    auto model_config                        = makeOutputVocabTestModelConfig();
    model_config.special_tokens.eos_token_id = 7;
    RuntimeConfig runtime_config;

    auto query                                  = make_shared<GenerateInput>();
    query->input_ids                            = hostIntBuffer({2});
    query->generate_config                      = make_shared<GenerateConfig>();
    query->generate_config->num_beams           = 4;
    query->generate_config->max_new_tokens      = 4;
    query->generate_config->in_think_mode       = true;
    query->generate_config->max_thinking_tokens = 1;
    query->generate_config->end_think_token_ids = {7};
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);

    EXPECT_TRUE(stream->hasError());
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_NE(stream->statusInfo().ToString().find("cannot expand beam width from 1 to 4"), std::string::npos);
}

TEST_F(NormalBatchStreamProcessorTest, testThinkHardTerminatorRejectsLaterDynamicBeamExpansion) {
    std::vector<int32_t> retained_tokens(9);
    std::iota(retained_tokens.begin(), retained_tokens.end(), 0);
    auto mapping = std::make_shared<OutputVocabMapping>(10, retained_tokens, "test-model", "digest");

    ResourceContext resource_context;
    resource_context.output_vocab_mapping    = mapping;
    auto model_config                        = makeOutputVocabTestModelConfig();
    model_config.special_tokens.eos_token_id = 7;
    RuntimeConfig runtime_config;

    auto query                                  = make_shared<GenerateInput>();
    query->input_ids                            = hostIntBuffer({2});
    query->generate_config                      = make_shared<GenerateConfig>();
    query->generate_config->variable_num_beams  = {1, 1, 2};
    query->generate_config->max_new_tokens      = 4;
    query->generate_config->in_think_mode       = true;
    query->generate_config->max_thinking_tokens = 2;
    query->generate_config->end_think_token_ids = {7, 8};
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);

    EXPECT_TRUE(stream->hasError());
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_NE(stream->statusInfo().ToString().find("cannot expand beam width from 1 to 2"), std::string::npos);
}

TEST_F(NormalBatchStreamProcessorTest, testThinkRejectsEmptyTerminatorWithoutOutputVocabPruning) {
    ResourceContext resource_context;
    auto            model_config = makeOutputVocabTestModelConfig();
    RuntimeConfig   runtime_config;

    auto query                                  = make_shared<GenerateInput>();
    query->input_ids                            = hostIntBuffer({2});
    query->generate_config                      = make_shared<GenerateConfig>();
    query->generate_config->in_think_mode       = true;
    query->generate_config->max_thinking_tokens = 4;
    query->generate_config->end_think_token_ids = {};
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);

    EXPECT_TRUE(stream->hasError());
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_NE(stream->statusInfo().ToString().find("requires non-empty end_think_token_ids"), std::string::npos);
}

TEST_F(NormalBatchStreamProcessorTest, testRecommendationAllowsEmptyTerminator) {
    auto mapping = std::make_shared<OutputVocabMapping>(10, std::vector<int32_t>{0, 2, 7}, "test-model", "digest");
    ResourceContext resource_context;
    resource_context.output_vocab_mapping = mapping;
    auto          model_config            = makeOutputVocabTestModelConfig();
    RuntimeConfig runtime_config;

    auto query                                  = make_shared<GenerateInput>();
    query->input_ids                            = hostIntBuffer({2});
    query->generate_config                      = make_shared<GenerateConfig>();
    query->generate_config->combo_token_size    = 3;
    query->generate_config->end_think_token_ids = {};
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);

    ASSERT_FALSE(stream->hasError());
    const auto processors = stream->getAllLogitsProcessorPtr();
    EXPECT_NE(std::find_if(processors.begin(),
                           processors.end(),
                           [](const auto& processor) {
                               return std::dynamic_pointer_cast<RecommendationLogitsProcessor>(processor) != nullptr;
                           }),
              processors.end());
}

TEST_F(NormalBatchStreamProcessorTest, testOutputVocabRejectsMissingThinkModeToken) {
    auto mapping = std::make_shared<OutputVocabMapping>(10, std::vector<int32_t>{0, 2, 7}, "test-model", "digest");

    ResourceContext resource_context;
    resource_context.output_vocab_mapping    = mapping;
    auto model_config                        = makeOutputVocabTestModelConfig();
    model_config.special_tokens.eos_token_id = 7;
    RuntimeConfig runtime_config;

    auto query                                  = make_shared<GenerateInput>();
    query->input_ids                            = hostIntBuffer({2});
    query->generate_config                      = make_shared<GenerateConfig>();
    query->generate_config->in_think_mode       = true;
    query->generate_config->max_thinking_tokens = 4;
    query->generate_config->end_think_token_ids = {5};
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);

    EXPECT_TRUE(stream->hasError());
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_NE(stream->statusInfo().ToString().find("think/recommendation terminator token [5]"), std::string::npos);
}

TEST_F(NormalBatchStreamProcessorTest, testOutputVocabRejectsMissingRecommendationTerminator) {
    auto mapping = std::make_shared<OutputVocabMapping>(10, std::vector<int32_t>{0, 2, 7}, "test-model", "digest");

    ResourceContext resource_context;
    resource_context.output_vocab_mapping    = mapping;
    auto model_config                        = makeOutputVocabTestModelConfig();
    model_config.special_tokens.eos_token_id = 7;
    RuntimeConfig runtime_config;

    auto query                                  = make_shared<GenerateInput>();
    query->input_ids                            = hostIntBuffer({2});
    query->generate_config                      = make_shared<GenerateConfig>();
    query->generate_config->combo_token_size    = 3;
    query->generate_config->end_think_token_ids = {5};
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);

    EXPECT_TRUE(stream->hasError());
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_NE(stream->statusInfo().ToString().find("think/recommendation terminator token [5]"), std::string::npos);
}

TEST_F(NormalBatchStreamProcessorTest, testOutputVocabAcceptsRetainedRecommendationTerminator) {
    auto mapping = std::make_shared<OutputVocabMapping>(10, std::vector<int32_t>{0, 2, 5, 7}, "test-model", "digest");

    ResourceContext resource_context;
    resource_context.output_vocab_mapping    = mapping;
    auto model_config                        = makeOutputVocabTestModelConfig();
    model_config.special_tokens.eos_token_id = 7;
    RuntimeConfig runtime_config;

    auto query                                  = make_shared<GenerateInput>();
    query->input_ids                            = hostIntBuffer({2});
    query->generate_config                      = make_shared<GenerateConfig>();
    query->generate_config->combo_token_size    = 3;
    query->generate_config->end_think_token_ids = {5};
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);

    ASSERT_FALSE(stream->hasError());
    auto processors = stream->getAllLogitsProcessorPtr();
    auto rec_it     = std::find_if(processors.begin(), processors.end(), [](const auto& processor) {
        return std::dynamic_pointer_cast<RecommendationLogitsProcessor>(processor) != nullptr;
    });
    ASSERT_NE(rec_it, processors.end());
}

TEST_F(NormalBatchStreamProcessorTest, testThinkBeamStartsWithOneProcessorStateThenExpands) {
    std::vector<int32_t> retained_tokens(9);
    std::iota(retained_tokens.begin(), retained_tokens.end(), 0);
    auto mapping = std::make_shared<OutputVocabMapping>(10, retained_tokens, "test-model", "digest");

    ResourceContext resource_context;
    resource_context.output_vocab_mapping    = mapping;
    auto model_config                        = makeOutputVocabTestModelConfig();
    model_config.special_tokens.eos_token_id = 7;
    RuntimeConfig runtime_config;

    auto query                                  = make_shared<GenerateInput>();
    query->input_ids                            = hostIntBuffer({2});
    query->generate_config                      = make_shared<GenerateConfig>();
    query->generate_config->num_beams           = 4;
    query->generate_config->in_think_mode       = true;
    query->generate_config->max_thinking_tokens = 4;
    query->generate_config->end_think_token_ids = {7};
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);

    ASSERT_FALSE(stream->hasError());
    auto processors = stream->getAllLogitsProcessorPtr();
    auto think_it   = std::find_if(processors.begin(), processors.end(), [](const auto& processor) {
        return std::dynamic_pointer_cast<ThinkModeLogitsProcessor>(processor) != nullptr;
    });
    ASSERT_NE(think_it, processors.end());
    auto think_processor = std::dynamic_pointer_cast<ThinkModeLogitsProcessor>(*think_it);
    ASSERT_EQ(think_processor->size(), 1);

    think_processor->updateMultiSeqStatus({0, 0, 0, 0});
    EXPECT_EQ(think_processor->size(), 4);
}

}  // namespace rtp_llm
