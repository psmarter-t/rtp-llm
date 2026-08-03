#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

#include <vector>

using namespace rtp_llm;

class RocmSamplerOpTest: public DeviceTestBase {};

TEST_F(RocmSamplerOpTest, DoSampleFalseSkipsNoRepeatInMixedBatch) {
    constexpr int64_t batch_size = 2;
    constexpr int64_t vocab_size = 4;
    constexpr size_t  step       = 1;

    auto logits = torch::tensor({0.0f, 1.0f, 5.0f, 0.0f, 0.0f, 1.0f, 5.0f, 0.0f}, torch::kFloat32)
                      .reshape({batch_size, vocab_size})
                      .to(torch::kCUDA);
    auto token_ids = torch::tensor({2, 0, 2, 0}, torch::kInt32)
                         .reshape({batch_size, static_cast<int64_t>(step + 1)})
                         .to(torch::kCUDA);
    auto input_lengths    = torch::tensor({1, 1}, torch::kInt32).to(torch::kCUDA);
    auto sequence_lengths = torch::tensor({0, 0}, torch::kInt32).to(torch::kCUDA);
    auto top_k            = torch::tensor({1, 1}, torch::kInt32).pin_memory();
    auto top_p            = torch::tensor({1.0f, 1.0f}, torch::kFloat32).pin_memory();
    auto temperature      = torch::tensor({1.0f, 1.0f}, torch::kFloat32).pin_memory();
    auto no_repeat        = torch::tensor({1, 1}, torch::kInt32).pin_memory();
    auto do_sample        = torch::tensor({false, true}, torch::kBool).pin_memory();

    std::vector<at::Generator> generators(batch_size);

    GreedyParams params({logits,
                         input_lengths,
                         sequence_lengths,
                         token_ids,
                         step,
                         top_k,
                         top_p,
                         temperature,
                         std::nullopt,
                         no_repeat,
                         std::nullopt,
                         std::nullopt,
                         false,
                         std::nullopt,
                         std::nullopt,
                         std::nullopt,
                         do_sample,
                         generators});

    execSampleGreedy(params);

    const auto output_ids = token_ids.cpu().contiguous();
    EXPECT_EQ(output_ids[0][step].item<int32_t>(), 2);
    EXPECT_EQ(output_ids[1][step].item<int32_t>(), 1);
}
