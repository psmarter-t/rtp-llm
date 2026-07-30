#include "rtp_llm/cpp/config/OutputVocabMapping.h"

#include <gtest/gtest.h>

namespace rtp_llm {

TEST(OutputVocabMappingTest, ConvertsBetweenLocalAndFullIds) {
    OutputVocabMapping mapping(10, {0, 2, 7, 9}, "test-model", "digest");

    EXPECT_EQ(mapping.fullVocabSize(), 10);
    EXPECT_EQ(mapping.size(), 4);
    EXPECT_EQ(mapping.toFull(2), 7);
    EXPECT_EQ(mapping.toLocal(9), 3);
    EXPECT_FALSE(mapping.toLocal(8).has_value());
    EXPECT_FALSE(mapping.toLocal(-1).has_value());
    EXPECT_FALSE(mapping.toLocal(10).has_value());
    EXPECT_TRUE(mapping.contains(0));
}

TEST(OutputVocabMappingTest, RejectsInvalidMappings) {
    EXPECT_THROW(OutputVocabMapping(10, {}, "model", "digest"), std::invalid_argument);
    EXPECT_THROW(OutputVocabMapping(10, {0, 2, 2}, "model", "digest"), std::invalid_argument);
    EXPECT_THROW(OutputVocabMapping(10, {0, 11}, "model", "digest"), std::invalid_argument);
    EXPECT_THROW(OutputVocabMapping(10, {2, 1}, "model", "digest"), std::invalid_argument);
}

TEST(OutputVocabMappingTest, RejectsOutOfRangeLocalId) {
    OutputVocabMapping mapping(10, {0, 2, 7}, "test-model", "digest");

    EXPECT_THROW(mapping.toFull(-1), std::out_of_range);
    EXPECT_THROW(mapping.toFull(3), std::out_of_range);
}

}  // namespace rtp_llm
