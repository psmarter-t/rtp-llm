load("//:def.bzl", "cc_test_wrapper", "copts")

cc_test = cc_test_wrapper

cc_test(
    name = "output_vocab_mapping_test",
    srcs = ["OutputVocabMappingTest.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/config:output_vocab_mapping",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
