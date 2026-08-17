load("//:def.bzl", "any_cuda_copts")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "topk_bench",
    srcs = ["topk_bench.cu"],
    copts = any_cuda_copts(),
    deps = [
        "//3rdparty/trt_beam_search:trt_beam_search_impl",
    ] + select({
        "@//:using_rocm": [
            "@local_config_rocm//rocm:rocm_headers",
            "@local_config_rocm//rocm:hip",
        ],
        "//conditions:default": [
            "@local_config_cuda//cuda:cuda",
            "@local_config_cuda//cuda:cudart",
        ],
    }),
)
