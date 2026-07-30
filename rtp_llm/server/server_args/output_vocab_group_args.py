def init_output_vocab_group_args(parser, output_vocab_config):
    output_vocab_group = parser.add_argument_group("Output Vocabulary")
    output_vocab_group.add_argument(
        "--output_vocab_config_path",
        env_name="OUTPUT_VOCAB_CONFIG_PATH",
        bind_to=(output_vocab_config, "config_path"),
        type=str,
        default="",
        help="Path to the model-level output vocabulary pruning JSON config.",
    )
    output_vocab_group.add_argument(
        "--output_vocab_model_identity",
        env_name="OUTPUT_VOCAB_MODEL_IDENTITY",
        bind_to=(output_vocab_config, "model_identity"),
        type=str,
        default="",
        help=(
            "Deployment-provided model/tokenizer revision or manifest identity. "
            "It must exactly match model_identity in the output vocabulary config."
        ),
    )
