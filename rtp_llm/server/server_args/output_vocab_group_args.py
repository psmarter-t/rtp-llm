from rtp_llm.server.server_args.util import str2bool


def init_output_vocab_group_args(parser, output_vocab_config):
    output_vocab_group = parser.add_argument_group("Output Vocabulary")
    output_vocab_group.add_argument(
        "--enable_output_vocab_pruning",
        env_name="ENABLE_OUTPUT_VOCAB_PRUNING",
        bind_to=(output_vocab_config, "enabled"),
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help=(
            "Enable output vocabulary pruning using output_vocab.json from the "
            "checkpoint directory."
        ),
    )
