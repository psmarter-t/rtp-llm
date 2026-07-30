import unittest
from types import SimpleNamespace
from typing import List

import torch

from rtp_llm.config.output_vocab_config import OutputVocabMapping
from rtp_llm.model_loader.model_weight_info import (
    ModelDeployWeightInfo,
    ModelWeightInfo,
    ModelWeights,
    select_output_vocab_rows,
)
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.models.base_model import BaseModel
from rtp_llm.utils.database import CkptDatabase
from rtp_llm.utils.model_weight import CkptWeightInfo, W, identity, sp_0_pad8


class FakeCkptFileInfo:
    def __init__(self, file_name: str, tensor_names: List[str], file_size: int = 1):
        self.file_name = file_name
        self._tensor_names = tensor_names
        self._file_size = file_size

    @property
    def file_size(self) -> int:
        return self._file_size

    def get_tensor_names(self) -> List[str]:
        return self._tensor_names


class FakeWeight:
    def __init__(self, ckpt_names: List[str]):
        self.weights = [CkptWeightInfo(name) for name in ckpt_names]

    def get_components(self):
        return [self]


class FakeCompositeWeight:
    def __init__(self, weights: List[FakeWeight]):
        self._weights = weights

    def get_components(self):
        return self._weights


def make_database(files: List[FakeCkptFileInfo]) -> CkptDatabase:
    database = CkptDatabase(None)
    database.pretrain_file_list = files
    database.finetune_file_list = []
    database._is_ft_style = False
    return database


def make_output_vocab_mapping(
    full_vocab_size: int = 11,
    local_to_full: tuple[int, ...] = (0, 2, 5, 7, 10),
) -> OutputVocabMapping:
    return OutputVocabMapping(
        full_vocab_size=full_vocab_size,
        local_to_full=local_to_full,
        model_identity="test-model",
        config_digest="test-digest",
        source_path="test-output-vocab.json",
    )


class RecordingDeployWeightInfo(ModelDeployWeightInfo):
    def __init__(self, database: CkptDatabase, returned_weight_info: ModelWeightInfo):
        self.database = database
        self.returned_weight_info = returned_weight_info
        self.output_vocab_mapping = None
        self.events = []

    def process_meta_from_ckpt(self, ckpt_metas):
        self.events.append(
            (
                "process_meta_from_ckpt",
                len(ckpt_metas),
                len(self.database.pretrain_file_list),
            )
        )

    def get_weight_info(self) -> ModelWeightInfo:
        self.events.append(("get_weight_info", len(self.database.pretrain_file_list)))
        return self.returned_weight_info


class ModelDeployWeightInfoCkptRegexTest(unittest.TestCase):
    def test_ckpt_tensor_name_regex_matches_layer_and_expert_placeholders(self):
        pattern = ModelDeployWeightInfo._ckpt_tensor_name_to_regex(
            "model.layers.{i_1}.mlp.experts.{expert_id}.down_proj.weight"
        )

        self.assertIsNotNone(
            pattern.fullmatch("model.layers.12.mlp.experts.3.down_proj.weight")
        )
        self.assertIsNone(
            pattern.fullmatch("model.layers.x.mlp.experts.3.down_proj.weight")
        )
        self.assertIsNone(
            pattern.fullmatch("model.layers.12.mlp.experts.3.down_proj.weight.extra")
        )

    def test_ckpt_tensor_name_regex_escapes_literal_dots(self):
        pattern = ModelDeployWeightInfo._ckpt_tensor_name_to_regex("lm_head.weight")

        self.assertIsNotNone(pattern.fullmatch("lm_head.weight"))
        self.assertIsNone(pattern.fullmatch("lm_headXweight"))

    def test_collect_ckpt_tensor_regexes_from_global_layer_and_composite_weights(self):
        weight_info = ModelWeightInfo(
            weights=[
                FakeWeight(["model.embed_tokens.weight"]),
                FakeCompositeWeight([FakeWeight(["lm_head.weight"])]),
            ],
            layer_weights=[
                [
                    FakeWeight(["model.layers.{i}.self_attn.q_proj.weight"]),
                    FakeCompositeWeight(
                        [FakeWeight(["model.layers.{i}.mlp.experts.{expert_id}.w1"])]
                    ),
                ]
            ],
        )

        patterns = ModelDeployWeightInfo._collect_ckpt_tensor_name_regexes(weight_info)

        self.assertEqual(len(patterns), 4)
        self.assertTrue(
            any(pattern.fullmatch("model.embed_tokens.weight") for pattern in patterns)
        )
        self.assertTrue(
            any(pattern.fullmatch("lm_head.weight") for pattern in patterns)
        )
        self.assertTrue(
            any(
                pattern.fullmatch("model.layers.0.self_attn.q_proj.weight")
                for pattern in patterns
            )
        )
        self.assertTrue(
            any(
                pattern.fullmatch("model.layers.1.mlp.experts.7.w1")
                for pattern in patterns
            )
        )

    def test_collect_ckpt_tensor_regexes_ignores_empty_weight_info(self):
        weight_info = ModelWeightInfo(weights=[], layer_weights=[])

        patterns = ModelDeployWeightInfo._collect_ckpt_tensor_name_regexes(weight_info)

        self.assertEqual(patterns, [])


class OutputVocabLmHeadSelectionTest(unittest.TestCase):
    def test_untied_lm_head_wrapper_selects_rows_after_original_process(self):
        mapping = make_output_vocab_mapping()
        lm_head = torch.arange(44, dtype=torch.float32).reshape(11, 4)

        def add_offset(tensors):
            return tensors[0] + 100

        deploy_weight_info = ModelDeployWeightInfo.__new__(ModelDeployWeightInfo)
        deploy_weight_info.model_config = SimpleNamespace(has_lm_head_bias=False)
        deploy_weight_info._hidden_size = 4
        deploy_weight_info.output_vocab_mapping = mapping
        weight_info = ModelWeightInfo(
            weights=[
                AtomicWeight(
                    W.embedding,
                    [CkptWeightInfo("model.embed_tokens.weight")],
                    identity,
                ),
                AtomicWeight(
                    W.lm_head,
                    [CkptWeightInfo("lm_head.weight")],
                    add_offset,
                ),
            ],
            layer_weights=[],
        )

        deploy_weight_info._fix_output_vocab_lm_head(weight_info)
        wrapped_lm_head = next(
            weight for weight in weight_info.weights if weight.name == W.lm_head
        )

        torch.testing.assert_close(
            wrapped_lm_head.process_fun([lm_head]),
            (lm_head + 100)[list(mapping.local_to_full)],
        )

    def test_tied_lm_head_fallback_is_resolved_before_row_selection(self):
        mapping = make_output_vocab_mapping()
        embedding = torch.arange(44, dtype=torch.float32).reshape(11, 4)
        deploy_weight_info = ModelDeployWeightInfo.__new__(ModelDeployWeightInfo)
        deploy_weight_info.model_config = SimpleNamespace(has_lm_head_bias=False)
        deploy_weight_info._hidden_size = 4
        deploy_weight_info.output_vocab_mapping = mapping
        weight_info = ModelWeightInfo(
            weights=[
                AtomicWeight(
                    W.embedding,
                    [CkptWeightInfo("model.embed_tokens.weight")],
                    identity,
                ),
                AtomicWeight(
                    W.lm_head,
                    [CkptWeightInfo("lm_head.weight")],
                    identity,
                ),
            ],
            layer_weights=[],
        )

        deploy_weight_info._fix_tie_lm_head(weight_info)
        deploy_weight_info._fix_output_vocab_lm_head(weight_info)
        wrapped_lm_head = next(
            weight for weight in weight_info.weights if weight.name == W.lm_head
        )
        input_embedding = next(
            weight for weight in weight_info.weights if weight.name == W.embedding
        )

        torch.testing.assert_close(
            wrapped_lm_head.process_fun([None, embedding]),
            embedding[list(mapping.local_to_full)],
        )
        torch.testing.assert_close(input_embedding.process_fun([embedding]), embedding)

    def test_selects_discrete_rows_and_returns_contiguous_tensor(self):
        lm_head = torch.arange(30, dtype=torch.float32).reshape(6, 5)

        result = select_output_vocab_rows(
            [lm_head],
            origin_func=identity,
            local_to_full=(0, 2, 5),
            full_vocab_size=6,
            hidden_size=5,
        )

        torch.testing.assert_close(result, lm_head[[0, 2, 5]])
        self.assertTrue(result.is_contiguous())

    def test_tp_split_gather_matches_full_lm_head_reference(self):
        full_vocab_size = 23
        hidden_size = 4
        tp_size = 3
        local_to_full = (
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            16,
            17,
            19,
            21,
            22,
        )
        lm_head = (
            torch.arange(
                full_vocab_size * hidden_size,
                dtype=torch.float32,
            ).reshape(full_vocab_size, hidden_size)
            / 100
        )
        hidden_states = (
            torch.arange(3 * hidden_size, dtype=torch.float32).reshape(3, hidden_size)
            / 10
        )

        pruned_lm_head = select_output_vocab_rows(
            [lm_head],
            origin_func=identity,
            local_to_full=local_to_full,
            full_vocab_size=full_vocab_size,
            hidden_size=hidden_size,
        )
        rank_lm_heads = [
            sp_0_pad8(pruned_lm_head, tp=tp_size, tp_rank=tp_rank)
            for tp_rank in range(tp_size)
        ]

        self.assertEqual([list(shard.shape) for shard in rank_lm_heads], [[8, 4]] * 3)
        gathered_logits_with_padding = torch.cat(
            [hidden_states @ shard.t() for shard in rank_lm_heads],
            dim=1,
        )
        logical_logits = gathered_logits_with_padding[:, : len(local_to_full)]

        full_logits = hidden_states @ lm_head.t()
        reference_logits = full_logits.index_select(
            1, torch.tensor(local_to_full, dtype=torch.long)
        )

        self.assertEqual(list(gathered_logits_with_padding.shape), [3, 24])
        torch.testing.assert_close(
            gathered_logits_with_padding[:, len(local_to_full) :],
            torch.zeros(3, 5),
        )
        torch.testing.assert_close(logical_logits, reference_logits)

    def test_pruned_logits_match_reference_for_greedy_sampling_and_beam(self):
        full_vocab_size = 13
        hidden_size = 4
        local_to_full = (0, 2, 5, 6, 9, 12)
        lm_head = (
            torch.arange(full_vocab_size * hidden_size, dtype=torch.float32)
            .reshape(full_vocab_size, hidden_size)
            .div(100)
        )
        hidden_states = torch.tensor(
            [[0.5, -0.25, 0.75, 1.0], [-0.3, 0.9, 0.2, 0.4]],
            dtype=torch.float32,
        )

        pruned_lm_head = select_output_vocab_rows(
            [lm_head],
            origin_func=identity,
            local_to_full=local_to_full,
            full_vocab_size=full_vocab_size,
            hidden_size=hidden_size,
        )
        physical_logits = hidden_states @ pruned_lm_head.t()
        reference_logits = (hidden_states @ lm_head.t()).index_select(
            1, torch.tensor(local_to_full, dtype=torch.long)
        )
        torch.testing.assert_close(physical_logits, reference_logits)

        local_to_full_tensor = torch.tensor(local_to_full, dtype=torch.long)
        physical_greedy = physical_logits.argmax(dim=-1)
        reference_greedy = reference_logits.argmax(dim=-1)
        torch.testing.assert_close(physical_greedy, reference_greedy)
        torch.testing.assert_close(
            local_to_full_tensor[physical_greedy],
            local_to_full_tensor[reference_greedy],
        )

        def sample_top_k(logits: torch.Tensor, seed: int) -> torch.Tensor:
            top_values, top_indices = torch.topk(logits, k=4)
            generator = torch.Generator().manual_seed(seed)
            sampled = torch.multinomial(
                torch.softmax(top_values / 0.8, dim=-1),
                num_samples=1,
                generator=generator,
            )
            return top_indices.gather(1, sampled).squeeze(1)

        physical_sample = sample_top_k(physical_logits, seed=20260728)
        reference_sample = sample_top_k(reference_logits, seed=20260728)
        torch.testing.assert_close(physical_sample, reference_sample)
        torch.testing.assert_close(
            local_to_full_tensor[physical_sample],
            local_to_full_tensor[reference_sample],
        )

        cumulative_scores = torch.tensor([-0.2, -1.1], dtype=torch.float32)
        physical_beam_scores = (
            torch.log_softmax(physical_logits, dim=-1) + cumulative_scores.unsqueeze(1)
        ).flatten()
        reference_beam_scores = (
            torch.log_softmax(reference_logits, dim=-1) + cumulative_scores.unsqueeze(1)
        ).flatten()
        physical_scores, physical_flat_ids = torch.topk(physical_beam_scores, k=3)
        reference_scores, reference_flat_ids = torch.topk(reference_beam_scores, k=3)
        logical_vocab_size = len(local_to_full)
        physical_token_ids = physical_flat_ids % logical_vocab_size
        reference_token_ids = reference_flat_ids % logical_vocab_size

        torch.testing.assert_close(physical_scores, reference_scores)
        torch.testing.assert_close(physical_flat_ids, reference_flat_ids)
        torch.testing.assert_close(
            local_to_full_tensor[physical_token_ids],
            local_to_full_tensor[reference_token_ids],
        )

    def test_tp_split_gather_matches_reference_when_k_is_smaller_than_tp_alignment(
        self,
    ):
        full_vocab_size = 11
        hidden_size = 4
        tp_size = 4
        local_to_full = (0, 2, 5, 7, 10)
        lm_head = torch.arange(
            full_vocab_size * hidden_size, dtype=torch.float32
        ).reshape(full_vocab_size, hidden_size)
        hidden_states = torch.arange(2 * hidden_size, dtype=torch.float32).reshape(
            2, hidden_size
        )

        pruned_lm_head = select_output_vocab_rows(
            [lm_head],
            origin_func=identity,
            local_to_full=local_to_full,
            full_vocab_size=full_vocab_size,
            hidden_size=hidden_size,
        )
        rank_lm_heads = [
            sp_0_pad8(pruned_lm_head, tp=tp_size, tp_rank=tp_rank)
            for tp_rank in range(tp_size)
        ]

        self.assertEqual([list(shard.shape) for shard in rank_lm_heads], [[8, 4]] * 4)
        gathered_logits = torch.cat(
            [hidden_states @ shard.t() for shard in rank_lm_heads], dim=1
        )
        reference_logits = (hidden_states @ lm_head.t()).index_select(
            1, torch.tensor(local_to_full, dtype=torch.long)
        )

        torch.testing.assert_close(
            gathered_logits[:, : len(local_to_full)], reference_logits
        )
        torch.testing.assert_close(
            gathered_logits[:, len(local_to_full) :], torch.zeros(2, 27)
        )

    def test_lensrecall_sized_tp2_split_gather_trims_padding(self):
        full_vocab_size = 217303
        output_vocab_start = 151643
        output_vocab_size = 65660
        hidden_size = 2
        tp_size = 2
        local_to_full = tuple(
            range(output_vocab_start, output_vocab_start + output_vocab_size)
        )
        lm_head = torch.arange(
            full_vocab_size * hidden_size, dtype=torch.float32
        ).reshape(full_vocab_size, hidden_size)
        hidden_states = torch.tensor([[0.25, -0.5], [1.0, 0.125]], dtype=torch.float32)

        pruned_lm_head = select_output_vocab_rows(
            [lm_head],
            origin_func=identity,
            local_to_full=local_to_full,
            full_vocab_size=full_vocab_size,
            hidden_size=hidden_size,
        )
        rank_lm_heads = [
            sp_0_pad8(pruned_lm_head, tp=tp_size, tp_rank=tp_rank)
            for tp_rank in range(tp_size)
        ]
        gathered_logits = torch.cat(
            [hidden_states @ shard.t() for shard in rank_lm_heads], dim=1
        )
        logical_logits = gathered_logits[:, :output_vocab_size]
        reference_logits = (hidden_states @ lm_head.t()).index_select(
            1, torch.tensor(local_to_full, dtype=torch.long)
        )

        self.assertEqual(
            [list(shard.shape) for shard in rank_lm_heads], [[32832, 2]] * 2
        )
        self.assertEqual(list(gathered_logits.shape), [2, 65664])
        torch.testing.assert_close(logical_logits, reference_logits)
        torch.testing.assert_close(
            gathered_logits[:, output_vocab_size:], torch.zeros(2, 4)
        )

    def test_sp_0_pad8_returns_equal_shards_for_k_and_tp_matrix(self):
        cases = (
            (1, 2),
            (5, 2),
            (10, 4),
            (16, 2),
            (17, 2),
            (33, 4),
        )
        for output_vocab_size, tp_size in cases:
            with self.subTest(output_vocab_size=output_vocab_size, tp_size=tp_size):
                tensor = torch.arange(
                    output_vocab_size * 3, dtype=torch.float32
                ).reshape(output_vocab_size, 3)
                shards = [
                    sp_0_pad8(tensor, tp=tp_size, tp_rank=tp_rank)
                    for tp_rank in range(tp_size)
                ]
                expected_per_rank = (
                    (output_vocab_size + tp_size * 8 - 1) // (tp_size * 8)
                ) * 8
                self.assertEqual(
                    [list(shard.shape) for shard in shards],
                    [[expected_per_rank, 3]] * tp_size,
                )

                gathered = torch.cat(shards, dim=0)
                torch.testing.assert_close(gathered[:output_vocab_size], tensor)
                torch.testing.assert_close(
                    gathered[output_vocab_size:],
                    torch.zeros(tp_size * expected_per_rank - output_vocab_size, 3),
                )

    def test_sp_0_pad8_supports_one_dimensional_tensor(self):
        tensor = torch.arange(5, dtype=torch.float32)
        shards = [sp_0_pad8(tensor, tp=4, tp_rank=rank) for rank in range(4)]

        self.assertEqual([list(shard.shape) for shard in shards], [[8]] * 4)
        gathered = torch.cat(shards)
        torch.testing.assert_close(gathered[:5], tensor)
        torch.testing.assert_close(gathered[5:], torch.zeros(27))

    def test_rejects_non_matrix_lm_head(self):
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            select_output_vocab_rows(
                [torch.arange(6)],
                origin_func=identity,
                local_to_full=(0, 2),
                full_vocab_size=6,
                hidden_size=4,
            )

    def test_rejects_lm_head_that_does_not_cover_vocab(self):
        with self.assertRaisesRegex(ValueError, "does not cover"):
            select_output_vocab_rows(
                [torch.zeros(5, 4)],
                origin_func=identity,
                local_to_full=(0, 2),
                full_vocab_size=6,
                hidden_size=4,
            )

    def test_rejects_non_floating_lm_head(self):
        with self.assertRaisesRegex(ValueError, "FP16, BF16, or FP32"):
            select_output_vocab_rows(
                [torch.zeros(6, 4, dtype=torch.int8)],
                origin_func=identity,
                local_to_full=(0, 2),
                full_vocab_size=6,
                hidden_size=4,
            )

    def test_rejects_transposed_lm_head_even_when_first_dimension_covers_vocab(self):
        with self.assertRaisesRegex(ValueError, r"row-major \[V,H\]"):
            select_output_vocab_rows(
                [torch.zeros(8, 6)],
                origin_func=identity,
                local_to_full=(0, 2),
                full_vocab_size=6,
                hidden_size=8,
            )


class OutputVocabWeightValidationTest(unittest.TestCase):
    @staticmethod
    def _make_model(
        mapping: OutputVocabMapping,
        lm_head: torch.Tensor,
        tp_size: int = 1,
        dp_size: int = 1,
        ep_size: int = 1,
        lm_head_tp_size: int | None = None,
        hidden_size: int | None = None,
    ) -> BaseModel:
        model = BaseModel.__new__(BaseModel)
        model.output_vocab_mapping = mapping
        model.model_config = SimpleNamespace(
            hidden_size=lm_head.size(1) if hidden_size is None else hidden_size
        )
        load_config = SimpleNamespace(
            tp_size=tp_size,
            dp_size=dp_size,
            ep_size=ep_size,
            lm_head_tp_size=(tp_size if lm_head_tp_size is None else lm_head_tp_size),
        )
        model.model_weights_loader = SimpleNamespace(
            get_load_config=lambda: load_config,
        )
        model.weight = ModelWeights(num_layers=0, device="cpu", dtype=torch.float32)
        model.weight.set_global_weight(
            W.embedding,
            torch.zeros(mapping.full_vocab_size, lm_head.size(1)),
        )
        model.weight.set_global_weight(W.lm_head, lm_head)
        return model

    def test_default_off_skips_loaded_weight_validation(self):
        model = BaseModel.__new__(BaseModel)
        model.output_vocab_mapping = None
        model.weight = None

        model._validate_output_vocab_weights()

    def test_tp1_accepts_unpadded_output_vocab_rows(self):
        mapping = make_output_vocab_mapping()
        model = self._make_model(mapping, torch.zeros(mapping.size, 4))

        model._validate_output_vocab_weights()

    def test_tp2_accepts_per_rank_padded_output_vocab_rows(self):
        mapping = make_output_vocab_mapping()
        pruned_lm_head = torch.zeros(mapping.size, 4)
        rank_lm_head = sp_0_pad8(pruned_lm_head, tp=2, tp_rank=1)
        model = self._make_model(mapping, rank_lm_head, tp_size=2)

        model._validate_output_vocab_weights()

    def test_effective_tp1_accepts_unsharded_rows_with_lm_head_tp2(self):
        mapping = make_output_vocab_mapping()
        model = self._make_model(
            mapping,
            torch.zeros(mapping.size, 4),
            tp_size=1,
            lm_head_tp_size=2,
        )

        model._validate_output_vocab_weights()

    def test_tp1_accepts_padding_when_dp_or_ep_invokes_weight_split(self):
        mapping = make_output_vocab_mapping()
        padded_lm_head = sp_0_pad8(
            torch.zeros(mapping.size, 4),
            tp=1,
            tp_rank=0,
        )

        for dp_size, ep_size in ((2, 1), (1, 2)):
            with self.subTest(dp_size=dp_size, ep_size=ep_size):
                model = self._make_model(
                    mapping,
                    padded_lm_head,
                    dp_size=dp_size,
                    ep_size=ep_size,
                )
                model._validate_output_vocab_weights()

    def test_rejects_loaded_lm_head_with_wrong_hidden_dimension(self):
        mapping = make_output_vocab_mapping()
        model = self._make_model(
            mapping,
            torch.zeros(mapping.size, 4),
            hidden_size=8,
        )

        with self.assertRaisesRegex(ValueError, r"row-major \[V,H\]"):
            model._validate_output_vocab_weights()


class CkptDatabaseFilterTest(unittest.TestCase):
    def test_get_max_file_size_returns_zero_for_empty_pretrain_files(self):
        database = make_database([])

        self.assertEqual(database.get_max_file_size(), 0)

    def test_filter_by_tensor_name_regexes_keeps_only_matching_files(self):
        database = make_database(
            [
                FakeCkptFileInfo(
                    "base.safetensors",
                    ["model.layers.0.self_attn.q_proj.weight"],
                ),
                FakeCkptFileInfo(
                    "mtp.safetensors",
                    ["mtp.layers.12.self_attn.q_proj.weight"],
                ),
                FakeCkptFileInfo(
                    "prefix_only.safetensors",
                    ["mtp.layers.12.self_attn.q_proj.weight.extra"],
                ),
            ]
        )
        patterns = [
            ModelDeployWeightInfo._ckpt_tensor_name_to_regex(
                "mtp.layers.{i}.self_attn.q_proj.weight"
            )
        ]

        database.filter_by_tensor_name_regexes(patterns)

        self.assertEqual(
            [ckpt.file_name for ckpt in database.pretrain_file_list],
            ["mtp.safetensors"],
        )

    def test_filter_by_tensor_name_regexes_is_noop_for_single_file(self):
        original_file = FakeCkptFileInfo(
            "single.safetensors",
            ["irrelevant.weight"],
        )
        database = make_database([original_file])
        patterns = [ModelDeployWeightInfo._ckpt_tensor_name_to_regex("required.weight")]

        database.filter_by_tensor_name_regexes(patterns)

        self.assertEqual(database.pretrain_file_list, [original_file])

    def test_filter_by_tensor_name_regexes_is_noop_for_empty_patterns(self):
        files = [
            FakeCkptFileInfo("a.safetensors", ["a.weight"]),
            FakeCkptFileInfo("b.safetensors", ["b.weight"]),
        ]
        database = make_database(files)

        database.filter_by_tensor_name_regexes([])

        self.assertEqual(database.pretrain_file_list, files)

    def test_filter_by_tensor_name_regexes_keeps_original_when_no_file_matches(self):
        files = [
            FakeCkptFileInfo("a.safetensors", ["a.weight"]),
            FakeCkptFileInfo("b.safetensors", ["b.weight"]),
        ]
        database = make_database(files)
        patterns = [ModelDeployWeightInfo._ckpt_tensor_name_to_regex("missing.weight")]

        database.filter_by_tensor_name_regexes(patterns)

        self.assertEqual(database.pretrain_file_list, files)


class CreateModelWeightInfoFilterOrderTest(unittest.TestCase):
    def test_create_model_weight_info_filters_after_meta_and_final_weight_info(self):
        database = make_database(
            [
                FakeCkptFileInfo(
                    "base.safetensors",
                    ["model.layers.0.self_attn.q_proj.weight"],
                ),
                FakeCkptFileInfo(
                    "mtp.safetensors",
                    ["mtp.layers.0.self_attn.q_proj.weight"],
                ),
            ]
        )
        returned_weight_info = ModelWeightInfo(
            weights=[],
            layer_weights=[[FakeWeight(["mtp.layers.{i}.self_attn.q_proj.weight"])]],
        )
        weight_info = RecordingDeployWeightInfo(database, returned_weight_info)

        result = weight_info.create_model_weight_info(database)

        self.assertIs(result, returned_weight_info)
        self.assertEqual(
            weight_info.events,
            [
                ("process_meta_from_ckpt", 2, 2),
                ("process_meta_from_ckpt", 0, 2),
                ("get_weight_info", 2),
            ],
        )
        self.assertEqual(
            [ckpt.file_name for ckpt in database.pretrain_file_list],
            ["mtp.safetensors"],
        )

    def test_create_model_weight_info_returns_none_for_ft_style_database(self):
        database = make_database([])
        database._is_ft_style = True
        weight_info = RecordingDeployWeightInfo(
            database,
            ModelWeightInfo(weights=[], layer_weights=[]),
        )

        self.assertIsNone(weight_info.create_model_weight_info(database))
        self.assertEqual(weight_info.events, [])

    def test_create_model_weight_info_raises_for_unknown_database_type(self):
        class UnknownDatabase:
            is_ft_style = False

        weight_info = RecordingDeployWeightInfo(
            make_database([]),
            ModelWeightInfo(weights=[], layer_weights=[]),
        )

        with self.assertRaisesRegex(Exception, "Unknown database class"):
            weight_info.create_model_weight_info(UnknownDatabase())


if __name__ == "__main__":
    unittest.main()
