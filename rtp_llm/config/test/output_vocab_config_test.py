import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rtp_llm.config.output_vocab_config import (
    OutputVocabConfig,
    OutputVocabMapping,
    OutputVocabRankState,
    load_output_vocab_mapping,
    validate_output_vocab_rank_states,
)
from rtp_llm.model_factory import ModelFactory


class OutputVocabConfigTest(unittest.TestCase):
    def _write_config(self, config):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        config_path = os.path.join(temp_dir.name, "output_vocab.json")
        with open(config_path, "w", encoding="utf-8") as writer:
            json.dump(config, writer)
        return config_path

    def _base_config(self, output_vocab):
        return {
            "version": 1,
            "model_identity": "test-model-revision",
            "model_vocab_size": 10,
            "output_vocab": output_vocab,
        }

    def test_mixed_ranges_and_tokens_are_sorted_and_deduplicated(self):
        config_path = self._write_config(
            self._base_config(
                {
                    "ranges": [
                        {"start_id": 5, "end_id": 8},
                        {"start_id": 1, "end_id": 3},
                    ],
                    "token_ids": [9, 2, 5, 0],
                }
            )
        )

        mapping = load_output_vocab_mapping(config_path, full_vocab_size=10)

        self.assertEqual(mapping.local_to_full, (0, 1, 2, 5, 6, 7, 9))
        self.assertEqual(mapping.size, 7)
        self.assertEqual(mapping.to_local(5), 3)
        self.assertIsNone(mapping.to_local(4))
        self.assertEqual(mapping.to_full(6), 9)
        self.assertTrue(mapping.contains(7))
        self.assertFalse(mapping.contains(8))
        self.assertEqual(len(mapping.config_digest), 64)

    def test_pure_discrete_tokens_are_supported(self):
        config_path = self._write_config(self._base_config({"token_ids": [9, 0, 4, 4]}))

        mapping = load_output_vocab_mapping(config_path, full_vocab_size=10)

        self.assertEqual(mapping.local_to_full, (0, 4, 9))

    def test_lensrecall_sized_range_has_expected_boundaries(self):
        config_path = self._write_config(
            {
                "version": 1,
                "model_identity": "LensRecall_nd_pg_attn@test",
                "model_vocab_size": 217303,
                "output_vocab": {
                    "ranges": [{"start_id": 151643, "end_id": 217303}],
                    "token_ids": [],
                },
            }
        )

        mapping = load_output_vocab_mapping(config_path, full_vocab_size=217303)

        self.assertEqual(mapping.size, 65660)
        self.assertEqual(mapping.local_to_full[0], 151643)
        self.assertEqual(mapping.local_to_full[-1], 217302)
        self.assertEqual(mapping.to_local(151643), 0)
        self.assertEqual(mapping.to_local(217302), 65659)
        self.assertIsNone(mapping.to_local(151642))

    def test_complete_vocab_disables_pruning(self):
        config_path = self._write_config(
            self._base_config({"ranges": [{"start_id": 0, "end_id": 10}]})
        )
        config = OutputVocabConfig(
            config_path=config_path, model_identity="test-model-revision"
        )

        self.assertIsNone(config.resolve(full_vocab_size=10))
        self.assertIsNone(config.resolve(full_vocab_size=10))
        self.assertEqual(len(config.config_digest), 64)
        self.assertIn("output_vocab_pruning: disabled", config.to_string())
        self.assertIn("test-model-revision", config.to_string())

    def test_input_embedding_must_cover_all_output_tokens(self):
        config_path = self._write_config(self._base_config({"token_ids": [0, 7]}))

        with self.assertRaisesRegex(ValueError, "input embedding size 6"):
            load_output_vocab_mapping(
                config_path, full_vocab_size=10, input_vocab_size=6
            )

    def test_invalid_configs_are_rejected(self):
        invalid_cases = [
            (self._base_config({}), "does not contain any token"),
            (
                self._base_config({"ranges": [{"start_id": 5, "end_id": 5}]}),
                "must satisfy",
            ),
            (self._base_config({"token_ids": [10]}), "outside"),
            (
                {
                    **self._base_config({"token_ids": [1]}),
                    "model_vocab_size": 11,
                },
                "does not match",
            ),
            (
                {
                    **self._base_config({"token_ids": [1]}),
                    "version": 2,
                },
                "unsupported",
            ),
            (
                {
                    **self._base_config({"token_ids": [1]}),
                    "model_identity": "",
                },
                "non-empty",
            ),
        ]

        for config, message in invalid_cases:
            with self.subTest(message=message):
                config_path = self._write_config(config)
                with self.assertRaisesRegex(ValueError, message):
                    load_output_vocab_mapping(config_path, full_vocab_size=10)

    def test_unknown_fields_are_rejected_at_every_object_level(self):
        invalid_cases = [
            (
                {
                    **self._base_config({"token_ids": [1]}),
                    "model_revision": "typo",
                },
                "output vocab config root.*model_revision",
            ),
            (
                self._base_config({"token_id": [1]}),
                "output_vocab.*token_id",
            ),
            (
                self._base_config({"ranges": [{"start_id": 1, "end_id": 2, "end": 2}]}),
                r"output_vocab\.ranges\[0\].*end",
            ),
        ]

        for config, message in invalid_cases:
            with self.subTest(message=message):
                config_path = self._write_config(config)
                with self.assertRaisesRegex(ValueError, message):
                    load_output_vocab_mapping(config_path, full_vocab_size=10)

    def test_local_id_bounds_are_checked(self):
        config_path = self._write_config(self._base_config({"token_ids": [0, 4, 9]}))
        mapping = load_output_vocab_mapping(config_path, full_vocab_size=10)

        with self.assertRaisesRegex(ValueError, "outside output vocab"):
            mapping.to_full(3)

    def test_cached_mapping_is_bound_to_input_vocab_size(self):
        config_path = self._write_config(self._base_config({"token_ids": [0, 4, 7]}))
        config = OutputVocabConfig(
            config_path=config_path, model_identity="test-model-revision"
        )

        self.assertIsNotNone(config.resolve(full_vocab_size=10, input_vocab_size=8))
        with self.assertRaisesRegex(ValueError, "current model uses"):
            config.resolve(full_vocab_size=10, input_vocab_size=9)

    def test_deployment_identity_is_required_and_must_match(self):
        config_path = self._write_config(self._base_config({"token_ids": [0, 4, 7]}))

        with self.assertRaisesRegex(ValueError, "must be set"):
            OutputVocabConfig(config_path=config_path).resolve(full_vocab_size=10)

        with self.assertRaisesRegex(ValueError, "does not match deployment identity"):
            OutputVocabConfig(
                config_path=config_path, model_identity="different-model"
            ).resolve(full_vocab_size=10)

        mapping = OutputVocabConfig(
            config_path=config_path, model_identity="test-model-revision"
        ).resolve(full_vocab_size=10)
        self.assertIsNotNone(mapping)

    def test_matching_rank_states_are_accepted(self):
        states = [
            OutputVocabRankState(0, True, True, 10, 3, "digest"),
            OutputVocabRankState(1, True, True, 10, 3, "digest"),
        ]

        validate_output_vocab_rank_states(states, expected_world_size=2)

    def test_rank_parse_error_is_reported_to_all_ranks(self):
        states = [
            OutputVocabRankState(0, True, True, 10, 3, "digest"),
            OutputVocabRankState(
                1, False, False, 10, 0, "", "ValueError: invalid token"
            ),
        ]

        with self.assertRaisesRegex(ValueError, "rank 1.*invalid token"):
            validate_output_vocab_rank_states(states, expected_world_size=2)

    def test_rank_semantic_mismatch_is_rejected(self):
        mismatch_cases = [
            OutputVocabRankState(1, True, False, 10, 10, ""),
            OutputVocabRankState(1, True, True, 10, 4, "digest"),
            OutputVocabRankState(1, True, True, 10, 3, "other-digest"),
        ]
        rank_zero = OutputVocabRankState(0, True, True, 10, 3, "digest")

        for rank_one in mismatch_cases:
            with self.subTest(rank_one=rank_one):
                with self.assertRaisesRegex(ValueError, "differs across ranks"):
                    validate_output_vocab_rank_states(
                        [rank_zero, rank_one], expected_world_size=2
                    )

    def test_disabled_rank_with_explicit_config_digest_is_not_unconfigured(self):
        states = [
            OutputVocabRankState(0, True, False, 10, 10, "full-digest"),
            OutputVocabRankState(1, True, False, 10, 10, ""),
        ]

        with self.assertRaisesRegex(ValueError, "differs across ranks"):
            validate_output_vocab_rank_states(states, expected_world_size=2)

    def test_invalid_rank_state_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid rank state"):
            validate_output_vocab_rank_states([None], expected_world_size=1)


class OutputVocabRankCollectiveTest(unittest.TestCase):
    def _resolve_with_peer(self, output_vocab_config, peer_state):
        model_config = SimpleNamespace(vocab_size=10, input_vocab_size=10)
        engine_config = SimpleNamespace(
            output_vocab_config=output_vocab_config,
            parallelism_config=SimpleNamespace(world_size=2),
        )

        def gather_rank_states(rank_states, local_state):
            rank_states[:] = [local_state, peer_state]

        with (
            patch(
                "rtp_llm.model_factory.torch.distributed.is_initialized",
                return_value=True,
            ),
            patch(
                "rtp_llm.model_factory.torch.distributed.get_world_size",
                return_value=2,
            ),
            patch("rtp_llm.model_factory.torch.distributed.get_rank", return_value=0),
            patch(
                "rtp_llm.model_factory.torch.distributed.all_gather_object",
                side_effect=gather_rank_states,
            ) as gather,
        ):
            result = ModelFactory._resolve_output_vocab_mapping(
                model_config, engine_config
            )
        return result, gather

    def test_disabled_rank_still_joins_collective(self):
        output_vocab_config = SimpleNamespace(
            resolve=Mock(return_value=None), config_digest=""
        )
        peer_state = OutputVocabRankState(1, True, True, 10, 3, "digest")

        with self.assertRaisesRegex(ValueError, "differs across ranks"):
            self._resolve_with_peer(output_vocab_config, peer_state)

    def test_local_parse_error_is_gathered_before_failure(self):
        output_vocab_config = SimpleNamespace(
            resolve=Mock(side_effect=ValueError("bad config")), config_digest=""
        )
        peer_state = OutputVocabRankState(1, True, False, 10, 10, "")

        with self.assertRaisesRegex(ValueError, "rank 0.*bad config"):
            self._resolve_with_peer(output_vocab_config, peer_state)

    def test_matching_rank_state_returns_mapping(self):
        mapping = OutputVocabMapping(
            full_vocab_size=10,
            local_to_full=(0, 4, 9),
            model_identity="test-model-revision",
            config_digest="digest",
            source_path="output-vocab.json",
        )
        output_vocab_config = SimpleNamespace(
            resolve=Mock(return_value=mapping), config_digest="digest"
        )
        peer_state = OutputVocabRankState(1, True, True, 10, 3, "digest")

        result, gather = self._resolve_with_peer(output_vocab_config, peer_state)

        self.assertIs(result, mapping)
        gather.assert_called_once()


if __name__ == "__main__":
    unittest.main()
