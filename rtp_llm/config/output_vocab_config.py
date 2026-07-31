import bisect
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Tuple


OUTPUT_VOCAB_FILENAME = "output_vocab.json"


def _require_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer, got {value!r}")
    return value


def _reject_unknown_fields(
    value: dict, allowed_fields: Sequence[str], field_name: str
) -> None:
    unknown_fields = sorted(set(value) - set(allowed_fields))
    if unknown_fields:
        raise ValueError(
            f"{field_name} contains unknown fields: {', '.join(unknown_fields)}"
        )


@dataclass(frozen=True)
class OutputVocabMapping:
    full_vocab_size: int
    local_to_full: Tuple[int, ...]
    config_digest: str
    source_path: str

    @property
    def size(self) -> int:
        return len(self.local_to_full)

    def contains(self, full_id: int) -> bool:
        pos = bisect.bisect_left(self.local_to_full, full_id)
        return pos < self.size and self.local_to_full[pos] == full_id

    def to_local(self, full_id: int) -> Optional[int]:
        pos = bisect.bisect_left(self.local_to_full, full_id)
        if pos < self.size and self.local_to_full[pos] == full_id:
            return pos
        return None

    def to_full(self, local_id: int) -> int:
        if local_id < 0 or local_id >= self.size:
            raise ValueError(
                f"local token id {local_id} is outside output vocab [0, {self.size})"
            )
        return self.local_to_full[local_id]


@dataclass(frozen=True)
class OutputVocabRankState:
    rank: int
    parse_ok: bool
    enabled: bool
    full_vocab_size: int
    output_vocab_size: int
    config_digest: str
    error: str = ""

    def semantic_key(self) -> Tuple[bool, int, int, str]:
        return (
            self.enabled,
            self.full_vocab_size,
            self.output_vocab_size,
            self.config_digest,
        )

    def summary(self) -> str:
        digest = self.config_digest or "<disabled>"
        return (
            f"rank {self.rank}: enabled={self.enabled}, V={self.full_vocab_size}, "
            f"K={self.output_vocab_size}, digest={digest}"
        )


def validate_output_vocab_rank_states(
    states: Sequence[OutputVocabRankState], expected_world_size: int
) -> None:
    if expected_world_size <= 0:
        raise ValueError(
            "output vocabulary rank validation requires a positive world size"
        )
    if len(states) != expected_world_size:
        raise ValueError(
            "output vocabulary rank validation expected "
            f"{expected_world_size} states, got {len(states)}"
        )
    if any(not isinstance(state, OutputVocabRankState) for state in states):
        raise ValueError(
            "output vocabulary rank validation received an invalid rank state"
        )

    ordered_states = sorted(states, key=lambda state: state.rank)
    actual_ranks = [state.rank for state in ordered_states]
    expected_ranks = list(range(expected_world_size))
    if actual_ranks != expected_ranks:
        raise ValueError(
            "output vocabulary rank validation expected ranks "
            f"{expected_ranks}, got {actual_ranks}"
        )

    failed_states = [state for state in ordered_states if not state.parse_ok]
    if failed_states:
        details = "; ".join(
            f"rank {state.rank}: {state.error or 'unknown configuration error'}"
            for state in failed_states
        )
        raise ValueError(f"output vocabulary configuration failed: {details}")

    expected_key = ordered_states[0].semantic_key()
    if any(state.semantic_key() != expected_key for state in ordered_states[1:]):
        details = "; ".join(state.summary() for state in ordered_states)
        raise ValueError(
            "output vocabulary configuration differs across ranks: " + details
        )


def _build_digest(
    version: int,
    full_vocab_size: int,
    local_to_full: Tuple[int, ...],
) -> str:
    canonical = json.dumps(
        {
            "version": version,
            "model_vocab_size": full_vocab_size,
            "local_to_full": local_to_full,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def load_output_vocab_mapping(
    config_path: str,
    full_vocab_size: int,
    input_vocab_size: Optional[int] = None,
) -> OutputVocabMapping:
    if not config_path:
        raise ValueError("output vocab config path is empty")
    if not os.path.isfile(config_path):
        raise ValueError(f"output vocab config does not exist: {config_path}")
    if full_vocab_size <= 0:
        raise ValueError(f"full_vocab_size must be positive, got {full_vocab_size}")

    with open(config_path, "r", encoding="utf-8") as reader:
        raw_config = json.load(reader)

    if not isinstance(raw_config, dict):
        raise ValueError("output vocab config root must be a JSON object")
    _reject_unknown_fields(
        raw_config,
        ("version", "model_vocab_size", "output_vocab"),
        "output vocab config root",
    )

    version = _require_int(raw_config.get("version"), "version")
    if version != 1:
        raise ValueError(f"unsupported output vocab config version: {version}")

    configured_vocab_size = _require_int(
        raw_config.get("model_vocab_size"), "model_vocab_size"
    )
    if configured_vocab_size != full_vocab_size:
        raise ValueError(
            "output vocab config model_vocab_size "
            f"{configured_vocab_size} does not match model vocab_size {full_vocab_size}"
        )

    output_vocab = raw_config.get("output_vocab")
    if not isinstance(output_vocab, dict):
        raise ValueError("output_vocab must be a JSON object")
    _reject_unknown_fields(output_vocab, ("ranges", "token_ids"), "output_vocab")

    ranges = output_vocab.get("ranges", [])
    token_ids = output_vocab.get("token_ids", [])
    if not isinstance(ranges, list):
        raise ValueError("output_vocab.ranges must be a list")
    if not isinstance(token_ids, list):
        raise ValueError("output_vocab.token_ids must be a list")

    expanded_ids = []
    for index, item in enumerate(ranges):
        field_prefix = f"output_vocab.ranges[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{field_prefix} must be a JSON object")
        _reject_unknown_fields(item, ("start_id", "end_id"), field_prefix)
        start_id = _require_int(item.get("start_id"), f"{field_prefix}.start_id")
        end_id = _require_int(item.get("end_id"), f"{field_prefix}.end_id")
        if not 0 <= start_id < end_id <= full_vocab_size:
            raise ValueError(
                f"{field_prefix} must satisfy 0 <= start_id < end_id <= "
                f"{full_vocab_size}, got [{start_id}, {end_id})"
            )
        expanded_ids.extend(range(start_id, end_id))

    for index, token_id_value in enumerate(token_ids):
        token_id = _require_int(token_id_value, f"output_vocab.token_ids[{index}]")
        if token_id < 0 or token_id >= full_vocab_size:
            raise ValueError(
                f"output_vocab.token_ids[{index}]={token_id} is outside "
                f"[0, {full_vocab_size})"
            )
        expanded_ids.append(token_id)

    if not expanded_ids:
        raise ValueError("output vocab config does not contain any token")

    local_to_full = tuple(sorted(set(expanded_ids)))
    if len(local_to_full) >= full_vocab_size:
        raise ValueError(
            "output vocab config must keep a proper subset of the model vocabulary, "
            f"got K={len(local_to_full)}, V={full_vocab_size}"
        )
    effective_input_vocab_size = (
        input_vocab_size
        if input_vocab_size is not None and input_vocab_size > 0
        else full_vocab_size
    )
    uncovered = next(
        (
            token_id
            for token_id in local_to_full
            if token_id >= effective_input_vocab_size
        ),
        None,
    )
    if uncovered is not None:
        raise ValueError(
            f"output token id {uncovered} is not covered by input embedding size "
            f"{effective_input_vocab_size}"
        )

    duplicate_count = len(expanded_ids) - len(local_to_full)
    digest = _build_digest(version, full_vocab_size, local_to_full)
    logging.info(
        "loaded output vocab config: path=%s, raw_tokens=%d, "
        "output_vocab_size=%d, duplicates=%d, digest=%s",
        config_path,
        len(expanded_ids),
        len(local_to_full),
        duplicate_count,
        digest,
    )
    return OutputVocabMapping(
        full_vocab_size=full_vocab_size,
        local_to_full=local_to_full,
        config_digest=digest,
        source_path=os.path.abspath(config_path),
    )


@dataclass
class OutputVocabConfig:
    enabled: bool = False
    _resolved: bool = field(default=False, init=False, repr=False)
    _full_vocab_size: int = field(default=0, init=False, repr=False)
    _input_vocab_size: int = field(default=0, init=False, repr=False)
    _resolved_config_path: str = field(default="", init=False, repr=False)
    _config_digest: str = field(default="", init=False, repr=False)
    _mapping: Optional[OutputVocabMapping] = field(default=None, init=False, repr=False)

    def resolve(
        self,
        checkpoint_path: str,
        full_vocab_size: int,
        input_vocab_size: Optional[int] = None,
    ) -> Optional[OutputVocabMapping]:
        if not self.enabled:
            return None
        if not isinstance(checkpoint_path, str) or not checkpoint_path:
            raise ValueError("checkpoint_path must be set for output vocabulary pruning")
        effective_input_vocab_size = (
            input_vocab_size
            if input_vocab_size is not None and input_vocab_size > 0
            else full_vocab_size
        )
        resolved_config_path = os.path.abspath(
            os.path.join(checkpoint_path, OUTPUT_VOCAB_FILENAME)
        )
        if self._resolved:
            if (
                self._full_vocab_size != full_vocab_size
                or self._input_vocab_size != effective_input_vocab_size
                or self._resolved_config_path != resolved_config_path
            ):
                raise ValueError(
                    "output vocab config was resolved for "
                    f"({self._full_vocab_size}, {self._input_vocab_size}, "
                    f"{self._resolved_config_path!r}), but the current model uses "
                    f"({full_vocab_size}, {effective_input_vocab_size}, "
                    f"{resolved_config_path!r})"
                )
            return self._mapping

        mapping = load_output_vocab_mapping(
            resolved_config_path,
            full_vocab_size=full_vocab_size,
            input_vocab_size=input_vocab_size,
        )
        self._mapping = mapping
        self._full_vocab_size = full_vocab_size
        self._input_vocab_size = effective_input_vocab_size
        self._resolved_config_path = resolved_config_path
        self._config_digest = mapping.config_digest
        self._resolved = True
        return self._mapping

    @property
    def config_digest(self) -> str:
        return self._config_digest

    def to_string(self) -> str:
        if not self.enabled:
            return "enable_output_vocab_pruning: false"
        if not self._resolved:
            return (
                "enable_output_vocab_pruning: true\n"
                f"output_vocab_filename: {OUTPUT_VOCAB_FILENAME}\n"
                "resolved: false"
            )
        return (
            "enable_output_vocab_pruning: true\n"
            f"output_vocab_config_path: {self._resolved_config_path}\n"
            "resolved: true\n"
            "output_vocab_pruning: enabled\n"
            f"output_vocab_size: {self._mapping.size}\n"
            f"config_digest: {self._mapping.config_digest}"
        )
