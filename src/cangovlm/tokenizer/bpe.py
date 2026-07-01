"""Byte Pair Encoding merge-rule training.

This module implements only BPE merge learning. It does not encode new text
with learned merges and it does not decode merged tokens back to text yet.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from cangovlm.tokenizer.bytes import text_to_utf8_bytes
from cangovlm.tokenizer.vocabulary import build_initial_byte_vocabulary

TokenSequence = list[int]
TokenPair = tuple[int, int]


@dataclass(frozen=True)
class BpeMergeRule:
    """One learned BPE merge rule.

    Attributes:
        rank: Zero-based merge order. Lower ranks are applied earlier later.
        pair: Adjacent token pair that was merged.
        new_token_id: Token ID assigned to the merged pair.
        count: Pair frequency at the time the rule was learned.
    """

    rank: int
    pair: TokenPair
    new_token_id: int
    count: int


def texts_to_byte_sequences(texts: Iterable[str]) -> list[TokenSequence]:
    """Convert raw texts into UTF-8 byte-token sequences."""

    return [text_to_utf8_bytes(text) for text in texts]


def count_adjacent_pairs(sequences: Sequence[Sequence[int]]) -> Counter[TokenPair]:
    """Count adjacent token pairs across token sequences.

    Empty and single-token sequences contribute no pairs.
    """

    pair_counts: Counter[TokenPair] = Counter()
    for sequence in sequences:
        pair_counts.update(zip(sequence, sequence[1:]))
    return pair_counts


def select_best_pair(pair_counts: Counter[TokenPair]) -> tuple[TokenPair, int] | None:
    """Select the best pair to merge.

    The most frequent pair wins. Ties are resolved by smaller token IDs so
    training remains deterministic across Python versions and platforms.
    """

    if not pair_counts:
        return None

    pair, count = min(pair_counts.items(), key=lambda item: (-item[1], item[0]))
    return pair, count


def replace_pair(sequence: Sequence[int], pair: TokenPair, new_token_id: int) -> TokenSequence:
    """Replace non-overlapping occurrences of ``pair`` in one sequence."""

    replaced: TokenSequence = []
    index = 0
    while index < len(sequence):
        if index < len(sequence) - 1 and (sequence[index], sequence[index + 1]) == pair:
            replaced.append(new_token_id)
            index += 2
        else:
            replaced.append(sequence[index])
            index += 1
    return replaced


def replace_pair_in_sequences(
    sequences: Sequence[Sequence[int]],
    pair: TokenPair,
    new_token_id: int,
) -> list[TokenSequence]:
    """Replace a pair across all token sequences."""

    return [replace_pair(sequence, pair, new_token_id) for sequence in sequences]


def learn_bpe_merge_rules(
    texts: Iterable[str],
    *,
    target_vocab_size: int,
) -> list[BpeMergeRule]:
    """Learn BPE merge rules from raw training texts.

    Args:
        texts: Training corpus as raw strings.
        target_vocab_size: Desired vocabulary size including the initial byte
            vocabulary and special tokens.

    Returns:
        Learned merge rules in rank order.

    Raises:
        ValueError: If ``target_vocab_size`` is smaller than the initial
            vocabulary size.
    """

    initial_vocab_size = build_initial_byte_vocabulary().size
    if target_vocab_size < initial_vocab_size:
        raise ValueError(
            "target_vocab_size must be at least the initial vocabulary size "
            f"({initial_vocab_size}), got {target_vocab_size}"
        )

    sequences = texts_to_byte_sequences(texts)
    merge_rules: list[BpeMergeRule] = []
    next_token_id = initial_vocab_size

    while next_token_id < target_vocab_size:
        pair_counts = count_adjacent_pairs(sequences)
        selected = select_best_pair(pair_counts)
        if selected is None:
            break

        pair, count = selected
        rule = BpeMergeRule(
            rank=len(merge_rules),
            pair=pair,
            new_token_id=next_token_id,
            count=count,
        )
        merge_rules.append(rule)
        sequences = replace_pair_in_sequences(sequences, pair, next_token_id)
        next_token_id += 1

    return merge_rules

