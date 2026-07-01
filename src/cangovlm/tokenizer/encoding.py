"""Tokenizer encoding with learned byte-level BPE merge rules.

Encoding replays previously learned merge rules over a UTF-8 byte sequence. It
does not learn new merges and it does not decode token IDs back to text.
"""

from __future__ import annotations

from collections.abc import Sequence

from cangovlm.tokenizer.bpe import BpeMergeRule, TokenSequence, replace_pair
from cangovlm.tokenizer.bytes import text_to_utf8_bytes
from cangovlm.tokenizer.vocabulary import build_initial_byte_vocabulary


def ordered_merge_rules(merge_rules: Sequence[BpeMergeRule]) -> list[BpeMergeRule]:
    """Return merge rules sorted by rank.

    Encoding must apply merges in the same order they were learned. This helper
    makes that requirement explicit and keeps callers from depending on input
    list ordering by accident.
    """

    return sorted(merge_rules, key=lambda rule: rule.rank)


def apply_bpe_merge_rules(
    token_ids: Sequence[int],
    merge_rules: Sequence[BpeMergeRule],
) -> TokenSequence:
    """Apply learned BPE merge rules to an existing token sequence."""

    merged = list(token_ids)
    for rule in ordered_merge_rules(merge_rules):
        merged = replace_pair(merged, rule.pair, rule.new_token_id)
    return merged


def encode_text(
    text: str,
    merge_rules: Sequence[BpeMergeRule],
    *,
    add_bos: bool = False,
    add_eos: bool = False,
) -> TokenSequence:
    """Encode text into token IDs using learned byte-level BPE rules.

    Args:
        text: Input Unicode text.
        merge_rules: Learned BPE merge rules.
        add_bos: If true, prepend the ``<bos>`` token ID.
        add_eos: If true, append the ``<eos>`` token ID.

    Returns:
        Encoded token IDs.
    """

    vocabulary = build_initial_byte_vocabulary()
    token_ids = apply_bpe_merge_rules(text_to_utf8_bytes(text), merge_rules)

    if add_bos:
        token_ids.insert(0, vocabulary.bos_id)
    if add_eos:
        token_ids.append(vocabulary.eos_id)

    return token_ids

