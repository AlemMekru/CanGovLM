"""Tokenizer decoding for byte-level BPE token IDs.

Decoding expands learned BPE token IDs back to primitive byte IDs, removes
special tokens by default, and decodes the resulting UTF-8 byte stream.
"""

from __future__ import annotations

from collections.abc import Sequence

from cangovlm.tokenizer.bpe import BpeMergeRule, TokenPair, TokenSequence
from cangovlm.tokenizer.bytes import utf8_bytes_to_text
from cangovlm.tokenizer.vocabulary import BYTE_VOCAB_SIZE, build_initial_byte_vocabulary


def build_merge_expansion_map(merge_rules: Sequence[BpeMergeRule]) -> dict[int, TokenPair]:
    """Build a mapping from learned token ID to the pair it represents."""

    return {rule.new_token_id: rule.pair for rule in merge_rules}


def expand_token_id(token_id: int, expansion_map: dict[int, TokenPair]) -> TokenSequence:
    """Expand one token ID into primitive byte IDs.

    Args:
        token_id: Byte token ID or learned BPE token ID.
        expansion_map: Mapping from learned token IDs to their child token pair.

    Returns:
        Primitive byte token IDs.

    Raises:
        ValueError: If ``token_id`` is not a byte token and is not present in
            the learned merge expansion map.
    """

    if 0 <= token_id < BYTE_VOCAB_SIZE:
        return [token_id]

    if token_id not in expansion_map:
        raise ValueError(f"Cannot decode unknown token ID: {token_id}")

    left, right = expansion_map[token_id]
    return expand_token_id(left, expansion_map) + expand_token_id(right, expansion_map)


def token_ids_to_byte_ids(
    token_ids: Sequence[int],
    merge_rules: Sequence[BpeMergeRule],
    *,
    skip_special_tokens: bool = True,
) -> TokenSequence:
    """Convert token IDs into primitive byte IDs."""

    vocabulary = build_initial_byte_vocabulary()
    special_token_ids = {vocabulary.pad_id, vocabulary.bos_id, vocabulary.eos_id}
    expansion_map = build_merge_expansion_map(merge_rules)
    byte_ids: TokenSequence = []

    for token_id in token_ids:
        if token_id in special_token_ids:
            if skip_special_tokens:
                continue
            raise ValueError(f"Cannot decode special token ID as text: {token_id}")
        byte_ids.extend(expand_token_id(token_id, expansion_map))

    return byte_ids


def decode_token_ids(
    token_ids: Sequence[int],
    merge_rules: Sequence[BpeMergeRule],
    *,
    skip_special_tokens: bool = True,
) -> str:
    """Decode token IDs back into UTF-8 text.

    Args:
        token_ids: Encoded token IDs.
        merge_rules: Learned BPE merge rules used to create the token IDs.
        skip_special_tokens: If true, ignore ``<pad>``, ``<bos>``, and
            ``<eos>``. If false, raise when a special token is encountered
            because special tokens do not correspond to UTF-8 bytes.

    Returns:
        Decoded text.
    """

    byte_ids = token_ids_to_byte_ids(
        token_ids,
        merge_rules,
        skip_special_tokens=skip_special_tokens,
    )
    return utf8_bytes_to_text(byte_ids)

