"""Tokenizer primitives for CanGovLM."""

from cangovlm.tokenizer.bpe import (
    BpeMergeRule,
    count_adjacent_pairs,
    learn_bpe_merge_rules,
    replace_pair,
    replace_pair_in_sequences,
    select_best_pair,
    texts_to_byte_sequences,
)
from cangovlm.tokenizer.bytes import text_to_utf8_bytes, utf8_bytes_to_text
from cangovlm.tokenizer.corpus import iter_corpus_texts, iter_text_files, read_text_file
from cangovlm.tokenizer.encoding import apply_bpe_merge_rules, encode_text, ordered_merge_rules
from cangovlm.tokenizer.vocabulary import (
    BOS_TOKEN,
    BYTE_VOCAB_SIZE,
    EOS_TOKEN,
    PAD_TOKEN,
    SPECIAL_TOKENS,
    ByteVocabulary,
    build_initial_byte_vocabulary,
    byte_token,
)

__all__ = [
    "BOS_TOKEN",
    "BYTE_VOCAB_SIZE",
    "EOS_TOKEN",
    "PAD_TOKEN",
    "SPECIAL_TOKENS",
    "ByteVocabulary",
    "BpeMergeRule",
    "apply_bpe_merge_rules",
    "build_initial_byte_vocabulary",
    "byte_token",
    "count_adjacent_pairs",
    "encode_text",
    "iter_corpus_texts",
    "iter_text_files",
    "learn_bpe_merge_rules",
    "ordered_merge_rules",
    "read_text_file",
    "replace_pair",
    "replace_pair_in_sequences",
    "select_best_pair",
    "text_to_utf8_bytes",
    "texts_to_byte_sequences",
    "utf8_bytes_to_text",
]
