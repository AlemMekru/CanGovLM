from unittest import TestCase

from cangovlm.tokenizer import (
    BpeMergeRule,
    build_initial_byte_vocabulary,
    build_merge_expansion_map,
    decode_token_ids,
    encode_text,
    expand_token_id,
    learn_bpe_merge_rules,
    text_to_utf8_bytes,
    token_ids_to_byte_ids,
)


class DecodingTests(TestCase):
    def test_build_merge_expansion_map_uses_new_token_ids(self) -> None:
        rule = BpeMergeRule(rank=0, pair=(ord("a"), ord("n")), new_token_id=259, count=4)

        self.assertEqual(build_merge_expansion_map([rule]), {259: (ord("a"), ord("n"))})

    def test_expand_token_id_returns_byte_token_as_itself(self) -> None:
        self.assertEqual(expand_token_id(ord("A"), {}), [ord("A")])

    def test_expand_token_id_recursively_expands_bpe_token(self) -> None:
        expansion_map = {
            259: (ord("a"), ord("n")),
            260: (259, ord("a")),
            261: (ord("b"), 260),
        }

        self.assertEqual(expand_token_id(261, expansion_map), text_to_utf8_bytes("bana"))

    def test_expand_token_id_rejects_unknown_non_byte_token(self) -> None:
        with self.assertRaises(ValueError):
            expand_token_id(999, {})

    def test_token_ids_to_byte_ids_reverses_learned_merges(self) -> None:
        initial_size = build_initial_byte_vocabulary().size
        rules = learn_bpe_merge_rules(["banana banana"], target_vocab_size=initial_size + 2)
        encoded = encode_text("banana", rules)

        self.assertEqual(token_ids_to_byte_ids(encoded, rules), text_to_utf8_bytes("banana"))

    def test_decode_token_ids_decodes_utf8_text_without_merges(self) -> None:
        text = "Québec"

        self.assertEqual(decode_token_ids(text_to_utf8_bytes(text), []), text)

    def test_decode_token_ids_reverses_bpe_merges(self) -> None:
        initial_size = build_initial_byte_vocabulary().size
        rules = learn_bpe_merge_rules(["banana banana"], target_vocab_size=initial_size + 4)

        self.assertEqual(decode_token_ids(encode_text("banana banana", rules), rules), "banana banana")

    def test_decode_token_ids_skips_special_tokens_by_default(self) -> None:
        vocabulary = build_initial_byte_vocabulary()

        decoded = decode_token_ids(
            [vocabulary.bos_id, ord("H"), ord("i"), vocabulary.eos_id, vocabulary.pad_id],
            [],
        )

        self.assertEqual(decoded, "Hi")

    def test_decode_token_ids_can_reject_special_tokens(self) -> None:
        vocabulary = build_initial_byte_vocabulary()

        with self.assertRaises(ValueError):
            decode_token_ids([vocabulary.bos_id, ord("H")], [], skip_special_tokens=False)

    def test_decode_encode_round_trip_preserves_text(self) -> None:
        initial_size = build_initial_byte_vocabulary().size
        rules = learn_bpe_merge_rules(["banana banana Québec"], target_vocab_size=initial_size + 8)
        text = "banana Québec"

        self.assertEqual(decode_token_ids(encode_text(text, rules), rules), text)

    def test_decode_encode_round_trip_with_boundary_tokens_preserves_text(self) -> None:
        initial_size = build_initial_byte_vocabulary().size
        rules = learn_bpe_merge_rules(["hello hello"], target_vocab_size=initial_size + 4)
        text = "hello"

        encoded = encode_text(text, rules, add_bos=True, add_eos=True)

        self.assertEqual(decode_token_ids(encoded, rules), text)

    def test_encode_decode_round_trip_for_canonical_encoded_tokens(self) -> None:
        initial_size = build_initial_byte_vocabulary().size
        rules = learn_bpe_merge_rules(["banana banana"], target_vocab_size=initial_size + 3)
        encoded = encode_text("banana", rules)

        self.assertEqual(encode_text(decode_token_ids(encoded, rules), rules), encoded)

