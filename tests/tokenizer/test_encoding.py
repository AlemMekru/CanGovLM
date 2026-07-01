from unittest import TestCase

from cangovlm.tokenizer import (
    BpeMergeRule,
    apply_bpe_merge_rules,
    build_initial_byte_vocabulary,
    encode_text,
    learn_bpe_merge_rules,
    ordered_merge_rules,
    text_to_utf8_bytes,
)


class EncodingTests(TestCase):
    def test_ordered_merge_rules_sorts_by_rank(self) -> None:
        rules = [
            BpeMergeRule(rank=2, pair=(3, 4), new_token_id=261, count=1),
            BpeMergeRule(rank=0, pair=(1, 2), new_token_id=259, count=3),
            BpeMergeRule(rank=1, pair=(2, 3), new_token_id=260, count=2),
        ]

        self.assertEqual([rule.rank for rule in ordered_merge_rules(rules)], [0, 1, 2])

    def test_apply_bpe_merge_rules_replays_rules_in_rank_order(self) -> None:
        byte_a = ord("a")
        byte_n = ord("n")
        rules = [
            BpeMergeRule(rank=1, pair=(259, byte_a), new_token_id=260, count=2),
            BpeMergeRule(rank=0, pair=(byte_a, byte_n), new_token_id=259, count=2),
        ]

        self.assertEqual(
            apply_bpe_merge_rules(text_to_utf8_bytes("anana"), rules),
            [259, 260],
        )

    def test_encode_text_without_merges_returns_utf8_byte_ids(self) -> None:
        text = "Québec"

        self.assertEqual(encode_text(text, []), text_to_utf8_bytes(text))

    def test_encode_text_uses_learned_rules_from_synthetic_corpus(self) -> None:
        initial_size = build_initial_byte_vocabulary().size
        rules = learn_bpe_merge_rules(["banana banana"], target_vocab_size=initial_size + 2)

        encoded = encode_text("banana", rules)

        self.assertEqual(encoded, [initial_size + 1, initial_size, ord("a")])

    def test_encode_text_can_add_bos_and_eos_tokens(self) -> None:
        vocabulary = build_initial_byte_vocabulary()

        encoded = encode_text("Hi", [], add_bos=True, add_eos=True)

        self.assertEqual(encoded, [vocabulary.bos_id, ord("H"), ord("i"), vocabulary.eos_id])

    def test_encode_text_can_add_only_one_boundary_token(self) -> None:
        vocabulary = build_initial_byte_vocabulary()

        self.assertEqual(encode_text("A", [], add_bos=True), [vocabulary.bos_id, ord("A")])
        self.assertEqual(encode_text("A", [], add_eos=True), [ord("A"), vocabulary.eos_id])
