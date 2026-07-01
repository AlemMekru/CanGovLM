from unittest import TestCase

from cangovlm.tokenizer import (
    build_initial_byte_vocabulary,
    count_adjacent_pairs,
    learn_bpe_merge_rules,
    replace_pair,
    replace_pair_in_sequences,
    select_best_pair,
    text_to_utf8_bytes,
    texts_to_byte_sequences,
)


class BpeMergeTrainingTests(TestCase):
    def test_texts_to_byte_sequences_converts_synthetic_corpus(self) -> None:
        self.assertEqual(texts_to_byte_sequences(["banana", "Canada"]), [
            text_to_utf8_bytes("banana"),
            text_to_utf8_bytes("Canada"),
        ])

    def test_count_adjacent_pairs_counts_across_sequences(self) -> None:
        byte_a = ord("a")
        byte_b = ord("b")
        byte_n = ord("n")

        counts = count_adjacent_pairs([
            [byte_b, byte_a, byte_n, byte_a],
            [byte_a, byte_n],
        ])

        self.assertEqual(counts[(byte_a, byte_n)], 2)
        self.assertEqual(counts[(byte_b, byte_a)], 1)
        self.assertEqual(counts[(byte_n, byte_a)], 1)

    def test_select_best_pair_uses_deterministic_tie_break(self) -> None:
        selected = select_best_pair({
            (4, 5): 2,
            (1, 9): 2,
            (1, 8): 1,
        })

        self.assertEqual(selected, ((1, 9), 2))

    def test_replace_pair_uses_non_overlapping_left_to_right_replacement(self) -> None:
        # In "aaaa", replacing "aa" should produce two merged tokens, not
        # three overlapping replacements.
        byte_a = ord("a")

        self.assertEqual(
            replace_pair([byte_a, byte_a, byte_a, byte_a], (byte_a, byte_a), 259),
            [259, 259],
        )

    def test_replace_pair_in_sequences_applies_to_each_sequence(self) -> None:
        self.assertEqual(
            replace_pair_in_sequences([[1, 2, 1, 2], [3, 1, 2]], (1, 2), 259),
            [[259, 259], [3, 259]],
        )

    def test_learn_bpe_merge_rules_stops_at_target_vocab_size(self) -> None:
        initial_size = build_initial_byte_vocabulary().size

        rules = learn_bpe_merge_rules(["banana banana"], target_vocab_size=initial_size + 3)

        self.assertEqual(len(rules), 3)
        self.assertEqual([rule.rank for rule in rules], [0, 1, 2])
        self.assertEqual([rule.new_token_id for rule in rules], [initial_size, initial_size + 1, initial_size + 2])

    def test_learn_bpe_merge_rules_first_banana_merge_is_most_frequent_pair(self) -> None:
        initial_size = build_initial_byte_vocabulary().size
        byte_a = ord("a")
        byte_n = ord("n")

        rules = learn_bpe_merge_rules(["banana banana"], target_vocab_size=initial_size + 1)

        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0].pair, (byte_a, byte_n))
        self.assertEqual(rules[0].count, 4)
        self.assertEqual(rules[0].new_token_id, initial_size)

    def test_learn_bpe_merge_rules_returns_no_rules_when_no_pairs_exist(self) -> None:
        initial_size = build_initial_byte_vocabulary().size

        rules = learn_bpe_merge_rules(["a", ""], target_vocab_size=initial_size + 10)

        self.assertEqual(rules, [])

    def test_learn_bpe_merge_rules_rejects_target_smaller_than_initial_vocab(self) -> None:
        initial_size = build_initial_byte_vocabulary().size

        with self.assertRaises(ValueError):
            learn_bpe_merge_rules(["banana"], target_vocab_size=initial_size - 1)

