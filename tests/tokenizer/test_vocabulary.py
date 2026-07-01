from unittest import TestCase

from cangovlm.tokenizer import (
    BOS_TOKEN,
    BYTE_VOCAB_SIZE,
    EOS_TOKEN,
    PAD_TOKEN,
    build_initial_byte_vocabulary,
    byte_token,
)


class VocabularyTests(TestCase):
    def test_initial_vocabulary_contains_all_byte_tokens(self) -> None:
        vocab = build_initial_byte_vocabulary()

        for byte_value in range(BYTE_VOCAB_SIZE):
            token = byte_token(byte_value)
            self.assertEqual(vocab.token_to_id[token], byte_value)
            self.assertEqual(vocab.id_to_token[byte_value], token)

    def test_initial_vocabulary_appends_special_tokens_after_bytes(self) -> None:
        vocab = build_initial_byte_vocabulary()

        self.assertEqual(vocab.token_to_id[PAD_TOKEN], 256)
        self.assertEqual(vocab.token_to_id[BOS_TOKEN], 257)
        self.assertEqual(vocab.token_to_id[EOS_TOKEN], 258)
        self.assertEqual(vocab.pad_id, 256)
        self.assertEqual(vocab.bos_id, 257)
        self.assertEqual(vocab.eos_id, 258)
        self.assertEqual(vocab.size, 259)

    def test_byte_token_rejects_values_outside_byte_range(self) -> None:
        with self.assertRaises(ValueError):
            byte_token(-1)

        with self.assertRaises(ValueError):
            byte_token(256)
