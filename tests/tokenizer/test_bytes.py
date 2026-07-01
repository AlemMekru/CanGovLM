from unittest import TestCase

from cangovlm.tokenizer import text_to_utf8_bytes, utf8_bytes_to_text


class ByteConversionTests(TestCase):
    def test_text_to_utf8_bytes_returns_byte_values(self) -> None:
        self.assertEqual(text_to_utf8_bytes("Canada"), [67, 97, 110, 97, 100, 97])


    def test_utf8_bytes_round_trip_preserves_french_accents(self) -> None:
        text = "Gouvernement du Québec"

        byte_ids = text_to_utf8_bytes(text)

        self.assertTrue(all(0 <= byte_id <= 255 for byte_id in byte_ids))
        self.assertEqual(utf8_bytes_to_text(byte_ids), text)


    def test_utf8_bytes_preserve_whitespace_and_newlines(self) -> None:
        text = "Canada\n  Revenue Agency"

        self.assertEqual(utf8_bytes_to_text(text_to_utf8_bytes(text)), text)
