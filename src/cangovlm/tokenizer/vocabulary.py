"""Initial byte vocabulary for the CanGovLM tokenizer.

Phase 1 defines only the primitive vocabulary:

- byte IDs 0..255 map directly to UTF-8 byte values
- special tokens are appended after byte IDs

No BPE merge tokens are created in this phase.
"""

from dataclasses import dataclass

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS = (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN)
BYTE_VOCAB_SIZE = 256


@dataclass(frozen=True)
class ByteVocabulary:
    """Vocabulary containing byte tokens and Phase 1 special tokens."""

    token_to_id: dict[str, int]
    id_to_token: dict[int, str]

    @property
    def pad_id(self) -> int:
        """Token ID for ``<pad>``."""

        return self.token_to_id[PAD_TOKEN]

    @property
    def bos_id(self) -> int:
        """Token ID for ``<bos>``."""

        return self.token_to_id[BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        """Token ID for ``<eos>``."""

        return self.token_to_id[EOS_TOKEN]

    @property
    def size(self) -> int:
        """Total vocabulary size."""

        return len(self.token_to_id)


def byte_token(byte_value: int) -> str:
    """Return the stable token string for a byte value.

    Args:
        byte_value: Integer byte value in the inclusive range 0..255.

    Returns:
        A printable token representation such as ``<byte:65>``.

    Raises:
        ValueError: If ``byte_value`` is outside the byte range.
    """

    if not 0 <= byte_value < BYTE_VOCAB_SIZE:
        raise ValueError(f"Byte value must be in range 0..255, got {byte_value}")
    return f"<byte:{byte_value}>"


def build_initial_byte_vocabulary() -> ByteVocabulary:
    """Build the Phase 1 tokenizer vocabulary.

    Byte token IDs intentionally match their byte values. Special tokens are
    appended afterward so they cannot collide with real UTF-8 bytes.
    """

    token_to_id = {byte_token(byte_value): byte_value for byte_value in range(BYTE_VOCAB_SIZE)}

    for offset, token in enumerate(SPECIAL_TOKENS):
        token_to_id[token] = BYTE_VOCAB_SIZE + offset

    id_to_token = {token_id: token for token, token_id in token_to_id.items()}
    return ByteVocabulary(token_to_id=token_to_id, id_to_token=id_to_token)

