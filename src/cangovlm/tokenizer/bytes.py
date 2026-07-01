"""UTF-8 byte conversion for the CanGovLM tokenizer.

The initial tokenizer alphabet is raw UTF-8 bytes. This gives the tokenizer a
complete, reversible base representation before any BPE merges are introduced.
"""


def text_to_utf8_bytes(text: str) -> list[int]:
    """Convert text into UTF-8 byte IDs.

    Args:
        text: Input Unicode string.

    Returns:
        A list of integers in the inclusive range 0..255.
    """

    return list(text.encode("utf-8"))


def utf8_bytes_to_text(byte_ids: list[int]) -> str:
    """Convert UTF-8 byte IDs back into text.

    Args:
        byte_ids: Integer byte values in the inclusive range 0..255.

    Returns:
        The decoded Unicode string.
    """

    return bytes(byte_ids).decode("utf-8")

