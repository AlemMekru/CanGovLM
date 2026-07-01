"""Corpus reading utilities for tokenizer training.

Phase 1 only needs reliable text loading. The tokenizer will learn from files
under ``corpus/`` later, so this module keeps file discovery deterministic and
limited to text-like inputs.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

DEFAULT_TEXT_EXTENSIONS = frozenset({".txt", ".md", ".text"})


def iter_text_files(
    corpus_dir: str | Path,
    *,
    extensions: frozenset[str] = DEFAULT_TEXT_EXTENSIONS,
) -> Iterator[Path]:
    """Yield text files under ``corpus_dir`` in deterministic order.

    Args:
        corpus_dir: Root directory containing corpus files.
        extensions: File extensions to include. Extensions should include the
            leading dot, for example ``.txt``.

    Yields:
        Paths to matching files, sorted by their full path.

    Raises:
        FileNotFoundError: If ``corpus_dir`` does not exist.
        NotADirectoryError: If ``corpus_dir`` is not a directory.
    """

    root = Path(corpus_dir)
    if not root.exists():
        raise FileNotFoundError(f"Corpus directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Corpus path is not a directory: {root}")

    normalized_extensions = frozenset(ext.lower() for ext in extensions)
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in normalized_extensions:
            yield path


def read_text_file(path: str | Path, *, encoding: str = "utf-8") -> str:
    """Read one UTF-8 text file.

    Args:
        path: File to read.
        encoding: Text encoding. CanGovLM uses UTF-8 for Phase 1.

    Returns:
        The file contents as a Python string.
    """

    return Path(path).read_text(encoding=encoding)


def iter_corpus_texts(corpus_dir: str | Path) -> Iterator[str]:
    """Yield text contents from all discovered corpus files."""

    for path in iter_text_files(corpus_dir):
        yield read_text_file(path)
