from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from cangovlm.tokenizer import iter_corpus_texts, iter_text_files, read_text_file


class CorpusTests(TestCase):
    def test_iter_text_files_recurses_and_sorts_supported_files(self) -> None:
        with TemporaryDirectory() as temp_dir:
            corpus_dir = Path(temp_dir) / "corpus"
            en_dir = corpus_dir / "en"
            fr_dir = corpus_dir / "fr"
            en_dir.mkdir(parents=True)
            fr_dir.mkdir(parents=True)
            (fr_dir / "b.txt").write_text("Bonjour", encoding="utf-8")
            (en_dir / "a.md").write_text("Hello", encoding="utf-8")
            (en_dir / "ignore.json").write_text("{}", encoding="utf-8")

            files = list(iter_text_files(corpus_dir))

            self.assertEqual(files, [en_dir / "a.md", fr_dir / "b.txt"])

    def test_read_text_file_uses_utf8(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.txt"
            path.write_text("Santé Canada", encoding="utf-8")

            self.assertEqual(read_text_file(path), "Santé Canada")

    def test_iter_corpus_texts_reads_discovered_files(self) -> None:
        with TemporaryDirectory() as temp_dir:
            corpus_dir = Path(temp_dir) / "corpus"
            corpus_dir.mkdir()
            (corpus_dir / "a.txt").write_text("one", encoding="utf-8")
            (corpus_dir / "b.txt").write_text("two", encoding="utf-8")

            self.assertEqual(list(iter_corpus_texts(corpus_dir)), ["one", "two"])

    def test_iter_text_files_rejects_missing_directory(self) -> None:
        with TemporaryDirectory() as temp_dir:
            with self.assertRaises(FileNotFoundError):
                list(iter_text_files(Path(temp_dir) / "missing"))
