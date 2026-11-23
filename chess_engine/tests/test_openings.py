import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chess_engine.openings import (
    convert_opening_source_to_tsv,
    convert_opening_sources_to_tsv,
    detect_opening,
    generate_opening_key,
    load_opening_book,
    next_opening_moves,
)


class OpeningTest(unittest.TestCase):
    def test_detect_opening_uses_longest_builtin_match(self):
        self.assertEqual(
            detect_opening(["e4", "e5", "Nf3", "Nc6", "Bb5"]),
            "C60 Ruy Lopez",
        )

    def test_detect_opening_returns_unknown_without_match(self):
        self.assertEqual(detect_opening(["a4"]), "Opening: unknown")

    def test_detect_opening_empty_history_does_not_build_position_index(self):
        self.assertEqual(detect_opening([]), "Opening: unknown")

    def test_load_opening_book_reads_tsv_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "openings.tsv"
            path.write_text("C99\tCustom Line\te4 e5 Qh5\n", encoding="utf-8")

            self.assertEqual(load_opening_book(path), {"e4 e5 Qh5": "C99 Custom Line"})
            self.assertEqual(detect_opening(["e4", "e5", "Qh5"], path), "C99 Custom Line")

    def test_next_opening_moves_returns_matching_continuations(self):
        self.assertIn("e5", next_opening_moves(["e4"]))

    def test_generate_opening_key_sorts_white_and_black_moves(self):
        self.assertEqual(generate_opening_key("e4 e5 Nc3"), "Nc3,e4|e5")
        self.assertEqual(generate_opening_key("1. Nc3 e5 2. e4"), "Nc3,e4|e5")

    def test_detect_opening_uses_transposed_position(self):
        self.assertEqual(
            detect_opening(["Nc3", "e5"]),
            "A00 Van Geet Opening: Reversed Nimzowitsch",
        )
        self.assertEqual(
            detect_opening(["Nc3", "e5", "e4"]),
            "C25 Vienna Game",
        )
        self.assertEqual(
            detect_opening(["Nc3", "e5", "e4", "Nf6"]),
            "C25 Vienna Game",
        )

    def test_convert_opening_source_to_tsv_accepts_eco_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.tsv"
            output = Path(directory) / "openings.tsv"
            source.write_text("C60\tRuy Lopez\t1. e4 e5 2. Nf3 Nc6 3. Bb5\n", encoding="utf-8")

            count = convert_opening_source_to_tsv(source, output)

            self.assertEqual(count, 1)
            self.assertEqual(output.read_text(encoding="utf-8"), "C60\tRuy Lopez\te4 e5 Nf3 Nc6 Bb5\n")

    def test_convert_multiple_opening_sources_to_one_tsv(self):
        with tempfile.TemporaryDirectory() as directory:
            source_a = Path(directory) / "a.tsv"
            source_b = Path(directory) / "b.tsv"
            output = Path(directory) / "openings.tsv"
            source_a.write_text("A00\tTest A\t1. a3\n", encoding="utf-8")
            source_b.write_text("B00\tTest B\t1. e4\n", encoding="utf-8")

            count = convert_opening_sources_to_tsv([source_a, source_b], output)

            self.assertEqual(count, 2)
            self.assertIn("A00\tTest A\ta3", output.read_text(encoding="utf-8"))
            self.assertIn("B00\tTest B\te4", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
