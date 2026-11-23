import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chess_engine.chess_core.square import Square


class SquareTest(unittest.TestCase):
    def test_square_converts_to_and_from_algebraic_notation(self):
        square = Square.from_algebraic("e4")

        self.assertEqual(square.file, 4)
        self.assertEqual(square.rank, 3)
        self.assertEqual(square.to_algebraic(), "e4")

    def test_square_rejects_invalid_coordinates(self):
        with self.assertRaises(ValueError):
            Square(8, 0)

        with self.assertRaises(ValueError):
            Square.from_algebraic("z9")

    def test_offset_returns_none_when_square_leaves_board(self):
        self.assertEqual(Square.from_algebraic("e4").offset(1, 1), Square.from_algebraic("f5"))
        self.assertIsNone(Square.from_algebraic("h8").offset(1, 0))


if __name__ == "__main__":
    unittest.main()

