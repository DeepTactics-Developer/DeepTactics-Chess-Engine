import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chess_engine.chess_core.enums import PieceType
from chess_engine.chess_core.move import Move
from chess_engine.chess_core.square import Square


class MoveTest(unittest.TestCase):
    def test_move_converts_to_and_from_uci(self):
        move = Move.from_uci("e2e4")

        self.assertEqual(move.from_square, Square.from_algebraic("e2"))
        self.assertEqual(move.to_square, Square.from_algebraic("e4"))
        self.assertIsNone(move.promotion)
        self.assertEqual(move.to_uci(), "e2e4")

    def test_promotion_move_converts_to_and_from_uci(self):
        move = Move.from_uci("a7a8q")

        self.assertEqual(move.promotion, PieceType.QUEEN)
        self.assertEqual(move.to_uci(), "a7a8q")

    def test_invalid_promotion_piece_is_rejected(self):
        with self.assertRaises(ValueError):
            Move(
                from_square=Square.from_algebraic("a7"),
                to_square=Square.from_algebraic("a8"),
                promotion=PieceType.KING,
            )


if __name__ == "__main__":
    unittest.main()

