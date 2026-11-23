import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chess_engine.chess_core.board import Board
from chess_engine.chess_core.enums import Color, PieceType
from chess_engine.chess_core.move import Move
from chess_engine.chess_core.piece import Piece
from chess_engine.chess_core.square import Square


class BoardTest(unittest.TestCase):
    def test_starting_position_has_expected_pieces(self):
        board = Board.starting_position()

        self.assertEqual(board.side_to_move, Color.WHITE)
        self.assertEqual(board.get_piece(Square.from_algebraic("e1")), Piece(Color.WHITE, PieceType.KING))
        self.assertEqual(board.get_piece(Square.from_algebraic("d8")), Piece(Color.BLACK, PieceType.QUEEN))
        self.assertEqual(board.get_piece(Square.from_algebraic("a2")), Piece(Color.WHITE, PieceType.PAWN))
        self.assertEqual(board.get_piece(Square.from_algebraic("h7")), Piece(Color.BLACK, PieceType.PAWN))
        self.assertIsNone(board.get_piece(Square.from_algebraic("e4")))

    def test_starting_position_contains_32_pieces(self):
        board = Board.starting_position()

        self.assertEqual(len(list(board.pieces())), 32)
        self.assertEqual(len(list(board.pieces(Color.WHITE))), 16)
        self.assertEqual(len(list(board.pieces(Color.BLACK))), 16)

    def test_apply_move_unchecked_moves_piece_and_advances_turn(self):
        board = Board.starting_position()
        move = Move.from_uci("e2e4")

        captured = board.apply_move_unchecked(move)

        self.assertIsNone(captured)
        self.assertIsNone(board.get_piece(Square.from_algebraic("e2")))
        self.assertEqual(board.get_piece(Square.from_algebraic("e4")), Piece(Color.WHITE, PieceType.PAWN))
        self.assertEqual(board.side_to_move, Color.BLACK)
        self.assertEqual(board.fullmove_number, 1)

    def test_black_move_advances_fullmove_number(self):
        board = Board.starting_position()
        board.apply_move_unchecked(Move.from_uci("e2e4"))
        board.apply_move_unchecked(Move.from_uci("e7e5"))

        self.assertEqual(board.side_to_move, Color.WHITE)
        self.assertEqual(board.fullmove_number, 2)

    def test_board_copy_is_independent(self):
        board = Board.starting_position()
        copied = board.copy()

        copied.apply_move_unchecked(Move.from_uci("g1f3"))

        self.assertEqual(board.get_piece(Square.from_algebraic("g1")), Piece(Color.WHITE, PieceType.KNIGHT))
        self.assertIsNone(board.get_piece(Square.from_algebraic("f3")))
        self.assertIsNone(copied.get_piece(Square.from_algebraic("g1")))
        self.assertEqual(copied.get_piece(Square.from_algebraic("f3")), Piece(Color.WHITE, PieceType.KNIGHT))

    def test_empty_board_allows_direct_piece_placement(self):
        board = Board.empty()
        square = Square.from_algebraic("d4")

        board.set_piece(square, Piece(Color.BLACK, PieceType.BISHOP))

        self.assertEqual(board.get_piece(square), Piece(Color.BLACK, PieceType.BISHOP))
        self.assertEqual(len(list(board.pieces())), 1)


if __name__ == "__main__":
    unittest.main()

