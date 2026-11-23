import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chess_engine.chess_core.board import Board
from chess_engine.chess_core.enums import Color, PieceType
from chess_engine.chess_core.fen import STANDARD_STARTING_FEN, board_from_fen, board_to_fen
from chess_engine.chess_core.move import Move
from chess_engine.chess_core.piece import Piece
from chess_engine.chess_core.square import Square


class FenTest(unittest.TestCase):
    def test_starting_position_exports_standard_fen(self):
        board = Board.starting_position()

        self.assertEqual(board_to_fen(board), STANDARD_STARTING_FEN)

    def test_standard_fen_imports_starting_position(self):
        board = board_from_fen(STANDARD_STARTING_FEN)

        self.assertEqual(board.side_to_move, Color.WHITE)
        self.assertEqual(board.fullmove_number, 1)
        self.assertEqual(board.halfmove_clock, 0)
        self.assertIsNone(board.en_passant_target)
        self.assertEqual(board.get_piece(Square.from_algebraic("e1")), Piece(Color.WHITE, PieceType.KING))
        self.assertEqual(board.get_piece(Square.from_algebraic("d8")), Piece(Color.BLACK, PieceType.QUEEN))
        self.assertTrue(board.castling_rights.white_king_side)
        self.assertTrue(board.castling_rights.white_queen_side)
        self.assertTrue(board.castling_rights.black_king_side)
        self.assertTrue(board.castling_rights.black_queen_side)

    def test_custom_fen_round_trips(self):
        fen = "r3k2r/pppq1ppp/2npbn2/3Np3/4P3/2N1B3/PPPQ1PPP/R3K2R b KQkq e3 7 12"

        board = board_from_fen(fen)

        self.assertEqual(board_to_fen(board), fen)

    def test_fen_reflects_moves_and_en_passant_target(self):
        board = Board.starting_position()

        board.apply_move_unchecked(Move.from_uci("e2e4"))

        self.assertEqual(
            board_to_fen(board),
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        )

    def test_fen_without_castling_rights_uses_dash(self):
        board = board_from_fen("8/8/8/8/8/8/8/4K2k w - - 10 42")

        self.assertEqual(board_to_fen(board), "8/8/8/8/8/8/8/4K2k w - - 10 42")

    def test_invalid_fen_field_count_is_rejected(self):
        with self.assertRaises(ValueError):
            board_from_fen("8/8/8/8/8/8/8/8 w - - 0")

    def test_invalid_piece_placement_is_rejected(self):
        with self.assertRaises(ValueError):
            board_from_fen("9/8/8/8/8/8/8/8 w - - 0 1")

    def test_invalid_side_to_move_is_rejected(self):
        with self.assertRaises(ValueError):
            board_from_fen("8/8/8/8/8/8/8/8 x - - 0 1")

    def test_invalid_castling_rights_are_rejected(self):
        with self.assertRaises(ValueError):
            board_from_fen("8/8/8/8/8/8/8/8 w KK - 0 1")

    def test_invalid_en_passant_square_is_rejected(self):
        with self.assertRaises(ValueError):
            board_from_fen("8/8/8/8/8/8/8/8 w - e4 0 1")


if __name__ == "__main__":
    unittest.main()

