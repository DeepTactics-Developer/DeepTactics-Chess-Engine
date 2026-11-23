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
from chess_engine.chess_core.rules import generate_piece_moves, generate_pseudo_legal_moves
from chess_engine.chess_core.square import Square


def uci_moves(moves):
    return {move.to_uci() for move in moves}


class PieceMovesTest(unittest.TestCase):
    def test_starting_position_has_twenty_white_moves(self):
        board = Board.starting_position()

        moves = generate_pseudo_legal_moves(board, Color.WHITE)

        self.assertEqual(len(moves), 20)
        self.assertIn("e2e4", uci_moves(moves))
        self.assertIn("g1f3", uci_moves(moves))

    def test_pawn_moves_one_or_two_squares_from_start(self):
        board = Board.starting_position()

        moves = generate_piece_moves(board, Square.from_algebraic("e2"))

        self.assertEqual(uci_moves(moves), {"e2e3", "e2e4"})

    def test_pawn_captures_diagonally(self):
        board = Board.empty()
        board.set_piece(Square.from_algebraic("e4"), Piece(Color.WHITE, PieceType.PAWN))
        board.set_piece(Square.from_algebraic("d5"), Piece(Color.BLACK, PieceType.KNIGHT))
        board.set_piece(Square.from_algebraic("f5"), Piece(Color.BLACK, PieceType.BISHOP))
        board.set_piece(Square.from_algebraic("e5"), Piece(Color.BLACK, PieceType.ROOK))

        moves = generate_piece_moves(board, Square.from_algebraic("e4"))

        self.assertEqual(uci_moves(moves), {"e4d5", "e4f5"})

    def test_pawn_promotion_generates_four_choices(self):
        board = Board.empty()
        board.set_piece(Square.from_algebraic("a7"), Piece(Color.WHITE, PieceType.PAWN))

        moves = generate_piece_moves(board, Square.from_algebraic("a7"))

        self.assertEqual(uci_moves(moves), {"a7a8q", "a7a8r", "a7a8b", "a7a8n"})

    def test_knight_can_jump_over_blockers(self):
        board = Board.starting_position()

        moves = generate_piece_moves(board, Square.from_algebraic("g1"))

        self.assertEqual(uci_moves(moves), {"g1f3", "g1h3"})

    def test_bishop_slides_until_blocked_or_capture(self):
        board = Board.empty()
        board.set_piece(Square.from_algebraic("d4"), Piece(Color.WHITE, PieceType.BISHOP))
        board.set_piece(Square.from_algebraic("f6"), Piece(Color.WHITE, PieceType.PAWN))
        board.set_piece(Square.from_algebraic("b6"), Piece(Color.BLACK, PieceType.PAWN))

        moves = generate_piece_moves(board, Square.from_algebraic("d4"))

        self.assertEqual(
            uci_moves(moves),
            {
                "d4e5",
                "d4c5",
                "d4b6",
                "d4e3",
                "d4f2",
                "d4g1",
                "d4c3",
                "d4b2",
                "d4a1",
            },
        )

    def test_rook_slides_orthogonally(self):
        board = Board.empty()
        board.set_piece(Square.from_algebraic("d4"), Piece(Color.WHITE, PieceType.ROOK))
        board.set_piece(Square.from_algebraic("d6"), Piece(Color.BLACK, PieceType.PAWN))
        board.set_piece(Square.from_algebraic("b4"), Piece(Color.WHITE, PieceType.PAWN))

        moves = generate_piece_moves(board, Square.from_algebraic("d4"))

        self.assertIn("d4d6", uci_moves(moves))
        self.assertNotIn("d4d7", uci_moves(moves))
        self.assertNotIn("d4b4", uci_moves(moves))
        self.assertNotIn("d4a4", uci_moves(moves))

    def test_queen_combines_rook_and_bishop_directions(self):
        board = Board.empty()
        board.set_piece(Square.from_algebraic("d4"), Piece(Color.WHITE, PieceType.QUEEN))

        moves = generate_piece_moves(board, Square.from_algebraic("d4"))

        self.assertEqual(len(moves), 27)
        self.assertIn("d4d8", uci_moves(moves))
        self.assertIn("d4h8", uci_moves(moves))
        self.assertIn("d4a4", uci_moves(moves))

    def test_king_moves_one_square(self):
        board = Board.empty()
        board.set_piece(Square.from_algebraic("e4"), Piece(Color.WHITE, PieceType.KING))

        moves = generate_piece_moves(board, Square.from_algebraic("e4"))

        self.assertEqual(
            uci_moves(moves),
            {"e4e5", "e4f5", "e4f4", "e4f3", "e4e3", "e4d3", "e4d4", "e4d5"},
        )

    def test_generated_moves_are_plain_move_objects_for_any_ui(self):
        board = Board.starting_position()

        moves = generate_pseudo_legal_moves(board)

        self.assertTrue(all(isinstance(move, Move) for move in moves))


if __name__ == "__main__":
    unittest.main()
