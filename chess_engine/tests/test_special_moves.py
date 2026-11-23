import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chess_engine.chess_core.board import Board, CastlingRights
from chess_engine.chess_core.enums import Color, PieceType
from chess_engine.chess_core.game import Game
from chess_engine.chess_core.move import Move
from chess_engine.chess_core.piece import Piece
from chess_engine.chess_core.rules import generate_legal_moves
from chess_engine.chess_core.square import Square


def uci_moves(moves):
    return {move.to_uci() for move in moves}


class SpecialMovesTest(unittest.TestCase):
    def test_castling_moves_are_generated_when_path_is_clear(self):
        board = Board.empty()
        board.castling_rights = CastlingRights(True, True, False, False)
        board.set_piece(Square.from_algebraic("e1"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("a1"), Piece(Color.WHITE, PieceType.ROOK))
        board.set_piece(Square.from_algebraic("h1"), Piece(Color.WHITE, PieceType.ROOK))
        board.set_piece(Square.from_algebraic("a8"), Piece(Color.BLACK, PieceType.KING))

        moves = generate_legal_moves(board, Color.WHITE)

        self.assertIn("e1g1", uci_moves(moves))
        self.assertIn("e1c1", uci_moves(moves))

    def test_castling_moves_king_and_rook(self):
        board = Board.empty()
        board.castling_rights = CastlingRights(True, False, False, False)
        board.set_piece(Square.from_algebraic("e1"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("h1"), Piece(Color.WHITE, PieceType.ROOK))
        board.set_piece(Square.from_algebraic("a8"), Piece(Color.BLACK, PieceType.KING))
        game = Game(board=board)

        game.apply_move(Move.from_uci("e1g1"))

        self.assertEqual(
            game.board.get_piece(Square.from_algebraic("g1")),
            Piece(Color.WHITE, PieceType.KING),
        )
        self.assertEqual(
            game.board.get_piece(Square.from_algebraic("f1")),
            Piece(Color.WHITE, PieceType.ROOK),
        )
        self.assertIsNone(game.board.get_piece(Square.from_algebraic("e1")))
        self.assertIsNone(game.board.get_piece(Square.from_algebraic("h1")))
        self.assertFalse(game.board.castling_rights.white_king_side)
        self.assertFalse(game.board.castling_rights.white_queen_side)

    def test_castling_is_not_generated_through_attacked_square(self):
        board = Board.empty()
        board.castling_rights = CastlingRights(True, False, False, False)
        board.set_piece(Square.from_algebraic("e1"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("h1"), Piece(Color.WHITE, PieceType.ROOK))
        board.set_piece(Square.from_algebraic("a8"), Piece(Color.BLACK, PieceType.KING))
        board.set_piece(Square.from_algebraic("f8"), Piece(Color.BLACK, PieceType.ROOK))

        moves = generate_legal_moves(board, Color.WHITE)

        self.assertNotIn("e1g1", uci_moves(moves))

    def test_castling_rights_are_revoked_after_rook_move(self):
        board = Board.empty()
        board.castling_rights = CastlingRights(True, True, False, False)
        board.set_piece(Square.from_algebraic("e1"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("h1"), Piece(Color.WHITE, PieceType.ROOK))
        board.set_piece(Square.from_algebraic("a1"), Piece(Color.WHITE, PieceType.ROOK))

        board.apply_move_unchecked(Move.from_uci("h1h2"))

        self.assertFalse(board.castling_rights.white_king_side)
        self.assertTrue(board.castling_rights.white_queen_side)

    def test_double_pawn_move_sets_en_passant_target(self):
        board = Board.starting_position()

        board.apply_move_unchecked(Move.from_uci("e2e4"))

        self.assertEqual(board.en_passant_target, Square.from_algebraic("e3"))

    def test_en_passant_move_is_generated_after_double_pawn_move(self):
        board = Board.empty()
        board.set_piece(Square.from_algebraic("e1"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("e2"), Piece(Color.WHITE, PieceType.PAWN))
        board.set_piece(Square.from_algebraic("d4"), Piece(Color.BLACK, PieceType.PAWN))
        board.set_piece(Square.from_algebraic("a8"), Piece(Color.BLACK, PieceType.KING))

        board.apply_move_unchecked(Move.from_uci("e2e4"))
        moves = generate_legal_moves(board, Color.BLACK)

        self.assertIn("d4e3", uci_moves(moves))

    def test_game_applies_en_passant_from_plain_uci_move(self):
        board = Board.empty()
        board.set_piece(Square.from_algebraic("e1"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("e2"), Piece(Color.WHITE, PieceType.PAWN))
        board.set_piece(Square.from_algebraic("d4"), Piece(Color.BLACK, PieceType.PAWN))
        board.set_piece(Square.from_algebraic("a8"), Piece(Color.BLACK, PieceType.KING))
        game = Game(board=board)

        game.apply_move(Move.from_uci("e2e4"))
        game.apply_move(Move.from_uci("d4e3"))

        self.assertEqual(
            game.board.get_piece(Square.from_algebraic("e3")),
            Piece(Color.BLACK, PieceType.PAWN),
        )
        self.assertIsNone(game.board.get_piece(Square.from_algebraic("e4")))
        self.assertIsNone(game.board.get_piece(Square.from_algebraic("d4")))
        self.assertIsNone(game.board.en_passant_target)


if __name__ == "__main__":
    unittest.main()

