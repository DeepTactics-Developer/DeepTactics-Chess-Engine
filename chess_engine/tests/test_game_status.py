import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chess_engine.chess_core.board import Board
from chess_engine.chess_core.enums import Color, GameStatus, PieceType
from chess_engine.chess_core.game import Game, has_insufficient_material
from chess_engine.chess_core.move import Move
from chess_engine.chess_core.piece import Piece
from chess_engine.chess_core.rules import is_checkmate, is_stalemate
from chess_engine.chess_core.square import Square


class GameStatusTest(unittest.TestCase):
    def test_custom_position_detects_checkmate(self):
        board = Board.empty()
        board.side_to_move = Color.BLACK
        board.set_piece(Square.from_algebraic("a8"), Piece(Color.BLACK, PieceType.KING))
        board.set_piece(Square.from_algebraic("c6"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("b7"), Piece(Color.WHITE, PieceType.QUEEN))

        self.assertTrue(is_checkmate(board, Color.BLACK))
        self.assertFalse(is_stalemate(board, Color.BLACK))

    def test_custom_position_detects_stalemate(self):
        board = Board.empty()
        board.side_to_move = Color.BLACK
        board.set_piece(Square.from_algebraic("a8"), Piece(Color.BLACK, PieceType.KING))
        board.set_piece(Square.from_algebraic("c6"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("c7"), Piece(Color.WHITE, PieceType.QUEEN))

        self.assertTrue(is_stalemate(board, Color.BLACK))
        self.assertFalse(is_checkmate(board, Color.BLACK))

    def test_starting_game_is_ongoing(self):
        game = Game.new()

        self.assertEqual(game.status(), GameStatus.ONGOING)
        self.assertFalse(game.is_over())
        self.assertIsNone(game.winner())

    def test_game_reports_checkmate_winner(self):
        board = Board.empty()
        board.side_to_move = Color.BLACK
        board.set_piece(Square.from_algebraic("a8"), Piece(Color.BLACK, PieceType.KING))
        board.set_piece(Square.from_algebraic("c6"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("b7"), Piece(Color.WHITE, PieceType.QUEEN))
        game = Game(board=board)

        self.assertEqual(game.status(), GameStatus.CHECKMATE)
        self.assertTrue(game.is_over())
        self.assertEqual(game.winner(), Color.WHITE)

    def test_game_reports_stalemate_as_draw_without_winner(self):
        board = Board.empty()
        board.side_to_move = Color.BLACK
        board.set_piece(Square.from_algebraic("a8"), Piece(Color.BLACK, PieceType.KING))
        board.set_piece(Square.from_algebraic("c6"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("c7"), Piece(Color.WHITE, PieceType.QUEEN))
        game = Game(board=board)

        self.assertEqual(game.status(), GameStatus.STALEMATE)
        self.assertTrue(game.is_over())
        self.assertIsNone(game.winner())

    def test_game_reports_threefold_repetition_as_draw(self):
        game = Game.new()
        moves = [
            "g1f3",
            "g8f6",
            "f3g1",
            "f6g8",
            "g1f3",
            "g8f6",
            "f3g1",
            "f6g8",
        ]

        for move in moves:
            game.apply_move(Move.from_uci(move))

        self.assertEqual(game.status(), GameStatus.THREEFOLD_REPETITION)
        self.assertTrue(game.is_over())
        self.assertIsNone(game.winner())

    def test_game_reports_fifty_move_rule_as_draw(self):
        game = Game.new()
        game.board.halfmove_clock = 100

        self.assertEqual(game.status(), GameStatus.FIFTY_MOVE_RULE)
        self.assertTrue(game.is_over())

    def test_game_reports_insufficient_material_as_draw(self):
        board = Board.empty()
        board.set_piece(Square.from_algebraic("e1"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("e8"), Piece(Color.BLACK, PieceType.KING))
        game = Game(board=board)

        self.assertTrue(has_insufficient_material(board))
        self.assertEqual(game.status(), GameStatus.INSUFFICIENT_MATERIAL)
        self.assertTrue(game.is_over())

    def test_game_applies_legal_move_and_records_history(self):
        game = Game.new()
        move = Move.from_uci("e2e4")

        game.apply_move(move)

        self.assertEqual(game.move_history, [move])
        self.assertEqual(game.board.side_to_move, Color.BLACK)
        self.assertIsNone(game.board.get_piece(Square.from_algebraic("e2")))
        self.assertEqual(
            game.board.get_piece(Square.from_algebraic("e4")),
            Piece(Color.WHITE, PieceType.PAWN),
        )

    def test_game_rejects_illegal_move(self):
        game = Game.new()

        with self.assertRaises(ValueError):
            game.apply_move(Move.from_uci("e2e5"))

    def test_game_can_undo_last_move(self):
        game = Game.new()
        game.apply_move(Move.from_uci("e2e4"))

        undone_move = game.undo_last_move()

        self.assertEqual(undone_move, Move.from_uci("e2e4"))
        self.assertEqual(game.board.side_to_move, Color.WHITE)
        self.assertEqual(game.move_history, [])
        self.assertEqual(game.san_history, [])
        self.assertEqual(game.board_history, [])
        self.assertEqual(
            game.board.get_piece(Square.from_algebraic("e2")),
            Piece(Color.WHITE, PieceType.PAWN),
        )
        self.assertIsNone(game.board.get_piece(Square.from_algebraic("e4")))

    def test_undo_returns_none_when_history_is_empty(self):
        game = Game.new()

        self.assertIsNone(game.undo_last_move())

    def test_game_reset_clears_history(self):
        game = Game.new()
        game.apply_move(Move.from_uci("e2e4"))

        game.reset()

        self.assertEqual(game.board.side_to_move, Color.WHITE)
        self.assertEqual(game.move_history, [])
        self.assertEqual(game.san_history, [])
        self.assertEqual(game.board_history, [])


if __name__ == "__main__":
    unittest.main()
