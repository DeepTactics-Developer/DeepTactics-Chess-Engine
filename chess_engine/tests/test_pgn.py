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
from chess_engine.chess_core.pgn import PgnGame, format_san_move_list, game_from_pgn, move_to_san, pgn_to_san_tokens
from chess_engine.chess_core.piece import Piece
from chess_engine.chess_core.square import Square


class PgnTest(unittest.TestCase):
    def test_pawn_move_is_written_as_destination_square(self):
        game = Game.new()

        game.apply_move(Move.from_uci("e2e4"))

        self.assertEqual(game.san_history, ["e4"])

    def test_piece_move_uses_piece_letter(self):
        game = Game.new()

        game.apply_move(Move.from_uci("g1f3"))

        self.assertEqual(game.san_history, ["Nf3"])

    def test_capture_uses_x(self):
        board = Board.empty()
        board.set_piece(Square.from_algebraic("e1"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("e8"), Piece(Color.BLACK, PieceType.KING))
        board.set_piece(Square.from_algebraic("c4"), Piece(Color.WHITE, PieceType.BISHOP))
        board.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KNIGHT))
        move = Move.from_uci("c4g8")

        self.assertEqual(move_to_san(board, move), "Bxg8")

    def test_check_suffix_is_added(self):
        board = Board.empty()
        board.set_piece(Square.from_algebraic("c1"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("a1"), Piece(Color.WHITE, PieceType.ROOK))
        board.set_piece(Square.from_algebraic("e8"), Piece(Color.BLACK, PieceType.KING))
        move = Move.from_uci("a1e1")

        self.assertEqual(move_to_san(board, move), "Re1+")

    def test_checkmate_suffix_is_added(self):
        board = Board.empty()
        board.set_piece(Square.from_algebraic("c6"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("c7"), Piece(Color.WHITE, PieceType.QUEEN))
        board.set_piece(Square.from_algebraic("a8"), Piece(Color.BLACK, PieceType.KING))
        move = Move.from_uci("c7b7")

        self.assertEqual(move_to_san(board, move), "Qb7#")

    def test_castling_uses_standard_san(self):
        board = Board.empty()
        board.castling_rights = CastlingRights(True, False, False, False)
        board.set_piece(Square.from_algebraic("e1"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("h1"), Piece(Color.WHITE, PieceType.ROOK))
        board.set_piece(Square.from_algebraic("a8"), Piece(Color.BLACK, PieceType.KING))
        game = Game(board=board)

        game.apply_move(Move.from_uci("e1g1"))

        self.assertEqual(game.san_history, ["O-O"])

    def test_promotion_uses_equals_piece(self):
        board = Board.empty()
        board.set_piece(Square.from_algebraic("e1"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("a7"), Piece(Color.WHITE, PieceType.PAWN))
        board.set_piece(Square.from_algebraic("h8"), Piece(Color.BLACK, PieceType.KING))
        move = Move.from_uci("a7a8q")

        self.assertEqual(move_to_san(board, move), "a8=Q+")

    def test_disambiguation_uses_file_when_needed(self):
        board = Board.empty()
        board.set_piece(Square.from_algebraic("e1"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("e8"), Piece(Color.BLACK, PieceType.KING))
        board.set_piece(Square.from_algebraic("b1"), Piece(Color.WHITE, PieceType.KNIGHT))
        board.set_piece(Square.from_algebraic("f1"), Piece(Color.WHITE, PieceType.KNIGHT))
        move = Move.from_uci("b1d2")

        self.assertEqual(move_to_san(board, move), "Nbd2")

    def test_format_san_move_list_adds_move_numbers(self):
        self.assertEqual(
            format_san_move_list(["e4", "e5", "Nf3"], "*"),
            "1. e4 e5 2. Nf3 *",
        )

    def test_game_exports_minimal_pgn(self):
        game = Game.new()
        game.apply_move(Move.from_uci("e2e4"))
        game.apply_move(Move.from_uci("e7e5"))
        game.apply_move(Move.from_uci("g1f3"))

        pgn = game.to_pgn()

        self.assertIn('[Result "*"]', pgn)
        self.assertTrue(pgn.endswith("1. e4 e5 2. Nf3 *"))

    def test_pgn_game_uses_custom_tags(self):
        pgn = PgnGame(
            san_moves=["e4"],
            tags={"White": "Human", "Black": "Engine"},
            result="*",
        ).to_pgn()

        self.assertIn('[White "Human"]', pgn)
        self.assertIn('[Black "Engine"]', pgn)
        self.assertTrue(pgn.endswith("1. e4 *"))

    def test_pgn_tokens_ignore_tags_comments_and_variations(self):
        text = '[Event "?"]\n\n1. e4 {comment} e5 (1... c5) 2. Nf3 Nc6 *'

        self.assertEqual(pgn_to_san_tokens(text), ["e4", "e5", "Nf3", "Nc6"])

    def test_game_from_pgn_replays_mainline(self):
        game = game_from_pgn("1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 *")

        self.assertEqual(game.san_history, ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"])
        self.assertEqual(len(game.move_history), 6)

    def test_game_from_pgn_respects_fen_start_tag(self):
        text = (
            '[SetUp "1"]\n'
            '[FEN "1Q6/7k/5K2/6pp/4PpP1/8/8/8 b - - 0 53"]\n\n'
            "1. hxg4 Qf8 2. f3 Qg7# 1-0"
        )

        game = game_from_pgn(text)

        self.assertEqual(game.san_history, ["hxg4", "Qf8", "f3", "Qg7#"])
        self.assertEqual(len(game.move_history), 4)


if __name__ == "__main__":
    unittest.main()
