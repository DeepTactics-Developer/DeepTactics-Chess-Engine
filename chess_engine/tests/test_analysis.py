import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chess_engine.analysis import (
    EngineAnalysis,
    analyze_game_moves,
    analyze_position,
    calculate_game_accuracy,
    calculate_move_accuracy,
    classify_move,
    cp_to_wp,
    detect_move_tactics,
    score_to_white_perspective,
    variation_to_san,
)
from chess_engine.ai.search import SearchResult
from chess_engine.chess_core.enums import Color
from chess_engine.chess_core.game import Game
from chess_engine.chess_core.board import Board
from chess_engine.chess_core.move import Move
from chess_engine.chess_core.piece import Piece
from chess_engine.chess_core.square import Square
from chess_engine.chess_core.enums import PieceType


class AnalysisTest(unittest.TestCase):
    def test_analyze_position_returns_best_move_and_san_line(self):
        analysis = analyze_position(Board.starting_position(), depth=1)

        self.assertIsNotNone(analysis.best_move)
        self.assertIsNotNone(analysis.best_move_san)
        self.assertEqual(len(analysis.principal_variation_san), 1)
        self.assertEqual(analysis.completed_depth, 1)

    def test_analyze_position_can_use_native_backend(self):
        analysis = analyze_position(Board.starting_position(), depth=1, backend="native")

        self.assertIsNotNone(analysis.best_move)
        self.assertIsNotNone(analysis.best_move_san)
        self.assertEqual(len(analysis.principal_variation_san), 1)

    def test_variation_to_san_applies_moves_in_order(self):
        board = Board.starting_position()
        variation = (Move.from_uci("e2e4"), Move.from_uci("e7e5"), Move.from_uci("g1f3"))

        san = variation_to_san(board, variation)

        self.assertEqual(san, ("e4", "e5", "Nf3"))

    def test_classify_move_uses_centipawn_loss_thresholds(self):
        classification = classify_move(
            player=Color.WHITE,
            before_score=0,
            after_score=-250,
            best_score=0,
            played_move=Move.from_uci("e2e4"),
            best_move=Move.from_uci("d2d4"),
        )

        self.assertEqual(classification.key, "blunder")
        self.assertEqual(classification.best_score_cp, 0)

    def test_classify_move_marks_forced_moves(self):
        classification = classify_move(
            player=Color.WHITE,
            before_score=-300,
            after_score=-300,
            best_score=900,
            played_move=Move.from_uci("e1e2"),
            best_move=Move.from_uci("a1a8"),
            is_forced=True,
        )

        self.assertEqual(classification.key, "forced")
        self.assertEqual(classification.loss_cp, 0)

    def test_detect_move_tactics_marks_king_fork_and_pin(self):
        fork_board = Board.empty()
        fork_board.side_to_move = Color.WHITE
        fork_board.set_piece(Square.from_algebraic("a1"), Piece(Color.WHITE, PieceType.KING))
        fork_board.set_piece(Square.from_algebraic("f5"), Piece(Color.BLACK, PieceType.KING))
        fork_board.set_piece(Square.from_algebraic("e2"), Piece(Color.WHITE, PieceType.KNIGHT))
        fork_board.set_piece(Square.from_algebraic("f3"), Piece(Color.BLACK, PieceType.ROOK))

        self.assertIn("king_fork", detect_move_tactics(fork_board, Move.from_uci("e2d4")))

        pin_board = Board.empty()
        pin_board.side_to_move = Color.WHITE
        pin_board.set_piece(Square.from_algebraic("a1"), Piece(Color.WHITE, PieceType.KING))
        pin_board.set_piece(Square.from_algebraic("e8"), Piece(Color.BLACK, PieceType.KING))
        pin_board.set_piece(Square.from_algebraic("c4"), Piece(Color.WHITE, PieceType.BISHOP))
        pin_board.set_piece(Square.from_algebraic("c6"), Piece(Color.BLACK, PieceType.KNIGHT))

        self.assertIn("pin", detect_move_tactics(pin_board, Move.from_uci("c4b5")))

    def test_detect_move_tactics_marks_skewer_and_discovered_attack(self):
        skewer_board = Board.empty()
        skewer_board.side_to_move = Color.WHITE
        skewer_board.set_piece(Square.from_algebraic("a1"), Piece(Color.WHITE, PieceType.KING))
        skewer_board.set_piece(Square.from_algebraic("e7"), Piece(Color.BLACK, PieceType.KING))
        skewer_board.set_piece(Square.from_algebraic("e8"), Piece(Color.BLACK, PieceType.QUEEN))
        skewer_board.set_piece(Square.from_algebraic("e1"), Piece(Color.WHITE, PieceType.ROOK))

        self.assertIn("skewer", detect_move_tactics(skewer_board, Move.from_uci("e1e4")))

        discovered_board = Board.empty()
        discovered_board.side_to_move = Color.WHITE
        discovered_board.set_piece(Square.from_algebraic("a1"), Piece(Color.WHITE, PieceType.KING))
        discovered_board.set_piece(Square.from_algebraic("h8"), Piece(Color.BLACK, PieceType.KING))
        discovered_board.set_piece(Square.from_algebraic("b1"), Piece(Color.WHITE, PieceType.BISHOP))
        discovered_board.set_piece(Square.from_algebraic("c2"), Piece(Color.WHITE, PieceType.KNIGHT))
        discovered_board.set_piece(Square.from_algebraic("h7"), Piece(Color.BLACK, PieceType.QUEEN))

        self.assertIn("discovered_attack", detect_move_tactics(discovered_board, Move.from_uci("c2d4")))

    def test_detect_move_tactics_marks_defender_patterns(self):
        removal_board = Board.empty()
        removal_board.side_to_move = Color.WHITE
        removal_board.set_piece(Square.from_algebraic("a1"), Piece(Color.WHITE, PieceType.KING))
        removal_board.set_piece(Square.from_algebraic("h8"), Piece(Color.BLACK, PieceType.KING))
        removal_board.set_piece(Square.from_algebraic("a4"), Piece(Color.WHITE, PieceType.ROOK))
        removal_board.set_piece(Square.from_algebraic("a7"), Piece(Color.BLACK, PieceType.BISHOP))
        removal_board.set_piece(Square.from_algebraic("d4"), Piece(Color.BLACK, PieceType.QUEEN))

        self.assertIn("remove_defender", detect_move_tactics(removal_board, Move.from_uci("a4a7")))

        overload_board = Board.empty()
        overload_board.side_to_move = Color.WHITE
        overload_board.set_piece(Square.from_algebraic("a1"), Piece(Color.WHITE, PieceType.KING))
        overload_board.set_piece(Square.from_algebraic("h8"), Piece(Color.BLACK, PieceType.KING))
        overload_board.set_piece(Square.from_algebraic("b5"), Piece(Color.WHITE, PieceType.BISHOP))
        overload_board.set_piece(Square.from_algebraic("e2"), Piece(Color.BLACK, PieceType.BISHOP))
        overload_board.set_piece(Square.from_algebraic("d3"), Piece(Color.BLACK, PieceType.QUEEN))
        overload_board.set_piece(Square.from_algebraic("f3"), Piece(Color.BLACK, PieceType.ROOK))

        self.assertIn("overload", detect_move_tactics(overload_board, Move.from_uci("b5c4")))

        deflection_board = Board.empty()
        deflection_board.side_to_move = Color.WHITE
        deflection_board.set_piece(Square.from_algebraic("a1"), Piece(Color.WHITE, PieceType.KING))
        deflection_board.set_piece(Square.from_algebraic("h8"), Piece(Color.BLACK, PieceType.KING))
        deflection_board.set_piece(Square.from_algebraic("b5"), Piece(Color.WHITE, PieceType.BISHOP))
        deflection_board.set_piece(Square.from_algebraic("d3"), Piece(Color.BLACK, PieceType.BISHOP))
        deflection_board.set_piece(Square.from_algebraic("e2"), Piece(Color.BLACK, PieceType.QUEEN))

        self.assertIn("deflection", detect_move_tactics(deflection_board, Move.from_uci("b5c4")))

    def test_detect_move_tactics_marks_double_check(self):
        board = Board.empty()
        board.side_to_move = Color.WHITE
        board.set_piece(Square.from_algebraic("a1"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("e8"), Piece(Color.BLACK, PieceType.KING))
        board.set_piece(Square.from_algebraic("e1"), Piece(Color.WHITE, PieceType.ROOK))
        board.set_piece(Square.from_algebraic("e2"), Piece(Color.WHITE, PieceType.BISHOP))

        self.assertIn("double_check", detect_move_tactics(board, Move.from_uci("e2b5")))

    def test_detect_move_tactics_marks_mating_net(self):
        board = Board.empty()
        board.side_to_move = Color.WHITE
        board.set_piece(Square.from_algebraic("f6"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("h8"), Piece(Color.BLACK, PieceType.KING))
        board.set_piece(Square.from_algebraic("g7"), Piece(Color.WHITE, PieceType.QUEEN))
        board.set_piece(Square.from_algebraic("a1"), Piece(Color.WHITE, PieceType.ROOK))

        tags = detect_move_tactics(board, Move.from_uci("a1h1"))
        self.assertIn("mating_net", tags)
        self.assertIn("decoy", tags)
        self.assertIn("zwischenzug", tags)

    def test_detect_move_tactics_marks_luft_and_closed_center_knight(self):
        luft_board = Board.empty()
        luft_board.side_to_move = Color.WHITE
        luft_board.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        luft_board.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        luft_board.set_piece(Square.from_algebraic("f2"), Piece(Color.WHITE, PieceType.PAWN))
        luft_board.set_piece(Square.from_algebraic("g2"), Piece(Color.WHITE, PieceType.PAWN))
        luft_board.set_piece(Square.from_algebraic("h2"), Piece(Color.WHITE, PieceType.PAWN))

        self.assertIn("back_rank_luft", detect_move_tactics(luft_board, Move.from_uci("h2h3")))

        knight_board = Board.empty()
        knight_board.side_to_move = Color.WHITE
        knight_board.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        knight_board.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        knight_board.set_piece(Square.from_algebraic("e4"), Piece(Color.WHITE, PieceType.PAWN))
        knight_board.set_piece(Square.from_algebraic("e5"), Piece(Color.BLACK, PieceType.PAWN))
        knight_board.set_piece(Square.from_algebraic("c3"), Piece(Color.WHITE, PieceType.KNIGHT))

        self.assertIn("closed_center_knight", detect_move_tactics(knight_board, Move.from_uci("c3d5")))

    def test_detect_move_tactics_marks_clearance(self):
        board = Board.empty()
        board.side_to_move = Color.WHITE
        board.set_piece(Square.from_algebraic("e2"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("e8"), Piece(Color.BLACK, PieceType.KING))
        board.set_piece(Square.from_algebraic("a1"), Piece(Color.WHITE, PieceType.ROOK))
        board.set_piece(Square.from_algebraic("d1"), Piece(Color.WHITE, PieceType.KNIGHT))
        board.set_piece(Square.from_algebraic("h1"), Piece(Color.BLACK, PieceType.QUEEN))

        self.assertIn("clearance", detect_move_tactics(board, Move.from_uci("d1f2")))

    def test_detect_move_tactics_marks_interference(self):
        board = Board.empty()
        board.side_to_move = Color.WHITE
        board.set_piece(Square.from_algebraic("e2"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("e8"), Piece(Color.BLACK, PieceType.KING))
        board.set_piece(Square.from_algebraic("d3"), Piece(Color.WHITE, PieceType.KNIGHT))
        board.set_piece(Square.from_algebraic("a1"), Piece(Color.BLACK, PieceType.ROOK))
        board.set_piece(Square.from_algebraic("h1"), Piece(Color.WHITE, PieceType.QUEEN))

        self.assertIn("interference", detect_move_tactics(board, Move.from_uci("d3d1")))

    def test_classification_can_carry_tactical_tags(self):
        classification = classify_move(
            player=Color.WHITE,
            before_score=0,
            after_score=0,
            best_score=0,
            played_move=Move.from_uci("e2e4"),
            best_move=Move.from_uci("e2e4"),
            tactical_tags=("fork",),
        )

        self.assertEqual(classification.tactical_tags, ("fork",))

    def test_search_score_can_be_normalized_to_white_perspective(self):
        self.assertEqual(score_to_white_perspective(120, Color.WHITE), 120)
        self.assertEqual(score_to_white_perspective(120, Color.BLACK), -120)

    def test_caps_win_probability_is_logistic_and_centered(self):
        self.assertAlmostEqual(cp_to_wp(0), 50.0, places=3)
        self.assertGreater(cp_to_wp(300), cp_to_wp(0))
        self.assertLess(cp_to_wp(-300), cp_to_wp(0))

    def test_caps_move_accuracy_uses_win_probability_loss(self):
        self.assertLess(calculate_move_accuracy(100, -900), 20.0)
        self.assertEqual(calculate_move_accuracy(900, -900), 0.0)
        self.assertEqual(calculate_move_accuracy(-900, -900), 100.0)
        self.assertEqual(calculate_move_accuracy(0, -100, is_book_move=True), 100.0)

    def test_caps_game_accuracy_penalizes_low_scores_with_harmonic_mean(self):
        harmonic = calculate_game_accuracy((100.0, 100.0, 20.0))

        self.assertLess(harmonic, (100.0 + 100.0 + 20.0) / 3)
        self.assertGreaterEqual(harmonic, 0.0)

    def test_analyze_game_moves_returns_report_and_accuracy(self):
        game = Game.new()
        game.apply_move(Move.from_uci("e2e4"))
        game.apply_move(Move.from_uci("e7e5"))

        report = analyze_game_moves(game, depth=1)

        self.assertEqual(len(report.moves), 2)
        self.assertEqual(report.moves[0].san, "e4")
        self.assertEqual(report.moves[0].classification.key, "book")
        self.assertEqual(report.moves[0].accuracy, 100.0)
        self.assertGreaterEqual(report.white_accuracy, 0.0)
        self.assertLessEqual(report.black_accuracy, 100.0)

    def test_analyze_game_moves_compares_searched_scores_not_static_after_position(self):
        game = Game.new()
        game.apply_move(Move.from_uci("a2a3"))
        before = EngineAnalysis(
            result=SearchResult(Move.from_uci("a2a3"), score=0, depth=2, completed_depth=2),
            best_move_san="a3",
            principal_variation_san=("a3",),
        )
        after = EngineAnalysis(
            result=SearchResult(Move.from_uci("e7e5"), score=-240, depth=2, completed_depth=2),
            best_move_san="e5",
            principal_variation_san=("e5",),
        )

        with patch("chess_engine.analysis.engine_analysis.analyze_position", side_effect=(before, after)):
            report = analyze_game_moves(game, depth=2)

        self.assertEqual(report.moves[0].score_before, 0)
        self.assertEqual(report.moves[0].score_after, 240)


if __name__ == "__main__":
    unittest.main()
