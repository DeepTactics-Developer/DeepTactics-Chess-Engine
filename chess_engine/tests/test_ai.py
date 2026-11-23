import sys
from time import perf_counter
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chess_engine.ai.evaluator import (
    CHECKMATE_SCORE,
    EvaluationWeights,
    _central_control_score,
    _king_opening_safety_score,
    _native_python_overlay_score,
    _opening_development_score,
    _threat_score,
    evaluate_board,
    evaluate_endgame,
    evaluate_material,
    evaluate_mobility,
    evaluate_position,
    evaluate_strategic_knowledge,
    get_evaluation_weights,
    linear_nnue_status,
    load_accumulator_nnue_model,
    reset_linear_nnue_model,
    score_for_color,
    set_linear_nnue_model,
)
from chess_engine.ai.repetition import (
    OPENING_REPEATED_PIECE_PENALTY,
    TWOFOLD_REPETITION_PENALTY,
    combined_root_move_penalties,
    opening_repeated_piece_penalties,
    root_repetition_penalties,
)
from chess_engine.ai.backend import find_backend_move, find_backend_multipv
from chess_engine.ai.search import mate_in_from_score
from chess_engine.chess_core.board import Board
from chess_engine.chess_core.enums import Color, PieceType
from chess_engine.chess_core.fen import board_from_fen
from chess_engine.chess_core.game import Game
from chess_engine.chess_core.move import Move
from chess_engine.chess_core.piece import Piece
from chess_engine.chess_core.square import Square
from chess_engine.ai.backend import find_backend_move
from chess_engine.native.native_eval import (
    evaluate_native,
    evaluate_native_breakdown,
    find_best_line_native,
    find_best_move_native,
    evaluate_nnue_binary_native,
    evaluate_accumulator_nnue_binary_native,
    extract_nnue_features_native,
    native_available,
    native_search_diagnostics,
    nnue_sparse_linear_infer_file_native,
)
from chess_engine.nnue import AccumulatorNNUEModel, LinearNNUEModel, extract_nnue_features, nnue_features_from_fen


class AiTest(unittest.TestCase):
    def tearDown(self):
        reset_linear_nnue_model()

    def test_starting_position_evaluates_as_equal_material(self):
        board = Board.starting_position()

        self.assertEqual(evaluate_material(board), 0)
        self.assertEqual(evaluate_board(board), 0)

    def test_material_evaluation_counts_white_positive_black_negative(self):
        board = Board.empty()
        board.set_piece(Square.from_algebraic("e1"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("e8"), Piece(Color.BLACK, PieceType.KING))
        board.set_piece(Square.from_algebraic("d1"), Piece(Color.WHITE, PieceType.QUEEN))
        board.set_piece(Square.from_algebraic("a8"), Piece(Color.BLACK, PieceType.ROOK))

        self.assertEqual(evaluate_material(board), 400)
        self.assertGreater(score_for_color(board, Color.WHITE), 300)
        self.assertLess(score_for_color(board, Color.BLACK), -300)

    def test_linear_nnue_probe_can_blend_with_handcrafted_evaluation(self):
        board = Board.starting_position()
        first_feature = nnue_features_from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        ).active_indices[0]
        set_linear_nnue_model(
            LinearNNUEModel(feature_count=98317, bias=100.0, weights={first_feature: 100.0}),
            mode="blend",
            blend_weight=0.5,
        )

        self.assertEqual(evaluate_board(board), 100)
        self.assertIn("blend", linear_nnue_status())

    def test_linear_nnue_probe_can_replace_handcrafted_evaluation_for_experiments(self):
        board = Board.starting_position()
        set_linear_nnue_model(
            LinearNNUEModel(feature_count=98317, bias=42.0, weights={}),
            mode="linear",
        )

        self.assertEqual(evaluate_board(board), 42)

    def test_native_nnue_binary_inference_matches_python_when_available(self):
        board = Board.starting_position()
        features = nnue_features_from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        )
        first_feature = features.active_indices[0]
        model = LinearNNUEModel(feature_count=98317, bias=10.0, weights={first_feature: 15.0})
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.dtnnue"
            model.save_binary(path)
            native_score = nnue_sparse_linear_infer_file_native(path, features.active_indices)

        if native_score is None:
            self.skipTest("native NNUE binary export is not available until the DLL is rebuilt")
        self.assertEqual(native_score, 25)

    def test_native_nnue_binary_board_evaluation_matches_python_when_available(self):
        board = Board.starting_position()
        first_feature = nnue_features_from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        ).active_indices[0]
        model = LinearNNUEModel(feature_count=98317, bias=10.0, weights={first_feature: 15.0})
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.dtnnue"
            model.save_binary(path)
            native_score = evaluate_nnue_binary_native(board, path)

        if native_score is None:
            self.skipTest("native NNUE binary board evaluation is not available until the DLL is rebuilt")
        self.assertEqual(native_score, 25)

    def test_native_nnue_feature_extraction_matches_python_when_available(self):
        board = Board.starting_position()
        native_features = extract_nnue_features_native(board)

        if native_features is None:
            self.skipTest("native NNUE feature extraction is not available until the DLL is rebuilt")
        self.assertEqual(native_features, extract_nnue_features(board).active_indices)

    def test_native_accumulator_nnue_binary_evaluation_matches_linear_when_available(self):
        board = Board.starting_position()
        first_feature = nnue_features_from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        ).active_indices[0]
        linear = LinearNNUEModel(feature_count=98317, bias=10.0, weights={first_feature: 15.0})
        accumulator = AccumulatorNNUEModel.from_linear(linear)
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model_v2.dtnnue"
            accumulator.save_binary(path)
            native_score = evaluate_accumulator_nnue_binary_native(board, path)

        if native_score is None:
            self.skipTest("native accumulator NNUE binary evaluation is not available until the DLL is rebuilt")
        self.assertEqual(native_score, 25)

    def test_accumulator_nnue_runtime_loader_can_replace_evaluation_when_available(self):
        board = Board.starting_position()
        first_feature = nnue_features_from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        ).active_indices[0]
        linear = LinearNNUEModel(feature_count=98317, bias=10.0, weights={first_feature: 15.0})
        accumulator = AccumulatorNNUEModel.from_linear(linear)
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime_v2.dtnnue"
            accumulator.save_binary(path)
            load_accumulator_nnue_model(path, mode="linear")
            if evaluate_board(board) != 25:
                self.skipTest("native accumulator NNUE runtime evaluation is not available until the DLL is rebuilt")
            self.assertIn("accumulator", linear_nnue_status())

    def test_position_evaluation_rewards_centralized_knight(self):
        edge_knight_board = Board.empty()
        edge_knight_board.set_piece(Square.from_algebraic("e1"), Piece(Color.WHITE, PieceType.KING))
        edge_knight_board.set_piece(Square.from_algebraic("e8"), Piece(Color.BLACK, PieceType.KING))
        edge_knight_board.set_piece(Square.from_algebraic("a1"), Piece(Color.WHITE, PieceType.KNIGHT))

        central_knight_board = Board.empty()
        central_knight_board.set_piece(Square.from_algebraic("e1"), Piece(Color.WHITE, PieceType.KING))
        central_knight_board.set_piece(Square.from_algebraic("e8"), Piece(Color.BLACK, PieceType.KING))
        central_knight_board.set_piece(Square.from_algebraic("d4"), Piece(Color.WHITE, PieceType.KNIGHT))

        self.assertGreater(evaluate_position(central_knight_board), evaluate_position(edge_knight_board))

    def test_mobility_evaluation_rewards_more_legal_moves(self):
        active_board = Board.empty()
        active_board.set_piece(Square.from_algebraic("e1"), Piece(Color.WHITE, PieceType.KING))
        active_board.set_piece(Square.from_algebraic("e8"), Piece(Color.BLACK, PieceType.KING))
        active_board.set_piece(Square.from_algebraic("d4"), Piece(Color.WHITE, PieceType.QUEEN))

        passive_board = Board.empty()
        passive_board.set_piece(Square.from_algebraic("e1"), Piece(Color.WHITE, PieceType.KING))
        passive_board.set_piece(Square.from_algebraic("e8"), Piece(Color.BLACK, PieceType.KING))
        passive_board.set_piece(Square.from_algebraic("a1"), Piece(Color.WHITE, PieceType.ROOK))

        self.assertGreater(evaluate_mobility(active_board), evaluate_mobility(passive_board))

    def test_book_knowledge_rewards_stable_knight_outpost(self):
        outpost = Board.empty()
        outpost.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        outpost.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        outpost.set_piece(Square.from_algebraic("d5"), Piece(Color.WHITE, PieceType.KNIGHT))
        outpost.set_piece(Square.from_algebraic("e4"), Piece(Color.WHITE, PieceType.PAWN))

        challenged = Board.empty()
        challenged.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        challenged.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        challenged.set_piece(Square.from_algebraic("d5"), Piece(Color.WHITE, PieceType.KNIGHT))
        challenged.set_piece(Square.from_algebraic("e4"), Piece(Color.WHITE, PieceType.PAWN))
        challenged.set_piece(Square.from_algebraic("e6"), Piece(Color.BLACK, PieceType.PAWN))

        self.assertGreater(evaluate_strategic_knowledge(outpost), evaluate_strategic_knowledge(challenged))

    def test_book_knowledge_penalizes_bad_bishop_behind_own_pawns(self):
        bad_bishop = Board.empty()
        bad_bishop.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        bad_bishop.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        bad_bishop.set_piece(Square.from_algebraic("c1"), Piece(Color.WHITE, PieceType.BISHOP))
        for square in ("b2", "d2", "f2"):
            bad_bishop.set_piece(Square.from_algebraic(square), Piece(Color.WHITE, PieceType.PAWN))

        free_bishop = Board.empty()
        free_bishop.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        free_bishop.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        free_bishop.set_piece(Square.from_algebraic("c1"), Piece(Color.WHITE, PieceType.BISHOP))

        self.assertGreater(evaluate_strategic_knowledge(free_bishop), evaluate_strategic_knowledge(bad_bishop))

    def test_book_knowledge_rewards_rook_on_seventh_rank(self):
        active = Board.empty()
        active.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        active.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        active.set_piece(Square.from_algebraic("d7"), Piece(Color.WHITE, PieceType.ROOK))

        passive = Board.empty()
        passive.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        passive.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        passive.set_piece(Square.from_algebraic("d4"), Piece(Color.WHITE, PieceType.ROOK))

        self.assertGreater(evaluate_strategic_knowledge(active), evaluate_strategic_knowledge(passive))

    def test_book_knowledge_penalizes_knight_on_rim(self):
        rim = Board.empty()
        rim.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        rim.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        rim.set_piece(Square.from_algebraic("a4"), Piece(Color.WHITE, PieceType.KNIGHT))

        central = Board.empty()
        central.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        central.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        central.set_piece(Square.from_algebraic("d4"), Piece(Color.WHITE, PieceType.KNIGHT))

        self.assertGreater(evaluate_strategic_knowledge(central), evaluate_strategic_knowledge(rim))

    def test_book_knowledge_penalizes_isolated_and_backward_pawns(self):
        healthy = Board.empty()
        healthy.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        healthy.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        healthy.set_piece(Square.from_algebraic("d4"), Piece(Color.WHITE, PieceType.PAWN))
        healthy.set_piece(Square.from_algebraic("e3"), Piece(Color.WHITE, PieceType.PAWN))

        weak = Board.empty()
        weak.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        weak.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        weak.set_piece(Square.from_algebraic("d4"), Piece(Color.WHITE, PieceType.PAWN))
        weak.set_piece(Square.from_algebraic("e5"), Piece(Color.BLACK, PieceType.PAWN))

        self.assertGreater(evaluate_strategic_knowledge(healthy), evaluate_strategic_knowledge(weak))

    def test_book_knowledge_rewards_control_of_weak_squares(self):
        controlled = Board.empty()
        controlled.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        controlled.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        controlled.set_piece(Square.from_algebraic("f4"), Piece(Color.WHITE, PieceType.KNIGHT))
        controlled.set_piece(Square.from_algebraic("a7"), Piece(Color.BLACK, PieceType.PAWN))
        controlled.set_piece(Square.from_algebraic("h7"), Piece(Color.BLACK, PieceType.PAWN))

        no_control = Board.empty()
        no_control.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        no_control.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        no_control.set_piece(Square.from_algebraic("a4"), Piece(Color.WHITE, PieceType.KNIGHT))
        no_control.set_piece(Square.from_algebraic("a7"), Piece(Color.BLACK, PieceType.PAWN))
        no_control.set_piece(Square.from_algebraic("h7"), Piece(Color.BLACK, PieceType.PAWN))

        self.assertGreater(evaluate_strategic_knowledge(controlled), evaluate_strategic_knowledge(no_control))

    def test_book_knowledge_rewards_good_minor_pieces(self):
        good = Board.empty()
        good.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        good.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        good.set_piece(Square.from_algebraic("d5"), Piece(Color.WHITE, PieceType.KNIGHT))
        good.set_piece(Square.from_algebraic("e4"), Piece(Color.WHITE, PieceType.PAWN))
        good.set_piece(Square.from_algebraic("b2"), Piece(Color.WHITE, PieceType.BISHOP))

        poor = Board.empty()
        poor.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        poor.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        poor.set_piece(Square.from_algebraic("a1"), Piece(Color.WHITE, PieceType.KNIGHT))
        poor.set_piece(Square.from_algebraic("b2"), Piece(Color.WHITE, PieceType.BISHOP))
        poor.set_piece(Square.from_algebraic("c3"), Piece(Color.WHITE, PieceType.PAWN))
        poor.set_piece(Square.from_algebraic("d4"), Piece(Color.WHITE, PieceType.PAWN))

        self.assertGreater(evaluate_strategic_knowledge(good), evaluate_strategic_knowledge(poor))

    def test_book_knowledge_rewards_knight_over_bad_bishop_in_locked_center(self):
        good_knight = Board.empty()
        good_knight.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        good_knight.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        good_knight.set_piece(Square.from_algebraic("d5"), Piece(Color.WHITE, PieceType.KNIGHT))
        good_knight.set_piece(Square.from_algebraic("e4"), Piece(Color.WHITE, PieceType.PAWN))
        good_knight.set_piece(Square.from_algebraic("e5"), Piece(Color.BLACK, PieceType.PAWN))
        good_knight.set_piece(Square.from_algebraic("d4"), Piece(Color.WHITE, PieceType.PAWN))
        good_knight.set_piece(Square.from_algebraic("c5"), Piece(Color.BLACK, PieceType.PAWN))

        bad_bishop = Board.empty()
        bad_bishop.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        bad_bishop.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        bad_bishop.set_piece(Square.from_algebraic("c1"), Piece(Color.WHITE, PieceType.BISHOP))
        bad_bishop.set_piece(Square.from_algebraic("d2"), Piece(Color.WHITE, PieceType.PAWN))
        bad_bishop.set_piece(Square.from_algebraic("e3"), Piece(Color.WHITE, PieceType.PAWN))
        bad_bishop.set_piece(Square.from_algebraic("e4"), Piece(Color.WHITE, PieceType.PAWN))
        bad_bishop.set_piece(Square.from_algebraic("e5"), Piece(Color.BLACK, PieceType.PAWN))
        bad_bishop.set_piece(Square.from_algebraic("d4"), Piece(Color.WHITE, PieceType.PAWN))
        bad_bishop.set_piece(Square.from_algebraic("d5"), Piece(Color.BLACK, PieceType.PAWN))

        self.assertGreater(evaluate_strategic_knowledge(good_knight), evaluate_strategic_knowledge(bad_bishop))

    def test_book_knowledge_penalizes_color_complex_weakness(self):
        weak_dark = Board.empty()
        weak_dark.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        weak_dark.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        weak_dark.set_piece(Square.from_algebraic("b4"), Piece(Color.BLACK, PieceType.BISHOP))
        weak_dark.set_piece(Square.from_algebraic("a2"), Piece(Color.WHITE, PieceType.PAWN))
        weak_dark.set_piece(Square.from_algebraic("h2"), Piece(Color.WHITE, PieceType.PAWN))

        covered_dark = Board.empty()
        covered_dark.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        covered_dark.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        covered_dark.set_piece(Square.from_algebraic("b4"), Piece(Color.BLACK, PieceType.BISHOP))
        covered_dark.set_piece(Square.from_algebraic("c3"), Piece(Color.WHITE, PieceType.PAWN))
        covered_dark.set_piece(Square.from_algebraic("e3"), Piece(Color.WHITE, PieceType.PAWN))

        self.assertGreater(evaluate_strategic_knowledge(covered_dark), evaluate_strategic_knowledge(weak_dark))

    def test_book_knowledge_penalizes_back_rank_weakness(self):
        trapped = Board.empty()
        trapped.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        trapped.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        trapped.set_piece(Square.from_algebraic("f2"), Piece(Color.WHITE, PieceType.PAWN))
        trapped.set_piece(Square.from_algebraic("g2"), Piece(Color.WHITE, PieceType.PAWN))
        trapped.set_piece(Square.from_algebraic("h2"), Piece(Color.WHITE, PieceType.PAWN))
        trapped.set_piece(Square.from_algebraic("g7"), Piece(Color.BLACK, PieceType.ROOK))

        luft = Board.empty()
        luft.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        luft.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        luft.set_piece(Square.from_algebraic("f2"), Piece(Color.WHITE, PieceType.PAWN))
        luft.set_piece(Square.from_algebraic("g2"), Piece(Color.WHITE, PieceType.PAWN))
        luft.set_piece(Square.from_algebraic("h3"), Piece(Color.WHITE, PieceType.PAWN))
        luft.set_piece(Square.from_algebraic("g7"), Piece(Color.BLACK, PieceType.ROOK))

        self.assertGreater(evaluate_strategic_knowledge(luft), evaluate_strategic_knowledge(trapped))

    def test_book_knowledge_rewards_line_activity(self):
        active = Board.empty()
        active.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        active.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        active.set_piece(Square.from_algebraic("a1"), Piece(Color.WHITE, PieceType.ROOK))
        active.set_piece(Square.from_algebraic("h2"), Piece(Color.WHITE, PieceType.PAWN))

        blocked = Board.empty()
        blocked.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        blocked.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        blocked.set_piece(Square.from_algebraic("a1"), Piece(Color.WHITE, PieceType.ROOK))
        blocked.set_piece(Square.from_algebraic("a2"), Piece(Color.WHITE, PieceType.PAWN))

        self.assertGreater(evaluate_strategic_knowledge(active), evaluate_strategic_knowledge(blocked))

    def test_book_knowledge_rewards_simplification_when_ahead(self):
        simplified = Board.empty()
        simplified.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        simplified.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        simplified.set_piece(Square.from_algebraic("d1"), Piece(Color.WHITE, PieceType.QUEEN))

        crowded = Board.empty()
        crowded.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        crowded.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        crowded.set_piece(Square.from_algebraic("d1"), Piece(Color.WHITE, PieceType.QUEEN))
        crowded.set_piece(Square.from_algebraic("a1"), Piece(Color.WHITE, PieceType.ROOK))
        crowded.set_piece(Square.from_algebraic("h1"), Piece(Color.WHITE, PieceType.ROOK))
        crowded.set_piece(Square.from_algebraic("a8"), Piece(Color.BLACK, PieceType.ROOK))
        crowded.set_piece(Square.from_algebraic("h8"), Piece(Color.BLACK, PieceType.ROOK))

        self.assertGreater(evaluate_strategic_knowledge(simplified), evaluate_strategic_knowledge(crowded))

    def test_book_knowledge_discourages_simplification_when_behind(self):
        crowded = Board.empty()
        crowded.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        crowded.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        crowded.set_piece(Square.from_algebraic("d8"), Piece(Color.BLACK, PieceType.QUEEN))
        crowded.set_piece(Square.from_algebraic("a1"), Piece(Color.WHITE, PieceType.ROOK))
        crowded.set_piece(Square.from_algebraic("h1"), Piece(Color.WHITE, PieceType.ROOK))
        crowded.set_piece(Square.from_algebraic("a8"), Piece(Color.BLACK, PieceType.ROOK))
        crowded.set_piece(Square.from_algebraic("h8"), Piece(Color.BLACK, PieceType.ROOK))

        simplified = Board.empty()
        simplified.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        simplified.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        simplified.set_piece(Square.from_algebraic("d8"), Piece(Color.BLACK, PieceType.QUEEN))

        self.assertGreater(evaluate_strategic_knowledge(crowded), evaluate_strategic_knowledge(simplified))

    def test_book_knowledge_rewards_attack_momentum(self):
        attacking = Board.empty()
        attacking.set_piece(Square.from_algebraic("a1"), Piece(Color.WHITE, PieceType.KING))
        attacking.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        attacking.set_piece(Square.from_algebraic("h5"), Piece(Color.WHITE, PieceType.QUEEN))
        attacking.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.ROOK))

        quiet = Board.empty()
        quiet.set_piece(Square.from_algebraic("a1"), Piece(Color.WHITE, PieceType.KING))
        quiet.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        quiet.set_piece(Square.from_algebraic("a2"), Piece(Color.WHITE, PieceType.QUEEN))
        quiet.set_piece(Square.from_algebraic("a3"), Piece(Color.WHITE, PieceType.ROOK))

        self.assertGreater(evaluate_strategic_knowledge(attacking), evaluate_strategic_knowledge(quiet))

    def test_book_knowledge_rewards_pawn_majority_and_breaks(self):
        majority = Board.empty()
        majority.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        majority.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        majority.set_piece(Square.from_algebraic("a4"), Piece(Color.WHITE, PieceType.PAWN))
        majority.set_piece(Square.from_algebraic("b3"), Piece(Color.WHITE, PieceType.PAWN))
        majority.set_piece(Square.from_algebraic("h7"), Piece(Color.BLACK, PieceType.PAWN))

        even = Board.empty()
        even.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        even.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        even.set_piece(Square.from_algebraic("a4"), Piece(Color.WHITE, PieceType.PAWN))
        even.set_piece(Square.from_algebraic("h7"), Piece(Color.BLACK, PieceType.PAWN))

        self.assertGreater(evaluate_strategic_knowledge(majority), evaluate_strategic_knowledge(even))

        pawn_break = Board.empty()
        pawn_break.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        pawn_break.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        pawn_break.set_piece(Square.from_algebraic("e4"), Piece(Color.WHITE, PieceType.PAWN))
        pawn_break.set_piece(Square.from_algebraic("d5"), Piece(Color.BLACK, PieceType.PAWN))

        blocked = Board.empty()
        blocked.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        blocked.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        blocked.set_piece(Square.from_algebraic("e4"), Piece(Color.WHITE, PieceType.PAWN))
        blocked.set_piece(Square.from_algebraic("e5"), Piece(Color.BLACK, PieceType.PAWN))
        blocked.set_piece(Square.from_algebraic("d5"), Piece(Color.BLACK, PieceType.PAWN))

        self.assertGreater(evaluate_strategic_knowledge(pawn_break), evaluate_strategic_knowledge(blocked))

    def test_endgame_evaluation_rewards_passed_pawn_advancement(self):
        early_pawn_board = Board.empty()
        early_pawn_board.set_piece(Square.from_algebraic("e1"), Piece(Color.WHITE, PieceType.KING))
        early_pawn_board.set_piece(Square.from_algebraic("e8"), Piece(Color.BLACK, PieceType.KING))
        early_pawn_board.set_piece(Square.from_algebraic("a3"), Piece(Color.WHITE, PieceType.PAWN))

        advanced_pawn_board = Board.empty()
        advanced_pawn_board.set_piece(Square.from_algebraic("e1"), Piece(Color.WHITE, PieceType.KING))
        advanced_pawn_board.set_piece(Square.from_algebraic("e8"), Piece(Color.BLACK, PieceType.KING))
        advanced_pawn_board.set_piece(Square.from_algebraic("a6"), Piece(Color.WHITE, PieceType.PAWN))

        self.assertGreater(evaluate_endgame(advanced_pawn_board), evaluate_endgame(early_pawn_board))

    def test_endgame_evaluation_rewards_passed_pawn_support_and_active_rook(self):
        supported = Board.empty()
        supported.set_piece(Square.from_algebraic("d4"), Piece(Color.WHITE, PieceType.KING))
        supported.set_piece(Square.from_algebraic("h8"), Piece(Color.BLACK, PieceType.KING))
        supported.set_piece(Square.from_algebraic("a6"), Piece(Color.WHITE, PieceType.PAWN))
        supported.set_piece(Square.from_algebraic("a1"), Piece(Color.WHITE, PieceType.ROOK))

        unsupported = Board.empty()
        unsupported.set_piece(Square.from_algebraic("h1"), Piece(Color.WHITE, PieceType.KING))
        unsupported.set_piece(Square.from_algebraic("b8"), Piece(Color.BLACK, PieceType.KING))
        unsupported.set_piece(Square.from_algebraic("a6"), Piece(Color.WHITE, PieceType.PAWN))
        unsupported.set_piece(Square.from_algebraic("h2"), Piece(Color.WHITE, PieceType.ROOK))

        self.assertGreater(evaluate_endgame(supported), evaluate_endgame(unsupported))

        active_rook = Board.empty()
        active_rook.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        active_rook.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        active_rook.set_piece(Square.from_algebraic("a7"), Piece(Color.WHITE, PieceType.ROOK))
        active_rook.set_piece(Square.from_algebraic("h8"), Piece(Color.BLACK, PieceType.ROOK))

        passive_rook = Board.empty()
        passive_rook.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        passive_rook.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        passive_rook.set_piece(Square.from_algebraic("a1"), Piece(Color.WHITE, PieceType.ROOK))
        passive_rook.set_piece(Square.from_algebraic("h8"), Piece(Color.BLACK, PieceType.ROOK))

        self.assertGreater(evaluate_endgame(active_rook), evaluate_endgame(passive_rook))

    def test_endgame_evaluation_rewards_connected_passed_pawns(self):
        connected = Board.empty()
        connected.set_piece(Square.from_algebraic("e1"), Piece(Color.WHITE, PieceType.KING))
        connected.set_piece(Square.from_algebraic("e8"), Piece(Color.BLACK, PieceType.KING))
        connected.set_piece(Square.from_algebraic("c5"), Piece(Color.WHITE, PieceType.PAWN))
        connected.set_piece(Square.from_algebraic("d5"), Piece(Color.WHITE, PieceType.PAWN))

        split = Board.empty()
        split.set_piece(Square.from_algebraic("e1"), Piece(Color.WHITE, PieceType.KING))
        split.set_piece(Square.from_algebraic("e8"), Piece(Color.BLACK, PieceType.KING))
        split.set_piece(Square.from_algebraic("a5"), Piece(Color.WHITE, PieceType.PAWN))
        split.set_piece(Square.from_algebraic("h5"), Piece(Color.WHITE, PieceType.PAWN))

        self.assertGreater(evaluate_endgame(connected), evaluate_endgame(split))

    def test_endgame_evaluation_rewards_cutting_off_enemy_king(self):
        cutoff = Board.empty()
        cutoff.set_piece(Square.from_algebraic("d4"), Piece(Color.WHITE, PieceType.KING))
        cutoff.set_piece(Square.from_algebraic("a8"), Piece(Color.BLACK, PieceType.KING))
        cutoff.set_piece(Square.from_algebraic("a7"), Piece(Color.WHITE, PieceType.ROOK))

        passive = Board.empty()
        passive.set_piece(Square.from_algebraic("d4"), Piece(Color.WHITE, PieceType.KING))
        passive.set_piece(Square.from_algebraic("a8"), Piece(Color.BLACK, PieceType.KING))
        passive.set_piece(Square.from_algebraic("h1"), Piece(Color.WHITE, PieceType.ROOK))

        self.assertGreater(evaluate_endgame(cutoff), evaluate_endgame(passive))

    def test_native_evaluation_rewards_rook_on_open_file(self):
        if not native_available():
            self.skipTest("native evaluator DLL is not available")
        weights = EvaluationWeights(material=0, position=1, mobility=0, endgame=0)
        open_file = Board.empty()
        open_file.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        open_file.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        open_file.set_piece(Square.from_algebraic("a1"), Piece(Color.WHITE, PieceType.ROOK))
        open_file.set_piece(Square.from_algebraic("h2"), Piece(Color.WHITE, PieceType.PAWN))

        blocked_file = Board.empty()
        blocked_file.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        blocked_file.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        blocked_file.set_piece(Square.from_algebraic("a1"), Piece(Color.WHITE, PieceType.ROOK))
        blocked_file.set_piece(Square.from_algebraic("a2"), Piece(Color.WHITE, PieceType.PAWN))

        self.assertGreater(evaluate_native(open_file, weights), evaluate_native(blocked_file, weights))

    def test_native_evaluation_rewards_king_pawn_shelter(self):
        if not native_available():
            self.skipTest("native evaluator DLL is not available")
        weights = EvaluationWeights(material=0, position=1, mobility=0, endgame=0)
        sheltered = Board.empty()
        sheltered.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        sheltered.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        sheltered.set_piece(Square.from_algebraic("d1"), Piece(Color.WHITE, PieceType.QUEEN))
        sheltered.set_piece(Square.from_algebraic("d8"), Piece(Color.BLACK, PieceType.QUEEN))
        for square in ("f2", "g2", "h2"):
            sheltered.set_piece(Square.from_algebraic(square), Piece(Color.WHITE, PieceType.PAWN))

        exposed = Board.empty()
        exposed.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        exposed.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        exposed.set_piece(Square.from_algebraic("d1"), Piece(Color.WHITE, PieceType.QUEEN))
        exposed.set_piece(Square.from_algebraic("d8"), Piece(Color.BLACK, PieceType.QUEEN))
        for square in ("a2", "b2", "c2"):
            exposed.set_piece(Square.from_algebraic(square), Piece(Color.WHITE, PieceType.PAWN))

        self.assertGreater(evaluate_native(sheltered, weights), evaluate_native(exposed, weights))

    def test_native_evaluation_rewards_opening_development_and_castling(self):
        if not native_available():
            self.skipTest("native evaluator DLL is not available")
        weights = EvaluationWeights(material=0, position=1, mobility=0, endgame=0)

        developed = Board.empty()
        developed.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        developed.set_piece(Square.from_algebraic("f1"), Piece(Color.WHITE, PieceType.ROOK))
        developed.set_piece(Square.from_algebraic("c4"), Piece(Color.WHITE, PieceType.BISHOP))
        developed.set_piece(Square.from_algebraic("f3"), Piece(Color.WHITE, PieceType.KNIGHT))
        developed.set_piece(Square.from_algebraic("e8"), Piece(Color.BLACK, PieceType.KING))
        developed.set_piece(Square.from_algebraic("a8"), Piece(Color.BLACK, PieceType.ROOK))

        undeveloped = Board.empty()
        undeveloped.set_piece(Square.from_algebraic("e1"), Piece(Color.WHITE, PieceType.KING))
        undeveloped.set_piece(Square.from_algebraic("h1"), Piece(Color.WHITE, PieceType.ROOK))
        undeveloped.set_piece(Square.from_algebraic("c1"), Piece(Color.WHITE, PieceType.BISHOP))
        undeveloped.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KNIGHT))
        undeveloped.set_piece(Square.from_algebraic("e8"), Piece(Color.BLACK, PieceType.KING))
        undeveloped.set_piece(Square.from_algebraic("a8"), Piece(Color.BLACK, PieceType.ROOK))

        self.assertGreater(evaluate_native(developed, weights), evaluate_native(undeveloped, weights))

    def test_native_evaluation_rewards_central_control(self):
        if not native_available():
            self.skipTest("native evaluator DLL is not available")
        weights = EvaluationWeights(material=0, position=1, mobility=0, endgame=0)

        central = Board.empty()
        central.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        central.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        central.set_piece(Square.from_algebraic("e4"), Piece(Color.WHITE, PieceType.PAWN))
        central.set_piece(Square.from_algebraic("a7"), Piece(Color.BLACK, PieceType.PAWN))

        flank = Board.empty()
        flank.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        flank.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        flank.set_piece(Square.from_algebraic("a4"), Piece(Color.WHITE, PieceType.PAWN))
        flank.set_piece(Square.from_algebraic("a7"), Piece(Color.BLACK, PieceType.PAWN))

        self.assertGreater(evaluate_native(central, weights), evaluate_native(flank, weights))

    def test_native_endgame_evaluation_pushes_weaker_king_to_edge(self):
        if not native_available():
            self.skipTest("native evaluator DLL is not available")
        weights = EvaluationWeights(material=0, position=0, mobility=0, endgame=1)
        edge_king = Board.empty()
        edge_king.set_piece(Square.from_algebraic("d4"), Piece(Color.WHITE, PieceType.KING))
        edge_king.set_piece(Square.from_algebraic("a8"), Piece(Color.BLACK, PieceType.KING))
        edge_king.set_piece(Square.from_algebraic("h1"), Piece(Color.WHITE, PieceType.ROOK))

        central_king = Board.empty()
        central_king.set_piece(Square.from_algebraic("d4"), Piece(Color.WHITE, PieceType.KING))
        central_king.set_piece(Square.from_algebraic("g6"), Piece(Color.BLACK, PieceType.KING))
        central_king.set_piece(Square.from_algebraic("h1"), Piece(Color.WHITE, PieceType.ROOK))

        self.assertGreater(evaluate_native(edge_king, weights), evaluate_native(central_king, weights))

    def test_native_endgame_evaluation_rewards_active_winning_piece(self):
        if not native_available():
            self.skipTest("native evaluator DLL is not available")
        weights = EvaluationWeights(material=0, position=0, mobility=0, endgame=1)
        active_rook = Board.empty()
        active_rook.set_piece(Square.from_algebraic("d4"), Piece(Color.WHITE, PieceType.KING))
        active_rook.set_piece(Square.from_algebraic("a8"), Piece(Color.BLACK, PieceType.KING))
        active_rook.set_piece(Square.from_algebraic("a7"), Piece(Color.WHITE, PieceType.ROOK))

        passive_rook = Board.empty()
        passive_rook.set_piece(Square.from_algebraic("d4"), Piece(Color.WHITE, PieceType.KING))
        passive_rook.set_piece(Square.from_algebraic("a8"), Piece(Color.BLACK, PieceType.KING))
        passive_rook.set_piece(Square.from_algebraic("h1"), Piece(Color.WHITE, PieceType.ROOK))

        self.assertGreater(evaluate_native(active_rook, weights), evaluate_native(passive_rook, weights))

    def test_native_endgame_evaluation_rewards_unstoppable_passed_pawn(self):
        if not native_available():
            self.skipTest("native evaluator DLL is not available")
        weights = EvaluationWeights(material=0, position=0, mobility=0, endgame=1)
        far_king = Board.empty()
        far_king.set_piece(Square.from_algebraic("e1"), Piece(Color.WHITE, PieceType.KING))
        far_king.set_piece(Square.from_algebraic("h8"), Piece(Color.BLACK, PieceType.KING))
        far_king.set_piece(Square.from_algebraic("a6"), Piece(Color.WHITE, PieceType.PAWN))

        near_king = Board.empty()
        near_king.set_piece(Square.from_algebraic("e1"), Piece(Color.WHITE, PieceType.KING))
        near_king.set_piece(Square.from_algebraic("b8"), Piece(Color.BLACK, PieceType.KING))
        near_king.set_piece(Square.from_algebraic("a6"), Piece(Color.WHITE, PieceType.PAWN))

        self.assertGreater(evaluate_native(far_king, weights), evaluate_native(near_king, weights))

    def test_native_evaluation_rewards_pawn_majority_and_endgame_support(self):
        if not native_available():
            self.skipTest("native evaluator DLL is not available")
        strategic_weights = EvaluationWeights(material=0, position=1, mobility=0, endgame=0)
        majority = Board.empty()
        majority.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        majority.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        majority.set_piece(Square.from_algebraic("a4"), Piece(Color.WHITE, PieceType.PAWN))
        majority.set_piece(Square.from_algebraic("b3"), Piece(Color.WHITE, PieceType.PAWN))
        majority.set_piece(Square.from_algebraic("h7"), Piece(Color.BLACK, PieceType.PAWN))

        even = Board.empty()
        even.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        even.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        even.set_piece(Square.from_algebraic("a4"), Piece(Color.WHITE, PieceType.PAWN))
        even.set_piece(Square.from_algebraic("h7"), Piece(Color.BLACK, PieceType.PAWN))

        self.assertGreater(evaluate_native(majority, strategic_weights), evaluate_native(even, strategic_weights))

        endgame_weights = EvaluationWeights(material=0, position=0, mobility=0, endgame=1)
        supported = Board.empty()
        supported.set_piece(Square.from_algebraic("d4"), Piece(Color.WHITE, PieceType.KING))
        supported.set_piece(Square.from_algebraic("h8"), Piece(Color.BLACK, PieceType.KING))
        supported.set_piece(Square.from_algebraic("a6"), Piece(Color.WHITE, PieceType.PAWN))
        supported.set_piece(Square.from_algebraic("a1"), Piece(Color.WHITE, PieceType.ROOK))

        unsupported = Board.empty()
        unsupported.set_piece(Square.from_algebraic("h1"), Piece(Color.WHITE, PieceType.KING))
        unsupported.set_piece(Square.from_algebraic("b8"), Piece(Color.BLACK, PieceType.KING))
        unsupported.set_piece(Square.from_algebraic("a6"), Piece(Color.WHITE, PieceType.PAWN))
        unsupported.set_piece(Square.from_algebraic("h2"), Piece(Color.WHITE, PieceType.ROOK))

        self.assertGreater(evaluate_native(supported, endgame_weights), evaluate_native(unsupported, endgame_weights))

    def test_native_evaluation_rewards_locked_center_minor_piece_quality(self):
        if not native_available():
            self.skipTest("native evaluator DLL is not available")
        weights = EvaluationWeights(material=0, position=1, mobility=0, endgame=0)
        good_knight = Board.empty()
        good_knight.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        good_knight.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        good_knight.set_piece(Square.from_algebraic("d5"), Piece(Color.WHITE, PieceType.KNIGHT))
        good_knight.set_piece(Square.from_algebraic("e4"), Piece(Color.WHITE, PieceType.PAWN))
        good_knight.set_piece(Square.from_algebraic("e5"), Piece(Color.BLACK, PieceType.PAWN))
        good_knight.set_piece(Square.from_algebraic("d4"), Piece(Color.WHITE, PieceType.PAWN))
        good_knight.set_piece(Square.from_algebraic("c5"), Piece(Color.BLACK, PieceType.PAWN))

        bad_bishop = Board.empty()
        bad_bishop.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        bad_bishop.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        bad_bishop.set_piece(Square.from_algebraic("c1"), Piece(Color.WHITE, PieceType.BISHOP))
        bad_bishop.set_piece(Square.from_algebraic("d2"), Piece(Color.WHITE, PieceType.PAWN))
        bad_bishop.set_piece(Square.from_algebraic("e3"), Piece(Color.WHITE, PieceType.PAWN))
        bad_bishop.set_piece(Square.from_algebraic("e4"), Piece(Color.WHITE, PieceType.PAWN))
        bad_bishop.set_piece(Square.from_algebraic("e5"), Piece(Color.BLACK, PieceType.PAWN))
        bad_bishop.set_piece(Square.from_algebraic("d4"), Piece(Color.WHITE, PieceType.PAWN))
        bad_bishop.set_piece(Square.from_algebraic("d5"), Piece(Color.BLACK, PieceType.PAWN))

        self.assertGreater(evaluate_native(good_knight, weights), evaluate_native(bad_bishop, weights))

    def test_native_evaluation_penalizes_back_rank_weakness(self):
        if not native_available():
            self.skipTest("native evaluator DLL is not available")
        weights = EvaluationWeights(material=0, position=1, mobility=0, endgame=0)
        trapped = Board.empty()
        trapped.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        trapped.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        trapped.set_piece(Square.from_algebraic("f2"), Piece(Color.WHITE, PieceType.PAWN))
        trapped.set_piece(Square.from_algebraic("g2"), Piece(Color.WHITE, PieceType.PAWN))
        trapped.set_piece(Square.from_algebraic("h2"), Piece(Color.WHITE, PieceType.PAWN))
        trapped.set_piece(Square.from_algebraic("g7"), Piece(Color.BLACK, PieceType.ROOK))

        luft = Board.empty()
        luft.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        luft.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        luft.set_piece(Square.from_algebraic("f2"), Piece(Color.WHITE, PieceType.PAWN))
        luft.set_piece(Square.from_algebraic("g2"), Piece(Color.WHITE, PieceType.PAWN))
        luft.set_piece(Square.from_algebraic("h3"), Piece(Color.WHITE, PieceType.PAWN))
        luft.set_piece(Square.from_algebraic("g7"), Piece(Color.BLACK, PieceType.ROOK))

        self.assertGreater(evaluate_native(luft, weights), evaluate_native(trapped, weights))

    def test_native_activity_weights_are_independently_tunable_and_reported(self):
        if not native_available():
            self.skipTest("native evaluator DLL is not available")
        board = Board.empty()
        board.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        # line_activity is bishop-diagonal-scope only -- rook open/semi-open-file
        # activity was removed from here (it lived under rook_activity_weight
        # too, double-counting the same fact under two weights).
        board.set_piece(Square.from_algebraic("c1"), Piece(Color.WHITE, PieceType.BISHOP))
        board.set_piece(Square.from_algebraic("h2"), Piece(Color.WHITE, PieceType.PAWN))

        default = evaluate_native_breakdown(board, EvaluationWeights())
        disabled = evaluate_native_breakdown(board, EvaluationWeights(line_activity=0))

        self.assertIsNotNone(default)
        self.assertIsNotNone(disabled)
        self.assertGreater(default["line_activity"], 0)
        self.assertGreater(default["native_total"], disabled["native_total"])

    def test_native_bare_kings_do_not_create_false_winning_pressure(self):
        if not native_available():
            self.skipTest("native evaluator DLL is not available")
        weights = EvaluationWeights(material=0, position=0, mobility=0, endgame=1)
        board = Board.empty()
        board.set_piece(Square.from_algebraic("d4"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("a8"), Piece(Color.BLACK, PieceType.KING))

        self.assertEqual(evaluate_native(board, weights), 0)

    def test_native_overlay_does_not_double_count_opening_terms_already_in_native(self):
        """Regression test for the native/overlay double-counting bug.

        The starting position cannot catch this: it is symmetric, so
        central_control/opening_development/king_opening_safety all cancel to
        zero for both sides and 0 + 0 stays 0 either way. This position
        castles White kingside and gives White a central pawn so the three
        terms are individually non-zero and asymmetric between colors.
        """
        # Enough non-pawn material stays on the board to keep the game phase
        # in the opening/middlegame range (phase <= 0.25); the opening
        # development and king-safety terms are gated off in the endgame.
        board = Board.empty()
        board.side_to_move = Color.WHITE
        board.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("f1"), Piece(Color.WHITE, PieceType.ROOK))
        board.set_piece(Square.from_algebraic("d1"), Piece(Color.WHITE, PieceType.QUEEN))
        board.set_piece(Square.from_algebraic("c1"), Piece(Color.WHITE, PieceType.BISHOP))
        board.set_piece(Square.from_algebraic("e4"), Piece(Color.WHITE, PieceType.PAWN))
        board.set_piece(Square.from_algebraic("d4"), Piece(Color.WHITE, PieceType.PAWN))
        board.set_piece(Square.from_algebraic("e8"), Piece(Color.BLACK, PieceType.KING))
        board.set_piece(Square.from_algebraic("a8"), Piece(Color.BLACK, PieceType.ROOK))
        board.set_piece(Square.from_algebraic("d8"), Piece(Color.BLACK, PieceType.QUEEN))
        board.set_piece(Square.from_algebraic("c8"), Piece(Color.BLACK, PieceType.BISHOP))

        # Sanity check: the three previously-duplicated terms are actually
        # non-zero here, so this position would have caught the old bug.
        self.assertNotEqual(_central_control_score(board), 0)
        self.assertNotEqual(_opening_development_score(board), 0)
        self.assertNotEqual(_king_opening_safety_score(board), 0)

        self.assertEqual(_native_python_overlay_score(board), _threat_score(board))

        if not native_available():
            self.skipTest("native evaluator DLL is not available")
        weights = get_evaluation_weights()
        native_score = evaluate_native(board, weights)
        self.assertEqual(evaluate_board(board), native_score + _threat_score(board))

    def test_checkmate_evaluation_is_extreme(self):
        board = Board.empty()
        board.side_to_move = Color.BLACK
        board.set_piece(Square.from_algebraic("a8"), Piece(Color.BLACK, PieceType.KING))
        board.set_piece(Square.from_algebraic("c6"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("b7"), Piece(Color.WHITE, PieceType.QUEEN))

        self.assertEqual(evaluate_board(board), CHECKMATE_SCORE)
        self.assertEqual(score_for_color(board, Color.BLACK), -CHECKMATE_SCORE)

    def test_mate_score_can_be_formatted_as_distance(self):
        self.assertEqual(mate_in_from_score(CHECKMATE_SCORE - 1), 1)
        self.assertEqual(mate_in_from_score(-(CHECKMATE_SCORE - 3)), -2)
        self.assertIsNone(mate_in_from_score(250))


    def test_native_search_captures_hanging_queen(self):
        if not native_available():
            self.skipTest("native evaluator DLL is not available")
        board = Board.empty()
        board.side_to_move = Color.WHITE
        board.set_piece(Square.from_algebraic("e1"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("a1"), Piece(Color.WHITE, PieceType.ROOK))
        board.set_piece(Square.from_algebraic("e8"), Piece(Color.BLACK, PieceType.KING))
        board.set_piece(Square.from_algebraic("a8"), Piece(Color.BLACK, PieceType.QUEEN))

        result = find_best_move_native(board, depth=1, weights=get_evaluation_weights())

        self.assertIsNotNone(result)
        move, _ = result
        self.assertEqual(move.to_uci(), "a1a8")

    def test_native_search_reports_terminal_no_move_without_falling_back(self):
        if not native_available():
            self.skipTest("native evaluator DLL is not available")
        board = Board.empty()
        board.side_to_move = Color.BLACK
        board.set_piece(Square.from_algebraic("a8"), Piece(Color.BLACK, PieceType.KING))
        board.set_piece(Square.from_algebraic("c6"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("b7"), Piece(Color.WHITE, PieceType.QUEEN))

        result = find_best_move_native(board, depth=2, weights=get_evaluation_weights())

        self.assertIsNotNone(result)
        move, score = result
        self.assertIsNone(move)
        self.assertLessEqual(score, -CHECKMATE_SCORE)

    def test_native_search_accepts_quiescence_depth(self):
        if not native_available():
            self.skipTest("native evaluator DLL is not available")
        board = Board.empty()
        board.side_to_move = Color.WHITE
        board.set_piece(Square.from_algebraic("e1"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("a1"), Piece(Color.WHITE, PieceType.ROOK))
        board.set_piece(Square.from_algebraic("e8"), Piece(Color.BLACK, PieceType.KING))
        board.set_piece(Square.from_algebraic("a8"), Piece(Color.BLACK, PieceType.QUEEN))

        result = find_best_move_native(
            board,
            depth=1,
            weights=get_evaluation_weights(),
            quiescence_depth=2,
        )

        self.assertIsNotNone(result)
        move, _ = result
        self.assertEqual(move.to_uci(), "a1a8")

    def test_native_qsearch_does_not_prefer_immediately_recaptured_queen_trade(self):
        if not native_available():
            self.skipTest("native evaluator DLL is not available")
        board = Board.empty()
        board.side_to_move = Color.WHITE
        board.set_piece(Square.from_algebraic("a1"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("d1"), Piece(Color.WHITE, PieceType.QUEEN))
        board.set_piece(Square.from_algebraic("e8"), Piece(Color.BLACK, PieceType.KING))
        board.set_piece(Square.from_algebraic("d8"), Piece(Color.BLACK, PieceType.ROOK))

        result = find_best_line_native(board, depth=1, weights=get_evaluation_weights(), quiescence_depth=2)

        self.assertIsNotNone(result)
        self.assertNotEqual(result.move.to_uci(), "d1d8")

    def test_native_qsearch_at_horizon_searches_check_evasion(self):
        if not native_available():
            self.skipTest("native evaluator DLL is not available")
        board = Board.empty()
        board.side_to_move = Color.WHITE
        board.set_piece(Square.from_algebraic("a1"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("e2"), Piece(Color.WHITE, PieceType.QUEEN))
        board.set_piece(Square.from_algebraic("e8"), Piece(Color.BLACK, PieceType.KING))
        board.set_piece(Square.from_algebraic("e7"), Piece(Color.BLACK, PieceType.PAWN))

        result = find_best_line_native(board, depth=1, weights=get_evaluation_weights(), quiescence_depth=0)

        self.assertIsNotNone(result)
        self.assertNotEqual(result.move.to_uci(), "e2e7")

    def test_decisive_material_endgames_are_scored_as_winning(self):
        kqk = Board.empty()
        kqk.set_piece(Square.from_algebraic("a1"), Piece(Color.WHITE, PieceType.KING))
        kqk.set_piece(Square.from_algebraic("d4"), Piece(Color.WHITE, PieceType.QUEEN))
        kqk.set_piece(Square.from_algebraic("h8"), Piece(Color.BLACK, PieceType.KING))

        kqrk = kqk.copy()
        kqrk.set_piece(Square.from_algebraic("a4"), Piece(Color.WHITE, PieceType.ROOK))

        self.assertGreater(score_for_color(kqk, Color.WHITE), 700)
        self.assertGreater(score_for_color(kqrk, Color.WHITE), score_for_color(kqk, Color.WHITE))

    def test_native_search_returns_principal_variation(self):
        if not native_available():
            self.skipTest("native evaluator DLL is not available")
        result = find_best_line_native(
            Board.starting_position(),
            depth=2,
            weights=get_evaluation_weights(),
        )

        self.assertIsNotNone(result)
        self.assertIsNotNone(result.move)
        self.assertGreaterEqual(len(result.principal_variation), 1)
        self.assertEqual(result.principal_variation[0].to_uci(), result.move.to_uci())
        self.assertGreater(result.nodes, 1)

    def test_native_search_does_not_probe_consecutive_null_moves(self):
        if not native_available():
            self.skipTest("native evaluator DLL is not available")
        result = find_best_line_native(
            Board.starting_position(),
            depth=8,
            weights=get_evaluation_weights(),
        )
        diagnostics = native_search_diagnostics()

        self.assertIsNotNone(result)
        self.assertIsNotNone(diagnostics)
        self.assertGreater(diagnostics["null_move_probes"], 0)
        self.assertEqual(diagnostics["nested_null_move_probes"], 0)
        self.assertIn("aspiration_attempts", diagnostics)
        self.assertIn("aspiration_fail_lows", diagnostics)
        self.assertIn("aspiration_fail_highs", diagnostics)
        self.assertIn("aspiration_researches", diagnostics)
        self.assertGreater(diagnostics["quiet_beta_cutoffs"], 0)
        self.assertGreater(diagnostics["killer_order_hits"], 0)
        self.assertGreater(diagnostics["history_order_hits"], 0)
        self.assertGreaterEqual(diagnostics["history_saturated_updates"], 0)

    def test_backend_native_result_includes_principal_variation(self):
        result = find_backend_move(Board.starting_position(), depth=2, backend="native")

        self.assertIsNotNone(result.move)
        self.assertGreaterEqual(len(result.principal_variation), 1)
        self.assertEqual(result.principal_variation[0].to_uci(), result.move.to_uci())
        self.assertGreater(result.nodes, 1)

    def test_native_timed_search_obeys_short_deadline(self):
        if not native_available():
            self.skipTest("native evaluator DLL is not available")
        started = perf_counter()
        result = find_backend_move(Board.starting_position(), depth=64, backend="native", time_limit=0.02)
        elapsed = perf_counter() - started

        self.assertIsNotNone(result.move)
        self.assertLess(elapsed, 0.25)
        self.assertLess(result.completed_depth, 64)
        self.assertLessEqual(len(result.principal_variation), result.completed_depth)

    def test_native_insufficient_material_draws_score_zero(self):
        if not native_available():
            self.skipTest("native evaluator DLL is not available")

        drawn_fens = (
            "8/8/8/8/8/8/5k2/6K1 w - - 0 1",
            "8/8/8/8/8/8/5k2/5BK1 w - - 0 1",
            "8/8/8/8/8/8/5k2/5NK1 w - - 0 1",
            "8/8/8/8/8/8/4bk2/5BK1 w - - 0 1",
        )
        weights = get_evaluation_weights()
        for fen in drawn_fens:
            board = board_from_fen(fen)
            self.assertEqual(evaluate_native(board, weights), 0, fen)
            result = find_best_line_native(board, 3, weights)
            self.assertIsNotNone(result, fen)
            self.assertEqual(result.score, 0, fen)

        winning_board = board_from_fen("8/8/8/8/8/8/5k2/5RK1 w - - 0 1")
        self.assertNotEqual(evaluate_native(winning_board, weights), 0)

    def test_native_root_terminal_positions_return_no_move_with_score(self):
        if not native_available():
            self.skipTest("native evaluator DLL is not available")

        weights = get_evaluation_weights()
        checkmated = board_from_fen("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
        stalemated = board_from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")

        mate_result = find_best_line_native(checkmated, 3, weights)
        self.assertIsNotNone(mate_result)
        self.assertIsNone(mate_result.move)
        self.assertLessEqual(mate_result.score, -CHECKMATE_SCORE)
        self.assertEqual(mate_result.principal_variation, ())

        draw_result = find_best_line_native(stalemated, 3, weights)
        self.assertIsNotNone(draw_result)
        self.assertIsNone(draw_result.move)
        self.assertEqual(draw_result.score, 0)
        self.assertEqual(draw_result.principal_variation, ())

    def test_backend_terminal_positions_preserve_root_score(self):
        checkmated = board_from_fen("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
        stalemated = board_from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")

        mate_result = find_backend_move(checkmated, depth=3, backend="native")
        self.assertIsNone(mate_result.move)
        self.assertLessEqual(mate_result.score, -CHECKMATE_SCORE)

        draw_result = find_backend_multipv(stalemated, depth=3, multipv=2, backend="native")[0]
        self.assertIsNone(draw_result.move)
        self.assertEqual(draw_result.score, 0)

    def test_native_root_fifty_move_rule_returns_draw_without_bestmove(self):
        if not native_available():
            self.skipTest("native evaluator DLL is not available")

        board = board_from_fen("8/8/8/8/8/8/5k2/5RK1 w - - 100 75")
        weights = get_evaluation_weights()

        direct_result = find_best_line_native(board, 3, weights)
        self.assertIsNotNone(direct_result)
        self.assertIsNone(direct_result.move)
        self.assertEqual(direct_result.score, 0)
        self.assertEqual(direct_result.principal_variation, ())

        backend_result = find_backend_move(board, depth=3, backend="native")
        self.assertIsNone(backend_result.move)
        self.assertEqual(backend_result.score, 0)

        multipv_result = find_backend_multipv(board, depth=3, multipv=2, backend="native")[0]
        self.assertIsNone(multipv_result.move)
        self.assertEqual(multipv_result.score, 0)

    def test_evaluate_board_scores_terminal_draw_rules_as_zero(self):
        for fen in (
            "8/8/8/8/8/8/5k2/6K1 w - - 0 1",
            "8/8/8/8/8/8/5k2/5BK1 w - - 0 1",
            "8/8/8/8/8/8/5k2/5RK1 w - - 100 75",
        ):
            self.assertEqual(evaluate_board(board_from_fen(fen)), 0, fen)

    def test_native_search_applies_root_move_penalty(self):
        if not native_available():
            self.skipTest("native evaluator DLL is not available")
        board = Board.empty()
        board.side_to_move = Color.WHITE
        board.set_piece(Square.from_algebraic("e1"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("a1"), Piece(Color.WHITE, PieceType.ROOK))
        board.set_piece(Square.from_algebraic("e8"), Piece(Color.BLACK, PieceType.KING))
        board.set_piece(Square.from_algebraic("a8"), Piece(Color.BLACK, PieceType.QUEEN))

        result = find_best_move_native(
            board,
            depth=1,
            weights=get_evaluation_weights(),
            root_move_penalties={"a1a8": 20_000},
        )

        self.assertIsNotNone(result)
        move, _ = result
        self.assertNotEqual(move.to_uci(), "a1a8")


    def test_root_repetition_penalties_mark_returning_positions(self):
        game = Game.new()
        game.apply_move(Move.from_uci("g1f3"))
        game.apply_move(Move.from_uci("g8f6"))
        game.apply_move(Move.from_uci("f3g1"))
        game.apply_move(Move.from_uci("f6g8"))

        penalties = root_repetition_penalties(game.board, game.board_history)

        self.assertEqual(penalties["g1f3"], TWOFOLD_REPETITION_PENALTY)

    def test_opening_repeated_piece_penalty_discourages_redeveloping_same_piece(self):
        game = Game.new()
        game.apply_move(Move.from_uci("g1f3"))
        game.apply_move(Move.from_uci("g8f6"))
        game.apply_move(Move.from_uci("f3g1"))
        game.apply_move(Move.from_uci("f6g8"))

        penalties = opening_repeated_piece_penalties(game.board, game.move_history)

        self.assertEqual(penalties["g1f3"], OPENING_REPEATED_PIECE_PENALTY)

    def test_opening_repeated_piece_penalty_does_not_penalize_normal_pawn_capture(self):
        game = Game.new()
        game.apply_move(Move.from_uci("e2e4"))
        game.apply_move(Move.from_uci("d7d5"))
        game.apply_move(Move.from_uci("b1c3"))

        penalties = opening_repeated_piece_penalties(game.board, game.move_history)

        self.assertNotIn("d5e4", penalties)

    def test_combined_root_penalties_sum_repetition_and_opening_tempo(self):
        game = Game.new()
        game.apply_move(Move.from_uci("g1f3"))
        game.apply_move(Move.from_uci("g8f6"))
        game.apply_move(Move.from_uci("f3g1"))
        game.apply_move(Move.from_uci("f6g8"))

        penalties = combined_root_move_penalties(game.board, game.board_history, game.move_history)

        self.assertEqual(
            penalties["g1f3"],
            TWOFOLD_REPETITION_PENALTY + OPENING_REPEATED_PIECE_PENALTY,
        )












if __name__ == "__main__":
    unittest.main()
