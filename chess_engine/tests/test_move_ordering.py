import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chess_engine.ai.move_ordering import MoveStage, SearchHeuristics, order_moves, score_move, staged_moves, static_exchange_score
from chess_engine.chess_core.board import Board
from chess_engine.chess_core.enums import Color, PieceType
from chess_engine.chess_core.move import Move
from chess_engine.chess_core.piece import Piece
from chess_engine.chess_core.square import Square
from chess_engine.native.native_eval import static_exchange_score_native


class MoveOrderingTest(unittest.TestCase):
    def test_mvv_lva_prefers_capturing_more_valuable_piece(self):
        board = Board.empty()
        board.side_to_move = Color.WHITE
        board.set_piece(Square.from_algebraic("e1"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("e8"), Piece(Color.BLACK, PieceType.KING))
        board.set_piece(Square.from_algebraic("d4"), Piece(Color.WHITE, PieceType.PAWN))
        board.set_piece(Square.from_algebraic("c5"), Piece(Color.BLACK, PieceType.QUEEN))
        board.set_piece(Square.from_algebraic("e5"), Piece(Color.BLACK, PieceType.PAWN))
        capture_queen = Move.from_uci("d4c5")
        capture_pawn = Move.from_uci("d4e5")

        ordered = order_moves(board, [capture_pawn, capture_queen])

        self.assertEqual(ordered[0], capture_queen)
        self.assertGreater(score_move(board, capture_queen), score_move(board, capture_pawn))

    def test_staged_move_picker_places_tt_then_good_capture_before_quiets(self):
        board = Board.empty()
        board.side_to_move = Color.WHITE
        board.set_piece(Square.from_algebraic("e1"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("e8"), Piece(Color.BLACK, PieceType.KING))
        board.set_piece(Square.from_algebraic("d4"), Piece(Color.WHITE, PieceType.PAWN))
        board.set_piece(Square.from_algebraic("c5"), Piece(Color.BLACK, PieceType.QUEEN))
        quiet = Move.from_uci("e1f1")
        capture = Move.from_uci("d4c5")
        preferred = Move.from_uci("e1d1")

        entries = staged_moves(board, [quiet, capture, preferred], preferred_move=preferred)

        self.assertEqual(entries[0].move, preferred)
        self.assertEqual(entries[0].stage, MoveStage.TT)
        self.assertEqual(entries[1].move, capture)
        self.assertEqual(entries[1].stage, MoveStage.GOOD_CAPTURE)

    def test_static_exchange_score_accounts_for_legal_recapture(self):
        board = Board.empty()
        board.side_to_move = Color.WHITE
        board.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        board.set_piece(Square.from_algebraic("c4"), Piece(Color.WHITE, PieceType.BISHOP))
        board.set_piece(Square.from_algebraic("e6"), Piece(Color.BLACK, PieceType.KNIGHT))
        board.set_piece(Square.from_algebraic("f7"), Piece(Color.BLACK, PieceType.PAWN))

        self.assertLess(static_exchange_score(board, Move.from_uci("c4e6")), 0)

    def test_static_exchange_score_accounts_for_revealed_xray_recapture(self):
        board = Board.empty()
        board.side_to_move = Color.WHITE
        board.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        board.set_piece(Square.from_algebraic("a1"), Piece(Color.WHITE, PieceType.ROOK))
        board.set_piece(Square.from_algebraic("a7"), Piece(Color.BLACK, PieceType.BISHOP))
        board.set_piece(Square.from_algebraic("a8"), Piece(Color.BLACK, PieceType.ROOK))

        self.assertLess(static_exchange_score(board, Move.from_uci("a1a7")), 0)

    def test_static_exchange_score_ignores_illegal_pinned_recapture(self):
        board = Board.empty()
        board.side_to_move = Color.WHITE
        board.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("e8"), Piece(Color.BLACK, PieceType.KING))
        board.set_piece(Square.from_algebraic("e1"), Piece(Color.WHITE, PieceType.ROOK))
        board.set_piece(Square.from_algebraic("g4"), Piece(Color.WHITE, PieceType.KNIGHT))
        board.set_piece(Square.from_algebraic("f6"), Piece(Color.BLACK, PieceType.KNIGHT))
        board.set_piece(Square.from_algebraic("e7"), Piece(Color.BLACK, PieceType.BISHOP))

        self.assertGreater(static_exchange_score(board, Move.from_uci("g4f6")), 250)

    def test_native_see_matches_python_for_recapture_xray_and_pin_cases(self):
        cases: list[tuple[Board, Move]] = []

        recapture = Board.empty()
        recapture.side_to_move = Color.WHITE
        recapture.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        recapture.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        recapture.set_piece(Square.from_algebraic("c4"), Piece(Color.WHITE, PieceType.BISHOP))
        recapture.set_piece(Square.from_algebraic("e6"), Piece(Color.BLACK, PieceType.KNIGHT))
        recapture.set_piece(Square.from_algebraic("f7"), Piece(Color.BLACK, PieceType.PAWN))
        cases.append((recapture, Move.from_uci("c4e6")))

        xray = Board.empty()
        xray.side_to_move = Color.WHITE
        xray.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        xray.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        xray.set_piece(Square.from_algebraic("a1"), Piece(Color.WHITE, PieceType.ROOK))
        xray.set_piece(Square.from_algebraic("a7"), Piece(Color.BLACK, PieceType.BISHOP))
        xray.set_piece(Square.from_algebraic("a8"), Piece(Color.BLACK, PieceType.ROOK))
        cases.append((xray, Move.from_uci("a1a7")))

        pinned = Board.empty()
        pinned.side_to_move = Color.WHITE
        pinned.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        pinned.set_piece(Square.from_algebraic("e8"), Piece(Color.BLACK, PieceType.KING))
        pinned.set_piece(Square.from_algebraic("e1"), Piece(Color.WHITE, PieceType.ROOK))
        pinned.set_piece(Square.from_algebraic("g4"), Piece(Color.WHITE, PieceType.KNIGHT))
        pinned.set_piece(Square.from_algebraic("f6"), Piece(Color.BLACK, PieceType.KNIGHT))
        pinned.set_piece(Square.from_algebraic("e7"), Piece(Color.BLACK, PieceType.BISHOP))
        cases.append((pinned, Move.from_uci("g4f6")))

        for board, move in cases:
            with self.subTest(move=move.to_uci()):
                self.assertEqual(static_exchange_score_native(board, move), static_exchange_score(board, move))

    def test_promotion_is_ordered_before_quiet_move(self):
        board = Board.empty()
        board.side_to_move = Color.WHITE
        board.set_piece(Square.from_algebraic("e1"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("e8"), Piece(Color.BLACK, PieceType.KING))
        board.set_piece(Square.from_algebraic("a7"), Piece(Color.WHITE, PieceType.PAWN))
        board.set_piece(Square.from_algebraic("h2"), Piece(Color.WHITE, PieceType.PAWN))
        promote = Move.from_uci("a7a8q")
        quiet = Move.from_uci("h2h3")

        ordered = order_moves(board, [quiet, promote])

        self.assertEqual(ordered[0], promote)

    def test_queen_promotion_is_ordered_before_underpromotion(self):
        board = Board.empty()
        board.side_to_move = Color.WHITE
        board.set_piece(Square.from_algebraic("e1"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("e8"), Piece(Color.BLACK, PieceType.KING))
        board.set_piece(Square.from_algebraic("a7"), Piece(Color.WHITE, PieceType.PAWN))
        queen = Move.from_uci("a7a8q")
        rook = Move.from_uci("a7a8r")

        ordered = order_moves(board, [rook, queen])

        self.assertEqual(ordered[0], queen)

    def test_checking_move_is_ordered_before_quiet_move(self):
        board = Board.empty()
        board.side_to_move = Color.WHITE
        board.set_piece(Square.from_algebraic("e1"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("e8"), Piece(Color.BLACK, PieceType.KING))
        board.set_piece(Square.from_algebraic("d1"), Piece(Color.WHITE, PieceType.QUEEN))
        checking_move = Move.from_uci("d1a4")
        quiet = Move.from_uci("d1d2")

        ordered = order_moves(board, [quiet, checking_move])

        self.assertEqual(ordered[0], checking_move)

    def test_killer_move_boosts_quiet_move_ordering(self):
        board = Board.empty()
        board.side_to_move = Color.WHITE
        board.set_piece(Square.from_algebraic("e1"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("e8"), Piece(Color.BLACK, PieceType.KING))
        board.set_piece(Square.from_algebraic("a1"), Piece(Color.WHITE, PieceType.ROOK))
        board.set_piece(Square.from_algebraic("h1"), Piece(Color.WHITE, PieceType.ROOK))
        quiet_a = Move.from_uci("a1a2")
        quiet_h = Move.from_uci("h1h2")
        heuristics = SearchHeuristics()
        heuristics.record_cutoff(quiet_h, depth=3, ply=0, is_quiet=True)

        ordered = order_moves(board, [quiet_a, quiet_h], heuristics=heuristics, ply=0)

        self.assertEqual(ordered[0], quiet_h)

    def test_preferred_transposition_move_is_ordered_first(self):
        board = Board.empty()
        board.side_to_move = Color.WHITE
        board.set_piece(Square.from_algebraic("e1"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("e8"), Piece(Color.BLACK, PieceType.KING))
        board.set_piece(Square.from_algebraic("a1"), Piece(Color.WHITE, PieceType.ROOK))
        quiet_a = Move.from_uci("a1a2")
        quiet_b = Move.from_uci("a1b1")

        ordered = order_moves(board, [quiet_a, quiet_b], preferred_move=quiet_b)

        self.assertEqual(ordered[0], quiet_b)

    def test_hanging_major_piece_is_ordered_after_safer_quiet_move(self):
        board = Board.empty()
        board.side_to_move = Color.WHITE
        board.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("d1"), Piece(Color.WHITE, PieceType.QUEEN))
        board.set_piece(Square.from_algebraic("a1"), Piece(Color.WHITE, PieceType.ROOK))
        board.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        board.set_piece(Square.from_algebraic("e8"), Piece(Color.BLACK, PieceType.ROOK))
        risky_queen = Move.from_uci("d1e2")
        safe_rook = Move.from_uci("a1a2")

        ordered = order_moves(board, [risky_queen, safe_rook])

        self.assertEqual(ordered[0], safe_rook)

    def test_fork_candidate_is_ordered_before_plain_quiet_move(self):
        board = Board.empty()
        board.side_to_move = Color.WHITE
        board.set_piece(Square.from_algebraic("a1"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("h8"), Piece(Color.BLACK, PieceType.KING))
        board.set_piece(Square.from_algebraic("e2"), Piece(Color.WHITE, PieceType.KNIGHT))
        board.set_piece(Square.from_algebraic("f5"), Piece(Color.BLACK, PieceType.QUEEN))
        board.set_piece(Square.from_algebraic("f3"), Piece(Color.BLACK, PieceType.ROOK))
        fork = Move.from_uci("e2d4")
        quiet = Move.from_uci("e2g1")

        ordered = order_moves(board, [quiet, fork])

        self.assertEqual(ordered[0], fork)
        self.assertGreater(score_move(board, fork), score_move(board, quiet))

    def test_king_fork_candidate_gets_extra_priority(self):
        board = Board.empty()
        board.side_to_move = Color.WHITE
        board.set_piece(Square.from_algebraic("a1"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("f5"), Piece(Color.BLACK, PieceType.KING))
        board.set_piece(Square.from_algebraic("e2"), Piece(Color.WHITE, PieceType.KNIGHT))
        board.set_piece(Square.from_algebraic("f3"), Piece(Color.BLACK, PieceType.ROOK))
        fork = Move.from_uci("e2d4")
        quiet = Move.from_uci("e2g1")

        self.assertGreater(score_move(board, fork), score_move(board, quiet) + 1_000)

    def test_pin_creation_candidate_is_ordered_before_plain_quiet_move(self):
        board = Board.empty()
        board.side_to_move = Color.WHITE
        board.set_piece(Square.from_algebraic("a1"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("e8"), Piece(Color.BLACK, PieceType.KING))
        board.set_piece(Square.from_algebraic("c4"), Piece(Color.WHITE, PieceType.BISHOP))
        board.set_piece(Square.from_algebraic("c6"), Piece(Color.BLACK, PieceType.KNIGHT))
        pin = Move.from_uci("c4b5")
        quiet = Move.from_uci("c4e2")

        ordered = order_moves(board, [quiet, pin])

        self.assertEqual(ordered[0], pin)
        self.assertGreater(score_move(board, pin), score_move(board, quiet))

    def test_tactical_pressure_is_limited_after_upper_search_plies(self):
        board = Board.empty()
        board.side_to_move = Color.WHITE
        board.set_piece(Square.from_algebraic("a1"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("h8"), Piece(Color.BLACK, PieceType.KING))
        board.set_piece(Square.from_algebraic("e2"), Piece(Color.WHITE, PieceType.KNIGHT))
        board.set_piece(Square.from_algebraic("f5"), Piece(Color.BLACK, PieceType.QUEEN))
        board.set_piece(Square.from_algebraic("f3"), Piece(Color.BLACK, PieceType.ROOK))
        fork = Move.from_uci("e2d4")

        self.assertGreater(score_move(board, fork, ply=0), score_move(board, fork, ply=3))

    def test_removing_defender_capture_gets_ordering_priority(self):
        board = Board.empty()
        board.side_to_move = Color.WHITE
        board.set_piece(Square.from_algebraic("a1"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("h8"), Piece(Color.BLACK, PieceType.KING))
        board.set_piece(Square.from_algebraic("a4"), Piece(Color.WHITE, PieceType.ROOK))
        board.set_piece(Square.from_algebraic("a7"), Piece(Color.BLACK, PieceType.BISHOP))
        board.set_piece(Square.from_algebraic("d4"), Piece(Color.BLACK, PieceType.QUEEN))
        remove_defender = Move.from_uci("a4a7")
        quiet = Move.from_uci("a4a5")

        self.assertGreater(score_move(board, remove_defender), score_move(board, quiet))

    def test_pawn_break_candidate_is_ordered_before_plain_pawn_push(self):
        board = Board.empty()
        board.side_to_move = Color.WHITE
        board.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("g8"), Piece(Color.BLACK, PieceType.KING))
        board.set_piece(Square.from_algebraic("e4"), Piece(Color.WHITE, PieceType.PAWN))
        board.set_piece(Square.from_algebraic("a2"), Piece(Color.WHITE, PieceType.PAWN))
        board.set_piece(Square.from_algebraic("d5"), Piece(Color.BLACK, PieceType.PAWN))
        pawn_break = Move.from_uci("e4e5")
        quiet = Move.from_uci("a2a3")

        ordered = order_moves(board, [quiet, pawn_break])

        self.assertEqual(ordered[0], pawn_break)
        self.assertGreater(score_move(board, pawn_break), score_move(board, quiet))


if __name__ == "__main__":
    unittest.main()
