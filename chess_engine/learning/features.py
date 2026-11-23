"""Feature extraction for lightweight evaluation learning.

Reads the same raw terms the native evaluator actually scores (via
evaluate_native_breakdown), rather than a separate Python reimplementation --
training on a divergent set of formulas from what the engine plays with
made the learned weights fit the wrong quantities.
"""

from __future__ import annotations

from dataclasses import dataclass

from chess_engine.ai.evaluator import EvaluationWeights
from chess_engine.chess_core.board import Board
from chess_engine.chess_core.fen import board_from_fen
from chess_engine.native.native_eval import evaluate_native_breakdown

_RAW_WEIGHTS = EvaluationWeights()


@dataclass(frozen=True, slots=True)
class PositionFeatures:
    """Handcrafted features measured from White's point of view."""

    material: int
    position: int
    mobility: int
    endgame: int
    pawn_structure: int
    bishop_pair: int
    rook_activity: int
    king_safety: int
    passed_pawn: int
    endgame_pressure: int
    line_activity: int
    attack_momentum: int


def extract_position_features(board: Board) -> PositionFeatures:
    """Extract the current handcrafted evaluation terms from a board."""
    breakdown = evaluate_native_breakdown(board, _RAW_WEIGHTS)
    if breakdown is None:
        raise RuntimeError(
            "native evaluator unavailable -- learning features require the native "
            "breakdown export so training measures the same terms the engine scores"
        )
    return PositionFeatures(
        material=breakdown["material"],
        position=breakdown["pst"],
        mobility=breakdown["mobility"],
        endgame=breakdown["endgame"],
        pawn_structure=breakdown["pawn_structure"],
        bishop_pair=breakdown["bishop_pair"],
        rook_activity=breakdown["rook_activity"],
        king_safety=breakdown["king_safety"],
        passed_pawn=breakdown["passed_pawn_raw"],
        endgame_pressure=breakdown["endgame_pressure_raw"],
        line_activity=breakdown["line_activity"],
        attack_momentum=breakdown["attack_momentum"],
    )


def features_from_fen(fen: str) -> PositionFeatures:
    """Extract features from a full FEN string."""
    return extract_position_features(board_from_fen(fen))
