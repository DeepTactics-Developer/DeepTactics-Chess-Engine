"""Board evaluation functions.

Scores are from White's point of view:

- positive means White is better
- negative means Black is better

The material evaluator is intentionally small and easy to extend. Positional
tables, mobility, endgame plans, and king safety can be added here later
without changing the search API.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from chess_engine.chess_core.board import Board
from chess_engine.chess_core.enums import Color, GameStatus, PieceType
from chess_engine.chess_core.game import has_insufficient_material
from chess_engine.chess_core.rules import generate_legal_moves, is_checkmate, is_square_attacked, is_stalemate
from chess_engine.chess_core.square import Square
from chess_engine.native.native_eval import evaluate_accumulator_nnue_binary_native, evaluate_native, native_available, native_status
from chess_engine.nnue.model import LinearNNUEModel, blend_evaluations, evaluate_linear_nnue


PIECE_VALUES = {
    PieceType.PAWN: 100,
    PieceType.KNIGHT: 320,
    PieceType.BISHOP: 330,
    PieceType.ROOK: 500,
    PieceType.QUEEN: 900,
    PieceType.KING: 0,
}

CHECKMATE_SCORE = 100_000
MOBILITY_WEIGHT = 2
ENDGAME_NON_PAWN_MATERIAL_THRESHOLD = 2_200


@dataclass(frozen=True, slots=True)
class EvaluationWeights:
    """Tunable weights applied to the handcrafted evaluation terms."""

    material: float = 1.0
    position: float = 1.0
    mobility: float = 1.0
    endgame: float = 1.0
    pawn_structure: float = 1.0
    bishop_pair: float = 1.0
    rook_activity: float = 1.0
    king_safety: float = 1.0
    passed_pawn: float = 1.0
    endgame_pressure: float = 1.0
    line_activity: float = 1.0
    attack_momentum: float = 1.0


ACTIVE_WEIGHTS = EvaluationWeights()
USE_NATIVE_EVALUATOR = native_available()
ACTIVE_LINEAR_NNUE_MODEL: LinearNNUEModel | None = None
ACTIVE_ACCUMULATOR_NNUE_PATH: Path | None = None
LINEAR_NNUE_MODE = "off"
LINEAR_NNUE_BLEND_WEIGHT = 0.35

PIECE_SQUARE_TABLES = {
    PieceType.PAWN: (
        (0, 0, 0, 0, 0, 0, 0, 0),
        (5, 10, 10, -20, -20, 10, 10, 5),
        (5, -5, -10, 0, 0, -10, -5, 5),
        (0, 0, 0, 20, 20, 0, 0, 0),
        (5, 5, 10, 25, 25, 10, 5, 5),
        (10, 10, 20, 30, 30, 20, 10, 10),
        (50, 50, 50, 50, 50, 50, 50, 50),
        (0, 0, 0, 0, 0, 0, 0, 0),
    ),
    PieceType.KNIGHT: (
        (-50, -40, -30, -30, -30, -30, -40, -50),
        (-40, -20, 0, 5, 5, 0, -20, -40),
        (-30, 5, 10, 15, 15, 10, 5, -30),
        (-30, 0, 15, 20, 20, 15, 0, -30),
        (-30, 5, 15, 20, 20, 15, 5, -30),
        (-30, 0, 10, 15, 15, 10, 0, -30),
        (-40, -20, 0, 0, 0, 0, -20, -40),
        (-50, -40, -30, -30, -30, -30, -40, -50),
    ),
    PieceType.BISHOP: (
        (-20, -10, -10, -10, -10, -10, -10, -20),
        (-10, 5, 0, 0, 0, 0, 5, -10),
        (-10, 10, 10, 10, 10, 10, 10, -10),
        (-10, 0, 10, 10, 10, 10, 0, -10),
        (-10, 5, 5, 10, 10, 5, 5, -10),
        (-10, 0, 5, 10, 10, 5, 0, -10),
        (-10, 0, 0, 0, 0, 0, 0, -10),
        (-20, -10, -10, -10, -10, -10, -10, -20),
    ),
    PieceType.ROOK: (
        (0, 0, 5, 10, 10, 5, 0, 0),
        (-5, 0, 0, 0, 0, 0, 0, -5),
        (-5, 0, 0, 0, 0, 0, 0, -5),
        (-5, 0, 0, 0, 0, 0, 0, -5),
        (-5, 0, 0, 0, 0, 0, 0, -5),
        (-5, 0, 0, 0, 0, 0, 0, -5),
        (5, 10, 10, 10, 10, 10, 10, 5),
        (0, 0, 0, 5, 5, 0, 0, 0),
    ),
    PieceType.QUEEN: (
        (-20, -10, -10, -5, -5, -10, -10, -20),
        (-10, 0, 5, 0, 0, 0, 0, -10),
        (-10, 5, 5, 5, 5, 5, 0, -10),
        (0, 0, 5, 5, 5, 5, 0, -5),
        (-5, 0, 5, 5, 5, 5, 0, -5),
        (-10, 0, 5, 5, 5, 5, 0, -10),
        (-10, 0, 0, 0, 0, 0, 0, -10),
        (-20, -10, -10, -5, -5, -10, -10, -20),
    ),
    PieceType.KING: (
        (20, 30, 10, 0, 0, 10, 30, 20),
        (20, 20, 0, 0, 0, 0, 20, 20),
        (-10, -20, -20, -20, -20, -20, -20, -10),
        (-20, -30, -30, -40, -40, -30, -30, -20),
        (-30, -40, -40, -50, -50, -40, -40, -30),
        (-30, -40, -40, -50, -50, -40, -40, -30),
        (-30, -40, -40, -50, -50, -40, -40, -30),
        (-30, -40, -40, -50, -50, -40, -40, -30),
    ),
}


def evaluate_material(board: Board) -> int:
    """Return a material-only score from White's perspective."""
    score = 0

    for _, piece in board.pieces():
        value = PIECE_VALUES[piece.piece_type]
        score += value if piece.color is Color.WHITE else -value

    return score


def evaluate_board(board: Board) -> int:
    """Return a board score from White's perspective."""
    status = _status_for_side_to_move(board)
    if status is GameStatus.CHECKMATE:
        return -CHECKMATE_SCORE if board.side_to_move is Color.WHITE else CHECKMATE_SCORE
    if status is not GameStatus.ONGOING:
        return 0

    if USE_NATIVE_EVALUATOR and LINEAR_NNUE_MODE != "linear":
        native_score = evaluate_native(board, ACTIVE_WEIGHTS)
        if native_score is not None:
            return _apply_linear_nnue(board, native_score + _native_python_overlay_score(board))

    phase = _endgame_phase(board)
    line_activity = _line_activity_score(board)
    attack_momentum = _attack_momentum_score(board)
    opening_score = (
        evaluate_position(board)
        + evaluate_strategic_knowledge(board)
        - line_activity
        - attack_momentum
        + _opening_development_score(board)
        + _central_control_score(board)
        + _king_opening_safety_score(board)
    )
    endgame_score = evaluate_endgame(board)
    tapered_position = opening_score * (1.0 - phase) + endgame_score * phase
    handcrafted_score = int(
        evaluate_material(board) * ACTIVE_WEIGHTS.material
        + tapered_position * ACTIVE_WEIGHTS.position
        + line_activity * (1.0 - phase) * ACTIVE_WEIGHTS.line_activity
        + attack_momentum * (1.0 - phase) * ACTIVE_WEIGHTS.attack_momentum
        + evaluate_mobility(board) * ACTIVE_WEIGHTS.mobility
        + _threat_score(board) * ACTIVE_WEIGHTS.position
    )
    return _apply_linear_nnue(board, handcrafted_score)


def evaluate_position(board: Board) -> int:
    """Return a piece-square-table score from White's perspective."""
    score = 0

    for square, piece in board.pieces():
        value = _piece_square_value(piece.piece_type, square, piece.color)
        score += value if piece.color is Color.WHITE else -value

    return score


def evaluate_mobility(board: Board) -> int:
    """Return a piece-aware activity score from White's perspective.

    Minor-piece activity matters most in the opening, rook and queen activity
    matter more once lines open, and early queen wandering is deliberately
    muted so the engine does not overrate flashy but premature queen moves.
    """

    phase = _endgame_phase(board)
    return _mobility_for_color(board, Color.WHITE, phase) - _mobility_for_color(board, Color.BLACK, phase)


def _mobility_for_color(board: Board, color: Color, phase: float) -> int:
    score = 0
    for move in generate_legal_moves(board, color):
        piece = board.get_piece(move.from_square)
        if piece is None:
            continue
        weight = _mobility_weight_for_piece(piece.piece_type, phase)
        if weight <= 0:
            continue
        if (
            piece.piece_type not in {PieceType.PAWN, PieceType.KING}
            and is_square_attacked(board, move.to_square, color.opponent)
        ):
            weight = max(0.5, weight - 1.0)
        score += int(round(weight * MOBILITY_WEIGHT))
    return score


def _mobility_weight_for_piece(piece_type: PieceType, phase: float) -> float:
    if piece_type is PieceType.KNIGHT:
        return 3.0
    if piece_type is PieceType.BISHOP:
        return 3.0
    if piece_type is PieceType.ROOK:
        return 1.8 + phase * 0.8
    if piece_type is PieceType.QUEEN:
        return 0.6 + phase * 0.8
    if piece_type is PieceType.KING:
        return 0.2 + phase * 0.6
    return 0.3


def evaluate_strategic_knowledge(board: Board) -> int:
    """Return book-derived strategic guidance from White's perspective.

    This is the first knowledge-ingestion layer from the strategy/tactics books:
    reward stable outposts and space, while penalizing bishops trapped behind
    their own pawn structure.
    """

    return (
        _knight_outpost_score(board)
        + _bad_bishop_score(board)
        + _space_advantage_score(board)
        + _advanced_pawn_structure_score(board)
        + _weak_square_score(board)
        + _piece_quality_score(board)
        + _locked_center_minor_score(board)
        + _color_complex_score(board)
        + _line_activity_score(board)
        + _back_rank_weakness_score(board)
        + _pawn_majority_score(board)
        + _pawn_break_score(board)
        + _exchange_strategy_score(board)
        + _attack_momentum_score(board)
        + _tactical_motif_score(board)
        + _endgame_theory_score(board)
        + _piece_coordination_score(board)
    )


def evaluate_breakdown(board: Board) -> dict[str, int]:
    """Return a debug-friendly evaluation breakdown from White's perspective."""
    phase = _endgame_phase(board)
    material = evaluate_material(board)
    pst = evaluate_position(board)
    strategic = evaluate_strategic_knowledge(board)
    mobility = evaluate_mobility(board)
    endgame = evaluate_endgame(board)
    threats = _threat_score(board)
    opening_development = _opening_development_score(board)
    central_control = _central_control_score(board)
    king_opening_safety = _king_opening_safety_score(board)
    line_activity = _line_activity_score(board)
    attack_momentum = _attack_momentum_score(board)
    piece_coordination = _piece_coordination_score(board)
    tactical_motif = _tactical_motif_score(board)
    opening = opening_development + central_control + king_opening_safety
    tapered = int((pst + strategic + opening) * (1.0 - phase) + endgame * phase)
    total = int(material + tapered + mobility + threats)
    return {
        "material": material,
        "pst": pst,
        "strategic": strategic,
        "opening": opening,
        "opening_development": opening_development,
        "central_control": central_control,
        "king_opening_safety": king_opening_safety,
        "line_activity": line_activity,
        "attack_momentum": attack_momentum,
        "piece_coordination": piece_coordination,
        "tactical_motif": tactical_motif,
        "mobility": mobility,
        "endgame": endgame,
        "threats": threats,
        "tapered": tapered,
        "total": total,
    }


def evaluate_weighted_breakdown(board: Board) -> dict[str, int]:
    """Return the terms used by the Python handcrafted weighted evaluation.

    Activity terms are separately weighted so training can adjust them
    without suppressing unrelated positional knowledge.
    """
    raw = evaluate_breakdown(board)
    material = int(raw["material"] * ACTIVE_WEIGHTS.material)
    phase = _endgame_phase(board)
    activity_factor = 1.0 - phase
    tapered_without_activity = raw["tapered"] - int(
        (raw["line_activity"] + raw["attack_momentum"]) * activity_factor
    )
    tapered_position = int(tapered_without_activity * ACTIVE_WEIGHTS.position)
    line_activity = int(raw["line_activity"] * activity_factor * ACTIVE_WEIGHTS.line_activity)
    attack_momentum = int(raw["attack_momentum"] * activity_factor * ACTIVE_WEIGHTS.attack_momentum)
    mobility = int(raw["mobility"] * ACTIVE_WEIGHTS.mobility)
    threats = int(raw["threats"] * ACTIVE_WEIGHTS.position)
    return {
        "material": material,
        "tapered_position_without_activity": tapered_position,
        "line_activity": line_activity,
        "attack_momentum": attack_momentum,
        "mobility": mobility,
        "threats_via_position_weight": threats,
        "python_handcrafted_total": material + tapered_position + line_activity + attack_momentum + mobility + threats,
    }


def _native_python_overlay_score(board: Board) -> int:
    """Small Python overlay for knowledge terms not yet ported to C++.

    ``_central_control_score``, ``_opening_development_score``, and
    ``_king_opening_safety_score`` are intentionally excluded here: native's
    ``evaluate_internal`` already folds these into ``opening_fundamentals``,
    so adding them again on top of ``evaluate_native`` double-counted them.
    Only ``_threat_score`` has no native counterpart (native's
    ``loose_piece_threat_score`` only implements the hanging-piece half).
    """
    return _threat_score(board)


def evaluate_endgame(board: Board) -> int:
    """Return endgame-specific guidance from White's perspective.

    The search still decides tactics. This term gives shallow searches useful
    strategic hints in simplified positions: centralize the king, advance
    passed pawns, and use the stronger king to approach the weaker king.
    """

    phase = _endgame_phase(board)
    if phase <= 0:
        return 0

    score = 0
    score += _king_centralization_score(board) * phase
    score += _passed_pawn_score(board) * phase
    score += _passed_pawn_support_score(board) * phase
    score += _connected_passed_pawn_score(board) * phase
    score += _rook_endgame_activity_score(board) * phase
    score += _king_cutoff_score(board) * phase
    score += _winning_king_proximity_score(board) * phase
    return int(score)


def score_for_color(board: Board, color: Color) -> int:
    """Return the evaluation from ``color``'s perspective."""
    score = evaluate_board(board)
    return score if color is Color.WHITE else -score


def set_evaluation_weights(weights: EvaluationWeights) -> None:
    """Set process-wide evaluation weights used by future searches."""
    global ACTIVE_WEIGHTS
    ACTIVE_WEIGHTS = weights


def get_evaluation_weights() -> EvaluationWeights:
    """Return the active evaluation weights."""
    return ACTIVE_WEIGHTS


def reset_evaluation_weights() -> None:
    """Restore the default handcrafted evaluation weights."""
    set_evaluation_weights(EvaluationWeights())


def load_linear_nnue_model(path: str | Path, mode: str = "blend", blend_weight: float = 0.35) -> LinearNNUEModel:
    """Load and activate the temporary sparse linear NNUE probe."""
    model = LinearNNUEModel.from_json(path)
    set_linear_nnue_model(model, mode=mode, blend_weight=blend_weight)
    return model


def load_accumulator_nnue_model(path: str | Path, mode: str = "blend", blend_weight: float = 0.35) -> Path:
    """Activate a native accumulator v2 ``.dtnnue`` model."""

    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"accumulator NNUE model not found: {model_path}")
    if mode not in {"off", "linear", "blend"}:
        raise ValueError("accumulator NNUE mode must be 'off', 'linear', or 'blend'")
    global ACTIVE_LINEAR_NNUE_MODEL, ACTIVE_ACCUMULATOR_NNUE_PATH, LINEAR_NNUE_MODE, LINEAR_NNUE_BLEND_WEIGHT
    ACTIVE_LINEAR_NNUE_MODEL = None
    ACTIVE_ACCUMULATOR_NNUE_PATH = model_path if mode != "off" else None
    LINEAR_NNUE_MODE = mode
    LINEAR_NNUE_BLEND_WEIGHT = max(0.0, min(1.0, blend_weight))
    return model_path


def set_linear_nnue_model(
    model: LinearNNUEModel | None,
    *,
    mode: str = "blend",
    blend_weight: float = 0.35,
) -> None:
    """Activate or clear the temporary sparse linear NNUE probe.

    Modes:

    - ``off``: ignore NNUE.
    - ``linear``: use only the sparse linear probe.
    - ``blend``: mix handcrafted/native score with the probe score.
    """
    if mode not in {"off", "linear", "blend"}:
        raise ValueError("linear NNUE mode must be 'off', 'linear', or 'blend'")
    global ACTIVE_LINEAR_NNUE_MODEL, ACTIVE_ACCUMULATOR_NNUE_PATH, LINEAR_NNUE_MODE, LINEAR_NNUE_BLEND_WEIGHT
    ACTIVE_LINEAR_NNUE_MODEL = model if mode != "off" else None
    ACTIVE_ACCUMULATOR_NNUE_PATH = None
    LINEAR_NNUE_MODE = mode
    LINEAR_NNUE_BLEND_WEIGHT = max(0.0, min(1.0, blend_weight))


def reset_linear_nnue_model() -> None:
    """Disable the temporary sparse linear NNUE probe."""
    set_linear_nnue_model(None, mode="off")


def linear_nnue_status() -> str:
    """Return a short status string for startup logs and diagnostics."""
    if ACTIVE_LINEAR_NNUE_MODEL is None or LINEAR_NNUE_MODE == "off":
        if ACTIVE_ACCUMULATOR_NNUE_PATH is not None and LINEAR_NNUE_MODE != "off":
            return f"{LINEAR_NNUE_MODE}; accumulator={ACTIVE_ACCUMULATOR_NNUE_PATH.name}; blend={LINEAR_NNUE_BLEND_WEIGHT:.2f}"
        return "off"
    return (
        f"{LINEAR_NNUE_MODE}; features={ACTIVE_LINEAR_NNUE_MODEL.feature_count}; "
        f"weights={len(ACTIVE_LINEAR_NNUE_MODEL.weights)}; blend={LINEAR_NNUE_BLEND_WEIGHT:.2f}"
    )


def use_native_evaluator(enabled: bool) -> None:
    """Enable or disable the optional C++ evaluator for future evaluations."""
    global USE_NATIVE_EVALUATOR
    USE_NATIVE_EVALUATOR = enabled and native_available()


def native_evaluator_enabled() -> bool:
    """Return whether the C++ evaluator is currently active."""
    return USE_NATIVE_EVALUATOR


def native_evaluator_available() -> bool:
    """Return whether the C++ evaluator DLL is available."""
    return native_available()


def native_evaluator_status() -> str:
    """Return a short native evaluator status string."""
    prefix = "enabled" if USE_NATIVE_EVALUATOR else "disabled"
    return f"{prefix}; {native_status()}"


def _apply_linear_nnue(board: Board, base_score: int) -> int:
    if LINEAR_NNUE_MODE == "off":
        return base_score
    if ACTIVE_ACCUMULATOR_NNUE_PATH is not None:
        nnue_score = evaluate_accumulator_nnue_binary_native(board, ACTIVE_ACCUMULATOR_NNUE_PATH)
        if nnue_score is None:
            return base_score
        if LINEAR_NNUE_MODE == "linear":
            return nnue_score
        return blend_evaluations(base_score, nnue_score, LINEAR_NNUE_BLEND_WEIGHT)
    if ACTIVE_LINEAR_NNUE_MODEL is None:
        return base_score
    nnue_score = evaluate_linear_nnue(board, ACTIVE_LINEAR_NNUE_MODEL)
    if LINEAR_NNUE_MODE == "linear":
        return nnue_score
    return blend_evaluations(base_score, nnue_score, LINEAR_NNUE_BLEND_WEIGHT)


def save_evaluation_weights(weights: EvaluationWeights, path: str | Path) -> None:
    """Save evaluation weights as JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(weights), indent=2), encoding="utf-8")


def load_evaluation_weights(path: str | Path, apply: bool = True) -> EvaluationWeights:
    """Load evaluation weights from JSON and optionally activate them."""
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    values = {field: float(data.get(field, 1.0)) for field in EvaluationWeights.__dataclass_fields__}
    # Old native files applied these motifs inside a strategic term that was
    # subsequently scaled by ``position`` again. Preserve that effective
    # position-squared scale until a newly trained file supplies the fields.
    legacy_position = float(data.get("position", 1.0))
    values["line_activity"] = float(data.get("line_activity", legacy_position * legacy_position))
    values["attack_momentum"] = float(data.get("attack_momentum", legacy_position * legacy_position))
    weights = EvaluationWeights(**values)
    if apply:
        set_evaluation_weights(weights)
    return weights


def _status_for_side_to_move(board: Board) -> GameStatus:
    if is_checkmate(board):
        return GameStatus.CHECKMATE
    if is_stalemate(board):
        return GameStatus.STALEMATE
    if board.halfmove_clock >= 100:
        return GameStatus.FIFTY_MOVE_RULE
    if has_insufficient_material(board):
        return GameStatus.INSUFFICIENT_MATERIAL
    return GameStatus.ONGOING


def _piece_square_value(piece_type: PieceType, square: Square, color: Color) -> int:
    rank = square.rank if color is Color.WHITE else 7 - square.rank
    return PIECE_SQUARE_TABLES[piece_type][rank][square.file]


def _endgame_phase(board: Board) -> float:
    non_pawn_material = 0
    for _, piece in board.pieces():
        if piece.piece_type not in {PieceType.KING, PieceType.PAWN}:
            non_pawn_material += PIECE_VALUES[piece.piece_type]
    return max(0.0, min(1.0, 1 - non_pawn_material / ENDGAME_NON_PAWN_MATERIAL_THRESHOLD))


def _king_centralization_score(board: Board) -> int:
    white_king = _king_square(board, Color.WHITE)
    black_king = _king_square(board, Color.BLACK)
    if white_king is None or black_king is None:
        return 0
    return _central_score(white_king) - _central_score(black_king)


def _passed_pawn_score(board: Board) -> int:
    score = 0
    for square, piece in board.pieces():
        if piece.piece_type is not PieceType.PAWN:
            continue
        if not _is_passed_pawn(board, square, piece.color):
            continue
        advancement = square.rank if piece.color is Color.WHITE else 7 - square.rank
        value = 8 + advancement * advancement * 2
        direction = 1 if piece.color is Color.WHITE else -1
        front_rank = square.rank + direction
        if 0 <= front_rank < 8:
            front = Square(square.file, front_rank)
            if board.get_piece(front) is None:
                value += 8 + advancement * 2
                if not is_square_attacked(board, front, piece.color.opponent):
                    value += 8
            enemy_king = _king_square(board, piece.color.opponent)
            if enemy_king is not None and enemy_king == front:
                value -= 18
        score += value if piece.color is Color.WHITE else -value
    return score


def _passed_pawn_support_score(board: Board) -> int:
    score = 0
    for square, piece in board.pieces():
        if piece.piece_type is not PieceType.PAWN or not _is_passed_pawn(board, square, piece.color):
            continue
        promotion = Square(square.file, 7 if piece.color is Color.WHITE else 0)
        king = _king_square(board, piece.color)
        enemy_king = _king_square(board, piece.color.opponent)
        value = 0
        if king is not None:
            value += max(0, 7 - _king_distance(king, square)) * 2
            value += max(0, 7 - _king_distance(king, promotion))
        if enemy_king is not None:
            value += max(0, _king_distance(enemy_king, promotion) - _king_distance(square, promotion)) * 4
        value += _major_piece_behind_passed_pawn(board, square, piece.color) * 10
        score += value if piece.color is Color.WHITE else -value
    return score


def _connected_passed_pawn_score(board: Board) -> int:
    """Reward connected passers because they often convert without help."""
    score = 0
    for square, piece in board.pieces():
        if piece.piece_type is not PieceType.PAWN or not _is_passed_pawn(board, square, piece.color):
            continue
        partner_bonus = 0
        for file in (square.file - 1, square.file + 1):
            if not 0 <= file < 8:
                continue
            for rank_delta in (-1, 0, 1):
                rank = square.rank + rank_delta
                if not 0 <= rank < 8:
                    continue
                partner = board.get_piece(Square(file, rank))
                if (
                    partner is not None
                    and partner.color is piece.color
                    and partner.piece_type is PieceType.PAWN
                    and _is_passed_pawn(board, Square(file, rank), piece.color)
                ):
                    advancement = square.rank if piece.color is Color.WHITE else 7 - square.rank
                    partner_bonus = max(partner_bonus, 10 + advancement * 4)
        score += partner_bonus if piece.color is Color.WHITE else -partner_bonus
    return score


def _rook_endgame_activity_score(board: Board) -> int:
    rooks = {Color.WHITE: 0, Color.BLACK: 0}
    non_rook_pieces = 0
    for _, piece in board.pieces():
        if piece.piece_type is PieceType.ROOK:
            rooks[piece.color] += 1
        elif piece.piece_type not in {PieceType.KING, PieceType.PAWN}:
            non_rook_pieces += 1
    if rooks[Color.WHITE] == 0 or rooks[Color.BLACK] == 0 or non_rook_pieces > 1:
        return 0

    score = 0
    for square, piece in board.pieces():
        if piece.piece_type is not PieceType.ROOK:
            continue
        value = 0
        seventh_rank = 6 if piece.color is Color.WHITE else 1
        if square.rank == seventh_rank:
            value += 18
        if not _file_has_pawn(board, square.file, piece.color):
            value += 8
        if _rook_behind_passed_pawn(board, square, piece.color):
            value += 14
        score += value if piece.color is Color.WHITE else -value
    return score


def _king_cutoff_score(board: Board) -> int:
    material_score = evaluate_material(board)
    if abs(material_score) < 300:
        return 0

    stronger = Color.WHITE if material_score > 0 else Color.BLACK
    weaker = stronger.opponent
    weaker_king = _king_square(board, weaker)
    if weaker_king is None:
        return 0

    score = 0
    for square, piece in board.pieces(stronger):
        if piece.piece_type not in {PieceType.ROOK, PieceType.QUEEN}:
            continue
        controlled_escape_squares = 0
        for target in _king_ring_squares(weaker_king):
            if _piece_has_line_to_square(board, square, piece, target):
                controlled_escape_squares += 1
        value = controlled_escape_squares * (8 if piece.piece_type is PieceType.ROOK else 10)
        if _piece_has_line_to_square(board, square, piece, weaker_king):
            value += 12
        score += min(42, value)
    return score if stronger is Color.WHITE else -score


def _winning_king_proximity_score(board: Board) -> int:
    material_score = evaluate_material(board)
    if abs(material_score) < 300:
        return 0

    stronger = Color.WHITE if material_score > 0 else Color.BLACK
    weaker = stronger.opponent
    stronger_king = _king_square(board, stronger)
    weaker_king = _king_square(board, weaker)
    if stronger_king is None or weaker_king is None:
        return 0

    distance = _king_distance(stronger_king, weaker_king)
    value = max(0, 14 - distance * 2)
    return value if stronger is Color.WHITE else -value


def _central_control_score(board: Board) -> int:
    center = {
        Square.from_algebraic("d4"),
        Square.from_algebraic("e4"),
        Square.from_algebraic("d5"),
        Square.from_algebraic("e5"),
    }
    score = 0
    for square in center:
        piece = board.get_piece(square)
        if piece is not None and piece.piece_type is PieceType.PAWN:
            score += 18 if piece.color is Color.WHITE else -18
        if is_square_attacked(board, square, Color.WHITE):
            score += 7
        if is_square_attacked(board, square, Color.BLACK):
            score -= 7
    return score


def _opening_development_score(board: Board) -> int:
    if _endgame_phase(board) > 0.25:
        return 0
    home_squares = {
        Color.WHITE: {
            Square.from_algebraic("b1"),
            Square.from_algebraic("g1"),
            Square.from_algebraic("c1"),
            Square.from_algebraic("f1"),
        },
        Color.BLACK: {
            Square.from_algebraic("b8"),
            Square.from_algebraic("g8"),
            Square.from_algebraic("c8"),
            Square.from_algebraic("f8"),
        },
    }
    score = 0
    for color in (Color.WHITE, Color.BLACK):
        developed = 0
        blocked = 0
        for square in home_squares[color]:
            piece = board.get_piece(square)
            if piece is None:
                developed += 1
            elif piece.piece_type in {PieceType.KNIGHT, PieceType.BISHOP}:
                blocked += 1
        value = developed * 10 - blocked * 2
        score += value if color is Color.WHITE else -value
    return score


def _king_opening_safety_score(board: Board) -> int:
    if _endgame_phase(board) > 0.35:
        return 0
    score = 0
    for color in (Color.WHITE, Color.BLACK):
        king = _king_square(board, color)
        if king is None:
            continue
        home_rank = 0 if color is Color.WHITE else 7
        value = 0
        if king.rank == home_rank and king.file in {6, 2}:
            value += 24
        elif king.rank == home_rank and king.file == 4:
            value -= 12
        score += value if color is Color.WHITE else -value
    return score


def _threat_score(board: Board) -> int:
    score = 0
    for square, piece in board.pieces():
        if piece.piece_type is PieceType.KING:
            continue
        attacked = is_square_attacked(board, square, piece.color.opponent)
        defended = is_square_attacked(board, square, piece.color)
        if attacked and not defended:
            value = min(90, PIECE_VALUES[piece.piece_type] // 8)
            score -= value if piece.color is Color.WHITE else -value

    for color in (Color.WHITE, Color.BLACK):
        for move in generate_legal_moves(board, color):
            attacker = board.get_piece(move.from_square)
            target = board.get_piece(move.to_square)
            if attacker is None or target is None or target.color is color:
                continue
            if PIECE_VALUES[attacker.piece_type] < PIECE_VALUES[target.piece_type]:
                value = min(70, (PIECE_VALUES[target.piece_type] - PIECE_VALUES[attacker.piece_type]) // 10)
                score += value if color is Color.WHITE else -value
    return score


def _is_passed_pawn(board: Board, square: Square, color: Color) -> bool:
    direction = 1 if color is Color.WHITE else -1
    enemy = color.opponent
    rank = square.rank + direction
    while 0 <= rank < 8:
        for file in range(max(0, square.file - 1), min(7, square.file + 1) + 1):
            piece = board.get_piece(Square(file, rank))
            if piece is not None and piece.color is enemy and piece.piece_type is PieceType.PAWN:
                return False
        rank += direction
    return True


def _king_square(board: Board, color: Color) -> Square | None:
    for square, piece in board.pieces(color):
        if piece.piece_type is PieceType.KING:
            return square
    return None


def _central_score(square: Square) -> int:
    file_distance = abs(square.file - 3.5)
    rank_distance = abs(square.rank - 3.5)
    return int(28 - (file_distance + rank_distance) * 6)


def _king_distance(first: Square, second: Square) -> int:
    return max(abs(first.file - second.file), abs(first.rank - second.rank))


def _king_ring_squares(square: Square) -> list[Square]:
    squares: list[Square] = []
    for file_delta in (-1, 0, 1):
        for rank_delta in (-1, 0, 1):
            if file_delta == 0 and rank_delta == 0:
                continue
            file = square.file + file_delta
            rank = square.rank + rank_delta
            if 0 <= file < 8 and 0 <= rank < 8:
                squares.append(Square(file, rank))
    return squares


def _knight_outpost_score(board: Board) -> int:
    score = 0
    for square, piece in board.pieces():
        if piece.piece_type is not PieceType.KNIGHT:
            continue
        if square.file not in {2, 3, 4, 5}:
            continue
        advancement = square.rank if piece.color is Color.WHITE else 7 - square.rank
        if advancement < 3:
            continue
        if not _is_supported_by_pawn(board, square, piece.color):
            continue
        if _is_attacked_by_pawn(board, square, piece.color.opponent):
            continue
        value = 18 + max(0, advancement - 3) * 5 + max(0, _central_score(square)) // 4
        score += value if piece.color is Color.WHITE else -value
    return score


def _bad_bishop_score(board: Board) -> int:
    score = 0
    for square, piece in board.pieces():
        if piece.piece_type is not PieceType.BISHOP:
            continue
        same_color_pawns = 0
        central_blockers = 0
        bishop_light_square = _is_light_square(square)
        for pawn_square, pawn in board.pieces(piece.color):
            if pawn.piece_type is not PieceType.PAWN:
                continue
            if _is_light_square(pawn_square) == bishop_light_square:
                same_color_pawns += 1
                if pawn_square.file in {2, 3, 4, 5}:
                    central_blockers += 1
        penalty = max(0, same_color_pawns - 2) * 3 + central_blockers * 4
        score -= penalty if piece.color is Color.WHITE else -penalty
    return score


def _space_advantage_score(board: Board) -> int:
    score = 0
    for square, piece in board.pieces():
        if piece.piece_type is not PieceType.PAWN:
            continue
        advancement = square.rank if piece.color is Color.WHITE else 7 - square.rank
        if advancement < 3:
            continue
        value = 3
        if square.file in {2, 3, 4, 5}:
            value += 5
        if advancement >= 4:
            value += 4
        score += value if piece.color is Color.WHITE else -value
    return score


def _advanced_pawn_structure_score(board: Board) -> int:
    score = 0
    for square, piece in board.pieces():
        if piece.piece_type is not PieceType.PAWN:
            continue
        penalty = 0
        if _is_isolated_pawn(board, square, piece.color):
            penalty += 10
        if _is_backward_pawn(board, square, piece.color):
            penalty += 8
        if _is_hanging_pawn(board, square, piece.color):
            penalty += 5
        score -= penalty if piece.color is Color.WHITE else -penalty
    return score


def _weak_square_score(board: Board) -> int:
    score = 0
    for file in range(2, 6):
        for rank in range(2, 6):
            square = Square(file, rank)
            weak_for_white = not _can_be_defended_by_pawn(board, square, Color.WHITE)
            weak_for_black = not _can_be_defended_by_pawn(board, square, Color.BLACK)
            if weak_for_black and _minor_piece_controls(board, square, Color.WHITE):
                score += 8
            if weak_for_white and _minor_piece_controls(board, square, Color.BLACK):
                score -= 8
    return score


def _piece_quality_score(board: Board) -> int:
    score = 0
    for square, piece in board.pieces():
        value = 0
        if piece.piece_type is PieceType.KNIGHT:
            value += max(0, _central_score(square)) // 3
            if _is_supported_by_pawn(board, square, piece.color):
                value += 6
            if _is_attacked_by_pawn(board, square, piece.color.opponent):
                value -= 8
        elif piece.piece_type is PieceType.BISHOP:
            value += _bishop_scope(board, square, piece.color)
        else:
            continue
        score += value if piece.color is Color.WHITE else -value
    return score


def _locked_center_minor_score(board: Board) -> int:
    locked_files = _locked_center_files(board)
    if not locked_files:
        return 0

    score = 0
    for square, piece in board.pieces():
        value = 0
        if piece.piece_type is PieceType.KNIGHT:
            if square.file in {2, 3, 4, 5} and _is_supported_by_pawn(board, square, piece.color):
                advancement = square.rank if piece.color is Color.WHITE else 7 - square.rank
                value += 10 + max(0, advancement - 2) * 4 + len(locked_files) * 3
        elif piece.piece_type is PieceType.BISHOP:
            blocked = _bishop_own_pawn_blockers(board, square, piece.color)
            if blocked:
                value -= 8 + blocked * 5 + len(locked_files) * 3
        score += value if piece.color is Color.WHITE else -value
    return score


def _color_complex_score(board: Board) -> int:
    score = 0
    for color in (Color.WHITE, Color.BLACK):
        for light in (False, True):
            undefended = _central_color_complex_weakness(board, color, light)
            enemy_bishop_pressure = _enemy_bishop_color_pressure(board, color, light)
            value = min(32, undefended * enemy_bishop_pressure * 3)
            score -= value if color is Color.WHITE else -value
    return score


def _line_activity_score(board: Board) -> int:
    score = 0
    for square, piece in board.pieces():
        value = 0
        if piece.piece_type is PieceType.ROOK:
            if not _file_has_pawn(board, square.file, piece.color) and not _file_has_pawn(
                board, square.file, piece.color.opponent
            ):
                value += 14
            elif not _file_has_pawn(board, square.file, piece.color):
                value += 8
            if _rook_behind_passed_pawn(board, square, piece.color):
                value += 10
        elif piece.piece_type is PieceType.BISHOP:
            value += max(0, _bishop_scope(board, square, piece.color) - 8)
        else:
            continue
        score += value if piece.color is Color.WHITE else -value
    return score


def _back_rank_weakness_score(board: Board) -> int:
    score = 0
    for color in (Color.WHITE, Color.BLACK):
        king = _king_square(board, color)
        if king is None:
            continue
        home_rank = 0 if color is Color.WHITE else 7
        if king.rank != home_rank:
            continue
        escape_rank = king.rank + (1 if color is Color.WHITE else -1)
        if not 0 <= escape_rank < 8:
            continue
        blocked = 0
        for file in range(max(0, king.file - 1), min(7, king.file + 1) + 1):
            piece = board.get_piece(Square(file, escape_rank))
            if piece is not None and piece.color is color:
                blocked += 1
        if blocked < 2:
            continue
        pressure = _enemy_major_back_rank_pressure(board, color, king)
        penalty = blocked * 7 + pressure * 8
        score -= penalty if color is Color.WHITE else -penalty
    return score


def _exchange_strategy_score(board: Board) -> int:
    material = evaluate_material(board)
    if abs(material) < 250:
        return 0
    phase = _endgame_phase(board)
    traded_non_pawn = int(ENDGAME_NON_PAWN_MATERIAL_THRESHOLD * phase)
    value = min(60, abs(material) // 60 + traded_non_pawn // 60)
    return value if material > 0 else -value


def _attack_momentum_score(board: Board) -> int:
    white_king = _king_square(board, Color.WHITE)
    black_king = _king_square(board, Color.BLACK)
    if white_king is None or black_king is None:
        return 0
    return _king_attack_pressure(board, Color.WHITE, black_king) - _king_attack_pressure(board, Color.BLACK, white_king)


def _tactical_motif_score(board: Board) -> int:
    """Reward common book tactics that can be detected statically."""

    score = 0
    for color in (Color.WHITE, Color.BLACK):
        value = 0
        enemy_king = _king_square(board, color.opponent)
        if enemy_king is not None:
            value += _queen_king_battery_score(board, color, enemy_king)
            value += _minor_piece_fork_pressure(board, color, enemy_king)
        value += _rook_seventh_rank_score(board, color)
        value += _knight_rim_penalty(board, color)
        score += value if color is Color.WHITE else -value
    return score


def _endgame_theory_score(board: Board) -> int:
    """Add small endgame principles from the books."""

    phase = _endgame_phase(board)
    if phase < 0.45:
        return 0
    score = 0
    score += int(_opposition_score(board) * phase)
    score += int(_outside_passer_score(board) * phase)
    score += int(_bishop_vs_knight_pawn_structure_score(board) * phase)
    return score


def _piece_coordination_score(board: Board) -> int:
    """Reward pieces that cooperate instead of acting alone."""

    score = 0
    for color in (Color.WHITE, Color.BLACK):
        value = 0
        for square, piece in board.pieces(color):
            if piece.piece_type in {PieceType.KNIGHT, PieceType.BISHOP, PieceType.ROOK, PieceType.QUEEN}:
                friendly_support = any(
                    support_square != square
                    and support_piece.piece_type is not PieceType.KING
                    and is_square_attacked(board, square, color)
                    for support_square, support_piece in board.pieces(color)
                )
                if friendly_support:
                    value += 4
                if piece.piece_type is PieceType.KNIGHT and square.file in {0, 7}:
                    value -= 10
        score += value if color is Color.WHITE else -value
    return score


def _queen_king_battery_score(board: Board, color: Color, enemy_king: Square) -> int:
    value = 0
    for square, piece in board.pieces(color):
        if piece.piece_type is not PieceType.QUEEN:
            continue
        if _piece_has_line_to_square(board, square, piece, enemy_king) and _king_distance(square, enemy_king) <= 5:
            value += 18
        for support_square, support_piece in board.pieces(color):
            if support_piece.piece_type not in {PieceType.BISHOP, PieceType.ROOK}:
                continue
            if _same_line(square, support_square, enemy_king):
                value += 10
    return min(36, value)


def _minor_piece_fork_pressure(board: Board, color: Color, enemy_king: Square) -> int:
    value = 0
    for square, piece in board.pieces(color):
        if piece.piece_type is not PieceType.KNIGHT:
            continue
        attacked_high_value = 0
        for move in generate_legal_moves(board, color):
            if move.from_square != square:
                continue
            target = board.get_piece(move.to_square)
            if target is not None and target.color is color.opponent and target.piece_type in {PieceType.ROOK, PieceType.QUEEN}:
                attacked_high_value += 1
            if move.to_square in _king_ring_squares(enemy_king):
                attacked_high_value += 1
        if attacked_high_value >= 2:
            value += 22
    return min(44, value)


def _rook_seventh_rank_score(board: Board, color: Color) -> int:
    target_rank = 6 if color is Color.WHITE else 1
    enemy_home_rank = 7 if color is Color.WHITE else 0
    value = 0
    for square, piece in board.pieces(color):
        if piece.piece_type is not PieceType.ROOK or square.rank != target_rank:
            continue
        value += 18
        enemy_king = _king_square(board, color.opponent)
        if enemy_king is not None and enemy_king.rank == enemy_home_rank:
            value += 10
    return min(42, value)


def _knight_rim_penalty(board: Board, color: Color) -> int:
    penalty = 0
    for square, piece in board.pieces(color):
        if piece.piece_type is PieceType.KNIGHT and (square.file in {0, 7} or square.rank in {0, 7}):
            penalty += 8
    return -penalty


def _opposition_score(board: Board) -> int:
    white_king = _king_square(board, Color.WHITE)
    black_king = _king_square(board, Color.BLACK)
    if white_king is None or black_king is None:
        return 0
    if white_king.file != black_king.file and white_king.rank != black_king.rank:
        return 0
    distance = _king_distance(white_king, black_king)
    if distance != 2:
        return 0
    return 10 if board.side_to_move is Color.BLACK else -10


def _outside_passer_score(board: Board) -> int:
    score = 0
    for square, piece in board.pieces():
        if piece.piece_type is not PieceType.PAWN or not _is_passed_pawn(board, square, piece.color):
            continue
        if square.file in {0, 1, 6, 7}:
            advancement = square.rank if piece.color is Color.WHITE else 7 - square.rank
            value = 6 + advancement * 2
            score += value if piece.color is Color.WHITE else -value
    return score


def _bishop_vs_knight_pawn_structure_score(board: Board) -> int:
    locked_pawns = 0
    for square, piece in board.pieces():
        if piece.piece_type is not PieceType.PAWN:
            continue
        front_rank = square.rank + (1 if piece.color is Color.WHITE else -1)
        if 0 <= front_rank < 8:
            blocker = board.get_piece(Square(square.file, front_rank))
            if blocker is not None and blocker.color is piece.color.opponent and blocker.piece_type is PieceType.PAWN:
                locked_pawns += 1
    if locked_pawns < 3:
        return 0
    score = 0
    for color in (Color.WHITE, Color.BLACK):
        knights = sum(1 for _, piece in board.pieces(color) if piece.piece_type is PieceType.KNIGHT)
        bishops = sum(1 for _, piece in board.pieces(color) if piece.piece_type is PieceType.BISHOP)
        value = knights * 8 - bishops * 4
        score += value if color is Color.WHITE else -value
    return score


def _king_attack_pressure(board: Board, color: Color, enemy_king: Square) -> int:
    pressure = 0
    for square, piece in board.pieces(color):
        if piece.piece_type in {PieceType.KING, PieceType.PAWN}:
            continue
        distance = _king_distance(square, enemy_king)
        value = max(0, 7 - distance)
        if piece.piece_type is PieceType.QUEEN:
            value *= 3
        elif piece.piece_type is PieceType.ROOK:
            value *= 2
        if _piece_has_line_to_square(board, square, piece, enemy_king) and (
            distance <= 5 or piece.piece_type is PieceType.ROOK
        ):
            value += 8
        pressure += value
    return min(45, pressure)


def _pawn_majority_score(board: Board) -> int:
    score = 0
    for files in (range(0, 4), range(4, 8)):
        white = _pawn_count_in_files(board, files, Color.WHITE)
        black = _pawn_count_in_files(board, files, Color.BLACK)
        if white > black:
            score += min(18, (white - black) * 8 + _most_advanced_pawn_in_files(board, files, Color.WHITE) * 2)
        elif black > white:
            score -= min(18, (black - white) * 8 + _most_advanced_pawn_in_files(board, files, Color.BLACK) * 2)
    return score


def _pawn_break_score(board: Board) -> int:
    score = 0
    for square, piece in board.pieces():
        if piece.piece_type is not PieceType.PAWN:
            continue
        if _can_create_pawn_break(board, square, piece.color):
            value = 8
            if square.file in {2, 3, 4, 5}:
                value += 4
            score += value if piece.color is Color.WHITE else -value
    return score


def _bishop_scope(board: Board, square: Square, color: Color) -> int:
    scope = 0
    for file_step, rank_step in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        file = square.file + file_step
        rank = square.rank + rank_step
        while 0 <= file < 8 and 0 <= rank < 8:
            piece = board.get_piece(Square(file, rank))
            if piece is None:
                scope += 2
            else:
                scope += 1 if piece.color is color.opponent else -2
                break
            file += file_step
            rank += rank_step
    return scope


def _locked_center_files(board: Board) -> set[int]:
    locked: set[int] = set()
    for file in (2, 3, 4, 5):
        for rank in range(1, 7):
            piece = board.get_piece(Square(file, rank))
            if piece is None or piece.piece_type is not PieceType.PAWN:
                continue
            direction = 1 if piece.color is Color.WHITE else -1
            front_rank = rank + direction
            if not 0 <= front_rank < 8:
                continue
            front = board.get_piece(Square(file, front_rank))
            if front is not None and front.color is piece.color.opponent and front.piece_type is PieceType.PAWN:
                locked.add(file)
    return locked


def _bishop_own_pawn_blockers(board: Board, square: Square, color: Color) -> int:
    blockers = 0
    for file_step, rank_step in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        file = square.file + file_step
        rank = square.rank + rank_step
        if not (0 <= file < 8 and 0 <= rank < 8):
            continue
        piece = board.get_piece(Square(file, rank))
        if piece is not None and piece.color is color and piece.piece_type is PieceType.PAWN:
            blockers += 1
    return blockers


def _enemy_major_back_rank_pressure(board: Board, color: Color, king: Square) -> int:
    pressure = 0
    for square, piece in board.pieces(color.opponent):
        if piece.piece_type not in {PieceType.ROOK, PieceType.QUEEN}:
            continue
        if square.rank == king.rank:
            pressure += 1
        if square.file in {king.file - 1, king.file, king.file + 1} and not _file_has_pawn(board, square.file, color):
            pressure += 1
    return min(3, pressure)


def _piece_has_line_to_square(board: Board, from_square: Square, piece, target: Square) -> bool:
    file_delta = target.file - from_square.file
    rank_delta = target.rank - from_square.rank
    abs_file = abs(file_delta)
    abs_rank = abs(rank_delta)
    if piece.piece_type is PieceType.BISHOP and abs_file != abs_rank:
        return False
    if piece.piece_type is PieceType.ROOK and file_delta != 0 and rank_delta != 0:
        return False
    if piece.piece_type is PieceType.QUEEN and not (file_delta == 0 or rank_delta == 0 or abs_file == abs_rank):
        return False
    if piece.piece_type not in {PieceType.BISHOP, PieceType.ROOK, PieceType.QUEEN}:
        return False
    file_step = 0 if file_delta == 0 else (1 if file_delta > 0 else -1)
    rank_step = 0 if rank_delta == 0 else (1 if rank_delta > 0 else -1)
    file = from_square.file + file_step
    rank = from_square.rank + rank_step
    while (file, rank) != (target.file, target.rank):
        if board.get_piece(Square(file, rank)) is not None:
            return False
        file += file_step
        rank += rank_step
    return True


def _same_line(first: Square, second: Square, third: Square) -> bool:
    return (
        first.file == second.file == third.file
        or first.rank == second.rank == third.rank
        or abs(first.file - second.file) == abs(first.rank - second.rank)
        and abs(first.file - third.file) == abs(first.rank - third.rank)
    )


def _major_piece_behind_passed_pawn(board: Board, pawn_square: Square, color: Color) -> int:
    count = 0
    ranks = range(0, pawn_square.rank) if color is Color.WHITE else range(pawn_square.rank + 1, 8)
    for rank in ranks:
        piece = board.get_piece(Square(pawn_square.file, rank))
        if piece is not None and piece.color is color and piece.piece_type in {PieceType.ROOK, PieceType.QUEEN}:
            count += 1
    return count


def _pawn_count_in_files(board: Board, files, color: Color) -> int:
    file_set = set(files)
    return sum(
        1
        for square, piece in board.pieces(color)
        if piece.piece_type is PieceType.PAWN and square.file in file_set
    )


def _most_advanced_pawn_in_files(board: Board, files, color: Color) -> int:
    file_set = set(files)
    advancements = [
        square.rank if color is Color.WHITE else 7 - square.rank
        for square, piece in board.pieces(color)
        if piece.piece_type is PieceType.PAWN and square.file in file_set
    ]
    return max(advancements, default=0)


def _is_hanging_pawn(board: Board, square: Square, color: Color) -> bool:
    same_rank_partner = False
    for file in (square.file - 1, square.file + 1):
        if not 0 <= file < 8:
            continue
        piece = board.get_piece(Square(file, square.rank))
        if piece is not None and piece.color is color and piece.piece_type is PieceType.PAWN:
            same_rank_partner = True
    if not same_rank_partner:
        return False
    for file in (square.file - 2, square.file + 2):
        if not 0 <= file < 8:
            continue
        if _file_has_pawn(board, file, color):
            return False
    return not _is_supported_by_pawn(board, square, color)


def _can_create_pawn_break(board: Board, square: Square, color: Color) -> bool:
    direction = 1 if color is Color.WHITE else -1
    target_rank = square.rank + direction
    if not 0 <= target_rank < 8:
        return False
    if board.get_piece(Square(square.file, target_rank)) is not None:
        return False
    for file in (square.file - 1, square.file + 1):
        if not 0 <= file < 8:
            continue
        target = board.get_piece(Square(file, target_rank + direction)) if 0 <= target_rank + direction < 8 else None
        adjacent = board.get_piece(Square(file, target_rank))
        if adjacent is not None and adjacent.color is color.opponent and adjacent.piece_type is PieceType.PAWN:
            return True
        if target is not None and target.color is color.opponent and target.piece_type is PieceType.PAWN:
            return True
    return False


def _file_has_pawn(board: Board, file: int, color: Color) -> bool:
    for rank in range(8):
        piece = board.get_piece(Square(file, rank))
        if piece is not None and piece.color is color and piece.piece_type is PieceType.PAWN:
            return True
    return False


def _rook_behind_passed_pawn(board: Board, rook_square: Square, color: Color) -> bool:
    for pawn_square, pawn in board.pieces(color):
        if pawn.piece_type is not PieceType.PAWN or pawn_square.file != rook_square.file:
            continue
        if not _is_passed_pawn(board, pawn_square, color):
            continue
        if color is Color.WHITE and rook_square.rank < pawn_square.rank:
            return True
        if color is Color.BLACK and rook_square.rank > pawn_square.rank:
            return True
    return False


def _central_color_complex_weakness(board: Board, color: Color, light: bool) -> int:
    count = 0
    for file in range(2, 6):
        for rank in range(2, 6):
            square = Square(file, rank)
            if _is_light_square(square) is not light:
                continue
            if not _can_be_defended_by_pawn(board, square, color):
                count += 1
    return count


def _enemy_bishop_color_pressure(board: Board, color: Color, light: bool) -> int:
    count = 0
    for square, piece in board.pieces(color.opponent):
        if piece.piece_type is PieceType.BISHOP and _is_light_square(square) is light:
            count += 1
    return count


def _is_isolated_pawn(board: Board, square: Square, color: Color) -> bool:
    for file in (square.file - 1, square.file + 1):
        if not 0 <= file < 8:
            continue
        for rank in range(8):
            piece = board.get_piece(Square(file, rank))
            if piece is not None and piece.color is color and piece.piece_type is PieceType.PAWN:
                return False
    return True


def _is_backward_pawn(board: Board, square: Square, color: Color) -> bool:
    if not _is_isolated_pawn(board, square, color):
        return False
    direction = 1 if color is Color.WHITE else -1
    front = square.rank + direction
    if not 0 <= front < 8:
        return False
    for file in (square.file - 1, square.file + 1):
        if not 0 <= file < 8:
            continue
        enemy = board.get_piece(Square(file, front))
        if enemy is not None and enemy.color is color.opponent and enemy.piece_type is PieceType.PAWN:
            return True
    return False


def _can_be_defended_by_pawn(board: Board, square: Square, color: Color) -> bool:
    direction = 1 if color is Color.WHITE else -1
    pawn_rank = square.rank - direction
    if not 0 <= pawn_rank < 8:
        return False
    for file in (square.file - 1, square.file + 1):
        if not 0 <= file < 8:
            continue
        piece = board.get_piece(Square(file, pawn_rank))
        if piece is not None and piece.color is color and piece.piece_type is PieceType.PAWN:
            return True
    return False


def _minor_piece_controls(board: Board, square: Square, color: Color) -> bool:
    for piece_square, piece in board.pieces(color):
        if piece.piece_type is PieceType.KNIGHT:
            if (abs(piece_square.file - square.file), abs(piece_square.rank - square.rank)) in {(1, 2), (2, 1)}:
                return True
        if piece.piece_type is PieceType.BISHOP:
            if abs(piece_square.file - square.file) == abs(piece_square.rank - square.rank):
                return True
    return False


def _is_supported_by_pawn(board: Board, square: Square, color: Color) -> bool:
    direction = 1 if color is Color.WHITE else -1
    pawn_rank = square.rank - direction
    if not 0 <= pawn_rank < 8:
        return False
    for file in (square.file - 1, square.file + 1):
        if not 0 <= file < 8:
            continue
        piece = board.get_piece(Square(file, pawn_rank))
        if piece is not None and piece.color is color and piece.piece_type is PieceType.PAWN:
            return True
    return False


def _is_attacked_by_pawn(board: Board, square: Square, by_color: Color) -> bool:
    direction = 1 if by_color is Color.WHITE else -1
    pawn_rank = square.rank - direction
    if not 0 <= pawn_rank < 8:
        return False
    for file in (square.file - 1, square.file + 1):
        if not 0 <= file < 8:
            continue
        piece = board.get_piece(Square(file, pawn_rank))
        if piece is not None and piece.color is by_color and piece.piece_type is PieceType.PAWN:
            return True
    return False


def _is_light_square(square: Square) -> bool:
    return (square.file + square.rank) % 2 == 1
