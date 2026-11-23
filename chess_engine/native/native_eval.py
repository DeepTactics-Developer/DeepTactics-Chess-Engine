"""ctypes bridge for the optional C++ evaluation backend."""

from __future__ import annotations

import ctypes
from pathlib import Path
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from chess_engine.chess_core.board import Board
from chess_engine.chess_core.enums import Color, PieceType
from chess_engine.chess_core.move import Move
from chess_engine.chess_core.square import Square


NATIVE_LIBRARY_PATH = Path(__file__).with_name("deeptactics_native_runtime_v2.dll")


class EvaluationWeightsLike(Protocol):
    material: float
    position: float
    mobility: float
    endgame: float
    pawn_structure: float
    bishop_pair: float
    rook_activity: float
    king_safety: float
    passed_pawn: float
    endgame_pressure: float
    line_activity: float
    attack_momentum: float

_PIECE_CODE_BY_VALUE = {
    (Color.WHITE, PieceType.PAWN): 1,
    (Color.WHITE, PieceType.KNIGHT): 2,
    (Color.WHITE, PieceType.BISHOP): 3,
    (Color.WHITE, PieceType.ROOK): 4,
    (Color.WHITE, PieceType.QUEEN): 5,
    (Color.WHITE, PieceType.KING): 6,
    (Color.BLACK, PieceType.PAWN): -1,
    (Color.BLACK, PieceType.KNIGHT): -2,
    (Color.BLACK, PieceType.BISHOP): -3,
    (Color.BLACK, PieceType.ROOK): -4,
    (Color.BLACK, PieceType.QUEEN): -5,
    (Color.BLACK, PieceType.KING): -6,
}

_library: ctypes.CDLL | None = None
_load_error: str | None = None
_native_internal_iterative_deepening = True
_BOARD_ARRAY = ctypes.c_int8 * 64
_MOVE_ARRAY = ctypes.c_int * 256
_PV_ARRAY = ctypes.c_int * 64
_PROMOTION_BY_CODE = {
    1: PieceType.QUEEN,
    2: PieceType.ROOK,
    3: PieceType.BISHOP,
    4: PieceType.KNIGHT,
}
_PROMOTION_CODE_BY_TYPE = {
    PieceType.QUEEN: 1,
    PieceType.ROOK: 2,
    PieceType.BISHOP: 3,
    PieceType.KNIGHT: 4,
}


@dataclass(frozen=True, slots=True)
class NativeSearchResult:
    move: Move | None
    score: int
    principal_variation: tuple[Move, ...]
    nodes: int = 0
    aborted: bool = False
    completed_depth: int = 0


def native_available() -> bool:
    """Return whether the native evaluator DLL can be loaded."""
    return _load_library() is not None


def native_status() -> str:
    """Return a short diagnostic for UI or benchmark output."""
    if native_available():
        return f"available ({NATIVE_LIBRARY_PATH.name})"
    return f"unavailable ({_load_error or 'DLL not built'})"


def native_build_status() -> str:
    """Return whether the loaded native binary is current with its C++ source."""
    source_path = Path(__file__).with_name("native_eval.cpp")
    if not NATIVE_LIBRARY_PATH.exists():
        return "missing"
    if not source_path.exists():
        return "source_missing"
    if NATIVE_LIBRARY_PATH.stat().st_mtime < source_path.stat().st_mtime:
        return "older_than_source"
    return "current"


def evaluate_native(board: Board, weights: EvaluationWeightsLike) -> int | None:
    """Return a native evaluation, or None if the backend is unavailable."""
    library = _load_library()
    if library is None:
        return None
    packed = _pack_board(board)
    return int(
        library.deeptactics_evaluate(
            packed,
            float(weights.material),
            float(weights.position),
            float(weights.mobility),
            float(weights.endgame),
            float(weights.pawn_structure),
            float(weights.bishop_pair),
            float(weights.rook_activity),
            float(weights.king_safety),
            float(weights.passed_pawn),
            float(weights.endgame_pressure),
            float(weights.line_activity),
            float(weights.attack_momentum),
        )
    )


def evaluate_native_breakdown(board: Board, weights: EvaluationWeightsLike) -> dict[str, int] | None:
    """Return native evaluator raw terms and final weighted score when exported."""
    library = _load_library()
    if library is None or not hasattr(library, "deeptactics_evaluate_breakdown"):
        return None
    packed = _pack_board(board)
    output = (ctypes.c_int * 15)()
    count = library.deeptactics_evaluate_breakdown(
        packed,
        float(weights.material),
        float(weights.position),
        float(weights.mobility),
        float(weights.endgame),
        float(weights.pawn_structure),
        float(weights.bishop_pair),
        float(weights.rook_activity),
        float(weights.king_safety),
        float(weights.passed_pawn),
        float(weights.endgame_pressure),
        float(weights.line_activity),
        float(weights.attack_momentum),
        output,
        len(output),
    )
    if count < 15:
        return None
    names = (
        "material", "pst", "mobility", "pawn_structure", "bishop_pair", "rook_activity",
        "king_safety", "book_knowledge_base", "line_activity", "attack_momentum",
        "endgame", "strategic_weighted", "native_total",
        "passed_pawn_raw", "endgame_pressure_raw",
    )
    return {name: int(output[index]) for index, name in enumerate(names)}


def _load_library() -> ctypes.CDLL | None:
    global _library, _load_error
    if _library is not None:
        return _library
    if not NATIVE_LIBRARY_PATH.exists():
        _load_error = "DLL not built"
        return None
    try:
        library = ctypes.CDLL(str(NATIVE_LIBRARY_PATH))
        library.deeptactics_evaluate.argtypes = [
            ctypes.POINTER(ctypes.c_int8),
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
        ]
        library.deeptactics_evaluate.restype = ctypes.c_int
        library.deeptactics_is_square_attacked.argtypes = [
            ctypes.POINTER(ctypes.c_int8),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        library.deeptactics_is_square_attacked.restype = ctypes.c_int
        library.deeptactics_generate_pseudo_legal_moves.argtypes = [
            ctypes.POINTER(ctypes.c_int8),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]
        library.deeptactics_generate_pseudo_legal_moves.restype = ctypes.c_int
        library.deeptactics_generate_legal_moves.argtypes = library.deeptactics_generate_pseudo_legal_moves.argtypes
        library.deeptactics_generate_legal_moves.restype = ctypes.c_int
        library.deeptactics_static_exchange_score.argtypes = [
            ctypes.POINTER(ctypes.c_int8),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        library.deeptactics_static_exchange_score.restype = ctypes.c_int
        library.deeptactics_find_best_move.argtypes = [
            ctypes.POINTER(ctypes.c_int8),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_uint64),
        ]
        library.deeptactics_find_best_move.restype = ctypes.c_int
        _configure_optional_search_control_exports(library)
        _configure_optional_breakdown_export(library)
        _configure_optional_accuracy_exports(library)
        _configure_optional_nnue_exports(library)
    except OSError as error:
        _load_error = str(error)
        return None
    _library = library
    _load_error = None
    return _library


def _configure_optional_accuracy_exports(library: ctypes.CDLL) -> None:
    try:
        library.deeptactics_cp_to_win_probability.argtypes = [ctypes.c_int]
        library.deeptactics_cp_to_win_probability.restype = ctypes.c_double
        library.deeptactics_move_accuracy.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        library.deeptactics_move_accuracy.restype = ctypes.c_double
        library.deeptactics_game_accuracy.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
        library.deeptactics_game_accuracy.restype = ctypes.c_double
    except AttributeError:
        # Older DLL builds can still run the engine. Python falls back to the
        # same formulas until the native library is rebuilt.
        return


def _configure_optional_search_control_exports(library: ctypes.CDLL) -> None:
    try:
        base = library.deeptactics_find_best_move.argtypes
        library.deeptactics_find_best_move_timed.argtypes = base[:22] + [ctypes.c_int] + base[22:] + [
            ctypes.POINTER(ctypes.c_int)
        ]
        library.deeptactics_find_best_move_timed.restype = ctypes.c_int
        library.deeptactics_find_best_move_iterative_timed.argtypes = base[:22] + [ctypes.c_int] + base[22:] + [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        library.deeptactics_find_best_move_iterative_timed.restype = ctypes.c_int
        library.deeptactics_request_stop.argtypes = []
        library.deeptactics_request_stop.restype = None
        library.deeptactics_set_search_experiments.argtypes = [ctypes.c_int, ctypes.c_int]
        library.deeptactics_set_search_experiments.restype = None
        library.deeptactics_set_defensive_mate_threat_ordering.argtypes = [ctypes.c_int]
        library.deeptactics_set_defensive_mate_threat_ordering.restype = None
        library.deeptactics_set_major_piece_safety_ordering.argtypes = [ctypes.c_int]
        library.deeptactics_set_major_piece_safety_ordering.restype = None
        library.deeptactics_set_clustered_tt_replacement.argtypes = [ctypes.c_int]
        library.deeptactics_set_clustered_tt_replacement.restype = None
        library.deeptactics_set_aspiration_windows.argtypes = [ctypes.c_int]
        library.deeptactics_set_aspiration_windows.restype = None
        library.deeptactics_set_pawn_endgame_extension_budget.argtypes = [ctypes.c_int]
        library.deeptactics_set_pawn_endgame_extension_budget.restype = None
        if hasattr(library, "deeptactics_set_check_extension_budget"):
            library.deeptactics_set_check_extension_budget.argtypes = [ctypes.c_int]
            library.deeptactics_set_check_extension_budget.restype = None
        if hasattr(library, "deeptactics_set_singular_extension"):
            library.deeptactics_set_singular_extension.argtypes = [ctypes.c_int]
            library.deeptactics_set_singular_extension.restype = None
        if hasattr(library, "deeptactics_set_futility_pruning"):
            library.deeptactics_set_futility_pruning.argtypes = [ctypes.c_int]
            library.deeptactics_set_futility_pruning.restype = None
        if hasattr(library, "deeptactics_set_lmr_formula"):
            library.deeptactics_set_lmr_formula.argtypes = [ctypes.c_int]
            library.deeptactics_set_lmr_formula.restype = None
        if hasattr(library, "deeptactics_set_razoring"):
            library.deeptactics_set_razoring.argtypes = [ctypes.c_int]
            library.deeptactics_set_razoring.restype = None
        if hasattr(library, "deeptactics_set_continuation_history"):
            library.deeptactics_set_continuation_history.argtypes = [ctypes.c_int]
            library.deeptactics_set_continuation_history.restype = None
        if hasattr(library, "deeptactics_set_probcut"):
            library.deeptactics_set_probcut.argtypes = [ctypes.c_int]
            library.deeptactics_set_probcut.restype = None
        library.deeptactics_set_hash_mb.argtypes = [ctypes.c_int]
        library.deeptactics_set_hash_mb.restype = None
        library.deeptactics_get_hash_mb.argtypes = []
        library.deeptactics_get_hash_mb.restype = ctypes.c_int
        library.deeptactics_get_hash_bucket_count.argtypes = []
        library.deeptactics_get_hash_bucket_count.restype = ctypes.c_int
        library.deeptactics_get_null_move_probes.argtypes = []
        library.deeptactics_get_null_move_probes.restype = ctypes.c_uint64
        library.deeptactics_get_null_move_cutoffs.argtypes = []
        library.deeptactics_get_null_move_cutoffs.restype = ctypes.c_uint64
        library.deeptactics_get_nested_null_move_probes.argtypes = []
        library.deeptactics_get_nested_null_move_probes.restype = ctypes.c_uint64
        library.deeptactics_get_aspiration_attempts.argtypes = []
        library.deeptactics_get_aspiration_attempts.restype = ctypes.c_uint64
        library.deeptactics_get_aspiration_fail_lows.argtypes = []
        library.deeptactics_get_aspiration_fail_lows.restype = ctypes.c_uint64
        library.deeptactics_get_aspiration_fail_highs.argtypes = []
        library.deeptactics_get_aspiration_fail_highs.restype = ctypes.c_uint64
        library.deeptactics_get_aspiration_researches.argtypes = []
        library.deeptactics_get_aspiration_researches.restype = ctypes.c_uint64
        library.deeptactics_get_quiet_beta_cutoffs.argtypes = []
        library.deeptactics_get_quiet_beta_cutoffs.restype = ctypes.c_uint64
        library.deeptactics_get_killer_order_hits.argtypes = []
        library.deeptactics_get_killer_order_hits.restype = ctypes.c_uint64
        library.deeptactics_get_history_order_hits.argtypes = []
        library.deeptactics_get_history_order_hits.restype = ctypes.c_uint64
        library.deeptactics_get_history_saturated_updates.argtypes = []
        library.deeptactics_get_history_saturated_updates.restype = ctypes.c_uint64
    except AttributeError:
        return


def _configure_optional_breakdown_export(library: ctypes.CDLL) -> None:
    try:
        library.deeptactics_evaluate_breakdown.argtypes = library.deeptactics_evaluate.argtypes + [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]
        library.deeptactics_evaluate_breakdown.restype = ctypes.c_int
    except AttributeError:
        return


def _configure_optional_nnue_exports(library: ctypes.CDLL) -> None:
    try:
        library.deeptactics_nnue_sparse_linear_infer.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_double,
        ]
        library.deeptactics_nnue_sparse_linear_infer.restype = ctypes.c_int
        library.deeptactics_nnue_sparse_linear_infer_file.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
        ]
        library.deeptactics_nnue_sparse_linear_infer_file.restype = ctypes.c_int
        library.deeptactics_nnue_extract_features.argtypes = [
            ctypes.POINTER(ctypes.c_int8),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]
        library.deeptactics_nnue_extract_features.restype = ctypes.c_int
        library.deeptactics_nnue_evaluate_board_file.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_int8),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
        ]
        library.deeptactics_nnue_evaluate_board_file.restype = ctypes.c_int
        library.deeptactics_nnue_accumulator_evaluate_board_file.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_int8),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
        ]
        library.deeptactics_nnue_accumulator_evaluate_board_file.restype = ctypes.c_int
    except AttributeError:
        return


def nnue_sparse_linear_infer_native(
    active_indices: list[int] | tuple[int, ...],
    feature_weights: list[float] | tuple[float, ...],
    bias: float = 0.0,
) -> int | None:
    """Run the native sparse NNUE-prototype inference kernel if available."""
    library = _load_library()
    if library is None or not hasattr(library, "deeptactics_nnue_sparse_linear_infer"):
        return None
    if not active_indices:
        return int(round(bias))
    index_array_type = ctypes.c_int * len(active_indices)
    weight_array_type = ctypes.c_double * len(feature_weights)
    index_array = index_array_type(*(int(index) for index in active_indices))
    weight_array = weight_array_type(*(float(weight) for weight in feature_weights))
    return int(
        library.deeptactics_nnue_sparse_linear_infer(
            index_array,
            len(active_indices),
            weight_array,
            len(feature_weights),
            float(bias),
        )
    )


def nnue_sparse_linear_infer_file_native(
    model_path: str | Path,
    active_indices: list[int] | tuple[int, ...],
) -> int | None:
    """Run native sparse inference by loading a .dtnnue binary model file."""
    library = _load_library()
    if library is None or not hasattr(library, "deeptactics_nnue_sparse_linear_infer_file"):
        return None
    index_array_type = ctypes.c_int * max(1, len(active_indices))
    index_array = index_array_type(*(int(index) for index in active_indices), *([0] if not active_indices else []))
    ok = ctypes.c_int(0)
    result = int(
        library.deeptactics_nnue_sparse_linear_infer_file(
            str(model_path).encode("utf-8"),
            index_array,
            len(active_indices),
            ctypes.byref(ok),
        )
    )
    if ok.value != 1:
        return None
    return result


def evaluate_nnue_binary_native(board: "Board", model_path: str | Path) -> int | None:
    """Evaluate ``board`` with the native cached .dtnnue sparse probe.

    C++ extracts the current sparse board features, reuses the cached binary
    model when the path is unchanged, and returns a White-perspective score.
    """
    library = _load_library()
    if library is None or not hasattr(library, "deeptactics_nnue_evaluate_board_file"):
        return None
    packed = _pack_board(board)
    ep_file = board.en_passant_target.file if board.en_passant_target is not None else -1
    ok = ctypes.c_int(0)
    result = int(
        library.deeptactics_nnue_evaluate_board_file(
            str(model_path).encode("utf-8"),
            packed,
            1 if board.side_to_move is Color.WHITE else -1,
            int(board.castling_rights.white_king_side),
            int(board.castling_rights.white_queen_side),
            int(board.castling_rights.black_king_side),
            int(board.castling_rights.black_queen_side),
            ep_file,
            ctypes.byref(ok),
        )
    )
    if ok.value != 1:
        return None
    return result


def extract_nnue_features_native(board: Board) -> tuple[int, ...] | None:
    """Return sparse NNUE feature indices from C++, or None if unavailable."""
    library = _load_library()
    if library is None or not hasattr(library, "deeptactics_nnue_extract_features"):
        return None
    packed = _pack_board(board)
    ep_file = board.en_passant_target.file if board.en_passant_target is not None else -1
    output_type = ctypes.c_int * 128
    output = output_type()
    count = int(
        library.deeptactics_nnue_extract_features(
            packed,
            1 if board.side_to_move is Color.WHITE else -1,
            int(board.castling_rights.white_king_side),
            int(board.castling_rights.white_queen_side),
            int(board.castling_rights.black_king_side),
            int(board.castling_rights.black_queen_side),
            ep_file,
            output,
            len(output),
        )
    )
    if count <= 0 or count > len(output):
        return None
    return tuple(int(output[index]) for index in range(count))


def evaluate_accumulator_nnue_binary_native(board: Board, model_path: str | Path) -> int | None:
    """Evaluate ``board`` with the native accumulator v2 .dtnnue model."""
    library = _load_library()
    if library is None or not hasattr(library, "deeptactics_nnue_accumulator_evaluate_board_file"):
        return None
    packed = _pack_board(board)
    ep_file = board.en_passant_target.file if board.en_passant_target is not None else -1
    ok = ctypes.c_int(0)
    result = int(
        library.deeptactics_nnue_accumulator_evaluate_board_file(
            str(model_path).encode("utf-8"),
            packed,
            1 if board.side_to_move is Color.WHITE else -1,
            int(board.castling_rights.white_king_side),
            int(board.castling_rights.white_queen_side),
            int(board.castling_rights.black_king_side),
            int(board.castling_rights.black_queen_side),
            ep_file,
            ctypes.byref(ok),
        )
    )
    if ok.value != 1:
        return None
    return result


def cp_to_win_probability_native(cp: int) -> float | None:
    library = _load_library()
    if library is None or not hasattr(library, "deeptactics_cp_to_win_probability"):
        return None
    return float(library.deeptactics_cp_to_win_probability(int(cp)))


def move_accuracy_native(cp_before: int, cp_after: int, is_book_move: bool = False) -> float | None:
    library = _load_library()
    if library is None or not hasattr(library, "deeptactics_move_accuracy"):
        return None
    return float(library.deeptactics_move_accuracy(int(cp_before), int(cp_after), int(is_book_move)))


def game_accuracy_native(accuracies: list[float] | tuple[float, ...]) -> float | None:
    library = _load_library()
    if library is None or not hasattr(library, "deeptactics_game_accuracy"):
        return None
    if not accuracies:
        return 100.0
    array_type = ctypes.c_double * len(accuracies)
    values = array_type(*(float(value) for value in accuracies))
    return float(library.deeptactics_game_accuracy(values, len(accuracies)))


def _pack_board(board: Board):
    packed = _BOARD_ARRAY()
    for rank, row in enumerate(board.grid):
        base = rank * 8
        for file, piece in enumerate(row):
            if piece is not None:
                packed[base + file] = _PIECE_CODE_BY_VALUE[(piece.color, piece.piece_type)]
    return packed


def is_square_attacked_native(board: Board, file: int, rank: int, by_color: Color) -> bool | None:
    """Return native attack detection, or None if the backend is unavailable."""
    library = _load_library()
    if library is None:
        return None
    packed = _pack_board(board)
    color_code = 1 if by_color is Color.WHITE else -1
    return bool(library.deeptactics_is_square_attacked(packed, file, rank, color_code))


def generate_pseudo_legal_moves_native(board: Board, color: Color) -> list[Move] | None:
    """Return native pseudo-legal moves, or None if the backend is unavailable."""
    return _generate_moves_native(board, color, legal=False)


def generate_legal_moves_native(board: Board, color: Color) -> list[Move] | None:
    """Return native legal moves, or None if the backend is unavailable."""
    return _generate_moves_native(board, color, legal=True)


def _generate_moves_native(board: Board, color: Color, legal: bool) -> list[Move] | None:
    library = _load_library()
    if library is None:
        return None
    packed = _pack_board(board)
    output = _MOVE_ARRAY()
    ep_file = board.en_passant_target.file if board.en_passant_target is not None else -1
    ep_rank = board.en_passant_target.rank if board.en_passant_target is not None else -1
    function = library.deeptactics_generate_legal_moves if legal else library.deeptactics_generate_pseudo_legal_moves
    count = function(
        packed,
        1 if color is Color.WHITE else -1,
        int(board.castling_rights.white_king_side),
        int(board.castling_rights.white_queen_side),
        int(board.castling_rights.black_king_side),
        int(board.castling_rights.black_queen_side),
        ep_file,
        ep_rank,
        output,
        len(output),
    )
    return [_decode_move(output[index]) for index in range(min(count, len(output)))]


def _decode_move(encoded: int) -> Move:
    from_index = encoded & 0x3F
    to_index = (encoded >> 6) & 0x3F
    promotion_code = (encoded >> 12) & 0xF
    flags = (encoded >> 16) & 0xF
    return Move(
        from_square=Square(from_index % 8, from_index // 8),
        to_square=Square(to_index % 8, to_index // 8),
        promotion=_PROMOTION_BY_CODE.get(promotion_code),
        is_en_passant=bool(flags & 1),
        is_castling=bool(flags & 2),
    )


def find_best_move_native(
    board: Board,
    depth: int,
    weights: EvaluationWeightsLike,
    quiescence_depth: int = 4,
    root_move_penalties: Mapping[str, int] | None = None,
) -> tuple[Move | None, int] | None:
    """Return a native search best move and score, or None if unavailable."""
    result = find_best_line_native(
        board,
        depth=depth,
        weights=weights,
        quiescence_depth=quiescence_depth,
        root_move_penalties=root_move_penalties,
    )
    if result is None:
        return None
    return result.move, result.score


def static_exchange_score_native(board: Board, move: Move) -> int | None:
    """Return native legal-recapture SEE for diagnostics and ordering parity tests."""
    library = _load_library()
    if library is None:
        return None
    packed = _pack_board(board)
    ep_file = board.en_passant_target.file if board.en_passant_target is not None else -1
    ep_rank = board.en_passant_target.rank if board.en_passant_target is not None else -1
    return int(
        library.deeptactics_static_exchange_score(
            packed,
            1 if board.side_to_move is Color.WHITE else -1,
            int(board.castling_rights.white_king_side),
            int(board.castling_rights.white_queen_side),
            int(board.castling_rights.black_king_side),
            int(board.castling_rights.black_queen_side),
            ep_file,
            ep_rank,
            _encode_move(move),
        )
    )


def find_best_line_native(
    board: Board,
    depth: int,
    weights: EvaluationWeightsLike,
    quiescence_depth: int = 4,
    root_move_penalties: Mapping[str, int] | None = None,
    time_limit_ms: int | None = None,
    iterative: bool = False,
) -> NativeSearchResult | None:
    """Return a native search result with principal variation, or None."""
    if board.halfmove_clock >= 100:
        from chess_engine.chess_core.rules import generate_legal_moves

        if generate_legal_moves(board):
            return NativeSearchResult(
                move=None,
                score=0,
                principal_variation=(),
                nodes=0,
                aborted=False,
                completed_depth=max(0, int(depth)),
            )

    library = _load_library()
    if library is None:
        return None
    packed = _pack_board(board)
    ep_file = board.en_passant_target.file if board.en_passant_target is not None else -1
    ep_rank = board.en_passant_target.rank if board.en_passant_target is not None else -1
    out_move = ctypes.c_int(0)
    out_score = ctypes.c_int(0)
    out_pv = _PV_ARRAY()
    out_pv_count = ctypes.c_int(0)
    out_nodes = ctypes.c_uint64(0)
    out_aborted = ctypes.c_int(0)
    out_completed_depth = ctypes.c_int(0)
    penalty_moves, penalties, penalty_count = _pack_root_move_penalties(root_move_penalties)
    common_args = (
        packed,
        1 if board.side_to_move is Color.WHITE else -1,
        int(board.castling_rights.white_king_side),
        int(board.castling_rights.white_queen_side),
        int(board.castling_rights.black_king_side),
        int(board.castling_rights.black_queen_side),
        ep_file,
        ep_rank,
        int(depth),
        int(quiescence_depth),
        float(weights.material),
        float(weights.position),
        float(weights.mobility),
        float(weights.endgame),
        float(weights.pawn_structure),
        float(weights.bishop_pair),
        float(weights.rook_activity),
        float(weights.king_safety),
        float(weights.passed_pawn),
        float(weights.endgame_pressure),
        float(weights.line_activity),
        float(weights.attack_momentum),
    )
    output_args = (
        penalty_moves,
        penalties,
        penalty_count,
        ctypes.byref(out_move),
        ctypes.byref(out_score),
        out_pv,
        len(out_pv),
        ctypes.byref(out_pv_count),
        ctypes.byref(out_nodes),
    )
    if iterative and time_limit_ms is not None and hasattr(library, "deeptactics_find_best_move_iterative_timed"):
        found = library.deeptactics_find_best_move_iterative_timed(
            *common_args,
            max(1, int(time_limit_ms)),
            *output_args,
            ctypes.byref(out_aborted),
            ctypes.byref(out_completed_depth),
        )
    elif hasattr(library, "deeptactics_find_best_move_timed"):
        found = library.deeptactics_find_best_move_timed(
            *common_args,
            -1 if time_limit_ms is None else max(1, int(time_limit_ms)),
            *output_args,
            ctypes.byref(out_aborted),
        )
    else:
        found = library.deeptactics_find_best_move(*common_args, *output_args)
    if not found:
        return None
    pv_count = max(0, min(out_pv_count.value, len(out_pv)))
    principal_variation = tuple(_decode_move(out_pv[index]) for index in range(pv_count))
    move = None if out_move.value == 0 else _decode_move(out_move.value)
    if not principal_variation and move is not None:
        principal_variation = (move,)
    return NativeSearchResult(
        move=move,
        score=int(out_score.value),
        principal_variation=principal_variation,
        nodes=int(out_nodes.value),
        aborted=bool(out_aborted.value),
        completed_depth=int(out_completed_depth.value) if iterative else int(depth),
    )


def request_native_search_stop() -> bool:
    """Ask a running native search to return at its next polling checkpoint."""
    library = _load_library()
    if library is None or not hasattr(library, "deeptactics_request_stop"):
        return False
    library.deeptactics_request_stop()
    return True


DEFAULT_FORCED_LINE_EXTENSION_BUDGET = 1
DEFAULT_MAJOR_PIECE_SAFETY_ORDERING = True


def set_native_search_experiments(
    *,
    forced_line_extension_budget: int = DEFAULT_FORCED_LINE_EXTENSION_BUDGET,
    mate_danger_ordering: bool = False,
    defensive_mate_threat_ordering: bool = False,
    major_piece_safety_ordering: bool = DEFAULT_MAJOR_PIECE_SAFETY_ORDERING,
    internal_iterative_deepening: bool = True,
    clustered_tt_replacement: bool = False,
    aspiration_windows: bool = True,
    pawn_endgame_extension_budget: int = 0,
    check_extension_budget: int = 0,
    singular_extension: bool = False,
    futility_pruning: bool = True,
    lmr_formula: bool = False,
    razoring: bool = False,
    continuation_history: bool = False,
    probcut: bool = False,
) -> bool:
    """Configure optional native search experiments for the current process."""
    global _native_internal_iterative_deepening
    _native_internal_iterative_deepening = bool(internal_iterative_deepening)
    library = _load_library()
    if library is None or not hasattr(library, "deeptactics_set_search_experiments"):
        return False
    library.deeptactics_set_search_experiments(
        max(0, min(2, int(forced_line_extension_budget))),
        int(bool(mate_danger_ordering)),
    )
    if hasattr(library, "deeptactics_set_defensive_mate_threat_ordering"):
        library.deeptactics_set_defensive_mate_threat_ordering(int(bool(defensive_mate_threat_ordering)))
    if hasattr(library, "deeptactics_set_major_piece_safety_ordering"):
        library.deeptactics_set_major_piece_safety_ordering(int(bool(major_piece_safety_ordering)))
    if hasattr(library, "deeptactics_set_clustered_tt_replacement"):
        library.deeptactics_set_clustered_tt_replacement(int(bool(clustered_tt_replacement)))
    if hasattr(library, "deeptactics_set_aspiration_windows"):
        library.deeptactics_set_aspiration_windows(int(bool(aspiration_windows)))
    if hasattr(library, "deeptactics_set_pawn_endgame_extension_budget"):
        library.deeptactics_set_pawn_endgame_extension_budget(
            max(0, min(2, int(pawn_endgame_extension_budget)))
        )
    if hasattr(library, "deeptactics_set_check_extension_budget"):
        library.deeptactics_set_check_extension_budget(max(0, min(2, int(check_extension_budget))))
    if hasattr(library, "deeptactics_set_singular_extension"):
        library.deeptactics_set_singular_extension(int(bool(singular_extension)))
    if hasattr(library, "deeptactics_set_futility_pruning"):
        library.deeptactics_set_futility_pruning(int(bool(futility_pruning)))
    if hasattr(library, "deeptactics_set_lmr_formula"):
        library.deeptactics_set_lmr_formula(int(bool(lmr_formula)))
    if hasattr(library, "deeptactics_set_razoring"):
        library.deeptactics_set_razoring(int(bool(razoring)))
    if hasattr(library, "deeptactics_set_continuation_history"):
        library.deeptactics_set_continuation_history(int(bool(continuation_history)))
    if hasattr(library, "deeptactics_set_probcut"):
        library.deeptactics_set_probcut(int(bool(probcut)))
    return True


def native_internal_iterative_deepening_enabled() -> bool:
    """Return whether timed native searches should reuse one C++ iterative session."""
    return _native_internal_iterative_deepening


def set_native_hash_mb(hash_mb: int) -> bool:
    """Set native transposition-table capacity for subsequent searches."""
    library = _load_library()
    if library is None or not hasattr(library, "deeptactics_set_hash_mb"):
        return False
    library.deeptactics_set_hash_mb(max(1, min(4096, int(hash_mb))))
    return True


def native_hash_status() -> dict[str, int] | None:
    """Return the configured native TT capacity when supported by the DLL."""
    library = _load_library()
    if (
        library is None or
        not hasattr(library, "deeptactics_get_hash_mb") or
        not hasattr(library, "deeptactics_get_hash_bucket_count")
    ):
        return None
    return {
        "hash_mb": int(library.deeptactics_get_hash_mb()),
        "bucket_count": int(library.deeptactics_get_hash_bucket_count()),
    }


def native_search_diagnostics() -> dict[str, int] | None:
    """Return counters from the most recent native search on this thread."""
    library = _load_library()
    if (
        library is None or
        not hasattr(library, "deeptactics_get_null_move_probes") or
        not hasattr(library, "deeptactics_get_null_move_cutoffs") or
        not hasattr(library, "deeptactics_get_nested_null_move_probes") or
        not hasattr(library, "deeptactics_get_aspiration_attempts") or
        not hasattr(library, "deeptactics_get_aspiration_fail_lows") or
        not hasattr(library, "deeptactics_get_aspiration_fail_highs") or
        not hasattr(library, "deeptactics_get_aspiration_researches") or
        not hasattr(library, "deeptactics_get_quiet_beta_cutoffs") or
        not hasattr(library, "deeptactics_get_killer_order_hits") or
        not hasattr(library, "deeptactics_get_history_order_hits") or
        not hasattr(library, "deeptactics_get_history_saturated_updates")
    ):
        return None
    return {
        "null_move_probes": int(library.deeptactics_get_null_move_probes()),
        "null_move_cutoffs": int(library.deeptactics_get_null_move_cutoffs()),
        "nested_null_move_probes": int(library.deeptactics_get_nested_null_move_probes()),
        "aspiration_attempts": int(library.deeptactics_get_aspiration_attempts()),
        "aspiration_fail_lows": int(library.deeptactics_get_aspiration_fail_lows()),
        "aspiration_fail_highs": int(library.deeptactics_get_aspiration_fail_highs()),
        "aspiration_researches": int(library.deeptactics_get_aspiration_researches()),
        "quiet_beta_cutoffs": int(library.deeptactics_get_quiet_beta_cutoffs()),
        "killer_order_hits": int(library.deeptactics_get_killer_order_hits()),
        "history_order_hits": int(library.deeptactics_get_history_order_hits()),
        "history_saturated_updates": int(library.deeptactics_get_history_saturated_updates()),
    }


def _pack_root_move_penalties(root_move_penalties: Mapping[str, int] | None):
    if not root_move_penalties:
        return None, None, 0
    move_values: list[int] = []
    penalty_values: list[int] = []
    for uci, penalty in root_move_penalties.items():
        try:
            move = Move.from_uci(uci)
        except ValueError:
            continue
        move_values.append(_encode_move_identity(move))
        penalty_values.append(int(penalty))
    count = len(move_values)
    if count == 0:
        return None, None, 0
    move_array_type = ctypes.c_int * count
    penalty_array_type = ctypes.c_int * count
    return move_array_type(*move_values), penalty_array_type(*penalty_values), count


def _encode_move_identity(move: Move) -> int:
    from_index = move.from_square.rank * 8 + move.from_square.file
    to_index = move.to_square.rank * 8 + move.to_square.file
    promotion_code = _PROMOTION_CODE_BY_TYPE.get(move.promotion, 0)
    return from_index | (to_index << 6) | (promotion_code << 12)


def _encode_move(move: Move) -> int:
    flags = int(move.is_en_passant) | (int(move.is_castling) << 1)
    return _encode_move_identity(move) | (flags << 16)
