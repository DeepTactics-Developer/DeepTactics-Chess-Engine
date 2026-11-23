"""Sparse NNUE feature extraction.

The first NNUE contract uses king-square conditioned piece-square occupancy:

- perspective: White and Black each receive their own feature plane
- king square: the perspective side's king location
- piece-square occupancy: every occupied square with color and piece type

This is deliberately close to classic NNUE input design while staying simple
enough to inspect in CSV form. Labels are stored from White's point of view so
they can be compared directly with the existing handcrafted evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass

from chess_engine.chess_core.board import Board
from chess_engine.chess_core.enums import Color, PieceType
from chess_engine.chess_core.fen import board_from_fen
from chess_engine.chess_core.square import Square


PIECE_ORDER: tuple[tuple[Color, PieceType], ...] = (
    (Color.WHITE, PieceType.PAWN),
    (Color.WHITE, PieceType.KNIGHT),
    (Color.WHITE, PieceType.BISHOP),
    (Color.WHITE, PieceType.ROOK),
    (Color.WHITE, PieceType.QUEEN),
    (Color.WHITE, PieceType.KING),
    (Color.BLACK, PieceType.PAWN),
    (Color.BLACK, PieceType.KNIGHT),
    (Color.BLACK, PieceType.BISHOP),
    (Color.BLACK, PieceType.ROOK),
    (Color.BLACK, PieceType.QUEEN),
    (Color.BLACK, PieceType.KING),
)

PIECE_INDEX = {piece: index for index, piece in enumerate(PIECE_ORDER)}
PIECE_SQUARE_COUNT = len(PIECE_ORDER) * 64
KING_CONDITIONED_COUNT = 2 * 64 * PIECE_SQUARE_COUNT
SIDE_TO_MOVE_FEATURE = KING_CONDITIONED_COUNT
CASTLING_FEATURE_OFFSET = SIDE_TO_MOVE_FEATURE + 1
EN_PASSANT_FILE_OFFSET = CASTLING_FEATURE_OFFSET + 4
NNUE_FEATURE_COUNT = EN_PASSANT_FILE_OFFSET + 8
RESULT_TO_CP = 600
MAX_LABEL_CP = 2_000
MATE_LABEL_CP = 1_500


@dataclass(frozen=True, slots=True)
class NNUEFeatureVector:
    """A sparse NNUE input vector."""

    active_indices: tuple[int, ...]
    feature_count: int = NNUE_FEATURE_COUNT

    def to_sparse_text(self) -> str:
        """Return active indices in a compact space-separated form."""
        return " ".join(str(index) for index in self.active_indices)


def extract_nnue_features(board: Board) -> NNUEFeatureVector:
    """Extract sparse NNUE features from a board position."""
    white_king = _king_square(board, Color.WHITE)
    black_king = _king_square(board, Color.BLACK)
    if white_king is None or black_king is None:
        raise ValueError("NNUE features require both kings to be present")

    indices: set[int] = set()
    for perspective, king_square in ((Color.WHITE, white_king), (Color.BLACK, black_king)):
        perspective_index = 0 if perspective is Color.WHITE else 1
        perspective_base = perspective_index * 64 * PIECE_SQUARE_COUNT
        king_base = perspective_base + _square_index(king_square) * PIECE_SQUARE_COUNT
        for square, piece in board.pieces():
            piece_index = PIECE_INDEX[(piece.color, piece.piece_type)]
            indices.add(king_base + piece_index * 64 + _square_index(square))

    if board.side_to_move is Color.BLACK:
        indices.add(SIDE_TO_MOVE_FEATURE)
    if board.castling_rights.white_king_side:
        indices.add(CASTLING_FEATURE_OFFSET)
    if board.castling_rights.white_queen_side:
        indices.add(CASTLING_FEATURE_OFFSET + 1)
    if board.castling_rights.black_king_side:
        indices.add(CASTLING_FEATURE_OFFSET + 2)
    if board.castling_rights.black_queen_side:
        indices.add(CASTLING_FEATURE_OFFSET + 3)
    if board.en_passant_target is not None:
        indices.add(EN_PASSANT_FILE_OFFSET + board.en_passant_target.file)

    return NNUEFeatureVector(tuple(sorted(indices)))


def nnue_features_from_fen(fen: str) -> NNUEFeatureVector:
    """Extract sparse NNUE features from a FEN string."""
    return extract_nnue_features(board_from_fen(normalize_fen(fen)))


def normalize_fen(fen: str) -> str:
    """Return a full six-field FEN.

    Reference evaluation rows contain the first four FEN fields only: piece
    placement, active color, castling rights, and en-passant square. The engine
    board loader expects full FEN, so halfmove/fullmove counters are appended
    when they are missing.
    """
    fields = fen.split()
    if len(fields) == 4:
        return f"{fen} 0 1"
    if len(fields) == 6:
        return fen
    raise ValueError("FEN must contain either 4 or 6 fields")


def label_from_training_row(row: dict[str, object], mode: str = "blended") -> int:
    """Return a White-perspective centipawn target for an NNUE sample.

    Supported modes:

    - ``score``: use the self-play search score column directly.
    - ``result``: convert final game result to a coarse centipawn target.
    - ``blended``: combine search score with final result for a less noisy
      first-generation training signal.
    """
    mode = mode.lower()
    score_cp = _safe_int(row.get("score"), 0)
    result_cp = int(round(_result_value(row) * RESULT_TO_CP))
    if mode == "score":
        return _clamp_int(score_cp, -MAX_LABEL_CP, MAX_LABEL_CP)
    if mode == "result":
        return _clamp_int(result_cp, -MAX_LABEL_CP, MAX_LABEL_CP)
    if mode == "blended":
        blended = int(round(score_cp * 0.65 + result_cp * 0.35))
        return _clamp_int(blended, -MAX_LABEL_CP, MAX_LABEL_CP)
    raise ValueError("label mode must be 'score', 'result', or 'blended'")


def _result_value(row: dict[str, object]) -> float:
    if "result_value" in row:
        try:
            return float(row["result_value"])
        except (TypeError, ValueError):
            pass
    result = str(row.get("result", "*"))
    if result == "1-0":
        return 1.0
    if result == "0-1":
        return -1.0
    return 0.0


def _king_square(board: Board, color: Color) -> Square | None:
    for square, piece in board.pieces(color):
        if piece.piece_type is PieceType.KING:
            return square
    return None


def _square_index(square: Square) -> int:
    return square.rank * 8 + square.file


def _safe_int(value: object, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _clamp_int(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))
