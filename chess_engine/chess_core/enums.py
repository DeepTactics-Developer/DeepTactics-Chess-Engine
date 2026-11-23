"""Shared chess enums."""

from __future__ import annotations

from enum import Enum


class Color(Enum):
    """The side a piece belongs to."""

    WHITE = "white"
    BLACK = "black"

    @property
    def opponent(self) -> "Color":
        """Return the opposite side."""
        return Color.BLACK if self is Color.WHITE else Color.WHITE


class PieceType(Enum):
    """The six standard chess piece types."""

    PAWN = "pawn"
    KNIGHT = "knight"
    BISHOP = "bishop"
    ROOK = "rook"
    QUEEN = "queen"
    KING = "king"


class CastlingSide(Enum):
    """The two castling directions."""

    KING_SIDE = "king_side"
    QUEEN_SIDE = "queen_side"


class GameStatus(Enum):
    """High-level game status for UI and orchestration code."""

    ONGOING = "ongoing"
    CHECKMATE = "checkmate"
    STALEMATE = "stalemate"
    THREEFOLD_REPETITION = "threefold_repetition"
    FIFTY_MOVE_RULE = "fifty_move_rule"
    INSUFFICIENT_MATERIAL = "insufficient_material"
