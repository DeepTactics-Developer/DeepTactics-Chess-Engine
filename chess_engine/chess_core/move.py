"""Move representation.

The engine represents moves as data, not UI actions. A console UI may parse
``e2e4`` into a Move, and a GUI may create the same Move from mouse
clicks. The core rules should not care where the Move came from.
"""

from __future__ import annotations

from dataclasses import dataclass

from chess_engine.chess_core.enums import PieceType
from chess_engine.chess_core.square import Square


PROMOTION_PIECE_TYPES = {
    PieceType.QUEEN,
    PieceType.ROOK,
    PieceType.BISHOP,
    PieceType.KNIGHT,
}

PROMOTION_TO_UCI = {
    PieceType.QUEEN: "q",
    PieceType.ROOK: "r",
    PieceType.BISHOP: "b",
    PieceType.KNIGHT: "n",
}

UCI_TO_PROMOTION = {value: key for key, value in PROMOTION_TO_UCI.items()}


@dataclass(frozen=True, slots=True)
class Move:
    """A single chess move.

    Special flags are explicit so later rule code can mark castling and
    en passant without making UI code guess from coordinates alone.
    """

    from_square: Square
    to_square: Square
    promotion: PieceType | None = None
    is_castling: bool = False
    is_en_passant: bool = False

    def __post_init__(self) -> None:
        if self.promotion is not None and self.promotion not in PROMOTION_PIECE_TYPES:
            raise ValueError("promotion must be queen, rook, bishop, or knight")

    @classmethod
    def from_uci(cls, value: str) -> "Move":
        """Create a move from UCI text such as 'e2e4' or 'a7a8q'."""
        if len(value) not in (4, 5):
            raise ValueError(f"move must look like e2e4 or a7a8q: {value!r}")

        from_square = Square.from_algebraic(value[:2])
        to_square = Square.from_algebraic(value[2:4])
        promotion = None
        if len(value) == 5:
            promotion_char = value[4].lower()
            if promotion_char not in UCI_TO_PROMOTION:
                raise ValueError(f"invalid promotion piece: {value[4]!r}")
            promotion = UCI_TO_PROMOTION[promotion_char]

        return cls(from_square=from_square, to_square=to_square, promotion=promotion)

    def to_uci(self) -> str:
        """Return UCI text such as 'e2e4' or 'a7a8q'."""
        text = f"{self.from_square}{self.to_square}"
        if self.promotion is not None:
            text += PROMOTION_TO_UCI[self.promotion]
        return text

    def __str__(self) -> str:
        return self.to_uci()

