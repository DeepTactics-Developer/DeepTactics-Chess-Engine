"""Chess piece models.

Pieces are plain data objects. They do not know about the board, the UI, or
whether a move is legal. Rule generation will live in ``rules.py`` later.
"""

from __future__ import annotations

from dataclasses import dataclass

from chess_engine.chess_core.enums import Color, PieceType


FEN_SYMBOLS = {
    PieceType.PAWN: "p",
    PieceType.KNIGHT: "n",
    PieceType.BISHOP: "b",
    PieceType.ROOK: "r",
    PieceType.QUEEN: "q",
    PieceType.KING: "k",
}

DISPLAY_SYMBOLS = {
    (Color.WHITE, PieceType.PAWN): "P",
    (Color.WHITE, PieceType.KNIGHT): "N",
    (Color.WHITE, PieceType.BISHOP): "B",
    (Color.WHITE, PieceType.ROOK): "R",
    (Color.WHITE, PieceType.QUEEN): "Q",
    (Color.WHITE, PieceType.KING): "K",
    (Color.BLACK, PieceType.PAWN): "p",
    (Color.BLACK, PieceType.KNIGHT): "n",
    (Color.BLACK, PieceType.BISHOP): "b",
    (Color.BLACK, PieceType.ROOK): "r",
    (Color.BLACK, PieceType.QUEEN): "q",
    (Color.BLACK, PieceType.KING): "k",
}


@dataclass(frozen=True, slots=True)
class Piece:
    """A chess piece with color and type."""

    color: Color
    piece_type: PieceType

    @property
    def fen_symbol(self) -> str:
        """Return this piece's FEN character."""
        symbol = FEN_SYMBOLS[self.piece_type]
        return symbol.upper() if self.color is Color.WHITE else symbol

    @property
    def display_symbol(self) -> str:
        """Return a simple text symbol suitable for console rendering."""
        return DISPLAY_SYMBOLS[(self.color, self.piece_type)]

    def __str__(self) -> str:
        return self.display_symbol

