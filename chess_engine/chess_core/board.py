"""Board state representation.

``Board`` is the core game-state object. It does not print the board, read
keyboard input, or know about any UI toolkit. UI layers should ask the board for state
and send ``Move`` objects back to the engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from chess_engine.chess_core.enums import CastlingSide, Color, PieceType
from chess_engine.chess_core.move import Move
from chess_engine.chess_core.piece import Piece
from chess_engine.chess_core.square import BOARD_SIZE, Square


Grid = list[list[Piece | None]]


@dataclass(slots=True)
class CastlingRights:
    """Tracks whether each side may still castle in each direction."""

    white_king_side: bool = True
    white_queen_side: bool = True
    black_king_side: bool = True
    black_queen_side: bool = True

    def can_castle(self, color: Color, side: CastlingSide) -> bool:
        """Return whether the given side still has this castling right."""
        if color is Color.WHITE and side is CastlingSide.KING_SIDE:
            return self.white_king_side
        if color is Color.WHITE and side is CastlingSide.QUEEN_SIDE:
            return self.white_queen_side
        if color is Color.BLACK and side is CastlingSide.KING_SIDE:
            return self.black_king_side
        return self.black_queen_side

    def revoke(self, color: Color, side: CastlingSide | None = None) -> None:
        """Remove one or both castling rights for a color."""
        if color is Color.WHITE:
            if side in (None, CastlingSide.KING_SIDE):
                self.white_king_side = False
            if side in (None, CastlingSide.QUEEN_SIDE):
                self.white_queen_side = False
            return

        if side in (None, CastlingSide.KING_SIDE):
            self.black_king_side = False
        if side in (None, CastlingSide.QUEEN_SIDE):
            self.black_queen_side = False


@dataclass(slots=True)
class Board:
    """Represents a chess board state.

    The board stores engine coordinates only. Rank 1 is index 0, rank 8 is
    index 7, file 'a' is index 0, and file 'h' is index 7.
    """

    grid: Grid = field(default_factory=lambda: Board.empty_grid())
    side_to_move: Color = Color.WHITE
    castling_rights: CastlingRights = field(default_factory=CastlingRights)
    en_passant_target: Square | None = None
    halfmove_clock: int = 0
    fullmove_number: int = 1

    @staticmethod
    def empty_grid() -> Grid:
        """Create an empty 8x8 grid."""
        return [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

    @classmethod
    def empty(cls) -> "Board":
        """Create an empty board."""
        return cls(
            grid=cls.empty_grid(),
            castling_rights=CastlingRights(False, False, False, False),
        )

    @classmethod
    def starting_position(cls) -> "Board":
        """Create the standard chess starting position."""
        board = cls()

        back_rank = [
            PieceType.ROOK,
            PieceType.KNIGHT,
            PieceType.BISHOP,
            PieceType.QUEEN,
            PieceType.KING,
            PieceType.BISHOP,
            PieceType.KNIGHT,
            PieceType.ROOK,
        ]

        for file, piece_type in enumerate(back_rank):
            board.set_piece(Square(file, 0), Piece(Color.WHITE, piece_type))
            board.set_piece(Square(file, 7), Piece(Color.BLACK, piece_type))

        for file in range(BOARD_SIZE):
            board.set_piece(Square(file, 1), Piece(Color.WHITE, PieceType.PAWN))
            board.set_piece(Square(file, 6), Piece(Color.BLACK, PieceType.PAWN))

        return board

    def copy(self) -> "Board":
        """Return a copy of the board state."""
        copied_grid = [rank.copy() for rank in self.grid]
        copied_rights = CastlingRights(
            white_king_side=self.castling_rights.white_king_side,
            white_queen_side=self.castling_rights.white_queen_side,
            black_king_side=self.castling_rights.black_king_side,
            black_queen_side=self.castling_rights.black_queen_side,
        )
        return Board(
            grid=copied_grid,
            side_to_move=self.side_to_move,
            castling_rights=copied_rights,
            en_passant_target=self.en_passant_target,
            halfmove_clock=self.halfmove_clock,
            fullmove_number=self.fullmove_number,
        )

    def get_piece(self, square: Square) -> Piece | None:
        """Return the piece on a square, or None if the square is empty."""
        return self.grid[square.rank][square.file]

    def set_piece(self, square: Square, piece: Piece | None) -> None:
        """Place a piece on a square, or clear it with None."""
        self.grid[square.rank][square.file] = piece

    def remove_piece(self, square: Square) -> Piece | None:
        """Remove and return the piece on a square."""
        piece = self.get_piece(square)
        self.set_piece(square, None)
        return piece

    def apply_move_unchecked(self, move: Move) -> Piece | None:
        """Apply a move without checking whether it is legal.

        Legal move validation lives in ``rules.py``. This method is a state
        transition helper used after a move has already been validated.
        """
        piece = self.get_piece(move.from_square)
        if piece is None:
            raise ValueError(f"no piece on {move.from_square}")

        captured_piece = self._captured_piece_for_move(move)
        previous_en_passant_target = self.en_passant_target

        self._update_castling_rights_for_move(move, piece, captured_piece)
        self.en_passant_target = None

        if move.is_castling:
            self._apply_castling_move(move, piece)
        else:
            self.remove_piece(move.from_square)
            if move.is_en_passant:
                self.remove_piece(Square(move.to_square.file, move.from_square.rank))

            self.set_piece(move.to_square, self._moved_piece_after_promotion(move, piece))

        self._update_en_passant_target(move, piece, previous_en_passant_target)
        self._update_halfmove_clock(piece, captured_piece)
        self._advance_turn()
        return captured_piece

    def _captured_piece_for_move(self, move: Move) -> Piece | None:
        if move.is_en_passant:
            return self.get_piece(Square(move.to_square.file, move.from_square.rank))
        return self.get_piece(move.to_square)

    def _moved_piece_after_promotion(self, move: Move, piece: Piece) -> Piece:
        moved_piece = piece
        if move.promotion is not None:
            moved_piece = Piece(piece.color, move.promotion)
        return moved_piece

    def _apply_castling_move(self, move: Move, king: Piece) -> None:
        self.remove_piece(move.from_square)
        self.set_piece(move.to_square, king)

        rank = move.from_square.rank
        if move.to_square.file == 6:
            rook_from = Square(7, rank)
            rook_to = Square(5, rank)
        else:
            rook_from = Square(0, rank)
            rook_to = Square(3, rank)

        rook = self.remove_piece(rook_from)
        if rook is None:
            raise ValueError(f"no rook on {rook_from} for castling")
        self.set_piece(rook_to, rook)

    def _update_en_passant_target(
        self,
        move: Move,
        piece: Piece,
        previous_target: Square | None,
    ) -> None:
        del previous_target
        if piece.piece_type is not PieceType.PAWN:
            return

        if abs(move.to_square.rank - move.from_square.rank) == 2:
            middle_rank = (move.from_square.rank + move.to_square.rank) // 2
            self.en_passant_target = Square(move.from_square.file, middle_rank)

    def _update_halfmove_clock(self, piece: Piece, captured_piece: Piece | None) -> None:
        if piece.piece_type is PieceType.PAWN or captured_piece is not None:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1

    def _update_castling_rights_for_move(
        self,
        move: Move,
        piece: Piece,
        captured_piece: Piece | None,
    ) -> None:
        if piece.piece_type is PieceType.KING:
            self.castling_rights.revoke(piece.color)

        if piece.piece_type is PieceType.ROOK:
            self._revoke_rook_castling_right(move.from_square, piece.color)

        if captured_piece is not None and captured_piece.piece_type is PieceType.ROOK:
            self._revoke_rook_castling_right(move.to_square, captured_piece.color)

    def _revoke_rook_castling_right(self, square: Square, color: Color) -> None:
        home_rank = 0 if color is Color.WHITE else 7
        if square.rank != home_rank:
            return

        if square.file == 0:
            self.castling_rights.revoke(color, CastlingSide.QUEEN_SIDE)
        elif square.file == 7:
            self.castling_rights.revoke(color, CastlingSide.KING_SIDE)

    def iter_squares(self) -> Iterator[tuple[Square, Piece | None]]:
        """Yield every square and its current piece."""
        for rank in range(BOARD_SIZE):
            for file in range(BOARD_SIZE):
                square = Square(file, rank)
                yield square, self.get_piece(square)

    def pieces(self, color: Color | None = None) -> Iterator[tuple[Square, Piece]]:
        """Yield occupied squares, optionally filtered by color."""
        for square, piece in self.iter_squares():
            if piece is not None and (color is None or piece.color is color):
                yield square, piece

    def _advance_turn(self) -> None:
        """Switch sides and update the fullmove number after black moves."""
        if self.side_to_move is Color.BLACK:
            self.fullmove_number += 1
        self.side_to_move = self.side_to_move.opponent
