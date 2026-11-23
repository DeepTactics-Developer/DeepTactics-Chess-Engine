"""Game orchestration.

``Game`` is the layer UI code should talk to. It owns a ``Board``, validates
incoming ``Move`` objects, records move history, and reports the current game
status. It does not know whether a move came from console text, a mouse click,
or an AI search.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from chess_engine.chess_core.board import Board
from chess_engine.chess_core.enums import Color, GameStatus, PieceType
from chess_engine.chess_core.fen import board_to_fen
from chess_engine.chess_core.move import Move
from chess_engine.chess_core.pgn import PgnGame, move_to_san, result_from_status
from chess_engine.chess_core.piece import Piece
from chess_engine.chess_core.rules import (
    generate_legal_moves,
    is_checkmate,
    is_in_check,
    is_stalemate,
)


@dataclass(slots=True)
class Game:
    """High-level game controller."""

    board: Board = field(default_factory=Board.starting_position)
    move_history: list[Move] = field(default_factory=list)
    san_history: list[str] = field(default_factory=list)
    board_history: list[Board] = field(default_factory=list)
    captured_history: list[Piece | None] = field(default_factory=list)

    @classmethod
    def new(cls) -> "Game":
        """Create a new game from the standard starting position."""
        return cls()

    def legal_moves(self) -> list[Move]:
        """Return legal moves for the side to move."""
        return generate_legal_moves(self.board)

    def apply_move(self, move: Move) -> None:
        """Validate and apply a move.

        UI layers should call this instead of mutating the board directly.
        """
        legal_move = self._matching_legal_move(move)
        if legal_move is None:
            raise ValueError(f"illegal move: {move}")

        san = move_to_san(self.board, legal_move)
        self.board_history.append(self.board.copy())
        captured_piece = self.board.apply_move_unchecked(legal_move)
        self.move_history.append(legal_move)
        self.san_history.append(san)
        self.captured_history.append(captured_piece)

    def undo_last_move(self) -> Move | None:
        """Undo the most recent move and return it."""
        if not self.board_history:
            return None

        undone_move = self.move_history.pop()
        self.san_history.pop()
        self.captured_history.pop()
        self.board = self.board_history.pop()
        return undone_move

    def reset(self) -> None:
        """Reset the game to the standard starting position."""
        self.board = Board.starting_position()
        self.move_history.clear()
        self.san_history.clear()
        self.board_history.clear()
        self.captured_history.clear()

    def captured_by(self, color: Color) -> list[Piece]:
        """Return pieces captured by ``color`` in move order."""
        return [
            piece
            for piece in self.captured_history
            if piece is not None and piece.color is color.opponent
        ]

    def status(self) -> GameStatus:
        """Return the current game status."""
        if is_checkmate(self.board):
            return GameStatus.CHECKMATE
        if is_stalemate(self.board):
            return GameStatus.STALEMATE
        if self.is_threefold_repetition():
            return GameStatus.THREEFOLD_REPETITION
        if self.board.halfmove_clock >= 100:
            return GameStatus.FIFTY_MOVE_RULE
        if has_insufficient_material(self.board):
            return GameStatus.INSUFFICIENT_MATERIAL
        return GameStatus.ONGOING

    def is_over(self) -> bool:
        """Return whether the game has ended."""
        return self.status() is not GameStatus.ONGOING

    def winner(self) -> Color | None:
        """Return the winner, or None if the game is ongoing or drawn."""
        if self.status() is GameStatus.CHECKMATE:
            return self.board.side_to_move.opponent
        return None

    def is_in_check(self) -> bool:
        """Return whether the side to move is currently in check."""
        return is_in_check(self.board, self.board.side_to_move)

    def is_threefold_repetition(self) -> bool:
        """Return whether the current position has occurred at least 3 times."""
        current_key = position_repetition_key(self.board)
        count = 1
        for board in self.board_history:
            if position_repetition_key(board) == current_key:
                count += 1
        return count >= 3

    def to_pgn(self) -> str:
        """Return a minimal PGN export for the current game."""
        return PgnGame(
            san_moves=self.san_history.copy(),
            result=result_from_status(self.status(), self.board.side_to_move),
        ).to_pgn()

    def _matching_legal_move(self, move: Move) -> Move | None:
        """Return the legal move matching user-facing UCI text."""
        for legal_move in self.legal_moves():
            if legal_move.to_uci() == move.to_uci():
                return legal_move
        return None


def position_repetition_key(board: Board) -> str:
    """Return the FEN fields relevant to threefold repetition."""
    return " ".join(board_to_fen(board).split()[:4])


def has_insufficient_material(board: Board) -> bool:
    """Return whether neither side has enough material to checkmate."""
    pieces = list(board.pieces())
    non_kings = [(square, piece) for square, piece in pieces if piece.piece_type is not PieceType.KING]
    if not non_kings:
        return True
    if len(non_kings) == 1:
        return non_kings[0][1].piece_type in {PieceType.BISHOP, PieceType.KNIGHT}
    if len(non_kings) == 2 and all(piece.piece_type is PieceType.BISHOP for _, piece in non_kings):
        first_square = non_kings[0][0]
        second_square = non_kings[1][0]
        return (first_square.file + first_square.rank) % 2 == (second_square.file + second_square.rank) % 2
    return False
