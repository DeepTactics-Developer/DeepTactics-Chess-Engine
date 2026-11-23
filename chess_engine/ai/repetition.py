"""Helpers that discourage avoidable repeated positions at the root.

These helpers do not change the chess rules. Threefold repetition is still
handled by ``Game``. The goal here is only practical engine behavior: when two
AI players have many equal-looking choices, prefer a fresh position over a
meaningless back-and-forth loop.
"""

from __future__ import annotations

from collections.abc import Iterable

from chess_engine.chess_core.board import Board
from chess_engine.chess_core.enums import Color, PieceType
from chess_engine.chess_core.game import position_repetition_key
from chess_engine.chess_core.move import Move
from chess_engine.chess_core.rules import generate_legal_moves


TWOFOLD_REPETITION_PENALTY = 80
THREEFOLD_REPETITION_PENALTY = 20_000
OPENING_REPEATED_PIECE_PENALTY = 45


def combined_root_move_penalties(
    board: Board,
    previous_boards: Iterable[Board],
    move_history: Iterable[Move],
) -> dict[str, int]:
    """Combine repetition and opening-tempo penalties without overwriting either."""

    penalties = root_repetition_penalties(board, previous_boards)
    for move_uci, penalty in opening_repeated_piece_penalties(board, move_history).items():
        penalties[move_uci] = penalties.get(move_uci, 0) + penalty
    return penalties


def root_repetition_penalties(
    board: Board,
    previous_boards: Iterable[Board],
    *,
    twofold_penalty: int = TWOFOLD_REPETITION_PENALTY,
    threefold_penalty: int = THREEFOLD_REPETITION_PENALTY,
) -> dict[str, int]:
    """Return root move penalties keyed by UCI move.

    A move that would create the second occurrence of a position receives a
    small penalty. A move that would create a third occurrence receives a very
    large penalty unless no better legal alternative exists.
    """

    previous_counts: dict[str, int] = {}
    for previous_board in previous_boards:
        key = position_repetition_key(previous_board)
        previous_counts[key] = previous_counts.get(key, 0) + 1

    penalties: dict[str, int] = {}
    for move in generate_legal_moves(board):
        penalties[move.to_uci()] = repetition_penalty_for_move(
            board,
            move,
            previous_counts,
            twofold_penalty=twofold_penalty,
            threefold_penalty=threefold_penalty,
        )
    return penalties


def opening_repeated_piece_penalties(
    board: Board,
    move_history: Iterable[Move],
    *,
    max_plies: int = 12,
    penalty: int = OPENING_REPEATED_PIECE_PENALTY,
) -> dict[str, int]:
    """Discourage moving the same developed piece again in the opening.

    A pure board position cannot tell whether a knight has already moved twice.
    This helper uses the current game's move history and applies a root-only
    practical penalty to legal moves that pick up a recently developed piece
    again. It is intentionally mild: tactics and checkmates still override it.
    """

    history = list(move_history)
    if len(history) >= max_plies:
        return {}

    side_to_move_offset = 0 if board.side_to_move is Color.WHITE else 1
    recent_destinations = {
        move.to_square
        for index, move in enumerate(history[-8:], start=max(0, len(history) - 8))
        if index % 2 == side_to_move_offset
    }
    if not recent_destinations:
        return {}

    penalties: dict[str, int] = {}
    for move in generate_legal_moves(board):
        piece = board.get_piece(move.from_square)
        is_capture = move.is_en_passant or board.get_piece(move.to_square) is not None
        # This is a development-tempo nudge, not a tactical restriction.
        # Normal pawn breaks/captures and forced exchanges must remain neutral.
        if (
            move.from_square in recent_destinations
            and piece is not None
            and piece.piece_type is not PieceType.PAWN
            and not is_capture
        ):
            penalties[move.to_uci()] = penalty
    return penalties


def repetition_penalty_for_move(
    board: Board,
    move: Move,
    previous_counts: dict[str, int],
    *,
    twofold_penalty: int = TWOFOLD_REPETITION_PENALTY,
    threefold_penalty: int = THREEFOLD_REPETITION_PENALTY,
) -> int:
    """Return a practical search penalty for one candidate move."""

    next_board = board.copy()
    next_board.apply_move_unchecked(move)
    previous_count = previous_counts.get(position_repetition_key(next_board), 0)
    if previous_count >= 2:
        return threefold_penalty
    if previous_count == 1:
        return twofold_penalty
    return 0
