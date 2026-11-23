"""Chess rule helpers.

This module starts with pseudo-legal move generation. Pseudo-legal moves follow
piece movement rules. Legal move generation then filters out moves that leave
the moving side's king in check.
"""

from __future__ import annotations

from collections.abc import Iterable

from chess_engine.chess_core.board import Board
from chess_engine.chess_core.enums import CastlingSide, Color, PieceType
from chess_engine.chess_core.move import Move
from chess_engine.chess_core.piece import Piece
from chess_engine.chess_core.square import Square
from chess_engine.native.native_eval import (
    generate_legal_moves_native,
    generate_pseudo_legal_moves_native,
    is_square_attacked_native,
    native_available,
)


ORTHOGONAL_DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIAGONAL_DIRECTIONS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
KNIGHT_OFFSETS = (
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
)
KING_OFFSETS = ORTHOGONAL_DIRECTIONS + DIAGONAL_DIRECTIONS
PROMOTION_CHOICES = (
    PieceType.QUEEN,
    PieceType.ROOK,
    PieceType.BISHOP,
    PieceType.KNIGHT,
)
USE_NATIVE_ATTACKS = native_available()


def generate_legal_moves(board: Board, color: Color | None = None) -> list[Move]:
    """Return moves that obey piece rules and do not leave the king in check."""
    moving_color = color or board.side_to_move
    if USE_NATIVE_ATTACKS:
        native_moves = generate_legal_moves_native(board, moving_color)
        if native_moves is not None:
            return native_moves
    return _generate_legal_moves_python(board, moving_color)


def _generate_legal_moves_python(board: Board, moving_color: Color) -> list[Move]:
    """Return Python legal moves, used as fallback and parity oracle."""
    legal_moves: list[Move] = []

    for move in _generate_pseudo_legal_moves_python(board, moving_color):
        next_board = board.copy()
        next_board.apply_move_unchecked(move)
        if not is_in_check(next_board, moving_color):
            legal_moves.append(move)

    return legal_moves


def has_legal_moves(board: Board, color: Color | None = None) -> bool:
    """Return whether a side has at least one legal move."""
    return bool(generate_legal_moves(board, color))


def is_checkmate(board: Board, color: Color | None = None) -> bool:
    """Return whether the given side is checkmated."""
    target_color = color or board.side_to_move
    return is_in_check(board, target_color) and not has_legal_moves(board, target_color)


def is_stalemate(board: Board, color: Color | None = None) -> bool:
    """Return whether the given side is stalemated."""
    target_color = color or board.side_to_move
    return not is_in_check(board, target_color) and not has_legal_moves(board, target_color)


def generate_pseudo_legal_moves(board: Board, color: Color | None = None) -> list[Move]:
    """Return all pseudo-legal moves for a side.

    If ``color`` is omitted, moves are generated for ``board.side_to_move``.
    """
    moving_color = color or board.side_to_move
    if USE_NATIVE_ATTACKS:
        native_moves = generate_pseudo_legal_moves_native(board, moving_color)
        if native_moves is not None:
            return native_moves
    return _generate_pseudo_legal_moves_python(board, moving_color)


def _generate_pseudo_legal_moves_python(board: Board, moving_color: Color) -> list[Move]:
    """Return Python pseudo-legal moves, used as fallback and parity oracle."""
    moves: list[Move] = []

    for square, piece in board.pieces(moving_color):
        moves.extend(generate_piece_moves(board, square, piece))

    return moves


def find_king(board: Board, color: Color) -> Square:
    """Return the square containing the given side's king."""
    for square, piece in board.pieces(color):
        if piece.piece_type is PieceType.KING:
            return square

    raise ValueError(f"{color.value} king is not on the board")


def is_in_check(board: Board, color: Color) -> bool:
    """Return whether the given side's king is currently attacked."""
    king_square = find_king(board, color)
    return is_square_attacked(board, king_square, color.opponent)


def is_square_attacked(board: Board, square: Square, by_color: Color) -> bool:
    """Return whether ``square`` is attacked by any piece of ``by_color``."""
    if USE_NATIVE_ATTACKS:
        native_result = is_square_attacked_native(board, square.file, square.rank, by_color)
        if native_result is not None:
            return native_result
    return _is_square_attacked_python(board, square, by_color)


def _is_square_attacked_python(board: Board, square: Square, by_color: Color) -> bool:
    """Return Python attack detection, used as fallback and parity oracle."""
    return (
        _is_attacked_by_pawn(board, square, by_color)
        or _is_attacked_by_knight(board, square, by_color)
        or _is_attacked_by_king(board, square, by_color)
        or _is_attacked_by_slider(
            board,
            square,
            by_color,
            DIAGONAL_DIRECTIONS,
            {PieceType.BISHOP, PieceType.QUEEN},
        )
        or _is_attacked_by_slider(
            board,
            square,
            by_color,
            ORTHOGONAL_DIRECTIONS,
            {PieceType.ROOK, PieceType.QUEEN},
        )
    )


def use_native_attacks(enabled: bool) -> None:
    """Enable or disable optional C++ attack detection."""
    global USE_NATIVE_ATTACKS
    USE_NATIVE_ATTACKS = enabled and native_available()


def native_attacks_enabled() -> bool:
    """Return whether optional C++ attack detection is active."""
    return USE_NATIVE_ATTACKS


def generate_piece_moves(board: Board, square: Square, piece: Piece | None = None) -> list[Move]:
    """Return pseudo-legal moves for a single piece on a square."""
    moving_piece = piece or board.get_piece(square)
    if moving_piece is None:
        return []

    if moving_piece.piece_type is PieceType.PAWN:
        return _pawn_moves(board, square, moving_piece.color)
    if moving_piece.piece_type is PieceType.KNIGHT:
        return _leaper_moves(board, square, moving_piece.color, KNIGHT_OFFSETS)
    if moving_piece.piece_type is PieceType.BISHOP:
        return _slider_moves(board, square, moving_piece.color, DIAGONAL_DIRECTIONS)
    if moving_piece.piece_type is PieceType.ROOK:
        return _slider_moves(board, square, moving_piece.color, ORTHOGONAL_DIRECTIONS)
    if moving_piece.piece_type is PieceType.QUEEN:
        return _slider_moves(
            board,
            square,
            moving_piece.color,
            ORTHOGONAL_DIRECTIONS + DIAGONAL_DIRECTIONS,
        )
    if moving_piece.piece_type is PieceType.KING:
        return _king_moves(board, square, moving_piece.color)

    return []


def _pawn_moves(board: Board, square: Square, color: Color) -> list[Move]:
    moves: list[Move] = []
    direction = 1 if color is Color.WHITE else -1
    start_rank = 1 if color is Color.WHITE else 6
    promotion_rank = 7 if color is Color.WHITE else 0

    one_step = square.offset(0, direction)
    if one_step is not None and board.get_piece(one_step) is None:
        moves.extend(_create_pawn_moves(square, one_step, promotion_rank))

        two_step = square.offset(0, direction * 2)
        if square.rank == start_rank and two_step is not None and board.get_piece(two_step) is None:
            moves.append(Move(square, two_step))

    for file_delta in (-1, 1):
        capture_square = square.offset(file_delta, direction)
        if capture_square is None:
            continue

        target = board.get_piece(capture_square)
        if target is not None and target.color is not color:
            moves.extend(_create_pawn_moves(square, capture_square, promotion_rank))
        elif board.en_passant_target == capture_square:
            moves.append(Move(square, capture_square, is_en_passant=True))

    return moves


def _king_moves(board: Board, square: Square, color: Color) -> list[Move]:
    moves = _leaper_moves(board, square, color, KING_OFFSETS)
    moves.extend(_castling_moves(board, square, color))
    return moves


def _castling_moves(board: Board, square: Square, color: Color) -> list[Move]:
    moves: list[Move] = []
    home_rank = 0 if color is Color.WHITE else 7
    if square != Square(4, home_rank):
        return moves

    king = board.get_piece(square)
    if king is None or king.piece_type is not PieceType.KING or king.color is not color:
        return moves

    if is_in_check(board, color):
        return moves

    moves.extend(_castling_move_for_side(board, square, color, CastlingSide.KING_SIDE))
    moves.extend(_castling_move_for_side(board, square, color, CastlingSide.QUEEN_SIDE))
    return moves


def _castling_move_for_side(
    board: Board,
    king_square: Square,
    color: Color,
    side: CastlingSide,
) -> list[Move]:
    if not board.castling_rights.can_castle(color, side):
        return []

    rank = king_square.rank
    if side is CastlingSide.KING_SIDE:
        rook_square = Square(7, rank)
        empty_squares = (Square(5, rank), Square(6, rank))
        safe_squares = empty_squares
        target_square = Square(6, rank)
    else:
        rook_square = Square(0, rank)
        empty_squares = (Square(1, rank), Square(2, rank), Square(3, rank))
        safe_squares = (Square(3, rank), Square(2, rank))
        target_square = Square(2, rank)

    rook = board.get_piece(rook_square)
    if rook != Piece(color, PieceType.ROOK):
        return []

    if any(board.get_piece(square) is not None for square in empty_squares):
        return []

    if any(is_square_attacked(board, square, color.opponent) for square in safe_squares):
        return []

    return [Move(king_square, target_square, is_castling=True)]


def _is_attacked_by_pawn(board: Board, square: Square, by_color: Color) -> bool:
    pawn_direction = 1 if by_color is Color.WHITE else -1

    for file_delta in (-1, 1):
        attacker_square = square.offset(file_delta, -pawn_direction)
        if _has_piece(board, attacker_square, by_color, PieceType.PAWN):
            return True

    return False


def _is_attacked_by_knight(board: Board, square: Square, by_color: Color) -> bool:
    for file_delta, rank_delta in KNIGHT_OFFSETS:
        attacker_square = square.offset(file_delta, rank_delta)
        if _has_piece(board, attacker_square, by_color, PieceType.KNIGHT):
            return True

    return False


def _is_attacked_by_king(board: Board, square: Square, by_color: Color) -> bool:
    for file_delta, rank_delta in KING_OFFSETS:
        attacker_square = square.offset(file_delta, rank_delta)
        if _has_piece(board, attacker_square, by_color, PieceType.KING):
            return True

    return False


def _is_attacked_by_slider(
    board: Board,
    square: Square,
    by_color: Color,
    directions: Iterable[tuple[int, int]],
    attacking_piece_types: set[PieceType],
) -> bool:
    for file_delta, rank_delta in directions:
        current = square
        while True:
            current = current.offset(file_delta, rank_delta)
            if current is None:
                break

            piece = board.get_piece(current)
            if piece is None:
                continue

            if piece.color is by_color and piece.piece_type in attacking_piece_types:
                return True
            break

    return False


def _has_piece(
    board: Board,
    square: Square | None,
    color: Color,
    piece_type: PieceType,
) -> bool:
    if square is None:
        return False

    piece = board.get_piece(square)
    return piece is not None and piece.color is color and piece.piece_type is piece_type


def _create_pawn_moves(from_square: Square, to_square: Square, promotion_rank: int) -> list[Move]:
    if to_square.rank != promotion_rank:
        return [Move(from_square, to_square)]

    return [
        Move(from_square, to_square, promotion=promotion)
        for promotion in PROMOTION_CHOICES
    ]


def _leaper_moves(
    board: Board,
    square: Square,
    color: Color,
    offsets: Iterable[tuple[int, int]],
) -> list[Move]:
    moves: list[Move] = []

    for file_delta, rank_delta in offsets:
        target_square = square.offset(file_delta, rank_delta)
        if target_square is None:
            continue

        target = board.get_piece(target_square)
        if target is None or target.color is not color:
            moves.append(Move(square, target_square))

    return moves


def _slider_moves(
    board: Board,
    square: Square,
    color: Color,
    directions: Iterable[tuple[int, int]],
) -> list[Move]:
    moves: list[Move] = []

    for file_delta, rank_delta in directions:
        current = square
        while True:
            current = current.offset(file_delta, rank_delta)
            if current is None:
                break

            target = board.get_piece(current)
            if target is None:
                moves.append(Move(square, current))
                continue

            if target.color is not color:
                moves.append(Move(square, current))
            break

    return moves
