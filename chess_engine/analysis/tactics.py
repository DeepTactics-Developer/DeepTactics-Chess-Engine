"""Lightweight tactical pattern detection for analysis labels.

These detectors translate book-style tactical ideas into compact tags. Search
still verifies the move; tags explain why a move has tactical character.
"""

from __future__ import annotations

from chess_engine.ai.evaluator import PIECE_VALUES
from chess_engine.chess_core.board import Board
from chess_engine.chess_core.enums import Color, PieceType
from chess_engine.chess_core.move import Move
from chess_engine.chess_core.piece import Piece
from chess_engine.chess_core.rules import generate_legal_moves, is_in_check
from chess_engine.chess_core.square import Square


TACTIC_LABELS = {
    "back_rank_luft": "Luft",
    "clearance": "Clearance",
    "closed_center_knight": "Closed-center knight",
    "decoy": "Decoy",
    "deflection": "Deflection",
    "discovered_attack": "Discovered attack",
    "double_check": "Double check",
    "fork": "Fork",
    "interference": "Interference",
    "king_fork": "King fork",
    "mating_net": "Mating net",
    "overload": "Overload",
    "pin": "Pin",
    "remove_defender": "Remove defender",
    "skewer": "Skewer",
    "zwischenzug": "Zwischenzug",
}


def detect_move_tactics(board: Board, move: Move) -> tuple[str, ...]:
    """Return tactical tags created by ``move`` on ``board``."""
    next_board = board.copy()
    try:
        next_board.apply_move_unchecked(move)
    except ValueError:
        return ()

    moved_piece = next_board.get_piece(move.to_square)
    if moved_piece is None:
        return ()

    tags: list[str] = []
    attacks = _attacked_enemy_values(next_board, move.to_square, moved_piece)
    gives_check = is_in_check(next_board, next_board.side_to_move)
    legal_replies = len(generate_legal_moves(next_board)) if gives_check else 0
    if gives_check and _king_attack_count(next_board, next_board.side_to_move) >= 2:
        tags.append("double_check")
    if gives_check and legal_replies <= 2:
        tags.append("mating_net")
    if gives_check and attacks:
        tags.append("king_fork")
    elif len(attacks) >= 2:
        tags.append("fork")
    if _creates_pin(next_board, move.to_square, moved_piece):
        tags.append("pin")
    if _creates_skewer(next_board, move.to_square, moved_piece):
        tags.append("skewer")
    if _creates_discovered_attack(board, next_board, move):
        tags.append("discovered_attack")
    if _creates_clearance(board, next_board, move, moved_piece.color):
        tags.append("clearance")
    if _creates_interference(board, next_board, move, moved_piece):
        tags.append("interference")
    if gives_check and (_is_capture_or_threat(board, next_board, move, moved_piece) or legal_replies <= 2):
        tags.append("zwischenzug")
    if _removes_defender(board, move):
        tags.append("remove_defender")
    if _attacks_overloaded_defender(next_board, move.to_square, moved_piece):
        tags.append("overload")
    if _attacks_valuable_defender(next_board, move.to_square, moved_piece):
        tags.append("deflection")
    if gives_check and (legal_replies <= 2 or _king_mobility(next_board, next_board.side_to_move) <= 2):
        tags.append("decoy")
    if _creates_luft(board, next_board, moved_piece.color):
        tags.append("back_rank_luft")
    if _creates_closed_center_knight(board, next_board, move, moved_piece):
        tags.append("closed_center_knight")
    return tuple(tags)


def tactic_labels(tags: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(TACTIC_LABELS.get(tag, tag.replace("_", " ").title()) for tag in tags)


def _attacked_enemy_values(board: Board, from_square: Square, piece: Piece) -> list[int]:
    values: list[int] = []
    for target_square, target_piece in board.pieces(piece.color.opponent):
        if target_piece.piece_type is PieceType.KING:
            continue
        if _piece_attacks_square(board, from_square, piece, target_square):
            values.append(PIECE_VALUES[target_piece.piece_type])
    values.sort(reverse=True)
    return values


def _king_attack_count(board: Board, king_color: Color) -> int:
    king_square = _king_square(board, king_color)
    if king_square is None:
        return 0
    count = 0
    for square, piece in board.pieces(king_color.opponent):
        if _piece_attacks_square(board, square, piece, king_square):
            count += 1
    return count


def _king_mobility(board: Board, king_color: Color) -> int:
    king_square = _king_square(board, king_color)
    if king_square is None:
        return 0
    count = 0
    for square in _adjacent_squares(king_square):
        piece = board.get_piece(square)
        if piece is not None and piece.color is king_color:
            continue
        count += 1
    return count


def _king_square(board: Board, color: Color) -> Square | None:
    for square, piece in board.pieces(color):
        if piece.piece_type is PieceType.KING:
            return square
    return None


def _adjacent_squares(square: Square) -> tuple[Square, ...]:
    squares: list[Square] = []
    for file_delta in (-1, 0, 1):
        for rank_delta in (-1, 0, 1):
            if file_delta == 0 and rank_delta == 0:
                continue
            file = square.file + file_delta
            rank = square.rank + rank_delta
            if 0 <= file < 8 and 0 <= rank < 8:
                squares.append(Square(file, rank))
    return tuple(squares)


def _creates_luft(before: Board, after: Board, color: Color) -> bool:
    before_blocked = _back_rank_blocked_escape_count(before, color)
    after_blocked = _back_rank_blocked_escape_count(after, color)
    return before_blocked >= 2 and after_blocked < before_blocked


def _back_rank_trapped(board: Board, color: Color) -> bool:
    return _back_rank_blocked_escape_count(board, color) >= 2


def _back_rank_blocked_escape_count(board: Board, color: Color) -> int:
    king = _king_square(board, color)
    if king is None:
        return 0
    home_rank = 0 if color is Color.WHITE else 7
    if king.rank != home_rank:
        return 0
    escape_rank = king.rank + (1 if color is Color.WHITE else -1)
    if not 0 <= escape_rank < 8:
        return 0
    blocked = 0
    for file in range(max(0, king.file - 1), min(7, king.file + 1) + 1):
        piece = board.get_piece(Square(file, escape_rank))
        if piece is not None and piece.color is color:
            blocked += 1
    return blocked


def _creates_closed_center_knight(before: Board, after: Board, move: Move, moved_piece: Piece) -> bool:
    if moved_piece.piece_type is not PieceType.KNIGHT:
        return False
    if move.to_square.file not in {2, 3, 4, 5}:
        return False
    if not _locked_center_files(after):
        return False
    before_piece = before.get_piece(move.from_square)
    was_supported = before_piece is not None and _is_supported_by_pawn(before, move.from_square, before_piece.color)
    return _is_supported_by_pawn(after, move.to_square, moved_piece.color) and not was_supported


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


def _removes_defender(board: Board, move: Move) -> bool:
    captured_square = _captured_square(move)
    captured = board.get_piece(captured_square)
    if captured is None or captured.piece_type is PieceType.KING:
        return False
    return max(_defended_own_piece_values(board, captured_square, captured), default=0) >= PIECE_VALUES[PieceType.KNIGHT]


def _attacks_overloaded_defender(board: Board, from_square: Square, piece: Piece) -> bool:
    for target_square, target_piece in board.pieces(piece.color.opponent):
        if target_piece.piece_type is PieceType.KING:
            continue
        if not _piece_attacks_square(board, from_square, piece, target_square):
            continue
        defended = _defended_own_piece_values(board, target_square, target_piece)
        valuable_defenses = [value for value in defended if value >= PIECE_VALUES[PieceType.KNIGHT]]
        if len(valuable_defenses) >= 2:
            return True
    return False


def _attacks_valuable_defender(board: Board, from_square: Square, piece: Piece) -> bool:
    for target_square, target_piece in board.pieces(piece.color.opponent):
        if target_piece.piece_type is PieceType.KING:
            continue
        if not _piece_attacks_square(board, from_square, piece, target_square):
            continue
        defended = _defended_own_piece_values(board, target_square, target_piece)
        if any(value > PIECE_VALUES[target_piece.piece_type] for value in defended):
            return True
    return False


def _defended_own_piece_values(board: Board, defender_square: Square, defender: Piece) -> list[int]:
    values: list[int] = []
    for target_square, target_piece in board.pieces(defender.color):
        if target_square == defender_square or target_piece.piece_type is PieceType.KING:
            continue
        if _piece_attacks_square(board, defender_square, defender, target_square):
            values.append(PIECE_VALUES[target_piece.piece_type])
    return values


def _captured_square(move: Move) -> Square:
    if move.is_en_passant:
        return Square(move.to_square.file, move.from_square.rank)
    return move.to_square


def _creates_pin(board: Board, from_square: Square, piece: Piece) -> bool:
    if piece.piece_type not in {PieceType.BISHOP, PieceType.ROOK, PieceType.QUEEN}:
        return False
    for file_step, rank_step in _slider_directions(piece.piece_type):
        if _ray_pins_piece_to_king(board, from_square, piece.color, file_step, rank_step):
            return True
    return False


def _creates_skewer(board: Board, from_square: Square, piece: Piece) -> bool:
    if piece.piece_type not in {PieceType.BISHOP, PieceType.ROOK, PieceType.QUEEN}:
        return False
    for file_step, rank_step in _slider_directions(piece.piece_type):
        if _ray_skewers_king_to_piece(board, from_square, piece.color, file_step, rank_step):
            return True
    return False


def _ray_skewers_king_to_piece(
    board: Board,
    start: Square,
    attacker_color: Color,
    file_step: int,
    rank_step: int,
) -> bool:
    file = start.file + file_step
    rank = start.rank + rank_step
    found_king = False
    while 0 <= file < 8 and 0 <= rank < 8:
        piece = board.get_piece(Square(file, rank))
        if piece is None:
            file += file_step
            rank += rank_step
            continue
        if piece.color is attacker_color:
            return False
        if not found_king:
            if piece.piece_type is not PieceType.KING:
                return False
            found_king = True
            file += file_step
            rank += rank_step
            continue
        return piece.color is attacker_color.opponent and piece.piece_type is not PieceType.KING
    return False


def _ray_pins_piece_to_king(
    board: Board,
    start: Square,
    attacker_color: Color,
    file_step: int,
    rank_step: int,
) -> bool:
    file = start.file + file_step
    rank = start.rank + rank_step
    found_enemy_piece = False
    while 0 <= file < 8 and 0 <= rank < 8:
        piece = board.get_piece(Square(file, rank))
        if piece is None:
            file += file_step
            rank += rank_step
            continue
        if piece.color is attacker_color:
            return False
        if not found_enemy_piece:
            if piece.piece_type is PieceType.KING:
                return False
            found_enemy_piece = True
            file += file_step
            rank += rank_step
            continue
        return piece.color is attacker_color.opponent and piece.piece_type is PieceType.KING
    return False


def _creates_discovered_attack(before: Board, after: Board, move: Move) -> bool:
    moved_piece = after.get_piece(move.to_square)
    if moved_piece is None:
        return False
    for file_step, rank_step in _slider_directions(PieceType.QUEEN):
        if _discovered_ray_attack(after, move.from_square, moved_piece.color, file_step, rank_step):
            return True
    return False


def _creates_clearance(before: Board, after: Board, move: Move, color: Color) -> bool:
    """Return whether moving away opens a line for a friendly slider."""
    if before.get_piece(move.from_square) is None:
        return False
    for file_step, rank_step in _slider_directions(PieceType.QUEEN):
        slider_square = _first_piece_on_ray(after, move.from_square, file_step, rank_step)
        if slider_square is None:
            continue
        slider = after.get_piece(slider_square)
        if slider is None or slider.color is not color or slider.piece_type not in {PieceType.BISHOP, PieceType.ROOK, PieceType.QUEEN}:
            continue
        target_square = _first_piece_on_ray(after, move.from_square, -file_step, -rank_step)
        if target_square is None:
            continue
        target = after.get_piece(target_square)
        if target is None or target.color is color or target.piece_type is PieceType.KING:
            continue
        if _piece_attacks_square(after, slider_square, slider, target_square):
            return True
    return False


def _creates_interference(before: Board, after: Board, move: Move, moved_piece: Piece) -> bool:
    """Return whether the move blocks an enemy slider from a valuable target."""
    if move.from_square == move.to_square:
        return False
    for file_step, rank_step in _slider_directions(PieceType.QUEEN):
        first_square = _first_piece_on_ray(after, move.to_square, file_step, rank_step)
        second_square = _first_piece_on_ray(after, move.to_square, -file_step, -rank_step)
        if first_square is None or second_square is None:
            continue
        first = after.get_piece(first_square)
        second = after.get_piece(second_square)
        if first is None or second is None:
            continue
        if _interferes_with_slider(before, after, first_square, first, second_square, second, moved_piece.color):
            return True
        if _interferes_with_slider(before, after, second_square, second, first_square, first, moved_piece.color):
            return True
    return False


def _interferes_with_slider(
    before: Board,
    after: Board,
    slider_square: Square,
    slider: Piece,
    target_square: Square,
    target: Piece,
    mover_color: Color,
) -> bool:
    if slider.color is mover_color or target.color is not mover_color:
        return False
    if target.piece_type not in {PieceType.KING, PieceType.QUEEN, PieceType.ROOK}:
        return False
    if slider.piece_type not in {PieceType.BISHOP, PieceType.ROOK, PieceType.QUEEN}:
        return False
    return _piece_attacks_square(before, slider_square, slider, target_square) and not _piece_attacks_square(
        after,
        slider_square,
        slider,
        target_square,
    )


def _is_capture_or_threat(before: Board, after: Board, move: Move, moved_piece: Piece) -> bool:
    captured = before.get_piece(_captured_square(move))
    if captured is not None and captured.color is moved_piece.color.opponent:
        return True
    return bool(_attacked_enemy_values(after, move.to_square, moved_piece))


def _discovered_ray_attack(
    board: Board,
    vacated: Square,
    attacker_color: Color,
    file_step: int,
    rank_step: int,
) -> bool:
    slider_square = _first_piece_on_ray(board, vacated, file_step, rank_step)
    target_square = _first_piece_on_ray(board, vacated, -file_step, -rank_step)
    if slider_square is None or target_square is None:
        return False
    slider = board.get_piece(slider_square)
    target = board.get_piece(target_square)
    if slider is None or target is None:
        return False
    if slider.color is not attacker_color or target.color is attacker_color:
        return False
    if slider.piece_type not in {PieceType.BISHOP, PieceType.ROOK, PieceType.QUEEN}:
        return False
    return _piece_attacks_square(board, slider_square, slider, target_square)


def _first_piece_on_ray(board: Board, start: Square, file_step: int, rank_step: int) -> Square | None:
    file = start.file + file_step
    rank = start.rank + rank_step
    while 0 <= file < 8 and 0 <= rank < 8:
        square = Square(file, rank)
        if board.get_piece(square) is not None:
            return square
        file += file_step
        rank += rank_step
    return None


def _piece_attacks_square(board: Board, from_square: Square, piece: Piece, target: Square) -> bool:
    file_delta = target.file - from_square.file
    rank_delta = target.rank - from_square.rank
    abs_file = abs(file_delta)
    abs_rank = abs(rank_delta)
    if piece.piece_type is PieceType.PAWN:
        direction = 1 if piece.color is Color.WHITE else -1
        return abs_file == 1 and rank_delta == direction
    if piece.piece_type is PieceType.KNIGHT:
        return (abs_file, abs_rank) in {(1, 2), (2, 1)}
    if piece.piece_type is PieceType.KING:
        return max(abs_file, abs_rank) == 1
    if piece.piece_type is PieceType.BISHOP:
        return abs_file == abs_rank and _line_is_clear(board, from_square, target)
    if piece.piece_type is PieceType.ROOK:
        return (file_delta == 0 or rank_delta == 0) and _line_is_clear(board, from_square, target)
    if piece.piece_type is PieceType.QUEEN:
        straight = file_delta == 0 or rank_delta == 0
        diagonal = abs_file == abs_rank
        return (straight or diagonal) and _line_is_clear(board, from_square, target)
    return False


def _line_is_clear(board: Board, start: Square, end: Square) -> bool:
    file_step = _sign(end.file - start.file)
    rank_step = _sign(end.rank - start.rank)
    file = start.file + file_step
    rank = start.rank + rank_step
    while (file, rank) != (end.file, end.rank):
        if board.get_piece(Square(file, rank)) is not None:
            return False
        file += file_step
        rank += rank_step
    return True


def _slider_directions(piece_type: PieceType) -> tuple[tuple[int, int], ...]:
    diagonal = ((1, 1), (1, -1), (-1, 1), (-1, -1))
    straight = ((1, 0), (-1, 0), (0, 1), (0, -1))
    if piece_type is PieceType.BISHOP:
        return diagonal
    if piece_type is PieceType.ROOK:
        return straight
    if piece_type is PieceType.QUEEN:
        return diagonal + straight
    return ()


def _sign(value: int) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0
