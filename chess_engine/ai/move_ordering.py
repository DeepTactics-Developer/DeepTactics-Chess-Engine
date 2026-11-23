"""Move ordering heuristics for alpha-beta search.

Alpha-beta pruning becomes stronger when promising moves are searched first.
The scores here do not decide the final move; they only choose the order in
which the real minimax search examines candidates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from chess_engine.ai.evaluator import PIECE_VALUES
from chess_engine.chess_core.board import Board
from chess_engine.chess_core.move import Move
from chess_engine.chess_core.piece import Piece
from chess_engine.chess_core.enums import Color, PieceType
from chess_engine.chess_core.rules import generate_legal_moves, is_in_check, is_square_attacked
from chess_engine.chess_core.square import Square


CENTER_SQUARES = {
    Square.from_algebraic("d4"),
    Square.from_algebraic("e4"),
    Square.from_algebraic("d5"),
    Square.from_algebraic("e5"),
}


class MoveStage(str, Enum):
    """Broad move ordering stages."""

    TT = "tt"
    GOOD_CAPTURE = "good_capture"
    GOOD_QUIET = "good_quiet"
    BAD_CAPTURE = "bad_capture"
    BAD_QUIET = "bad_quiet"


@dataclass(slots=True)
class SearchHeuristics:
    """Mutable move-ordering memory shared during one search."""

    killer_moves: dict[int, list[Move]] = field(default_factory=dict)
    history_scores: dict[str, int] = field(default_factory=dict)

    def record_cutoff(self, move: Move, depth: int, ply: int, is_quiet: bool) -> None:
        """Remember a quiet move that caused beta cutoff."""
        if not is_quiet:
            return

        killers = self.killer_moves.setdefault(ply, [])
        if move not in killers:
            killers.insert(0, move)
            del killers[2:]
        self.history_scores[move.to_uci()] = self.history_scores.get(move.to_uci(), 0) + depth * depth

    def killer_score(self, move: Move, ply: int) -> int:
        killers = self.killer_moves.get(ply, [])
        if move in killers:
            return 7_000 - killers.index(move) * 500
        return 0

    def history_score(self, move: Move) -> int:
        return min(5_000, self.history_scores.get(move.to_uci(), 0))


@dataclass(frozen=True, slots=True)
class StagedMove:
    """A move with its staged ordering metadata."""

    move: Move
    stage: MoveStage
    score: int


class StagedMovePicker:
    """Order moves in TT, good capture, good quiet, bad capture, bad quiet stages."""

    def __init__(
        self,
        board: Board,
        moves: list[Move],
        heuristics: SearchHeuristics | None = None,
        ply: int = 0,
        preferred_move: Move | None = None,
    ) -> None:
        self.board = board
        self.moves = moves
        self.heuristics = heuristics
        self.ply = ply
        self.preferred_move = preferred_move
        self._ordered = self._build_ordered_moves()

    def __iter__(self):
        return (entry.move for entry in self._ordered)

    def staged(self) -> list[StagedMove]:
        """Return staged entries, useful for tests and debug tooling."""

        return list(self._ordered)

    def _build_ordered_moves(self) -> list[StagedMove]:
        buckets: dict[MoveStage, list[StagedMove]] = {
            MoveStage.TT: [],
            MoveStage.GOOD_CAPTURE: [],
            MoveStage.GOOD_QUIET: [],
            MoveStage.BAD_CAPTURE: [],
            MoveStage.BAD_QUIET: [],
        }
        for move in self.moves:
            score = score_move(
                self.board,
                move,
                heuristics=self.heuristics,
                ply=self.ply,
                preferred_move=self.preferred_move,
            )
            stage = self._stage_for(move)
            buckets[stage].append(StagedMove(move=move, stage=stage, score=score))

        ordered: list[StagedMove] = []
        for stage in (MoveStage.TT, MoveStage.GOOD_CAPTURE, MoveStage.GOOD_QUIET, MoveStage.BAD_CAPTURE, MoveStage.BAD_QUIET):
            ordered.extend(sorted(buckets[stage], key=lambda entry: entry.score, reverse=True))
        return ordered

    def _stage_for(self, move: Move) -> MoveStage:
        if self.preferred_move is not None and move.to_uci() == self.preferred_move.to_uci():
            return MoveStage.TT
        if is_capture(self.board, move) or move.promotion is not None:
            return MoveStage.GOOD_CAPTURE if static_exchange_score(self.board, move) >= -40 else MoveStage.BAD_CAPTURE
        score = 0
        if self.heuristics is not None:
            score += self.heuristics.killer_score(move, self.ply)
            score += self.heuristics.history_score(move)
        if _move_gives_check(self.board, move) or move.is_castling or move.to_square in CENTER_SQUARES:
            score += 1
        return MoveStage.GOOD_QUIET if score > 0 else MoveStage.BAD_QUIET


def order_moves(
    board: Board,
    moves: list[Move],
    heuristics: SearchHeuristics | None = None,
    ply: int = 0,
    preferred_move: Move | None = None,
) -> list[Move]:
    """Return moves sorted by staged search priority."""

    return list(StagedMovePicker(board, moves, heuristics=heuristics, ply=ply, preferred_move=preferred_move))


def staged_moves(
    board: Board,
    moves: list[Move],
    heuristics: SearchHeuristics | None = None,
    ply: int = 0,
    preferred_move: Move | None = None,
) -> list[StagedMove]:
    """Return move ordering metadata for tests and debug output."""

    return StagedMovePicker(
        board,
        moves,
        heuristics=heuristics,
        ply=ply,
        preferred_move=preferred_move,
    ).staged()


def tactical_moves(board: Board, moves: list[Move]) -> list[Move]:
    """Return captures and promotions ordered for quiescence search."""
    return [
        move
        for move in order_moves(board, moves)
        if move.promotion is not None or (is_capture(board, move) and static_exchange_score(board, move) >= -120)
    ]


def is_capture(board: Board, move: Move) -> bool:
    """Return whether ``move`` captures a piece."""
    return _captured_piece(board, move) is not None


def score_move(
    board: Board,
    move: Move,
    heuristics: SearchHeuristics | None = None,
    ply: int = 0,
    preferred_move: Move | None = None,
) -> int:
    """Return a heuristic ordering score for one move.

    Captures use MVV-LVA: Most Valuable Victim, Least Valuable Attacker. A pawn
    taking a queen should be searched before a queen taking a pawn.
    """
    score = 0
    attacker = board.get_piece(move.from_square)

    if preferred_move is not None and move.to_uci() == preferred_move.to_uci():
        score += 50_000

    captured_piece = _captured_piece(board, move)
    if captured_piece is not None and attacker is not None:
        score += _capture_score(attacker, captured_piece)
        see = static_exchange_score(board, move)
        if see >= 0:
            score += 1_500 + see
        else:
            score -= 1_500 + abs(see) * 2

    if move.promotion is not None:
        score += PIECE_VALUES[move.promotion] + 8_000
        if move.promotion is PieceType.QUEEN:
            score += 600

    if _move_gives_check(board, move):
        score += 600

    if attacker is not None and ply <= 2:
        score += _tactical_pressure_score(board, move, attacker)
        score += _pin_creation_score(board, move)
        score += _defender_tactic_score(board, move, attacker)

    if move.is_castling:
        score += 250

    if attacker is not None and attacker.piece_type is PieceType.PAWN:
        score += _pawn_progress_score(move, attacker)
        score += _pawn_break_order_score(board, move, attacker)

    if attacker is not None:
        score -= _hanging_piece_risk(board, move, attacker, captured_piece)

    if move.to_square in CENTER_SQUARES:
        score += 30

    if heuristics is not None and captured_piece is None and move.promotion is None:
        score += heuristics.killer_score(move, ply)
        score += heuristics.history_score(move)

    return score


def _capture_score(attacker: Piece, captured_piece: Piece) -> int:
    victim_value = PIECE_VALUES[captured_piece.piece_type]
    attacker_value = PIECE_VALUES[attacker.piece_type]
    return 10_000 + victim_value * 10 - attacker_value


def static_exchange_score(board: Board, move: Move) -> int:
    """Return a SEE-style material swing for ``move`` from the mover's view.

    The implementation deliberately uses the engine's legal move generator
    instead of a hand-rolled attack list. That is slower than a bitboard SEE,
    but it correctly respects pins, discovered checks, en-passant side effects,
    and x-ray attackers revealed after each simulated capture.
    """
    attacker = board.get_piece(move.from_square)
    if attacker is None:
        return 0
    captured = _captured_piece(board, move)
    gain = PIECE_VALUES[captured.piece_type] if captured is not None else 0
    if move.promotion is not None:
        gain += PIECE_VALUES[move.promotion] - PIECE_VALUES[PieceType.PAWN]

    next_board = board.copy()
    try:
        next_board.apply_move_unchecked(move)
    except ValueError:
        return gain
    return gain - _best_legal_capture_gain(next_board, move.to_square, depth=0)


def _best_legal_capture_gain(board: Board, target: Square, depth: int) -> int:
    """Return the best material gain for the side to move by capturing target."""

    if depth >= 16:
        return 0
    target_piece = board.get_piece(target)
    if target_piece is None:
        return 0

    best_gain = 0
    candidates = [
        candidate
        for candidate in generate_legal_moves(board, board.side_to_move)
        if candidate.to_square == target and is_capture(board, candidate)
    ]
    candidates.sort(key=lambda candidate: _moving_piece_value(board, candidate))

    for candidate in candidates:
        next_board = board.copy()
        try:
            next_board.apply_move_unchecked(candidate)
        except ValueError:
            continue
        immediate_gain = PIECE_VALUES[target_piece.piece_type]
        if candidate.promotion is not None:
            immediate_gain += PIECE_VALUES[candidate.promotion] - PIECE_VALUES[PieceType.PAWN]
        reply_gain = _best_legal_capture_gain(next_board, target, depth + 1)
        best_gain = max(best_gain, immediate_gain - max(0, reply_gain))
    return best_gain


def _moving_piece_value(board: Board, move: Move) -> int:
    piece = board.get_piece(move.from_square)
    if piece is None:
        return 0
    return PIECE_VALUES[piece.piece_type]


def _pawn_progress_score(move: Move, pawn: Piece) -> int:
    rank = move.to_square.rank if pawn.color is Color.WHITE else 7 - move.to_square.rank
    if rank >= 5:
        return 80 + rank * 15
    return 0


def _pawn_break_order_score(board: Board, move: Move, pawn: Piece) -> int:
    if move.from_square.file != move.to_square.file:
        return 0
    if board.get_piece(move.to_square) is not None:
        return 0

    direction = 1 if pawn.color is Color.WHITE else -1
    score = 0
    for file in (move.to_square.file - 1, move.to_square.file + 1):
        if not 0 <= file < 8:
            continue
        adjacent = board.get_piece(Square(file, move.to_square.rank))
        if adjacent is not None and adjacent.color is pawn.color.opponent and adjacent.piece_type is PieceType.PAWN:
            score += 220
        next_rank = move.to_square.rank + direction
        if not 0 <= next_rank < 8:
            continue
        target = board.get_piece(Square(file, next_rank))
        if target is not None and target.color is pawn.color.opponent and target.piece_type is PieceType.PAWN:
            score += 220
    if score and move.to_square.file in {2, 3, 4, 5}:
        score += 80
    return min(520, score)


def _hanging_piece_risk(
    board: Board,
    move: Move,
    attacker: Piece,
    captured_piece: Piece | None,
) -> int:
    if attacker.piece_type is PieceType.KING:
        return 0

    next_board = board.copy()
    try:
        next_board.apply_move_unchecked(move)
    except ValueError:
        return 0

    moved_piece = next_board.get_piece(move.to_square)
    if moved_piece is None:
        return 0
    if not is_square_attacked(next_board, move.to_square, moved_piece.color.opponent):
        return 0

    moved_value = PIECE_VALUES[moved_piece.piece_type]
    captured_value = PIECE_VALUES[captured_piece.piece_type] if captured_piece is not None else 0
    risk = moved_value - captured_value
    if risk <= 0:
        return 0
    return min(2_000, risk * 2)


def _captured_piece(board: Board, move: Move) -> Piece | None:
    if move.is_en_passant:
        return board.get_piece(Square(move.to_square.file, move.from_square.rank))
    return board.get_piece(move.to_square)


def _move_gives_check(board: Board, move: Move) -> bool:
    next_board = board.copy()
    try:
        next_board.apply_move_unchecked(move)
    except ValueError:
        return False
    return is_in_check(next_board, next_board.side_to_move)


def _tactical_pressure_score(board: Board, move: Move, attacker: Piece) -> int:
    next_board = board.copy()
    try:
        next_board.apply_move_unchecked(move)
    except ValueError:
        return 0

    moved_piece = next_board.get_piece(move.to_square)
    if moved_piece is None:
        return 0

    attacked_values: list[int] = []
    attacks_king = False
    for target_square, target_piece in next_board.pieces(moved_piece.color.opponent):
        if not _piece_attacks_square(next_board, move.to_square, moved_piece, target_square):
            continue
        if target_piece.piece_type is PieceType.KING:
            attacks_king = True
        else:
            attacked_values.append(PIECE_VALUES[target_piece.piece_type])

    if not attacked_values:
        return 0

    attacked_values.sort(reverse=True)
    score = 0
    # Forks and double attacks are forcing tactical candidates, so search them
    # earlier. This does not decide the move; it only improves alpha-beta order.
    if attacks_king:
        score += 900 + attacked_values[0]
    if len(attacked_values) >= 2:
        score += 700 + sum(attacked_values[:2]) // 2
    if attacked_values[0] >= PIECE_VALUES[PieceType.QUEEN]:
        score += 250
    return score


def _pin_creation_score(board: Board, move: Move) -> int:
    next_board = board.copy()
    try:
        next_board.apply_move_unchecked(move)
    except ValueError:
        return 0

    moved_piece = next_board.get_piece(move.to_square)
    if moved_piece is None or moved_piece.piece_type not in {PieceType.BISHOP, PieceType.ROOK, PieceType.QUEEN}:
        return 0

    best = 0
    for file_step, rank_step in _slider_directions(moved_piece.piece_type):
        pinned_value = _ray_pin_value(next_board, move.to_square, moved_piece.color, file_step, rank_step)
        best = max(best, pinned_value)
    if best <= 0:
        return 0
    return 650 + best


def _defender_tactic_score(board: Board, move: Move, attacker: Piece) -> int:
    del attacker
    captured_piece = _captured_piece(board, move)
    if captured_piece is None:
        return 0
    captured_square = Square(move.to_square.file, move.from_square.rank) if move.is_en_passant else move.to_square
    defended = _defended_own_piece_values(board, captured_square, captured_piece)
    if not defended:
        return 0
    return 450 + max(defended)


def _defended_own_piece_values(board: Board, defender_square: Square, defender: Piece) -> list[int]:
    values: list[int] = []
    for target_square, target_piece in board.pieces(defender.color):
        if target_square == defender_square or target_piece.piece_type is PieceType.KING:
            continue
        if _piece_attacks_square(board, defender_square, defender, target_square):
            values.append(PIECE_VALUES[target_piece.piece_type])
    return values


def _ray_pin_value(
    board: Board,
    start: Square,
    attacker_color: Color,
    file_step: int,
    rank_step: int,
) -> int:
    file = start.file + file_step
    rank = start.rank + rank_step
    first_enemy: Piece | None = None
    while 0 <= file < 8 and 0 <= rank < 8:
        piece = board.get_piece(Square(file, rank))
        if piece is None:
            file += file_step
            rank += rank_step
            continue
        if piece.color is attacker_color:
            return 0
        if first_enemy is None:
            if piece.piece_type is PieceType.KING:
                return 0
            first_enemy = piece
            file += file_step
            rank += rank_step
            continue
        if piece.color is attacker_color.opponent and piece.piece_type is PieceType.KING:
            return PIECE_VALUES[first_enemy.piece_type]
        return 0
    return 0


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


def _sign(value: int) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0
