"""Engine-backed position analysis.

The analysis views and future PGN review tools should depend on this
module instead of directly knowing search details.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from chess_engine.ai.backend import SearchBackend, find_backend_move
from chess_engine.ai.evaluator import evaluate_material
from chess_engine.ai.search import SearchResult
from chess_engine.chess_core.board import Board
from chess_engine.chess_core.enums import Color
from chess_engine.chess_core.game import Game
from chess_engine.chess_core.move import Move
from chess_engine.chess_core.pgn import move_to_san
from chess_engine.chess_core.rules import generate_legal_moves
from chess_engine.native.native_eval import (
    cp_to_win_probability_native,
    game_accuracy_native,
    move_accuracy_native,
)
from chess_engine.openings import detect_opening
from chess_engine.analysis.tactics import detect_move_tactics


@dataclass(frozen=True, slots=True)
class EngineAnalysis:
    """One engine recommendation for a position."""

    result: SearchResult
    best_move_san: str | None
    principal_variation_san: tuple[str, ...]

    @property
    def best_move(self) -> Move | None:
        return self.result.move

    @property
    def score(self) -> int:
        return self.result.score

    @property
    def completed_depth(self) -> int:
        return self.result.completed_depth


@dataclass(frozen=True, slots=True)
class MoveClassification:
    """Move quality classification."""

    key: str
    label: str
    symbol: str
    loss_cp: int
    best_score_cp: int | None = None
    tactical_tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AnalyzedMove:
    """Analysis result for one move in a completed game."""

    ply: int
    san: str
    played_move: Move
    player: Color
    analysis: EngineAnalysis
    classification: MoveClassification
    best_move_san: str | None
    score_before: int
    score_after: int
    accuracy: float | None


@dataclass(frozen=True, slots=True)
class GameAnalysisReport:
    """Whole-game analysis summary."""

    moves: tuple[AnalyzedMove, ...]
    white_accuracy: float
    black_accuracy: float


CLASSIFICATION_LABELS = {
    "brilliant": ("Brilliant", "!!"),
    "great": ("Great", "!"),
    "book": ("Book", "B"),
    "forced": ("Forced", "F"),
    "best": ("Best", "*"),
    "excellent": ("Excellent", "+"),
    "good": ("Good", "OK"),
    "inaccuracy": ("Inaccuracy", "?!"),
    "mistake": ("Mistake", "?"),
    "missed_win": ("Missed Win", "x"),
    "blunder": ("Blunder", "??"),
}


def analyze_position(
    board: Board,
    depth: int = 2,
    time_limit: float | None = None,
    adaptive: bool = False,
    backend: SearchBackend = "native",
) -> EngineAnalysis:
    """Analyze a board and return the best move plus principal variation."""
    board_snapshot = board.copy()
    result = find_backend_move(
        board_snapshot,
        depth=depth,
        backend=backend,
        time_limit=time_limit,
        adaptive=adaptive,
    )

    variation = result.principal_variation or ((result.move,) if result.move is not None else ())
    pv_san = variation_to_san(board, variation)
    best_san = pv_san[0] if pv_san else None
    return EngineAnalysis(result=result, best_move_san=best_san, principal_variation_san=pv_san)


def analyze_game_moves(
    game: Game,
    depth: int = 1,
    time_limit: float | None = None,
    adaptive: bool = False,
    backend: SearchBackend = "native",
) -> GameAnalysisReport:
    """Analyze every played move and return classifications plus accuracy."""
    current = Game.new()
    analyzed: list[AnalyzedMove] = []
    position_analysis = analyze_position(
        current.board,
        depth=depth,
        time_limit=time_limit,
        adaptive=adaptive,
        backend=backend,
    )

    for index, move in enumerate(game.move_history, start=1):
        before_board = current.board.copy()
        player = before_board.side_to_move
        best = position_analysis
        best_score = score_to_white_perspective(best.score, player)
        before_score = best_score
        current.apply_move(move)
        if current.is_over():
            after_score = _quick_score(current.board)
            position_analysis = best
        else:
            position_analysis = analyze_position(
                current.board,
                depth=depth,
                time_limit=time_limit,
                adaptive=adaptive,
                backend=backend,
            )
            after_score = score_to_white_perspective(position_analysis.score, current.board.side_to_move)
        is_book = _is_book_position(current.san_history)
        classification = classify_move(
            player=player,
            before_score=before_score,
            after_score=after_score,
            best_score=best_score,
            played_move=move,
            best_move=best.best_move,
            is_forced=_is_forced_move(before_board),
            is_book=is_book,
            is_sacrifice=_is_material_sacrifice(player, before_board, current.board),
            tactical_tags=detect_move_tactics(before_board, move),
        )
        move_accuracy = calculate_move_accuracy(
            _score_from_player_perspective(before_score, player),
            _score_from_player_perspective(after_score, player),
            is_book_move=is_book,
        )
        analyzed.append(
            AnalyzedMove(
                ply=index,
                san=current.san_history[-1],
                played_move=move,
                player=player,
                analysis=best,
                classification=classification,
                best_move_san=best.best_move_san,
                score_before=before_score,
                score_after=after_score,
                accuracy=move_accuracy,
            )
        )

    return GameAnalysisReport(
        moves=tuple(analyzed),
        white_accuracy=_accuracy_for_color(analyzed, Color.WHITE),
        black_accuracy=_accuracy_for_color(analyzed, Color.BLACK),
    )


def variation_to_san(board: Board, variation: tuple[Move, ...]) -> tuple[str, ...]:
    """Convert a move variation to SAN by applying it on a copied board."""
    current = board.copy()
    san_moves: list[str] = []
    for move in variation:
        san_moves.append(move_to_san(current, move))
        current.apply_move_unchecked(move)
    return tuple(san_moves)


def classify_move(
    player: Color,
    before_score: int,
    after_score: int,
    best_score: int,
    played_move: Move,
    best_move: Move | None,
    is_forced: bool = False,
    is_book: bool = False,
    is_sacrifice: bool = False,
    tactical_tags: tuple[str, ...] = (),
) -> MoveClassification:
    """Classify a played move from the mover's perspective.

    Scores are centipawns from White's perspective. The loss is normalized so a
    worse move for Black is counted the same way as a worse move for White.
    """
    if is_book:
        return _classification("book", 0, best_score, tactical_tags)
    if is_forced:
        return _classification("forced", 0, best_score, tactical_tags)

    loss = _centipawn_loss(player, after_score, best_score)
    softened_loss = _soften_loss_for_decided_positions(loss, before_score)

    if best_move is not None and played_move.to_uci() == best_move.to_uci():
        return _classification("best", 0, best_score, tactical_tags)
    if is_sacrifice and softened_loss <= 15:
        return _classification("brilliant", softened_loss, best_score, tactical_tags)
    if _missed_win(player, before_score, after_score, best_score):
        return _classification("missed_win", softened_loss, best_score, tactical_tags)
    if softened_loss <= 15 and abs(best_score - before_score) >= 150:
        return _classification("great", softened_loss, best_score, tactical_tags)
    if softened_loss <= 25:
        return _classification("excellent", softened_loss, best_score, tactical_tags)
    if softened_loss <= 50:
        return _classification("good", softened_loss, best_score, tactical_tags)
    if softened_loss <= 100:
        return _classification("inaccuracy", softened_loss, best_score, tactical_tags)
    if softened_loss <= 200:
        return _classification("mistake", softened_loss, best_score, tactical_tags)
    return _classification("blunder", softened_loss, best_score, tactical_tags)


def _centipawn_loss(player: Color, played_score: int, best_score: int) -> int:
    if player is Color.WHITE:
        return max(0, best_score - played_score)
    return max(0, played_score - best_score)


def _soften_loss_for_decided_positions(loss: int, before_score: int) -> int:
    if abs(before_score) < 1000 or loss < 50:
        return loss
    if loss <= 150:
        return min(loss, 90)
    return loss


def _missed_win(player: Color, before_score: int, after_score: int, best_score: int) -> bool:
    if player is Color.WHITE:
        return best_score >= 700 and after_score < 300 and best_score - after_score > 200
    return best_score <= -700 and after_score > -300 and after_score - best_score > 200


def _classification(
    key: str,
    loss: int,
    best_score: int | None = None,
    tactical_tags: tuple[str, ...] = (),
) -> MoveClassification:
    label, symbol = CLASSIFICATION_LABELS[key]
    return MoveClassification(
        key=key,
        label=label,
        symbol=symbol,
        loss_cp=loss,
        best_score_cp=best_score,
        tactical_tags=tactical_tags,
    )


def score_to_white_perspective(score: int, mover: Color) -> int:
    """Convert a search score from mover perspective to White perspective."""

    return score if mover is Color.WHITE else -score


def _is_book_position(san_history: list[str]) -> bool:
    if len(san_history) > 12:
        return False
    return detect_opening(san_history) != "Opening: unknown"


def _is_forced_move(board: Board) -> bool:
    return len(generate_legal_moves(board)) == 1


def _is_material_sacrifice(player: Color, before_board: Board, after_board: Board) -> bool:
    before_material = evaluate_material(before_board)
    after_material = evaluate_material(after_board)
    material_change = after_material - before_material
    if player is Color.BLACK:
        material_change = -material_change
    return material_change <= -100


def _quick_score(board: Board) -> int:
    from chess_engine.ai.evaluator import score_for_color

    return score_for_color(board, Color.WHITE)


def cp_to_wp(cp: int) -> float:
    """Convert mover-perspective centipawns to estimated win probability."""
    native = cp_to_win_probability_native(cp)
    if native is not None:
        return native
    return 50.0 + 50.0 * ((2.0 / (1.0 + math.exp(-0.004 * cp))) - 1.0)


def calculate_move_accuracy(cp_before: int, cp_after: int, is_book_move: bool = False) -> float:
    """Return CAPS-style accuracy for one move.

    ``cp_before`` and ``cp_after`` are from the player-to-move's perspective.
    """
    native = move_accuracy_native(cp_before, cp_after, is_book_move)
    if native is not None:
        return round(max(0.0, min(100.0, native)), 1)
    if is_book_move:
        return 100.0
    wp_loss = max(0.0, cp_to_wp(cp_before) - cp_to_wp(cp_after))
    return round(max(0.0, min(100.0, 100.0 - wp_loss * 1.5)), 1)


def calculate_game_accuracy(accuracies: list[float] | tuple[float, ...]) -> float:
    """Combine move accuracies with a harmonic mean so bad misses matter."""
    if not accuracies:
        return 100.0
    native = game_accuracy_native(accuracies)
    if native is not None:
        return round(max(0.0, min(100.0, native)), 1)
    reciprocal_sum = 0.0
    for accuracy in accuracies:
        reciprocal_sum += 1.0 / max(1e-6, max(0.0, min(100.0, accuracy)))
    return round(max(0.0, min(100.0, len(accuracies) / reciprocal_sum)), 1)


def _score_from_player_perspective(score: int, player: Color) -> int:
    return score if player is Color.WHITE else -score


def _accuracy_for_color(moves: list[AnalyzedMove], color: Color) -> float:
    color_moves = [move for move in moves if move.player is color]
    if not color_moves:
        return 100.0

    return calculate_game_accuracy(tuple(move.accuracy for move in color_moves if move.accuracy is not None))
