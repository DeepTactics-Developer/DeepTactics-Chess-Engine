"""Run AI-vs-AI games for future self-play experiments."""

from __future__ import annotations

import random
from collections.abc import Callable

from chess_engine.ai.backend import SearchBackend, find_backend_move
from chess_engine.ai.evaluator import evaluate_board
from chess_engine.ai.repetition import combined_root_move_penalties
from chess_engine.ai.move_ordering import order_moves
from chess_engine.chess_core.enums import Color, GameStatus
from chess_engine.chess_core.fen import board_to_fen
from chess_engine.chess_core.game import Game, position_repetition_key
from chess_engine.chess_core.pgn import move_from_san
from chess_engine.chess_core.rules import generate_legal_moves
from chess_engine.chess_core.pgn import result_from_status
from chess_engine.openings import next_opening_moves
from chess_engine.self_play.training_data import SelfPlayGameResult, SelfPlayPosition, SelfPlayProgress

DRAW_ESCAPE_SCORE_THRESHOLD = -250
ADJUDICATION_WIN_THRESHOLD = 250
ADJUDICATION_MIN_PLY = 40


def run_self_play_game(
    depth: int = 1,
    max_plies: int = 120,
    time_limit: float | None = None,
    adaptive: bool = False,
    opening_book_path: str | None = "assets/openings/openings.tsv",
    opening_plies: int = 8,
    seed: int | None = None,
    game_index: int = 1,
    search_backend: SearchBackend = "native",
    progress_callback: Callable[[SelfPlayProgress], None] | None = None,
    adjudication_threshold: int | None = ADJUDICATION_WIN_THRESHOLD,
    adjudication_min_ply: int = ADJUDICATION_MIN_PLY,
    weights_label: str = "",
    exploration_cp: int = 0,
    exploration_plies: int = 24,
) -> SelfPlayGameResult:
    """Run one AI-vs-AI game and return recorded data."""
    if depth < 1:
        raise ValueError("depth must be at least 1")
    if max_plies < 1:
        raise ValueError("max_plies must be at least 1")
    if time_limit is not None and time_limit <= 0:
        raise ValueError("time_limit must be positive")
    if adjudication_threshold is not None and adjudication_threshold < 0:
        raise ValueError("adjudication_threshold must be non-negative")
    if adjudication_min_ply < 1:
        raise ValueError("adjudication_min_ply must be at least 1")
    if exploration_cp < 0:
        raise ValueError("exploration_cp must be non-negative")
    if exploration_plies < 0:
        raise ValueError("exploration_plies must be non-negative")

    game = Game.new()
    positions: list[SelfPlayPosition] = []
    reached_move_limit = False
    adjudicated_result: str | None = None
    rng = random.Random(seed)

    for ply in range(1, max_plies + 1):
        if game.is_over():
            break

        fen_before = board_to_fen(game.board)
        side_to_move = game.board.side_to_move.value
        book_move = _opening_book_move(game, opening_book_path, opening_plies, rng)
        if book_move is not None:
            selected_move = book_move
            score = 0
            source = "book"
        else:
            search_result = _self_play_search(game, depth, time_limit, adaptive, search_backend)
            if search_result.move is None:
                break
            selected_move = _avoid_immediate_repetition(game, search_result.move, search_result.score)
            score = search_result.score
            source = f"d{search_result.completed_depth or search_result.depth}"
            explored = _explore_self_play_move(
                game,
                preferred_move=selected_move,
                rng=rng,
                exploration_cp=exploration_cp,
                exploration_plies=exploration_plies,
                ply=ply,
            )
            if explored is not None:
                selected_move, score = explored
                source = f"explore:{source}"

        game.apply_move(selected_move)
        fen_after = board_to_fen(game.board)
        san = game.san_history[-1]
        positions.append(
            SelfPlayPosition(
                fen_before=fen_before,
                move_uci=selected_move.to_uci(),
                move_san=san,
                score=score,
                ply=ply,
                side_to_move=side_to_move,
                source=source,
                fen_after=fen_after,
                search_backend=search_backend,
                depth=depth,
                time_limit="" if time_limit is None else str(time_limit),
                adaptive=adaptive,
                weights=weights_label,
            )
        )
        if progress_callback is not None:
            progress_callback(
                SelfPlayProgress(
                    game_index=game_index,
                    ply=ply,
                    san=san,
                    move_uci=selected_move.to_uci(),
                    score=score,
                    source=source,
                )
            )
        if (
            adjudication_threshold is not None
            and ply >= adjudication_min_ply
            and not game.is_over()
        ):
            possible_result = _adjudicated_limit_result(game, adjudication_threshold)
            if possible_result in {"1-0", "0-1"}:
                adjudicated_result = possible_result
                break
    else:
        reached_move_limit = not game.is_over()

    status = game.status()
    result = adjudicated_result or result_from_status(status, game.board.side_to_move)
    if reached_move_limit and status is GameStatus.ONGOING:
        result = _adjudicated_limit_result(game, adjudication_threshold)

    return SelfPlayGameResult(
        positions=positions,
        pgn=game.to_pgn(),
        result=result,
        status=status,
        plies_played=len(positions),
        reached_move_limit=reached_move_limit,
    )


def run_self_play_games(
    game_count: int,
    depth: int = 1,
    max_plies: int = 120,
    time_limit: float | None = None,
    adaptive: bool = False,
    opening_book_path: str | None = "assets/openings/openings.tsv",
    opening_plies: int = 8,
    seed: int | None = None,
    search_backend: SearchBackend = "native",
    progress_callback: Callable[[SelfPlayProgress], None] | None = None,
    adjudication_threshold: int | None = ADJUDICATION_WIN_THRESHOLD,
    adjudication_min_ply: int = ADJUDICATION_MIN_PLY,
    weights_label: str = "",
    exploration_cp: int = 0,
    exploration_plies: int = 24,
) -> list[SelfPlayGameResult]:
    """Run multiple self-play games."""
    if game_count < 1:
        raise ValueError("game_count must be at least 1")

    return [
        run_self_play_game(
            depth=depth,
            max_plies=max_plies,
            time_limit=time_limit,
            adaptive=adaptive,
            opening_book_path=opening_book_path,
            opening_plies=opening_plies,
            seed=None if seed is None else seed + index,
            game_index=index + 1,
            search_backend=search_backend,
            progress_callback=progress_callback,
            adjudication_threshold=adjudication_threshold,
            adjudication_min_ply=adjudication_min_ply,
            weights_label=weights_label,
            exploration_cp=exploration_cp,
            exploration_plies=exploration_plies,
        )
        for index in range(game_count)
    ]


def _adjudicated_limit_result(game: Game, threshold: int | None) -> str:
    """Return a practical training result when a self-play game hits its ply cap."""
    if threshold is None:
        return "*"

    score = evaluate_board(game.board)
    if score >= threshold:
        return "1-0"
    if score <= -threshold:
        return "0-1"
    return "1/2-1/2"


def _self_play_search(
    game: Game,
    depth: int,
    time_limit: float | None,
    adaptive: bool,
    search_backend: SearchBackend = "native",
):
    penalties = combined_root_move_penalties(game.board, game.board_history, game.move_history)
    return find_backend_move(
        game.board,
        depth=depth,
        backend=search_backend,
        time_limit=time_limit,
        adaptive=adaptive,
        root_move_penalties=penalties,
    )


def _avoid_immediate_repetition(game: Game, preferred_move, preferred_score: int = 0):
    """Prefer a non-repeating move when the search choice repeats a position."""
    if preferred_score <= DRAW_ESCAPE_SCORE_THRESHOLD:
        return preferred_move
    if not _would_repeat_position(game, preferred_move):
        return preferred_move

    legal_moves = generate_legal_moves(game.board)
    for move in order_moves(game.board, legal_moves):
        if not _would_repeat_position(game, move):
            return move
    return preferred_move


def _explore_self_play_move(
    game: Game,
    *,
    preferred_move,
    rng: random.Random,
    exploration_cp: int,
    exploration_plies: int,
    ply: int,
):
    """Pick a near-equal legal move for data diversity during self-play only."""
    if exploration_cp <= 0 or ply > exploration_plies:
        return None

    legal_moves = generate_legal_moves(game.board)
    if len(legal_moves) < 2:
        return None

    mover = game.board.side_to_move
    scored: list[tuple[object, int]] = []
    for move in legal_moves:
        if _would_repeat_position(game, move):
            continue
        next_board = game.board.copy()
        next_board.apply_move_unchecked(move)
        raw_score = evaluate_board(next_board)
        mover_score = raw_score if mover is Color.WHITE else -raw_score
        scored.append((move, mover_score))

    if len(scored) < 2:
        return None

    best_score = max(score for _, score in scored)
    candidates = [(move, score) for move, score in scored if best_score - score <= exploration_cp]
    if not any(move.to_uci() == preferred_move.to_uci() for move, _ in candidates):
        candidates.append((preferred_move, best_score))
    if len(candidates) < 2:
        return None

    # Weighted random among near-equal candidates: stronger moves remain more
    # likely, but self-play no longer collapses into one deterministic line.
    temperature = max(1.0, exploration_cp / 2)
    weights = [2.718281828 ** ((score - best_score) / temperature) for _, score in candidates]
    chosen_move, chosen_score = rng.choices(candidates, weights=weights, k=1)[0]
    if chosen_move.to_uci() == preferred_move.to_uci():
        return None
    return chosen_move, chosen_score if mover is Color.WHITE else -chosen_score


def _would_repeat_position(game: Game, move) -> bool:
    next_board = game.board.copy()
    next_board.apply_move_unchecked(move)
    next_key = position_repetition_key(next_board)
    count = 1
    for board in game.board_history:
        if position_repetition_key(board) == next_key:
            count += 1
    return count >= 2


def _opening_book_move(
    game: Game,
    opening_book_path: str | None,
    opening_plies: int,
    rng: random.Random,
):
    if opening_book_path is None or opening_plies <= 0 or len(game.move_history) >= opening_plies:
        return None
    continuations = next_opening_moves(game.san_history, opening_book_path)
    rng.shuffle(continuations)
    for san in continuations:
        try:
            return move_from_san(game.board, san)
        except ValueError:
            continue
    return None
