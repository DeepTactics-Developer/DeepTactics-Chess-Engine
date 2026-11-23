"""Search backend selection helpers.

Search entry points used by the UI, UCI loop and self-play. All searching is
done by the native C++ library.
"""

from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter
from typing import Literal

from chess_engine.ai.evaluator import get_evaluation_weights, score_for_color
from chess_engine.ai.search import SearchResult
from chess_engine.chess_core.board import Board
from chess_engine.chess_core.rules import generate_legal_moves
from chess_engine.native.native_eval import find_best_line_native, native_internal_iterative_deepening_enabled


SearchBackend = Literal["native"]


def find_backend_move(
    board: Board,
    *,
    depth: int,
    backend: SearchBackend = "native",
    time_limit: float | None = None,
    adaptive: bool = False,
    quiescence_depth: int = 4,
    root_move_penalties: Mapping[str, int] | None = None,
) -> SearchResult:
    """Return a search result from the selected backend."""

    legal_moves = generate_legal_moves(board)
    if not legal_moves:
        return SearchResult(
            move=None,
            score=score_for_color(board, board.side_to_move),
            depth=depth,
            completed_depth=0,
        )

    if backend != "native":
        raise ValueError(f"unsupported search backend: {backend}")
    if adaptive:
        return _find_native_adaptive_move(
            board,
            max_depth=depth,
            max_time_seconds=time_limit or 2.0,
            quiescence_depth=quiescence_depth,
            root_move_penalties=root_move_penalties,
        )
    if time_limit is not None:
        return _find_native_iterative_move(
            board,
            max_depth=depth,
            time_limit_seconds=time_limit,
            quiescence_depth=quiescence_depth,
            root_move_penalties=root_move_penalties,
        )
    return _find_native_fixed_depth_move(
        board,
        depth=depth,
        quiescence_depth=quiescence_depth,
        root_move_penalties=root_move_penalties,
    )


def find_backend_multipv(
    board: Board,
    *,
    depth: int,
    multipv: int,
    backend: SearchBackend = "native",
    quiescence_depth: int = 4,
    root_move_penalties: Mapping[str, int] | None = None,
    native_time_limit: float | None = None,
) -> list[SearchResult]:
    """Return up to ``multipv`` root alternatives with PVs.

    This is intentionally root-level and conservative. Each selected move is
    excluded with a heavy root penalty before the next search, so the normal
    backend selector and TT/PV path stay intact.
    """

    if multipv < 1:
        raise ValueError("multipv must be at least 1")
    legal_moves = generate_legal_moves(board)
    if not legal_moves:
        return [
            SearchResult(
                move=None,
                score=score_for_color(board, board.side_to_move),
                depth=depth,
                completed_depth=0,
            )
        ]

    results: list[SearchResult] = []
    base_penalties = dict(root_move_penalties or {})
    exclusion_penalties: dict[str, int] = {}
    for _ in range(min(multipv, len(legal_moves))):
        penalties = dict(base_penalties)
        for move_uci, penalty in exclusion_penalties.items():
            penalties[move_uci] = penalties.get(move_uci, 0) + penalty
        if backend == "native" and native_time_limit is not None:
            result = _find_native_fixed_depth_move(
                board,
                depth=depth,
                quiescence_depth=quiescence_depth,
                root_move_penalties=penalties,
                time_limit_seconds=native_time_limit / max(1, multipv),
            )
        else:
            result = find_backend_move(
                board,
                depth=depth,
                backend=backend,
                time_limit=None,
                adaptive=False,
                quiescence_depth=quiescence_depth,
                root_move_penalties=penalties,
            )
        if result.move is None:
            if not results:
                results.append(result)
            break
        results.append(result)
        exclusion_penalties[result.move.to_uci()] = 1_000_000
    return results


def _find_native_fixed_depth_move(
    board: Board,
    *,
    depth: int,
    quiescence_depth: int,
    root_move_penalties: Mapping[str, int] | None,
    time_limit_seconds: float | None = None,
    iterative: bool = False,
) -> SearchResult:
    native_result = find_best_line_native(
        board,
        depth=depth,
        weights=get_evaluation_weights(),
        quiescence_depth=quiescence_depth,
        root_move_penalties=root_move_penalties,
        time_limit_ms=None if time_limit_seconds is None else max(1, int(time_limit_seconds * 1000)),
        iterative=iterative,
    )
    if native_result is not None:
        completed_depth = native_result.completed_depth if iterative else (0 if native_result.aborted else depth)
        has_completed_result = completed_depth > 0
        return SearchResult(
            move=native_result.move,
            score=native_result.score if has_completed_result else 0,
            depth=depth if has_completed_result else 0,
            completed_depth=completed_depth,
            principal_variation=native_result.principal_variation,
            nodes=max(1, native_result.nodes),
        )
    raise RuntimeError("native search library is not available; build chess_engine/native first")


def _find_native_iterative_move(
    board: Board,
    *,
    max_depth: int,
    time_limit_seconds: float,
    quiescence_depth: int,
    root_move_penalties: Mapping[str, int] | None,
) -> SearchResult:
    if time_limit_seconds <= 0:
        raise ValueError("time_limit_seconds must be positive")
    if max_depth < 1:
        raise ValueError("max_depth must be at least 1")

    started = perf_counter()
    if native_internal_iterative_deepening_enabled():
        return _find_native_fixed_depth_move(
            board,
            depth=max_depth,
            quiescence_depth=quiescence_depth,
            root_move_penalties=root_move_penalties,
            time_limit_seconds=time_limit_seconds,
            iterative=True,
        )
    best_result: SearchResult | None = None
    interrupted_result: SearchResult | None = None
    completed_depth = 0
    total_nodes = 0
    for current_depth in range(1, max_depth + 1):
        remaining = time_limit_seconds - (perf_counter() - started)
        if remaining <= 0:
            break
        result = _find_native_fixed_depth_move(
            board,
            depth=current_depth,
            quiescence_depth=quiescence_depth,
            root_move_penalties=root_move_penalties,
            time_limit_seconds=remaining,
        )
        total_nodes += result.nodes
        if result.completed_depth == 0:
            interrupted_result = result
            break
        best_result = result
        completed_depth = current_depth
        if perf_counter() - started >= time_limit_seconds:
            break

    if best_result is None:
        best_result = interrupted_result
        if best_result is None:
            best_result = _find_native_fixed_depth_move(
                board,
                depth=1,
                quiescence_depth=quiescence_depth,
                root_move_penalties=root_move_penalties,
                time_limit_seconds=time_limit_seconds,
            )
        completed_depth = best_result.completed_depth
    return SearchResult(
        move=best_result.move,
        score=best_result.score,
        depth=max_depth,
        completed_depth=completed_depth,
        principal_variation=best_result.principal_variation,
        mate_in=best_result.mate_in,
        nodes=max(1, total_nodes or best_result.nodes),
        time_ms=max(1, int((perf_counter() - started) * 1000)),
    )


def _find_native_adaptive_move(
    board: Board,
    *,
    max_depth: int,
    max_time_seconds: float,
    quiescence_depth: int,
    root_move_penalties: Mapping[str, int] | None,
) -> SearchResult:
    piece_count = sum(1 for row in board.grid for piece in row if piece is not None)
    if piece_count > 24:
        allocated_time = max_time_seconds * 0.45
    elif piece_count > 12:
        allocated_time = max_time_seconds * 0.7
    else:
        allocated_time = max_time_seconds
    return _find_native_iterative_move(
        board,
        max_depth=max_depth,
        time_limit_seconds=max(0.05, allocated_time),
        quiescence_depth=quiescence_depth,
        root_move_penalties=root_move_penalties,
    )
