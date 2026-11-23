"""Search result types shared by the native search backend and the UI."""

from __future__ import annotations

from dataclasses import dataclass

from chess_engine.ai.evaluator import CHECKMATE_SCORE
from chess_engine.chess_core.move import Move


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Result of an engine search."""

    move: Move | None
    score: int
    depth: int
    completed_depth: int = 0
    table_hits: int = 0
    table_stores: int = 0
    tablebase_hits: int = 0
    principal_variation: tuple[Move, ...] = ()
    mate_in: int | None = None
    nodes: int = 0
    time_ms: int = 0


def mate_in_from_score(score: int) -> int | None:
    """Return signed mate distance estimate from a mate score."""
    if abs(score) < CHECKMATE_SCORE - 1_000:
        return None
    plies = max(0, CHECKMATE_SCORE - abs(score))
    moves = max(1, (plies + 1) // 2)
    return moves if score > 0 else -moves
