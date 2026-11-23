"""Training data structures for future self-play experiments."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from chess_engine.chess_core.enums import GameStatus


@dataclass(frozen=True, slots=True)
class SelfPlayPosition:
    """One recorded position from a self-play game."""

    fen_before: str
    move_uci: str
    move_san: str
    score: int
    ply: int
    side_to_move: str = ""
    source: str = ""
    fen_after: str = ""
    search_backend: str = ""
    depth: int = 0
    time_limit: str = ""
    adaptive: bool = False
    weights: str = ""


@dataclass(frozen=True, slots=True)
class SelfPlayProgress:
    """One live progress event emitted after a self-play move."""

    game_index: int
    ply: int
    san: str
    move_uci: str
    score: int
    source: str


@dataclass(frozen=True, slots=True)
class SelfPlayGameResult:
    """Summary and records from one self-play game."""

    positions: list[SelfPlayPosition] = field(default_factory=list)
    pgn: str = ""
    result: str = "*"
    status: GameStatus = GameStatus.ONGOING
    plies_played: int = 0
    reached_move_limit: bool = False

    def to_training_rows(self) -> list[dict[str, str | int]]:
        """Return simple row dictionaries for later CSV/JSON export."""
        return [
            {
                "fen": position.fen_before,
                "move": position.move_uci,
                "san": position.move_san,
                "score": position.score,
                "ply": position.ply,
                "side": position.side_to_move,
                "source": position.source,
                "fen_after": position.fen_after,
                "search_backend": position.search_backend,
                "depth": position.depth,
                "time_limit": position.time_limit,
                "adaptive": int(position.adaptive),
                "weights": position.weights,
                "result": self.result,
                "result_value": _result_value(self.result),
            }
            for position in self.positions
        ]


def results_to_training_rows(
    results: list[SelfPlayGameResult],
) -> list[dict[str, str | int]]:
    """Flatten many self-play results into training rows."""
    rows: list[dict[str, str | int]] = []
    for game_index, result in enumerate(results, start=1):
        for row in result.to_training_rows():
            rows.append({"game": game_index, **row})
    return rows


def save_training_rows_json(
    results: list[SelfPlayGameResult],
    path: str | Path,
) -> None:
    """Save self-play rows as JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = results_to_training_rows(results)
    output_path.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_training_rows_csv(
    results: list[SelfPlayGameResult],
    path: str | Path,
) -> None:
    """Save self-play rows as CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = results_to_training_rows(results)
    fieldnames = [
        "game",
        "ply",
        "side",
        "fen",
        "move",
        "san",
        "score",
        "source",
        "fen_after",
        "search_backend",
        "depth",
        "time_limit",
        "adaptive",
        "weights",
        "result",
        "result_value",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_self_play_results(
    results: list[SelfPlayGameResult],
    path: str | Path,
    output_format: str | None = None,
) -> None:
    """Save self-play results as JSON or CSV.

    If ``output_format`` is omitted, the file extension decides the format.
    """
    selected_format = output_format or Path(path).suffix.lower().lstrip(".")
    if selected_format == "json":
        save_training_rows_json(results, path)
        return
    if selected_format == "csv":
        save_training_rows_csv(results, path)
        return
    raise ValueError("output format must be 'json' or 'csv'")


def deduplicate_training_rows(
    rows: list[dict[str, str | int | float]],
    max_per_position: int = 2,
) -> list[dict[str, str | int | float]]:
    """Return rows capped by normalized position to keep cumulative data useful.

    The key ignores halfmove/fullmove counters, so repeated transpositions do
    not dominate training. Keeping a small cap preserves examples where the
    same position was reached in different games.
    """
    if max_per_position < 1:
        raise ValueError("max_per_position must be at least 1")
    counts: dict[str, int] = {}
    filtered: list[dict[str, str | int | float]] = []
    for row in rows:
        key = _training_position_key(row)
        count = counts.get(key, 0)
        if count >= max_per_position:
            continue
        counts[key] = count + 1
        filtered.append(row)
    return filtered


def _training_position_key(row: dict[str, str | int | float]) -> str:
    fen = str(row.get("fen", ""))
    parts = fen.split()
    if len(parts) >= 4:
        return " ".join(parts[:4])
    return fen


def _result_value(result: str) -> float:
    if result == "1-0":
        return 1.0
    if result == "0-1":
        return -1.0
    if result == "1/2-1/2":
        return 0.0
    return 0.0
