"""Training data quality reporting."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from chess_engine.learning.trainer import load_training_rows


@dataclass(frozen=True, slots=True)
class TrainingDataReport:
    total_rows: int
    unique_positions: int
    duplicate_rows: int
    white_wins: int
    black_wins: int
    draws: int
    unfinished: int
    average_score: float
    average_abs_score: float

    def lines(self) -> list[str]:
        return [
            f"Rows: {self.total_rows}",
            f"Unique positions: {self.unique_positions}",
            f"Duplicate rows: {self.duplicate_rows}",
            f"Results: white={self.white_wins}, black={self.black_wins}, draws={self.draws}, unfinished={self.unfinished}",
            f"Average score: {self.average_score:.1f}cp",
            f"Average abs score: {self.average_abs_score:.1f}cp",
        ]


def build_training_report(path: str | Path) -> TrainingDataReport:
    rows = load_training_rows(path)
    position_counts = Counter(_normalized_fen(str(row.get("fen", ""))) for row in rows)
    scores = [_parse_float(row.get("score", 0)) for row in rows]
    total = len(rows)
    return TrainingDataReport(
        total_rows=total,
        unique_positions=len(position_counts),
        duplicate_rows=sum(max(0, count - 1) for count in position_counts.values()),
        white_wins=sum(1 for row in rows if str(row.get("result", "")) == "1-0"),
        black_wins=sum(1 for row in rows if str(row.get("result", "")) == "0-1"),
        draws=sum(1 for row in rows if str(row.get("result", "")) == "1/2-1/2"),
        unfinished=sum(1 for row in rows if str(row.get("result", "")) in {"", "*"}),
        average_score=sum(scores) / total if total else 0.0,
        average_abs_score=sum(abs(score) for score in scores) / total if total else 0.0,
    )


def _normalized_fen(fen: str) -> str:
    parts = fen.split()
    if len(parts) >= 4:
        return " ".join(parts[:4])
    return fen


def _parse_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
