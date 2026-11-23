"""Quality reports for NNUE probe models."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

from chess_engine.chess_core.fen import board_from_fen
from chess_engine.nnue.model import AccumulatorNNUEModel, LinearNNUEModel, evaluate_accumulator_nnue, evaluate_linear_nnue


@dataclass(frozen=True, slots=True)
class NNUECompareReport:
    """Comparison between dataset labels, handcrafted/native eval, and NNUE."""

    rows_seen: int
    rows_evaluated: int
    rows_skipped: int
    nnue_mae: float
    nnue_rmse: float
    baseline_mae: float | None
    baseline_rmse: float | None
    average_target: float
    average_nnue: float
    average_baseline: float | None
    model_path: str

    def lines(self) -> list[str]:
        baseline_mae = "-" if self.baseline_mae is None else f"{self.baseline_mae:.2f}cp"
        baseline_rmse = "-" if self.baseline_rmse is None else f"{self.baseline_rmse:.2f}cp"
        average_baseline = "-" if self.average_baseline is None else f"{self.average_baseline:.2f}cp"
        return [
            f"Rows seen: {self.rows_seen}",
            f"Rows evaluated: {self.rows_evaluated}",
            f"Rows skipped: {self.rows_skipped}",
            f"NNUE MAE: {self.nnue_mae:.2f}cp",
            f"NNUE RMSE: {self.nnue_rmse:.2f}cp",
            f"Baseline MAE: {baseline_mae}",
            f"Baseline RMSE: {baseline_rmse}",
            f"Average target: {self.average_target:.2f}cp",
            f"Average NNUE: {self.average_nnue:.2f}cp",
            f"Average baseline: {average_baseline}",
            f"Model: {self.model_path}",
        ]

    def to_dict(self) -> dict[str, float | int | str | None]:
        return {
            "rows_seen": self.rows_seen,
            "rows_evaluated": self.rows_evaluated,
            "rows_skipped": self.rows_skipped,
            "nnue_mae": self.nnue_mae,
            "nnue_rmse": self.nnue_rmse,
            "baseline_mae": self.baseline_mae,
            "baseline_rmse": self.baseline_rmse,
            "average_target": self.average_target,
            "average_nnue": self.average_nnue,
            "average_baseline": self.average_baseline,
            "model_path": self.model_path,
        }


def compare_linear_nnue_to_dataset(
    input_paths: list[str | Path],
    model_path: str | Path,
    *,
    max_rows: int | None = None,
    include_baseline: bool = True,
) -> NNUECompareReport:
    """Compare a sparse linear NNUE probe against labelled dataset rows."""
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be positive when provided")
    model = LinearNNUEModel.from_json(model_path)

    rows_seen = 0
    rows_evaluated = 0
    rows_skipped = 0
    nnue_abs = 0.0
    nnue_sq = 0.0
    baseline_abs = 0.0
    baseline_sq = 0.0
    baseline_count = 0
    target_sum = 0.0
    nnue_sum = 0.0
    baseline_sum = 0.0

    for row in _iter_rows(input_paths):
        if max_rows is not None and rows_evaluated >= max_rows:
            break
        rows_seen += 1
        parsed = _parse_row(row)
        if parsed is None:
            rows_skipped += 1
            continue
        fen, target = parsed
        try:
            board = board_from_fen(fen)
            nnue_score = evaluate_linear_nnue(board, model)
        except ValueError:
            rows_skipped += 1
            continue
        rows_evaluated += 1
        target_sum += target
        nnue_sum += nnue_score
        nnue_error = nnue_score - target
        nnue_abs += abs(nnue_error)
        nnue_sq += nnue_error * nnue_error

        if include_baseline:
            baseline_score = _baseline_score(board)
            baseline_sum += baseline_score
            baseline_error = baseline_score - target
            baseline_abs += abs(baseline_error)
            baseline_sq += baseline_error * baseline_error
            baseline_count += 1

    return NNUECompareReport(
        rows_seen=rows_seen,
        rows_evaluated=rows_evaluated,
        rows_skipped=rows_skipped,
        nnue_mae=nnue_abs / rows_evaluated if rows_evaluated else 0.0,
        nnue_rmse=math.sqrt(nnue_sq / rows_evaluated) if rows_evaluated else 0.0,
        baseline_mae=baseline_abs / baseline_count if baseline_count else None,
        baseline_rmse=math.sqrt(baseline_sq / baseline_count) if baseline_count else None,
        average_target=target_sum / rows_evaluated if rows_evaluated else 0.0,
        average_nnue=nnue_sum / rows_evaluated if rows_evaluated else 0.0,
        average_baseline=baseline_sum / baseline_count if baseline_count else None,
        model_path=str(model_path),
    )


def compare_accumulator_nnue_to_dataset(
    input_paths: list[str | Path],
    model_path: str | Path,
    *,
    max_rows: int | None = None,
    include_baseline: bool = True,
) -> NNUECompareReport:
    """Compare an accumulator NNUE model against labelled dataset rows."""
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be positive when provided")
    model = AccumulatorNNUEModel.from_binary(model_path)

    rows_seen = 0
    rows_evaluated = 0
    rows_skipped = 0
    nnue_abs = 0.0
    nnue_sq = 0.0
    baseline_abs = 0.0
    baseline_sq = 0.0
    baseline_count = 0
    target_sum = 0.0
    nnue_sum = 0.0
    baseline_sum = 0.0

    for row in _iter_rows(input_paths):
        if max_rows is not None and rows_evaluated >= max_rows:
            break
        rows_seen += 1
        parsed = _parse_row(row)
        if parsed is None:
            rows_skipped += 1
            continue
        fen, target = parsed
        try:
            board = board_from_fen(fen)
            nnue_score = evaluate_accumulator_nnue(board, model)
        except ValueError:
            rows_skipped += 1
            continue
        rows_evaluated += 1
        target_sum += target
        nnue_sum += nnue_score
        nnue_error = nnue_score - target
        nnue_abs += abs(nnue_error)
        nnue_sq += nnue_error * nnue_error

        if include_baseline:
            baseline_score = _baseline_score(board)
            baseline_sum += baseline_score
            baseline_error = baseline_score - target
            baseline_abs += abs(baseline_error)
            baseline_sq += baseline_error * baseline_error
            baseline_count += 1

    return NNUECompareReport(
        rows_seen=rows_seen,
        rows_evaluated=rows_evaluated,
        rows_skipped=rows_skipped,
        nnue_mae=nnue_abs / rows_evaluated if rows_evaluated else 0.0,
        nnue_rmse=math.sqrt(nnue_sq / rows_evaluated) if rows_evaluated else 0.0,
        baseline_mae=baseline_abs / baseline_count if baseline_count else None,
        baseline_rmse=math.sqrt(baseline_sq / baseline_count) if baseline_count else None,
        average_target=target_sum / rows_evaluated if rows_evaluated else 0.0,
        average_nnue=nnue_sum / rows_evaluated if rows_evaluated else 0.0,
        average_baseline=baseline_sum / baseline_count if baseline_count else None,
        model_path=str(model_path),
    )


def save_nnue_compare_report(report: NNUECompareReport, path: str | Path, output_format: str | None = None) -> None:
    """Save an NNUE comparison report as JSON or CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_format = output_format or output_path.suffix.lower().lstrip(".")
    if selected_format == "json":
        output_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        return
    if selected_format == "csv":
        data = report.to_dict()
        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(data.keys()))
            writer.writeheader()
            writer.writerow(data)
        return
    raise ValueError("NNUE comparison output format must be 'json' or 'csv'")


def _baseline_score(board) -> int:
    from chess_engine.ai.evaluator import evaluate_board, reset_linear_nnue_model

    reset_linear_nnue_model()
    return evaluate_board(board)


def _parse_row(row: dict[str, object]) -> tuple[str, float] | None:
    fen = row.get("fen")
    label = row.get("label_cp")
    if not isinstance(fen, str) or not fen.strip():
        return None
    try:
        target = float(label)
    except (TypeError, ValueError):
        return None
    return fen, target


def _iter_rows(input_paths: list[str | Path]):
    for input_path in input_paths:
        path = Path(input_path)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            with path.open("r", newline="", encoding="utf-8") as file:
                yield from csv.DictReader(file)
        elif suffix in {".jsonl", ".jsonlines"}:
            with path.open("r", encoding="utf-8") as file:
                for line in file:
                    if line.strip():
                        yield json.loads(line)
        else:
            raise ValueError("NNUE comparison input must be .csv or .jsonl")
