"""Convert self-play rows into NNUE training datasets."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from chess_engine.learning import load_training_rows
from chess_engine.nnue.features import NNUE_FEATURE_COUNT, label_from_training_row, nnue_features_from_fen
from chess_engine.self_play.training_data import deduplicate_training_rows


@dataclass(frozen=True, slots=True)
class NNUEDatasetStats:
    """Summary of an NNUE dataset conversion."""

    input_rows: int
    written_rows: int
    skipped_rows: int
    feature_count: int
    label_mode: str
    output_path: str

    def lines(self) -> list[str]:
        """Return human-readable summary lines."""
        return [
            f"Input rows: {self.input_rows}",
            f"Written rows: {self.written_rows}",
            f"Skipped rows: {self.skipped_rows}",
            f"Feature count: {self.feature_count}",
            f"Label mode: {self.label_mode}",
            f"Output: {self.output_path}",
        ]


def convert_self_play_to_nnue_dataset(
    input_path: str | Path,
    output_path: str | Path,
    *,
    label_mode: str = "blended",
    output_format: str | None = None,
    max_rows: int | None = None,
    deduplicate: bool = True,
    max_position_duplicates: int = 2,
) -> NNUEDatasetStats:
    """Convert cumulative self-play CSV/JSON rows into sparse NNUE samples."""
    rows = load_training_rows(input_path)
    input_count = len(rows)
    if deduplicate:
        rows = deduplicate_training_rows(rows, max_per_position=max_position_duplicates)
    if max_rows is not None:
        rows = rows[:max_rows]

    samples: list[dict[str, object]] = []
    skipped = 0
    for index, row in enumerate(rows, start=1):
        fen = row.get("fen")
        if not isinstance(fen, str) or not fen.strip():
            skipped += 1
            continue
        try:
            features = nnue_features_from_fen(fen)
            label_cp = label_from_training_row(row, label_mode)
        except (ValueError, TypeError):
            skipped += 1
            continue
        samples.append(
            {
                "row": index,
                "game": row.get("game", ""),
                "ply": row.get("ply", ""),
                "side": row.get("side", ""),
                "fen": fen,
                "move": row.get("move", ""),
                "san": row.get("san", ""),
                "label_mode": label_mode,
                "label_cp": label_cp,
                "score_cp": row.get("score", ""),
                "result_value": row.get("result_value", ""),
                "result": row.get("result", ""),
                "feature_count": NNUE_FEATURE_COUNT,
                "active_count": len(features.active_indices),
                "features": features.to_sparse_text(),
            }
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    selected_format = output_format or output.suffix.lower().lstrip(".") or "csv"
    if selected_format == "csv":
        _write_csv(samples, output)
    elif selected_format in {"jsonl", "jsonlines"}:
        _write_jsonl(samples, output)
    else:
        raise ValueError("NNUE dataset format must be 'csv' or 'jsonl'")

    return NNUEDatasetStats(
        input_rows=input_count,
        written_rows=len(samples),
        skipped_rows=skipped,
        feature_count=NNUE_FEATURE_COUNT,
        label_mode=label_mode,
        output_path=str(output),
    )


def _write_csv(samples: list[dict[str, object]], output: Path) -> None:
    fieldnames = [
        "row",
        "game",
        "ply",
        "side",
        "fen",
        "move",
        "san",
        "label_mode",
        "label_cp",
        "score_cp",
        "result_value",
        "result",
        "feature_count",
        "active_count",
        "features",
    ]
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(samples)


def _write_jsonl(samples: list[dict[str, object]], output: Path) -> None:
    with output.open("w", encoding="utf-8") as file:
        for sample in samples:
            file.write(json.dumps(sample, ensure_ascii=False))
            file.write("\n")
