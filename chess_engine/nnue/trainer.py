"""Streaming trainer for the first NNUE-compatible sparse model."""

from __future__ import annotations

import csv
import glob
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from time import time

from chess_engine.nnue.features import NNUE_FEATURE_COUNT
from chess_engine.nnue.model import (
    AccumulatorNNUEModel,
    LinearNNUEModel,
    NNUE_ACTIVATION_CLIPPED_RELU,
)

CLIPPED_RELU_MAX = 127.0


@dataclass(frozen=True, slots=True)
class NNUELinearTrainingStats:
    """Summary from one sparse linear training run."""

    rows_seen: int
    rows_trained: int
    rows_skipped: int
    average_loss: float
    output_path: str
    input_model_path: str
    epochs: int

    def lines(self) -> list[str]:
        return [
            f"Rows seen: {self.rows_seen}",
            f"Rows trained: {self.rows_trained}",
            f"Rows skipped: {self.rows_skipped}",
            f"Average loss: {self.average_loss:.2f}",
            f"Epochs: {self.epochs}",
            f"Input model: {self.input_model_path or '-'}",
            f"Output model: {self.output_path}",
        ]


@dataclass(frozen=True, slots=True)
class NNUEShardTrainingStats:
    """Summary from training many NNUE shards in sorted order."""

    shard_count: int
    rows_seen: int
    rows_trained: int
    rows_skipped: int
    output_path: str
    report_dir: str
    reports_written: int
    skipped_shards: int
    checkpoint_path: str

    def lines(self) -> list[str]:
        return [
            f"Shard count: {self.shard_count}",
            f"Skipped shards: {self.skipped_shards}",
            f"Rows seen: {self.rows_seen}",
            f"Rows trained: {self.rows_trained}",
            f"Rows skipped: {self.rows_skipped}",
            f"Output model: {self.output_path}",
            f"Report dir: {self.report_dir or '-'}",
            f"Reports written: {self.reports_written}",
            f"Checkpoint: {self.checkpoint_path or '-'}",
        ]


@dataclass(frozen=True, slots=True)
class NNUEShardCheckpointStatus:
    """Human-readable status for an NNUE shard checkpoint."""

    checkpoint_path: str
    input_pattern: str
    output_model: str
    total_shards: int
    completed_count: int
    remaining_count: int

    def lines(self) -> list[str]:
        percent = 0.0 if self.total_shards == 0 else self.completed_count / self.total_shards * 100
        return [
            f"Checkpoint: {self.checkpoint_path}",
            f"Input pattern: {self.input_pattern or '-'}",
            f"Output model: {self.output_model or '-'}",
            f"Total shards: {self.total_shards}",
            f"Completed shards: {self.completed_count}",
            f"Remaining shards: {self.remaining_count}",
            f"Progress: {percent:.1f}%",
        ]


def train_sparse_linear_nnue(
    input_paths: list[str | Path],
    output_path: str | Path,
    *,
    input_model_path: str | Path | None = None,
    epochs: int = 1,
    learning_rate: float = 0.001,
    l2: float = 0.000001,
    max_rows: int | None = None,
    reference_weight: float = 1.0,
    self_play_weight: float = 0.35,
    progress_every: int = 0,
) -> NNUELinearTrainingStats:
    """Train or continue a sparse linear NNUE probe from NNUE dataset files."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be positive when provided")

    model = (
        LinearNNUEModel.from_json(input_model_path)
        if input_model_path is not None and Path(input_model_path).exists()
        else LinearNNUEModel(feature_count=NNUE_FEATURE_COUNT, bias=0.0, weights={}, metadata={})
    )
    weights = dict(model.weights)
    bias = float(model.bias)
    total_loss = 0.0
    rows_seen = 0
    rows_trained = 0
    rows_skipped = 0

    for epoch in range(1, epochs + 1):
        epoch_trained = 0
        for row in _iter_dataset_rows(input_paths):
            if max_rows is not None and rows_trained >= max_rows:
                break
            rows_seen += 1
            parsed = _parse_training_row(row)
            if parsed is None:
                rows_skipped += 1
                continue
            active_indices, target_cp, label_mode = parsed
            sample_weight = _sample_weight(label_mode, reference_weight, self_play_weight)
            prediction = bias + sum(weights.get(index, 0.0) for index in active_indices)
            error = prediction - target_cp
            weighted_error = error * sample_weight
            total_loss += weighted_error * weighted_error
            rows_trained += 1
            epoch_trained += 1

            scale = learning_rate * weighted_error / max(1, len(active_indices))
            bias -= learning_rate * weighted_error * 0.01
            for index in active_indices:
                current = weights.get(index, 0.0)
                current -= scale + learning_rate * l2 * current
                if abs(current) < 1e-12:
                    weights.pop(index, None)
                else:
                    weights[index] = current

            if progress_every > 0 and rows_trained % progress_every == 0:
                avg_loss = total_loss / rows_trained
                print(
                    f"  epoch={epoch}, trained={rows_trained}, skipped={rows_skipped}, avg_loss={avg_loss:.2f}",
                    flush=True,
                )
        if max_rows is not None and rows_trained >= max_rows:
            break
        if epoch_trained == 0:
            break

    metadata = {
        "trained_at_unix": int(time()),
        "epochs": epochs,
        "rows_seen": rows_seen,
        "rows_trained": rows_trained,
        "rows_skipped": rows_skipped,
        "learning_rate": learning_rate,
        "l2": l2,
        "reference_weight": reference_weight,
        "self_play_weight": self_play_weight,
        "input_files": [str(path) for path in input_paths],
        "input_model": str(input_model_path) if input_model_path else "",
        "note": "Sparse linear NNUE probe. Reference labels receive higher default trust than self-play labels.",
    }
    output_model = LinearNNUEModel(
        feature_count=NNUE_FEATURE_COUNT,
        bias=bias,
        weights=weights,
        metadata=metadata,
    )
    output_model.save_json(output_path)
    return NNUELinearTrainingStats(
        rows_seen=rows_seen,
        rows_trained=rows_trained,
        rows_skipped=rows_skipped,
        average_loss=total_loss / rows_trained if rows_trained else 0.0,
        output_path=str(output_path),
        input_model_path=str(input_model_path) if input_model_path else "",
        epochs=epochs,
    )


def train_sparse_linear_nnue_shards(
    input_pattern: str | Path,
    output_path: str | Path,
    *,
    input_model_path: str | Path | None = None,
    report_dir: str | Path | None = None,
    report_format: str = "json",
    compare_max_rows: int | None = 10000,
    include_baseline: bool = False,
    epochs: int = 1,
    learning_rate: float = 0.001,
    l2: float = 0.000001,
    max_rows_per_shard: int | None = None,
    reference_weight: float = 1.0,
    self_play_weight: float = 0.35,
    progress_every: int = 0,
    checkpoint_path: str | Path | None = None,
    resume: bool = True,
) -> NNUEShardTrainingStats:
    """Train through sorted shard files and optionally save a report per shard.

    The output model is reused as the input model for the next shard, so a long
    Reference run can progress through ``nnue_reference_000001.csv``,
    ``nnue_reference_000002.csv`` and so on without manual bookkeeping.
    """

    shard_paths = _resolve_shard_paths(input_pattern)
    if not shard_paths:
        raise FileNotFoundError(f"No NNUE shard files matched: {input_pattern}")
    if report_format not in {"json", "csv"}:
        raise ValueError("report_format must be 'json' or 'csv'")

    output_path = Path(output_path)
    checkpoint = Path(checkpoint_path) if checkpoint_path else output_path.with_suffix(output_path.suffix + ".shards.json")
    completed_shards = _load_completed_shards(checkpoint) if resume else set()
    current_input_model = Path(input_model_path) if input_model_path else None
    if current_input_model is None and output_path.exists():
        current_input_model = output_path

    total_seen = 0
    total_trained = 0
    total_skipped = 0
    reports_written = 0
    skipped_shards = 0
    report_path = Path(report_dir) if report_dir else None
    if report_path is not None:
        report_path.mkdir(parents=True, exist_ok=True)

    for index, shard_path in enumerate(shard_paths, start=1):
        shard_key = _checkpoint_key(shard_path)
        if shard_key in completed_shards:
            skipped_shards += 1
            print(f"Shard {index}/{len(shard_paths)} skipped: {shard_path}", flush=True)
            continue
        print(f"Shard {index}/{len(shard_paths)}: {shard_path}", flush=True)
        stats = train_sparse_linear_nnue(
            [shard_path],
            output_path,
            input_model_path=current_input_model,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
            max_rows=max_rows_per_shard,
            reference_weight=reference_weight,
            self_play_weight=self_play_weight,
            progress_every=progress_every,
        )
        total_seen += stats.rows_seen
        total_trained += stats.rows_trained
        total_skipped += stats.rows_skipped
        current_input_model = output_path

        if report_path is not None:
            from chess_engine.nnue.report import compare_linear_nnue_to_dataset, save_nnue_compare_report

            report = compare_linear_nnue_to_dataset(
                [shard_path],
                output_path,
                max_rows=compare_max_rows,
                include_baseline=include_baseline,
            )
            shard_report_path = report_path / f"{shard_path.stem}_compare.{report_format}"
            save_nnue_compare_report(report, shard_report_path, report_format)
            reports_written += 1
            print(f"  Report: {shard_report_path}", flush=True)

        completed_shards.add(shard_key)
        _save_shard_checkpoint(
            checkpoint,
            completed_shards,
            output_path=output_path,
            input_pattern=input_pattern,
            total_shards=len(shard_paths),
        )

    return NNUEShardTrainingStats(
        shard_count=len(shard_paths),
        rows_seen=total_seen,
        rows_trained=total_trained,
        rows_skipped=total_skipped,
        output_path=str(output_path),
        report_dir=str(report_path) if report_path else "",
        reports_written=reports_written,
        skipped_shards=skipped_shards,
        checkpoint_path=str(checkpoint),
    )


def train_deep_accumulator_nnue_from_linear(
    input_model_path: str | Path,
    output_path: str | Path,
    *,
    hidden_dim: int = 32,
    activation: int = NNUE_ACTIVATION_CLIPPED_RELU,
) -> AccumulatorNNUEModel:
    """Expand a sparse linear probe into a deeper accumulator model shell.

    This is not yet backpropagation through a hidden network. It creates a
    multi-hidden accumulator binary that can be consumed by the runtime and
    later fine-tuned by a real NNUE trainer. The trained sparse signal is
    distributed deterministically across hidden units so inference remains
    numerically close to the source probe.
    """

    if hidden_dim < 1:
        raise ValueError("hidden_dim must be at least 1")
    linear = LinearNNUEModel.from_json(input_model_path)
    feature_weights: dict[tuple[int, int], float] = {}
    for feature_index, value in linear.weights.items():
        share = float(value) / hidden_dim
        for hidden_index in range(hidden_dim):
            feature_weights[(int(feature_index), hidden_index)] = share
    model = AccumulatorNNUEModel(
        feature_count=linear.feature_count,
        hidden_dim=hidden_dim,
        output_bias=linear.bias,
        feature_weights=feature_weights,
        output_weights=tuple(1.0 for _ in range(hidden_dim)),
        activation=activation,
        metadata={
            "format": "deeptactics_accumulator_deep_shell_v1",
            "source": str(input_model_path),
            "hidden_dim": hidden_dim,
            "note": "Expanded from sparse linear probe; ready for future hidden-layer fine tuning.",
        },
    )
    model.save_binary(output_path)
    return model


@dataclass(frozen=True, slots=True)
class NNUEAccumulatorTrainingStats:
    """Summary from one real accumulator (hidden-layer) training run."""

    rows_seen: int
    rows_trained: int
    rows_skipped: int
    average_loss: float
    output_path: str
    input_model_path: str
    hidden_dim: int
    epochs: int

    def lines(self) -> list[str]:
        return [
            f"Rows seen: {self.rows_seen}",
            f"Rows trained: {self.rows_trained}",
            f"Rows skipped: {self.rows_skipped}",
            f"Average loss: {self.average_loss:.2f}",
            f"Hidden dim: {self.hidden_dim}",
            f"Epochs: {self.epochs}",
            f"Input model: {self.input_model_path}",
            f"Output model: {self.output_path}",
        ]


def train_accumulator_nnue(
    input_paths: list[str | Path],
    output_path: str | Path,
    *,
    input_model_path: str | Path,
    epochs: int = 1,
    learning_rate: float = 0.001,
    l2: float = 0.000001,
    max_rows: int | None = None,
    reference_weight: float = 1.0,
    self_play_weight: float = 0.35,
    progress_every: int = 0,
    symmetry_break_seed: int = 20260817,
    symmetry_break_scale: float = 0.01,
    error_clip: float = 1000.0,
    lr_decay: float = 0.0001,
) -> NNUEAccumulatorTrainingStats:
    """Train a hidden-layer accumulator model with real forward/backward SGD.

    Unlike ``train_deep_accumulator_nnue_from_linear`` (a static shell that
    splits a linear probe's weights evenly across hidden units with no
    gradient descent), this runs actual sparse-input -> hidden(clipped ReLU)
    -> output-scalar backprop. Warm-starts from an existing accumulator
    binary (typically produced by that shell function) and perturbs the
    output layer slightly: identical hidden units would otherwise receive an
    identical gradient every step and never diverge, wasting the extra
    hidden capacity.

    ``error_clip`` bounds the per-row error used in weight updates (the
    reported loss stays unclipped). Real NNUE datasets mix in near-mate/
    blowout labels far outside the typical few-hundred-cp range; without
    clipping, one such row multiplies straight through output_bias,
    every output_weight, and every active feature_weight in one step and
    reliably explodes this 2-layer model to NaN within a few thousand rows,
    even at a learning_rate that is safe for the (single-layer) linear
    trainer.

    ``lr_decay`` applies a 1/(1 + lr_decay * rows_trained) schedule to every
    update. output_bias gets no L2 pull (matching the linear trainer, since
    it is a single scalar with no overfitting risk of its own), so a
    constant step size can walk it arbitrarily far over a large dataset even
    with error_clip bounding each individual step -- real training sets run
    to 100k+ rows, where that drift is large enough to dominate the
    prediction. Decaying the step size keeps early learning fast while
    preventing that late-training drift.
    """
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be positive when provided")

    base_model = AccumulatorNNUEModel.from_binary(input_model_path)
    hidden_dim = base_model.hidden_dim
    feature_weights = dict(base_model.feature_weights)
    rng = random.Random(symmetry_break_seed)
    output_weights = [
        value * (1.0 + rng.uniform(-symmetry_break_scale, symmetry_break_scale))
        for value in base_model.output_weights
    ]
    output_bias = float(base_model.output_bias)

    total_loss = 0.0
    rows_seen = 0
    rows_trained = 0
    rows_skipped = 0

    for epoch in range(1, epochs + 1):
        epoch_trained = 0
        for row in _iter_dataset_rows(input_paths):
            if max_rows is not None and rows_trained >= max_rows:
                break
            rows_seen += 1
            parsed = _parse_training_row(row)
            if parsed is None:
                rows_skipped += 1
                continue
            active_indices, target_cp, label_mode = parsed
            sample_weight = _sample_weight(label_mode, reference_weight, self_play_weight)

            accumulator = [0.0] * hidden_dim
            for feature_index in active_indices:
                for hidden_index in range(hidden_dim):
                    accumulator[hidden_index] += feature_weights.get((feature_index, hidden_index), 0.0)
            activated = [max(0.0, min(CLIPPED_RELU_MAX, value)) for value in accumulator]
            prediction = output_bias + sum(
                value * weight for value, weight in zip(activated, output_weights)
            )

            error = prediction - target_cp
            weighted_error = error * sample_weight
            total_loss += weighted_error * weighted_error
            rows_trained += 1
            epoch_trained += 1

            update_error = max(-error_clip, min(error_clip, weighted_error))
            effective_lr = learning_rate / (1.0 + lr_decay * rows_trained)
            output_bias -= effective_lr * update_error + effective_lr * l2 * output_bias
            active_count = max(1, len(active_indices))
            # activated[h] ranges up to CLIPPED_RELU_MAX (127), unlike a
            # typical normalized activation -- dividing by that ceiling
            # keeps this step comparable in scale to the feature-layer step
            # below (which normalizes by active_count instead). Without it,
            # output_weights take up to ~127x the intended step and run
            # away within a few hundred rows.
            for hidden_index in range(hidden_dim):
                output_weights[hidden_index] -= (
                    effective_lr * update_error * activated[hidden_index] / CLIPPED_RELU_MAX
                    + effective_lr * l2 * output_weights[hidden_index]
                )
                in_linear_region = 0.0 < accumulator[hidden_index] < CLIPPED_RELU_MAX
                if not in_linear_region:
                    continue
                grad_hidden = update_error * output_weights[hidden_index]
                feature_scale = effective_lr * grad_hidden / active_count
                for feature_index in active_indices:
                    key = (feature_index, hidden_index)
                    current = feature_weights.get(key, 0.0)
                    current -= feature_scale + effective_lr * l2 * current
                    if abs(current) < 1e-12:
                        feature_weights.pop(key, None)
                    else:
                        feature_weights[key] = current

            if progress_every > 0 and rows_trained % progress_every == 0:
                avg_loss = total_loss / rows_trained
                print(
                    f"  epoch={epoch}, trained={rows_trained}, skipped={rows_skipped}, avg_loss={avg_loss:.2f}",
                    flush=True,
                )
        if max_rows is not None and rows_trained >= max_rows:
            break
        if epoch_trained == 0:
            break

    if not math.isfinite(output_bias) or any(not math.isfinite(value) for value in output_weights):
        raise FloatingPointError(
            "accumulator training diverged (non-finite weights) -- lower learning_rate or l2"
        )

    metadata = {
        "format": "deeptactics_accumulator_trained_v1",
        "trained_at_unix": int(time()),
        "epochs": epochs,
        "rows_seen": rows_seen,
        "rows_trained": rows_trained,
        "rows_skipped": rows_skipped,
        "learning_rate": learning_rate,
        "l2": l2,
        "hidden_dim": hidden_dim,
        "input_model": str(input_model_path),
        "note": "Real forward/backward SGD through a hidden(clipped ReLU) accumulator, warm-started from an existing accumulator binary with output-layer symmetry breaking.",
    }
    model = AccumulatorNNUEModel(
        feature_count=base_model.feature_count,
        hidden_dim=hidden_dim,
        output_bias=output_bias,
        feature_weights=feature_weights,
        output_weights=tuple(output_weights),
        activation=NNUE_ACTIVATION_CLIPPED_RELU,
        metadata=metadata,
    )
    model.save_binary(output_path)
    return NNUEAccumulatorTrainingStats(
        rows_seen=rows_seen,
        rows_trained=rows_trained,
        rows_skipped=rows_skipped,
        average_loss=total_loss / rows_trained if rows_trained else 0.0,
        output_path=str(output_path),
        input_model_path=str(input_model_path),
        hidden_dim=hidden_dim,
        epochs=epochs,
    )


def _iter_dataset_rows(input_paths: list[str | Path]):
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
            raise ValueError("NNUE training input must be .csv or .jsonl")


def _parse_training_row(row: dict[str, object]) -> tuple[tuple[int, ...], float, str] | None:
    feature_text = row.get("features")
    label = row.get("label_cp")
    if not isinstance(feature_text, str):
        return None
    try:
        active_indices = tuple(int(value) for value in feature_text.split() if value)
        target_cp = float(label)
    except (TypeError, ValueError):
        return None
    if not active_indices:
        return None
    label_mode = str(row.get("label_mode", ""))
    return active_indices, target_cp, label_mode


def _sample_weight(label_mode: str, reference_weight: float, self_play_weight: float) -> float:
    if label_mode == "reference_eval":
        return reference_weight
    return self_play_weight


def _resolve_shard_paths(input_pattern: str | Path) -> list[Path]:
    pattern = Path(input_pattern)
    if pattern.is_dir():
        candidates = list(pattern.glob("*.csv")) + list(pattern.glob("*.jsonl"))
        return sorted(path for path in candidates if path.is_file())
    if any(char in str(input_pattern) for char in "*?[]"):
        return sorted(Path(path) for path in glob.glob(str(input_pattern)) if Path(path).is_file())
    return [pattern] if pattern.is_file() else []


def _checkpoint_key(path: Path) -> str:
    return str(path.resolve())


def _load_completed_shards(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    values = data.get("completed_shards", [])
    if not isinstance(values, list):
        return set()
    return {str(value) for value in values}


def _save_shard_checkpoint(
    path: Path,
    completed_shards: set[str],
    *,
    output_path: Path,
    input_pattern: str | Path,
    total_shards: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "format": "deeptactics_nnue_shard_checkpoint_v1",
        "updated_at_unix": int(time()),
        "input_pattern": str(input_pattern),
        "output_model": str(output_path),
        "total_shards": total_shards,
        "completed_count": len(completed_shards),
        "completed_shards": sorted(completed_shards),
    }
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temp_path.replace(path)


def nnue_shard_checkpoint_status(
    checkpoint_path: str | Path,
    *,
    input_pattern: str | Path | None = None,
) -> NNUEShardCheckpointStatus:
    path = Path(checkpoint_path)
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    completed = set(str(value) for value in data.get("completed_shards", []))
    pattern = str(input_pattern or data.get("input_pattern", ""))
    if pattern:
        total_shards = len(_resolve_shard_paths(pattern))
        if total_shards == 0:
            total_shards = int(data.get("total_shards", 0))
    else:
        total_shards = int(data.get("total_shards", 0))
    completed_count = len(completed)
    return NNUEShardCheckpointStatus(
        checkpoint_path=str(path),
        input_pattern=pattern,
        output_model=str(data.get("output_model", "")),
        total_shards=total_shards,
        completed_count=completed_count,
        remaining_count=max(0, total_shards - completed_count),
    )
