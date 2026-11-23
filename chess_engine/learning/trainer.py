"""Train lightweight evaluation weights from self-play rows."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from chess_engine.ai.evaluator import EvaluationWeights, load_evaluation_weights, save_evaluation_weights
from chess_engine.learning.features import PositionFeatures, features_from_fen

TrainingRow = dict[str, Any]

FEATURE_NORMALIZERS = {
    "material": 900.0,
    "position": 120.0,
    "mobility": 80.0,
    "endgame": 80.0,
    "pawn_structure": 120.0,
    "bishop_pair": 50.0,
    "rook_activity": 80.0,
    "king_safety": 80.0,
    "passed_pawn": 100.0,
    "endgame_pressure": 120.0,
    "line_activity": 80.0,
    "attack_momentum": 80.0,
}

# A run's estimate is blended toward the previous weight rather than replacing
# it outright, scaled down when either the batch is small or the correlation
# signal itself is weak -- see _regularized_update. Without this, every
# training run recomputed each weight from scratch and discarded whatever the
# previous run (or a hand-tuned default) had found, and a weak-but-nonzero
# correlation still nudged the weight every time, so repeated runs on
# similarly-weak signal could random-walk rather than converge.
MIN_SAMPLES_FOR_FULL_TRUST = 2000
MIN_SIGNAL_FOR_FULL_TRUST = 0.15
MAX_STEP_PER_RUN = 0.3


def load_training_rows(path: str | Path) -> list[TrainingRow]:
    """Load self-play rows saved by the self-play command."""
    input_path = Path(path)
    extension = input_path.suffix.lower()

    if extension == ".json":
        data = json.loads(input_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("training JSON must contain a list of rows")
        return [dict(row) for row in data]

    if extension == ".csv":
        with input_path.open("r", newline="", encoding="utf-8") as file:
            return [dict(row) for row in csv.DictReader(file)]

    raise ValueError("training data must be a .json or .csv file")


def train_weights_from_file(
    input_path: str | Path,
    output_path: str | Path,
    target_mode: str = "result",
) -> EvaluationWeights:
    """Train weights from a saved self-play file and write them to JSON.

    If ``output_path`` already holds a previously trained weights file, it is
    loaded as the starting point so this run updates it rather than
    recomputing from scratch -- see train_weights_from_rows.
    """
    previous_weights = EvaluationWeights()
    output = Path(output_path)
    if output.exists():
        try:
            previous_weights = load_evaluation_weights(output, apply=False)
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    weights = train_weights_from_rows(
        load_training_rows(input_path), target_mode=target_mode, previous_weights=previous_weights
    )
    save_evaluation_weights(weights, output_path)
    return weights


def train_weights_from_rows(
    rows: list[TrainingRow],
    target_mode: str = "result",
    previous_weights: EvaluationWeights | None = None,
) -> EvaluationWeights:
    """Return evaluation weights updated from self-play rows.

    Positions from games White won nudge features that favored White upward,
    and positions from games Black won nudge features that favored Black
    upward. Drawn or unfinished rows are treated as neutral.

    Each weight is a regularized update from ``previous_weights`` (defaults
    to EvaluationWeights(), i.e. all 1.0), not a fresh recomputation -- see
    _regularized_update for why: a single run's correlation estimate is
    noisy, especially with weak or small-sample signal, so it's blended
    toward the previous value rather than replacing it outright.
    """
    if target_mode not in {"result", "move-loss", "blended"}:
        raise ValueError("target_mode must be 'result', 'move-loss', or 'blended'")
    if previous_weights is None:
        previous_weights = EvaluationWeights()
    samples: list[tuple[PositionFeatures, float]] = []
    for row in rows:
        target = _training_target(row, target_mode)
        fen = row.get("fen", row.get("fen_before"))
        if target == 0.0 or not isinstance(fen, str):
            continue
        samples.append((features_from_fen(fen), target))

    if not samples:
        return previous_weights

    return EvaluationWeights(
        **{
            feature_name: _regularized_update(
                samples, feature_name, normalizer, getattr(previous_weights, feature_name)
            )
            for feature_name, normalizer in FEATURE_NORMALIZERS.items()
        }
    )


def _result_target(result: str) -> float:
    if result == "1-0":
        return 1.0
    if result == "0-1":
        return -1.0
    return 0.0


def _training_target(row: TrainingRow, target_mode: str) -> float:
    result_target = _result_target(str(row.get("result", row.get("game_result", "*"))))
    if target_mode == "result":
        return result_target

    loss_raw = row.get("move_loss_cp")
    try:
        loss = max(0.0, float(loss_raw))
    except (TypeError, ValueError):
        loss = 0.0
    side = str(row.get("side", row.get("side_to_move", ""))).lower()
    mover_penalty = min(1.0, loss / 300.0)
    move_target = -mover_penalty if side in {"w", "white"} else mover_penalty if side in {"b", "black"} else 0.0
    if target_mode == "move-loss":
        return move_target
    if move_target == 0.0:
        return result_target
    return result_target * 0.65 + move_target * 0.35


def _regularized_update(
    samples: list[tuple[PositionFeatures, float]],
    feature_name: str,
    normalizer: float,
    previous_value: float,
) -> float:
    correlation = 0.0
    for features, target in samples:
        feature_value = getattr(features, feature_name)
        correlation += target * _clamp(feature_value / normalizer, -1.0, 1.0)

    average = correlation / len(samples)
    candidate = _clamp(1.0 + average * 0.35, 0.5, 1.5)

    # Trust this run's estimate more when there's more data AND when the
    # correlation itself is meaningfully nonzero -- a large batch with a
    # near-zero correlation should still barely move the weight, since a
    # confident measurement of "no signal" is not the same as "move toward
    # 1.0 confidently".
    sample_confidence = _clamp(len(samples) / MIN_SAMPLES_FOR_FULL_TRUST, 0.0, 1.0)
    signal_confidence = _clamp(abs(average) / MIN_SIGNAL_FOR_FULL_TRUST, 0.0, 1.0)
    step = MAX_STEP_PER_RUN * sample_confidence * signal_confidence

    updated = previous_value + step * (candidate - previous_value)
    return round(_clamp(updated, 0.5, 1.5), 3)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
