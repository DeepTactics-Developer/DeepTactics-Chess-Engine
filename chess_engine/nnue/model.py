"""Small NNUE-compatible inference scaffolding.

This is not the final NNUE architecture. It is a deliberately tiny sparse
linear probe that lets the project test loading, inference, and blending before
we commit to a full accumulator-based network.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

from chess_engine.chess_core.board import Board
from chess_engine.chess_core.move import Move

NNUE_FEATURE_COUNT = 98_317
NNUE_BINARY_MAGIC = b"DTNNUE1\0"
NNUE_ACCUMULATOR_BINARY_MAGIC = b"DTNNU2\0\0"
NNUE_BINARY_FORMAT = "deeptactics_sparse_linear_binary_v1"
NNUE_ACCUMULATOR_BINARY_FORMAT = "deeptactics_accumulator_binary_v2"
NNUE_ACTIVATION_IDENTITY = 0
NNUE_ACTIVATION_CLIPPED_RELU = 1


@dataclass(frozen=True, slots=True)
class LinearNNUEModel:
    """Sparse linear model with the same feature indices as the NNUE dataset."""

    feature_count: int
    bias: float
    weights: dict[int, float]
    metadata: dict[str, object] | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> "LinearNNUEModel":
        """Load a simple sparse model JSON."""
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        weights = {int(index): float(value) for index, value in dict(data.get("weights", {})).items()}
        return cls(
            feature_count=int(data.get("feature_count", NNUE_FEATURE_COUNT)),
            bias=float(data.get("bias", 0.0)),
            weights=weights,
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def from_binary(cls, path: str | Path) -> "LinearNNUEModel":
        """Load the experimental sparse binary format used by native tests."""
        data = Path(path).read_bytes()
        if len(data) < 24 or data[:8] != NNUE_BINARY_MAGIC:
            raise ValueError("invalid DeepTactics NNUE binary model")
        feature_count, bias, weight_count = struct.unpack_from("<IdI", data, 8)
        offset = 24
        weights: dict[int, float] = {}
        for _ in range(weight_count):
            if offset + 12 > len(data):
                raise ValueError("truncated DeepTactics NNUE binary model")
            index, value = struct.unpack_from("<Id", data, offset)
            weights[int(index)] = float(value)
            offset += 12
        return cls(
            feature_count=int(feature_count),
            bias=float(bias),
            weights=weights,
            metadata={
                "format": NNUE_BINARY_FORMAT,
                "source": str(path),
                "next_format_plan": NNUE_ACCUMULATOR_BINARY_FORMAT,
            },
        )

    def save_json(self, path: str | Path) -> None:
        """Save the sparse linear model as JSON."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "deeptactics_sparse_linear_nnue_probe_v1",
            "feature_count": self.feature_count,
            "bias": self.bias,
            "weights": {str(index): value for index, value in sorted(self.weights.items())},
            "metadata": self.metadata or {},
        }
        temp_path = output_path.with_name(f"{output_path.name}.tmp")
        temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temp_path.replace(output_path)

    def save_binary(self, path: str | Path) -> None:
        """Save a compact experimental binary format for future C++ loading."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        items = sorted(self.weights.items())
        payload = bytearray()
        payload += NNUE_BINARY_MAGIC
        payload += struct.pack("<IdI", self.feature_count, self.bias, len(items))
        for index, value in items:
            payload += struct.pack("<Id", int(index), float(value))
        temp_path = output_path.with_name(f"{output_path.name}.tmp")
        temp_path.write_bytes(payload)
        temp_path.replace(output_path)


@dataclass(frozen=True, slots=True)
class AccumulatorNNUEModel:
    """Accumulator-style sparse NNUE model.

    This is the first real accumulator binary contract. It stores sparse
    feature-to-hidden weights, dense output weights, and an activation mode.
    A future trainer can learn a larger hidden layer without changing the
    inference interface.
    """

    feature_count: int
    hidden_dim: int
    output_bias: float
    feature_weights: dict[tuple[int, int], float]
    output_weights: tuple[float, ...]
    activation: int = NNUE_ACTIVATION_IDENTITY
    metadata: dict[str, object] | None = None

    @classmethod
    def from_linear(cls, model: LinearNNUEModel) -> "AccumulatorNNUEModel":
        """Lift a sparse linear probe into a one-neuron accumulator model."""
        return cls(
            feature_count=model.feature_count,
            hidden_dim=1,
            output_bias=model.bias,
            feature_weights={(int(index), 0): float(value) for index, value in model.weights.items()},
            output_weights=(1.0,),
            activation=NNUE_ACTIVATION_IDENTITY,
            metadata={
                "format": NNUE_ACCUMULATOR_BINARY_FORMAT,
                "source_format": NNUE_BINARY_FORMAT,
                "note": "one-neuron identity accumulator converted from sparse linear probe",
            },
        )

    @classmethod
    def from_binary(cls, path: str | Path) -> "AccumulatorNNUEModel":
        """Load the accumulator v2 binary format."""
        data = Path(path).read_bytes()
        if len(data) < 32 or data[:8] != NNUE_ACCUMULATOR_BINARY_MAGIC:
            raise ValueError("invalid DeepTactics accumulator NNUE binary model")
        feature_count, hidden_dim, activation = struct.unpack_from("<III", data, 8)
        output_bias = struct.unpack_from("<d", data, 20)[0]
        transform_count = struct.unpack_from("<I", data, 28)[0]
        if feature_count <= 0 or hidden_dim <= 0:
            raise ValueError("invalid accumulator NNUE dimensions")
        offset = 32
        feature_weights: dict[tuple[int, int], float] = {}
        for _ in range(transform_count):
            if offset + 16 > len(data):
                raise ValueError("truncated accumulator NNUE transform weights")
            feature_index, hidden_index, value = struct.unpack_from("<IId", data, offset)
            offset += 16
            if feature_index < feature_count and hidden_index < hidden_dim:
                feature_weights[(int(feature_index), int(hidden_index))] = float(value)
        if offset + hidden_dim * 8 > len(data):
            raise ValueError("truncated accumulator NNUE output weights")
        output_weights = struct.unpack_from(f"<{hidden_dim}d", data, offset)
        return cls(
            feature_count=int(feature_count),
            hidden_dim=int(hidden_dim),
            output_bias=float(output_bias),
            feature_weights=feature_weights,
            output_weights=tuple(float(value) for value in output_weights),
            activation=int(activation),
            metadata={"format": NNUE_ACCUMULATOR_BINARY_FORMAT, "source": str(path)},
        )

    def save_binary(self, path: str | Path) -> None:
        """Save the accumulator v2 binary format."""
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be at least 1")
        if len(self.output_weights) != self.hidden_dim:
            raise ValueError("output_weights length must match hidden_dim")
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        items = sorted(self.feature_weights.items())
        payload = bytearray()
        payload += NNUE_ACCUMULATOR_BINARY_MAGIC
        payload += struct.pack("<III", self.feature_count, self.hidden_dim, self.activation)
        payload += struct.pack("<dI", self.output_bias, len(items))
        for (feature_index, hidden_index), value in items:
            payload += struct.pack("<IId", int(feature_index), int(hidden_index), float(value))
        payload += struct.pack(f"<{self.hidden_dim}d", *(float(value) for value in self.output_weights))
        temp_path = output_path.with_name(f"{output_path.name}.tmp")
        temp_path.write_bytes(payload)
        temp_path.replace(output_path)


def evaluate_linear_nnue(board: Board, model: LinearNNUEModel) -> int:
    """Evaluate a board with the temporary sparse NNUE-compatible model."""
    from chess_engine.nnue.features import extract_nnue_features

    features = extract_nnue_features(board)
    score = model.bias
    for index in features.active_indices:
        score += model.weights.get(index, 0.0)
    return int(round(score))


def evaluate_accumulator_nnue(board: Board, model: AccumulatorNNUEModel) -> int:
    """Evaluate a board with the accumulator v2 NNUE model."""
    from chess_engine.nnue.features import extract_nnue_features

    features = extract_nnue_features(board)
    accumulator = [0.0] * model.hidden_dim
    for feature_index in features.active_indices:
        for hidden_index in range(model.hidden_dim):
            accumulator[hidden_index] += model.feature_weights.get((feature_index, hidden_index), 0.0)
    activated = [_activate_accumulator(value, model.activation) for value in accumulator]
    score = model.output_bias
    for hidden_value, output_weight in zip(activated, model.output_weights):
        score += hidden_value * output_weight
    return int(round(score))


@dataclass(slots=True)
class NNUEAccumulatorState:
    """Incrementally maintain accumulator values for one board.

    This is the Python-side contract for true accumulator updates. The native
    search still needs make/unmake wiring, but this class proves the feature
    add/remove delta path and gives tests a stable reference implementation.
    """

    model: AccumulatorNNUEModel
    active_features: set[int]
    accumulator: list[float]

    @classmethod
    def from_board(cls, board: Board, model: AccumulatorNNUEModel) -> "NNUEAccumulatorState":
        from chess_engine.nnue.features import extract_nnue_features

        active = set(extract_nnue_features(board).active_indices)
        return cls(model=model, active_features=active, accumulator=_accumulator_from_features(active, model))

    def evaluate(self) -> int:
        score = self.model.output_bias
        for value, output_weight in zip(self.accumulator, self.model.output_weights):
            score += _activate_accumulator(value, self.model.activation) * output_weight
        return int(round(score))

    def replace_features(self, new_features: set[int]) -> None:
        removed = self.active_features - new_features
        added = new_features - self.active_features
        for feature_index in removed:
            self._add_feature(feature_index, sign=-1.0)
        for feature_index in added:
            self._add_feature(feature_index, sign=1.0)
        self.active_features = set(new_features)

    def update_after_move(self, board_after_move: Board, move: Move | None = None) -> None:
        """Apply feature delta after a move.

        ``move`` is accepted for the native make/unmake API shape. The current
        sparse feature extractor is board based, so the delta is computed by
        comparing feature sets.
        """

        from chess_engine.nnue.features import extract_nnue_features

        self.replace_features(set(extract_nnue_features(board_after_move).active_indices))

    def _add_feature(self, feature_index: int, *, sign: float) -> None:
        for hidden_index in range(self.model.hidden_dim):
            self.accumulator[hidden_index] += sign * self.model.feature_weights.get((feature_index, hidden_index), 0.0)


def _accumulator_from_features(features: set[int], model: AccumulatorNNUEModel) -> list[float]:
    accumulator = [0.0] * model.hidden_dim
    for feature_index in features:
        for hidden_index in range(model.hidden_dim):
            accumulator[hidden_index] += model.feature_weights.get((feature_index, hidden_index), 0.0)
    return accumulator


def _activate_accumulator(value: float, activation: int) -> float:
    if activation == NNUE_ACTIVATION_IDENTITY:
        return value
    if activation == NNUE_ACTIVATION_CLIPPED_RELU:
        return max(0.0, min(127.0, value))
    raise ValueError(f"unsupported accumulator activation: {activation}")


def blend_evaluations(handcrafted_score: int, nnue_score: int | None, nnue_weight: float = 0.5) -> int:
    """Blend handcrafted and NNUE scores while NNUE is being validated."""
    if nnue_score is None:
        return handcrafted_score
    weight = max(0.0, min(1.0, nnue_weight))
    return int(round(handcrafted_score * (1.0 - weight) + nnue_score * weight))
