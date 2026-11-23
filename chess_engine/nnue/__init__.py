"""NNUE preparation utilities for DeepTactics Engine.

Exports are loaded lazily so the evaluator can import the lightweight model
module without pulling in the learning pipeline and creating circular imports.
"""

from __future__ import annotations

_EXPORTS = {
    "NNUEDatasetStats": ("chess_engine.nnue.dataset", "NNUEDatasetStats"),
    "convert_self_play_to_nnue_dataset": ("chess_engine.nnue.dataset", "convert_self_play_to_nnue_dataset"),
    "NNUEFeatureVector": ("chess_engine.nnue.features", "NNUEFeatureVector"),
    "NNUE_FEATURE_COUNT": ("chess_engine.nnue.features", "NNUE_FEATURE_COUNT"),
    "extract_nnue_features": ("chess_engine.nnue.features", "extract_nnue_features"),
    "label_from_training_row": ("chess_engine.nnue.features", "label_from_training_row"),
    "normalize_fen": ("chess_engine.nnue.features", "normalize_fen"),
    "nnue_features_from_fen": ("chess_engine.nnue.features", "nnue_features_from_fen"),
    "LinearNNUEModel": ("chess_engine.nnue.model", "LinearNNUEModel"),
    "AccumulatorNNUEModel": ("chess_engine.nnue.model", "AccumulatorNNUEModel"),
    "NNUEAccumulatorState": ("chess_engine.nnue.model", "NNUEAccumulatorState"),
    "blend_evaluations": ("chess_engine.nnue.model", "blend_evaluations"),
    "evaluate_accumulator_nnue": ("chess_engine.nnue.model", "evaluate_accumulator_nnue"),
    "evaluate_linear_nnue": ("chess_engine.nnue.model", "evaluate_linear_nnue"),
    "NNUELinearTrainingStats": ("chess_engine.nnue.trainer", "NNUELinearTrainingStats"),
    "NNUEShardCheckpointStatus": ("chess_engine.nnue.trainer", "NNUEShardCheckpointStatus"),
    "NNUEShardTrainingStats": ("chess_engine.nnue.trainer", "NNUEShardTrainingStats"),
    "nnue_shard_checkpoint_status": ("chess_engine.nnue.trainer", "nnue_shard_checkpoint_status"),
    "train_sparse_linear_nnue": ("chess_engine.nnue.trainer", "train_sparse_linear_nnue"),
    "train_sparse_linear_nnue_shards": ("chess_engine.nnue.trainer", "train_sparse_linear_nnue_shards"),
    "train_deep_accumulator_nnue_from_linear": (
        "chess_engine.nnue.trainer",
        "train_deep_accumulator_nnue_from_linear",
    ),
    "NNUEAccumulatorTrainingStats": ("chess_engine.nnue.trainer", "NNUEAccumulatorTrainingStats"),
    "train_accumulator_nnue": ("chess_engine.nnue.trainer", "train_accumulator_nnue"),
    "NNUECompareReport": ("chess_engine.nnue.report", "NNUECompareReport"),
    "compare_linear_nnue_to_dataset": ("chess_engine.nnue.report", "compare_linear_nnue_to_dataset"),
    "compare_accumulator_nnue_to_dataset": ("chess_engine.nnue.report", "compare_accumulator_nnue_to_dataset"),
    "save_nnue_compare_report": ("chess_engine.nnue.report", "save_nnue_compare_report"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    """Load NNUE helpers only when they are requested."""
    if name not in _EXPORTS:
        raise AttributeError(f"module 'chess_engine.nnue' has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    from importlib import import_module

    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
