"""Self-play learning helpers.

This package is intentionally small for now. It turns recorded self-play rows
into tunable evaluation weights, which can be loaded by the search engine.
"""

from chess_engine.learning.features import PositionFeatures, extract_position_features, features_from_fen
from chess_engine.learning.trainer import (
    load_training_rows,
    train_weights_from_file,
    train_weights_from_rows,
)
from chess_engine.learning.report import TrainingDataReport, build_training_report

__all__ = [
    "PositionFeatures",
    "extract_position_features",
    "features_from_fen",
    "load_training_rows",
    "train_weights_from_file",
    "train_weights_from_rows",
    "TrainingDataReport",
    "build_training_report",
]
