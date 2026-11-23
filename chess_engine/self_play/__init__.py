"""Self-play experiment package."""

from chess_engine.self_play.self_play_runner import run_self_play_game, run_self_play_games
from chess_engine.self_play.training_data import (
    SelfPlayGameResult,
    SelfPlayPosition,
    SelfPlayProgress,
    deduplicate_training_rows,
    results_to_training_rows,
    save_self_play_results,
    save_training_rows_csv,
    save_training_rows_json,
)

__all__ = [
    "SelfPlayGameResult",
    "SelfPlayPosition",
    "SelfPlayProgress",
    "deduplicate_training_rows",
    "results_to_training_rows",
    "run_self_play_game",
    "run_self_play_games",
    "save_self_play_results",
    "save_training_rows_csv",
    "save_training_rows_json",
]
