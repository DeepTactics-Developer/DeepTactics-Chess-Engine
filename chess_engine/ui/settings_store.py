"""Persistent settings storage for the Qt desktop interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


DEFAULT_GUI_SETTINGS: dict[str, Any] = {
    "theme": "Dark",
    "language": "English",
    "board_theme": "Wood",
    "piece_set_path": "assets/pieces",
    "start_with": "Play",
    "max_think_time": 60,
    "check_for_updates": False,
    "default_analysis_depth": 4,
    "analysis_mode": "Fixed Depth",
    "default_move_time": 5,
    "analysis_multipv": 3,
    "score_perspective": "White perspective",
    "nnue_mode": "Off",
    "auto_analyze_after_move": True,
    "show_move_classification": True,
    "use_analysis_cache": True,
    "review_full_game": True,
    "live_analysis_after_review": True,
    "show_arrows": True,
    "enable_premoves": True,
    "highlight_last_move": True,
    "show_move_numbers": True,
    "show_coordinates": True,
    "confirm_resign": True,
    "restore_clock_on_undo": False,
    "auto_promote_queen": False,
    "minimum_move_cost_ms": 100,
    "auto_save_pgn": True,
    "show_tooltips": True,
    "pgn_auto_save": "artifacts/pgn",
    "analysis_export": "artifacts/analysis",
    "database": "artifacts/database",
    "engine_files": "engines",
    "screenshots": "artifacts/screenshots",
    "logs": "artifacts/logs",
    "opening_books": "assets/openings",
    "syzygy_path": "",
    "use_tablebase": False,
    "tablebase_max_pieces": 7,
    "training_outputs": "artifacts/training",
    "interface_scale": 100,
    "compact_tables": False,
    "show_raw_score_tooltip": False,
    "sound_move": False,
    "sound_capture": False,
    "sound_check": False,
    "sound_game_end": False,
    "sound_engine_ready": False,
    "sound_mute_all": True,
    "sound_volume": 70,
    "debug_mode": False,
    "show_uci_log": False,
    "save_uci_logs": False,
    "show_raw_engine_output": False,
    "default_analysis_engine": "DeepTactics",
    "default_play_engine": "DeepTactics",
    "default_ai_match_engine": "DeepTactics",
    "ai_match_search_mode": "Fixed Depth",
    "ai_match_depth": 5,
    "ai_match_movetime_ms": 500,
    "engines": [
        {
            "name": "DeepTactics",
            "type": "UCI",
            "path": "DeepTactics-UCI.cmd",
            "status": "Built-in",
            "nnue": False,
            "threads": 1,
            "hash": 256,
            "multipv": 1,
            "use_tablebase": False,
            "syzygy_path": "",
            "tablebase_max_pieces": 7,
        }
    ],
}


class SettingsStore:
    """Read and write Qt settings without coupling widgets to JSON details."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> dict[str, Any]:
        settings = _copy_defaults()
        if not self.path.exists():
            return settings
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return settings
        if isinstance(loaded, dict):
            settings.update(loaded)
        return settings

    def save(self, values: Mapping[str, Any]) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        settings = _copy_defaults()
        settings.update(dict(values))
        self.path.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
        return self.path

    def reset(self) -> dict[str, Any]:
        defaults = _copy_defaults()
        self.save(defaults)
        return defaults


def _copy_defaults() -> dict[str, Any]:
    defaults = dict(DEFAULT_GUI_SETTINGS)
    defaults["engines"] = [dict(engine) for engine in DEFAULT_GUI_SETTINGS["engines"]]
    return defaults
