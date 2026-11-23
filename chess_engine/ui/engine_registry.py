"""Registered UCI engines shared by the Qt pages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Signal

from chess_engine.ui.settings_store import DEFAULT_GUI_SETTINGS, SettingsStore


class EngineRegistry(QObject):
    """Hold user-visible engine registrations without starting processes."""

    changed = Signal(list)

    def __init__(self, settings_store: SettingsStore) -> None:
        super().__init__()
        self.settings_store = settings_store
        stored = settings_store.load().get("engines")
        self.entries: list[dict[str, Any]] = (
            [_normalized_engine_entry(entry) for entry in stored]
            if isinstance(stored, list) and stored
            else [_normalized_engine_entry(entry) for entry in DEFAULT_GUI_SETTINGS["engines"]]
        )

    def names(self) -> list[str]:
        return [str(entry["name"]) for entry in self.entries]

    def supports_nnue(self, name: str) -> bool:
        # Built-in DeepTactics currently exposes handcrafted/native evaluation only.
        if name == "DeepTactics":
            return False
        for entry in self.entries:
            if entry["name"] == name:
                return bool(entry.get("nnue", False))
        return False

    def add_engine(self, path: str) -> str | None:
        name = Path(path).stem.replace("_", " ")
        if name and name not in self.names():
            self.entries.append({
                "name": name, "type": "UCI", "path": path, "status": "Registered / Not tested",
                "nnue": False, "threads": 1, "hash": 256, "multipv": 1,
                "use_tablebase": False, "syzygy_path": "", "tablebase_max_pieces": 7,
            })
            self.changed.emit(self.names())
            return name
        return None

    def remove_engine(self, name: str) -> None:
        if name != "DeepTactics":
            self.entries = [entry for entry in self.entries if entry["name"] != name]
            self.changed.emit(self.names())

    def unload_external(self) -> None:
        for entry in self.entries:
            if entry["name"] != "DeepTactics":
                entry["status"] = "Stopped"
        self.changed.emit(self.names())

    def entry(self, name: str) -> dict[str, Any] | None:
        return next((entry for entry in self.entries if entry.get("name") == name), None)

    def update_profile(self, name: str, **values: Any) -> None:
        entry = self.entry(name)
        if entry is not None:
            entry.update(values)
            self.changed.emit(self.names())


def _normalized_engine_entry(entry: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(entry)
    normalized.setdefault("nnue", False)
    normalized.setdefault("threads", 1)
    normalized.setdefault("hash", 256)
    normalized.setdefault("multipv", 1)
    normalized.setdefault("use_tablebase", False)
    normalized.setdefault("syzygy_path", "")
    normalized.setdefault("tablebase_max_pieces", 7)
    return normalized
