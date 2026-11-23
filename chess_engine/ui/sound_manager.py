"""Small Qt sound facade used by Play, AI Match, and analysis completion."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from PySide6.QtCore import QObject, QUrl
from PySide6.QtWidgets import QApplication

try:
    from PySide6.QtMultimedia import QSoundEffect
except ImportError:  # pragma: no cover - optional Qt multimedia component
    QSoundEffect = None


class SoundManager(QObject):
    """Play configured GUI sounds with a system-beep fallback."""

    def __init__(self, assets_root: Path, settings: Mapping[str, object], parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.assets_root = Path(assets_root)
        self.settings: dict[str, object] = {}
        self.effects: dict[str, object] = {}
        self.configure(settings)

    def configure(self, settings: Mapping[str, object]) -> None:
        self.settings = dict(settings)
        self.effects.clear()
        if QSoundEffect is None:
            return
        for event in ("move", "capture", "check", "game_end", "engine_ready"):
            path = self.assets_root / "sounds" / f"{event}.wav"
            if not path.exists():
                continue
            effect = QSoundEffect(self)
            effect.setSource(QUrl.fromLocalFile(str(path)))
            effect.setVolume(max(0.0, min(1.0, int(self.settings.get("sound_volume", 70)) / 100.0)))
            self.effects[event] = effect

    def play(self, event: str) -> bool:
        if bool(self.settings.get("sound_mute_all", True)):
            return False
        if not bool(self.settings.get(f"sound_{event}", False)):
            return False
        effect = self.effects.get(event)
        if effect is not None:
            effect.play()
        else:
            QApplication.beep()
        return True

