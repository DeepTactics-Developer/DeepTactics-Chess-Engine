"""Small JSON-backed translation catalog for the Qt desktop interface."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtWidgets import QAbstractButton, QLabel, QWidget


class TranslationCatalog:
    def __init__(self, assets_root: Path, language: str = "English") -> None:
        self.assets_root = Path(assets_root)
        self.language = language
        self.messages: dict[str, str] = {}
        self.load(language)

    def load(self, language: str) -> None:
        self.language = language
        code = {"English": "en", "Korean": "ko"}.get(language, "en")
        path = self.assets_root / "i18n" / f"{code}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        self.messages = {str(key): str(value) for key, value in payload.items()} if isinstance(payload, dict) else {}

    def text(self, key: str, fallback: str | None = None) -> str:
        return self.messages.get(key, fallback or key)

    def apply_widgets(self, root: QWidget) -> None:
        """Translate stable visible labels without changing internal option values."""
        replacements = {
            key.removeprefix("visible."): value
            for key, value in self.messages.items()
            if key.startswith("visible.")
        }
        for widget in [root, *root.findChildren(QWidget)]:
            if not isinstance(widget, (QLabel, QAbstractButton)):
                continue
            text = widget.text()
            for source, translated in replacements.items():
                text = text.replace(source, translated)
            widget.setText(text)
