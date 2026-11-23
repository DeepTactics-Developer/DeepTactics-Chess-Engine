"""Lazy bitmap asset cache for the Qt interface."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QPixmap

from chess_engine.ui.builtin_pieces import draw_piece

PIECE_NAMES = ("pawn", "knight", "bishop", "rook", "queen", "king")

class AssetManager:
    """Load piece images once and share pixmaps between board widgets."""

    def __init__(self, assets_root: Path) -> None:
        self.assets_root = assets_root
        self._piece_cache: dict[Path, dict[str, QPixmap]] = {}

    def piece_images(self, folder: str | Path | None = None) -> dict[str, QPixmap]:
        source = Path(folder) if folder is not None else self.assets_root / "pieces"
        if not source.is_absolute():
            source = self.assets_root.parent / source
        source = source.resolve()
        if source not in self._piece_cache:
            images = {asset.stem: QPixmap(str(asset)) for asset in source.glob("*.png")}
            if not images:
                images = {
                    f"{color}_{name}": draw_piece(name, color == "white")
                    for color in ("white", "black")
                    for name in PIECE_NAMES
                }
            self._piece_cache[source] = images
        return self._piece_cache[source]

    def clear_piece_cache(self) -> None:
        self._piece_cache.clear()
