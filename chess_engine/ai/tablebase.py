"""Optional Syzygy tablebase root probing.

This module is intentionally an adapter layer.  DeepTactics does not require
``python-chess`` at install time, so tablebase support must stay optional and
truthful: when the dependency or files are missing, callers receive no move
instead of a fake tablebase result.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TablebaseRootResult:
    move_uci: str
    wdl: int
    dtz: int | None = None

    @property
    def score_cp(self) -> int:
        """Map WDL to a display/search score for the side to move.

        Syzygy WDL is exact but not a centipawn evaluation.  The mapping is
        only used at the UCI root result boundary so GUIs get a clear direction
        while the raw WDL remains available in the info string.
        """
        if self.wdl >= 2:
            return 10_000
        if self.wdl == 1:
            return 5_000
        if self.wdl == -1:
            return -5_000
        if self.wdl <= -2:
            return -10_000
        return 0


def python_chess_available() -> bool:
    try:
        import chess  # noqa: F401
        import chess.syzygy  # noqa: F401
    except ImportError:
        return False
    return True


def probe_syzygy_root(fen: str, tablebase_path: str | Path, *, max_pieces: int = 7) -> TablebaseRootResult | None:
    """Return an exact root move from Syzygy tables, if available.

    The returned WDL is from the current root side's perspective:
    ``2`` means root side wins, ``0`` draw, and ``-2`` root side loses.
    """
    path = Path(tablebase_path)
    if not path.exists() or not path.is_dir():
        return None
    try:
        import chess
        import chess.syzygy
    except ImportError:
        return None

    board = chess.Board(fen)
    if len(board.piece_map()) > max_pieces:
        return None

    best_move: Any | None = None
    best_wdl = -3
    best_dtz: int | None = None
    try:
        with chess.syzygy.open_tablebase(str(path)) as tablebase:
            for move in board.legal_moves:
                board.push(move)
                try:
                    child_wdl = tablebase.probe_wdl(board)
                    root_wdl = -int(child_wdl)
                    child_dtz = tablebase.probe_dtz(board)
                    root_dtz = None if child_dtz is None else -int(child_dtz)
                except (KeyError, OSError, ValueError):
                    board.pop()
                    continue
                board.pop()
                if _is_better_tablebase_move(root_wdl, root_dtz, best_wdl, best_dtz):
                    best_move = move
                    best_wdl = root_wdl
                    best_dtz = root_dtz
    except (OSError, ValueError):
        return None

    if best_move is None:
        return None
    return TablebaseRootResult(move_uci=best_move.uci(), wdl=best_wdl, dtz=best_dtz)


def _is_better_tablebase_move(
    wdl: int,
    dtz: int | None,
    best_wdl: int,
    best_dtz: int | None,
) -> bool:
    if wdl != best_wdl:
        return wdl > best_wdl
    if dtz is None or best_dtz is None:
        return best_dtz is None and dtz is not None
    if wdl > 0:
        return abs(dtz) < abs(best_dtz)
    if wdl < 0:
        return abs(dtz) > abs(best_dtz)
    return abs(dtz) < abs(best_dtz)
