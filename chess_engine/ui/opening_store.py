"""Opening-library records and persisted Qt user state.

The Opening page reads the existing ECO TSV as a reference library. Statistics
or model games are deliberately not invented when the source does not contain
them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chess_engine.chess_core.fen import STANDARD_STARTING_FEN, board_to_fen
from chess_engine.chess_core.game import Game
from chess_engine.chess_core.pgn import move_from_san, pgn_to_san_tokens


@dataclass(slots=True)
class OpeningRecord:
    record_id: str
    eco: str
    name: str
    san_moves: tuple[str, ...]
    side: str
    source: str
    _uci_moves: tuple[str, ...] | None = None
    _positions: tuple[str, ...] | None = None

    @property
    def display_name(self) -> str:
        return self.name

    @property
    def summary(self) -> str:
        return "Opening book line"

    def materialize(self) -> bool:
        if self._uci_moves is not None:
            return True
        game = Game.new()
        moves: list[str] = []
        positions = [STANDARD_STARTING_FEN]
        try:
            for san in self.san_moves:
                move = move_from_san(game.board, san)
                game.apply_move(move)
                moves.append(move.to_uci())
                positions.append(board_to_fen(game.board))
        except ValueError:
            return False
        self._uci_moves = tuple(moves)
        self._positions = tuple(positions)
        return True

    @property
    def uci_moves(self) -> tuple[str, ...]:
        return self._uci_moves or ()

    @property
    def positions(self) -> tuple[str, ...]:
        return self._positions or (STANDARD_STARTING_FEN,)


def load_opening_records(path: Path) -> list[OpeningRecord]:
    """Load all available ECO lines without doing expensive board replay up-front."""
    records: list[OpeningRecord] = []
    if not path.exists():
        return records
    for row_index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.lower().startswith("eco\t"):
            continue
        fields = stripped.split("\t")
        if len(fields) < 3:
            continue
        eco, name, move_text = (field.strip() for field in fields[:3])
        san_moves = tuple(pgn_to_san_tokens(move_text) or move_text.split())
        if not san_moves:
            continue
        side = "Black" if any(word in name for word in ("Defense", "Defence", "Countergambit")) else "White"
        records.append(
            OpeningRecord(
                record_id=f"{eco}:{row_index}",
                eco=eco,
                name=name,
                san_moves=san_moves,
                side=side,
                source=str(path),
            )
        )
    return records


def opening_user_state(settings: dict[str, Any]) -> dict[str, Any]:
    """Return mutable Opening-specific settings in a backwards-compatible form."""
    return {
        "favorites": set(settings.get("opening_favorites", [])),
        "repertoire": set(settings.get("opening_repertoire", [])),
        "notes": {
            str(name): list(notes)
            for name, notes in settings.get("opening_notes", {}).items()
            if isinstance(notes, list)
        },
        "recent": list(settings.get("opening_recent", [])),
    }


def write_opening_user_state(settings: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    updated = dict(settings)
    updated["opening_favorites"] = sorted(state["favorites"])
    updated["opening_repertoire"] = sorted(state["repertoire"])
    updated["opening_notes"] = state["notes"]
    updated["opening_recent"] = list(state["recent"])[:12]
    return updated
