"""Persistent PGN library support for the Qt Games page."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from chess_engine.chess_core.fen import board_to_fen
from chess_engine.chess_core.pgn import game_from_pgn


HEADER_RE = re.compile(r'^\[(?P<name>[A-Za-z0-9_]+)\s+"(?P<value>.*)"\]\s*$', re.MULTILINE)


@dataclass(slots=True)
class GameRecord:
    """Metadata stored in the PGN library index."""

    id: str
    name: str
    white: str
    black: str
    result: str
    date: str
    event: str
    site: str = "?"
    round: str = "?"
    game_type: str = "Imported"
    opening: str = "Unknown"
    eco: str = ""
    tags: list[str] = field(default_factory=list)
    managed_path: str = ""
    source_path: str = ""
    source_type: str = "Imported"
    imported_at: str = ""
    last_opened_at: str = ""
    favorite: bool = False
    notes: str = ""
    move_count: int = 0
    ply_count: int = 0
    final_fen: str = ""
    analyzed: bool = False
    collections: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, values: dict) -> "GameRecord":
        known = {field.name for field in cls.__dataclass_fields__.values()}
        payload = {key: value for key, value in values.items() if key in known}
        return cls(**payload)

    def to_dict(self) -> dict:
        return asdict(self)


class GameLibraryStore:
    """Store user PGNs separately from engine-development datasets."""

    def __init__(self, library_root: str | Path) -> None:
        self.root = Path(library_root)
        self.games_dir = self.root / "library"
        self.index_path = self.root / "games_index.json"
        self.collections_path = self.root / "collections.json"
        self.records: list[GameRecord] = []
        self.collections: dict[str, str] = {}
        self.load()

    def load(self) -> list[GameRecord]:
        self.records = []
        if not self.index_path.exists():
            self._load_collections()
            return self.records
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return self.records
        if isinstance(payload, list):
            for values in payload:
                if isinstance(values, dict):
                    try:
                        self.records.append(GameRecord.from_dict(values))
                    except TypeError:
                        continue
        self._load_collections()
        return self.records

    def _load_collections(self) -> None:
        self.collections = {}
        if not self.collections_path.exists():
            return
        try:
            payload = json.loads(self.collections_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if isinstance(payload, dict):
            self.collections = {str(key): str(value) for key, value in payload.items()}

    def import_file(self, path: str | Path, *, source_type: str = "Imported") -> tuple[list[GameRecord], list[str]]:
        source = Path(path)
        try:
            text = source.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            return [], [f"{source.name}: {error}"]
        return self.import_text(text, source_path=str(source), source_type=source_type)

    def import_folder(self, folder: str | Path) -> tuple[list[GameRecord], list[str]]:
        added: list[GameRecord] = []
        warnings: list[str] = []
        for path in sorted(Path(folder).glob("*.pgn")):
            imported, errors = self.import_file(path)
            added.extend(imported)
            warnings.extend(errors)
        return added, warnings

    def import_text(
        self,
        text: str,
        *,
        source_path: str = "",
        source_type: str = "Imported",
        name_override: str | None = None,
    ) -> tuple[list[GameRecord], list[str]]:
        added: list[GameRecord] = []
        warnings: list[str] = []
        for index, pgn_text in enumerate(_split_pgn_games(text), start=1):
            try:
                game = game_from_pgn(pgn_text)
            except ValueError as error:
                warnings.append(f"Game {index}: {error}")
                continue
            headers = _headers(pgn_text)
            record_id = uuid.uuid4().hex
            now = datetime.now().isoformat(timespec="seconds")
            white = headers.get("White", "White")
            black = headers.get("Black", "Black")
            record = GameRecord(
                id=record_id,
                name=name_override or headers.get("Event") or f"{white} vs {black}",
                white=white,
                black=black,
                result=headers.get("Result", "*"),
                date=headers.get("Date", now[:10]).replace(".", "-"),
                event=headers.get("Event", "Imported Game"),
                site=headers.get("Site", "?"),
                round=headers.get("Round", "?"),
                game_type=headers.get("Type", _game_type(source_type)),
                opening=headers.get("Opening", "Unknown"),
                eco=headers.get("ECO", ""),
                tags=_record_tags(headers.get("Tags", ""), source_type),
                managed_path=str(self.games_dir / f"{record_id}.pgn"),
                source_path=source_path,
                source_type=source_type,
                imported_at=now,
                last_opened_at=now,
                move_count=(len(game.move_history) + 1) // 2,
                ply_count=len(game.move_history),
                final_fen=board_to_fen(game.board),
                analyzed=source_type == "Analyzed",
            )
            self.games_dir.mkdir(parents=True, exist_ok=True)
            Path(record.managed_path).write_text(pgn_text.strip() + "\n", encoding="utf-8")
            self.records.insert(0, record)
            added.append(record)
        if added:
            self.save()
        return added, warnings

    def pgn_text(self, record: GameRecord) -> str:
        return Path(record.managed_path).read_text(encoding="utf-8")

    def uci_moves(self, record: GameRecord) -> tuple[str, ...]:
        game = game_from_pgn(self.pgn_text(record))
        return tuple(move.to_uci() for move in game.move_history)

    def mark_opened(self, record: GameRecord) -> None:
        record.last_opened_at = datetime.now().isoformat(timespec="seconds")
        self.save()

    def toggle_favorite(self, record: GameRecord) -> None:
        record.favorite = not record.favorite
        self.save()

    def set_notes(self, record: GameRecord, notes: str) -> None:
        record.notes = notes
        self.save()

    def remove_from_library(self, record: GameRecord) -> None:
        """Remove only the index entry, preserving imported and managed PGN files."""
        self.records = [current for current in self.records if current.id != record.id]
        self.save()

    def create_collection(self, name: str) -> str:
        cleaned = name.strip()
        if not cleaned:
            raise ValueError("Collection name cannot be empty.")
        collection_id = uuid.uuid4().hex
        self.collections[collection_id] = cleaned
        self.save()
        return collection_id

    def add_to_collection(self, record: GameRecord, collection_id: str) -> None:
        if collection_id not in self.collections:
            raise ValueError("Collection does not exist.")
        if collection_id not in record.collections:
            record.collections.append(collection_id)
            self.save()

    def remove_from_collection(self, record: GameRecord, collection_id: str) -> None:
        record.collections = [current for current in record.collections if current != collection_id]
        self.save()

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.index_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps([record.to_dict() for record in self.records], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.index_path)
        collections_temporary = self.collections_path.with_suffix(".json.tmp")
        collections_temporary.write_text(json.dumps(self.collections, ensure_ascii=False, indent=2), encoding="utf-8")
        collections_temporary.replace(self.collections_path)


def _headers(text: str) -> dict[str, str]:
    return {match.group("name"): match.group("value") for match in HEADER_RE.finditer(text)}


def _split_pgn_games(text: str) -> list[str]:
    """Split conventional multi-game PGN files without parsing side variations."""
    cleaned = text.strip()
    if not cleaned:
        return []
    starts = [match.start() for match in re.finditer(r"(?m)^\[Event\s+\"", cleaned)]
    if len(starts) <= 1:
        return [cleaned]
    starts.append(len(cleaned))
    return [cleaned[starts[index] : starts[index + 1]].strip() for index in range(len(starts) - 1)]


def _game_type(source_type: str) -> str:
    return {
        "Saved": "Saved",
        "Human vs AI": "Human vs AI",
        "AI Match": "AI Match",
        "Analyzed": "Analyzed",
    }.get(source_type, "Imported")


def _record_tags(header_tags: str, source_type: str) -> list[str]:
    tags = [part.strip() for part in header_tags.split(",") if part.strip()]
    if source_type not in tags:
        tags.append(source_type)
    return tags
