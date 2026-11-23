"""Lightweight artifact indexing for the Qt Database page.

The Database page inventories engine-development files without loading large
training or match datasets at application startup. Expensive jobs remain CLI
operations until an asynchronous task runner is connected.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


SUPPORTED_SUFFIXES = {".csv", ".json", ".jsonl", ".pgn", ".md", ".bin", ".epd"}
IGNORED_DIRECTORIES = {"qt_mockup", "__pycache__"}


@dataclass(slots=True)
class DatasetRecord:
    path: Path
    root: Path
    data_type: str
    size_bytes: int
    updated_at: float
    status: str = "Unvalidated"
    tags: list[str] = field(default_factory=list)
    records: str = "--"
    notes: str = ""
    favorite: bool = False
    archived: bool = False

    @property
    def record_id(self) -> str:
        return str(self.path.resolve()).lower()

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def relative_path(self) -> str:
        try:
            return str(self.path.relative_to(self.root))
        except ValueError:
            return str(self.path)

    @property
    def size_text(self) -> str:
        amount = float(self.size_bytes)
        for unit in ("B", "KB", "MB", "GB"):
            if amount < 1024 or unit == "GB":
                return f"{amount:.1f} {unit}" if unit != "B" else f"{int(amount)} B"
            amount /= 1024
        return f"{amount:.1f} GB"

    @property
    def updated_text(self) -> str:
        value = datetime.fromtimestamp(self.updated_at)
        today = datetime.now().date()
        if value.date() == today:
            return value.strftime("Today %H:%M")
        return value.strftime("%Y-%m-%d %H:%M")

    @property
    def description(self) -> str:
        return {
            "Self-Play": "Self-play or engine match output used for evaluation and training evidence.",
            "Annotated": "Annotated analysis output.",
            "Loss Audit": "Loss-audit evidence, confirmation, or pattern report.",
            "Regression": "Regression positions or native search verification result.",
            "Training": "NNUE or training dataset/model artifact.",
            "PGN": "PGN source intended for engine analysis or development workflows.",
            "Book": "Opening book or line source.",
            "Weights": "Evaluation weight candidate or active weight artifact.",
            "Report": "Engine comparison or diagnostic report.",
        }.get(self.data_type, "Engine development artifact.")


class DatabaseStore:
    """Scan real development artifacts and persist only user annotations."""

    def __init__(self, roots: list[Path], user_state: dict[str, Any] | None = None) -> None:
        self.roots = _unique_existing_roots(roots)
        state = user_state or {}
        self.favorites = set(state.get("database_favorites", []))
        self.archived = set(state.get("database_archived", []))
        self.notes = dict(state.get("database_notes", {}))
        self.statuses = dict(state.get("database_statuses", {}))
        self.record_counts = dict(state.get("database_record_counts", {}))
        self.records: list[DatasetRecord] = []
        self.scan()

    def scan(self) -> list[DatasetRecord]:
        records: list[DatasetRecord] = []
        for root in self.roots:
            for path in root.rglob("*"):
                if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                    continue
                if any(part in IGNORED_DIRECTORIES for part in path.parts):
                    continue
                try:
                    stat = path.stat()
                except OSError:
                    continue
                record_id = str(path.resolve()).lower()
                record = DatasetRecord(
                    path=path,
                    root=root,
                    data_type=_classify(path),
                    size_bytes=stat.st_size,
                    updated_at=stat.st_mtime,
                    tags=_tags(path),
                    status=self.statuses.get(record_id, "Unvalidated"),
                    records=self.record_counts.get(record_id, "--"),
                    favorite=record_id in self.favorites,
                    archived=record_id in self.archived,
                    notes=self.notes.get(record_id, ""),
                )
                records.append(record)
        self.records = sorted(records, key=lambda item: item.updated_at, reverse=True)
        return self.records

    def user_state(self) -> dict[str, Any]:
        return {
            "database_favorites": sorted(self.favorites),
            "database_archived": sorted(self.archived),
            "database_notes": dict(self.notes),
            "database_statuses": dict(self.statuses),
            "database_record_counts": dict(self.record_counts),
        }

    def toggle_favorite(self, record: DatasetRecord) -> None:
        record.favorite = not record.favorite
        if record.favorite:
            self.favorites.add(record.record_id)
        else:
            self.favorites.discard(record.record_id)

    def archive(self, record: DatasetRecord) -> None:
        record.archived = True
        self.archived.add(record.record_id)
        record.status = "Archived"
        self.statuses[record.record_id] = record.status

    def set_notes(self, record: DatasetRecord, notes: str) -> None:
        record.notes = notes
        self.notes[record.record_id] = notes

    def update_validation(self, record: DatasetRecord, status: str) -> None:
        record.status = status
        self.statuses[record.record_id] = status
        self.record_counts[record.record_id] = record.records

    def preview(self, record: DatasetRecord) -> str:
        suffix = record.path.suffix.lower()
        try:
            if suffix == ".csv":
                return _csv_preview(record.path)
            if suffix in {".json", ".jsonl"}:
                return _json_preview(record.path, suffix)
            if suffix in {".pgn", ".md", ".epd"}:
                return _text_preview(record.path)
            return "Binary training/model file.\n\nPreview is unavailable; use metadata and validation only."
        except (OSError, UnicodeError, json.JSONDecodeError, csv.Error) as error:
            return f"Preview unavailable: {error}"

    def validate(self, record: DatasetRecord) -> tuple[str, str]:
        """Perform a deliberately bounded readability/schema check."""
        suffix = record.path.suffix.lower()
        try:
            if suffix == ".csv":
                with record.path.open("r", encoding="utf-8", newline="") as stream:
                    reader = csv.reader(stream)
                    header = next(reader, [])
                    if not header:
                        return "Warning", "CSV has no header row."
                    sample_count = sum(1 for _row in _take(reader, 1000))
                    record.records = f">={sample_count:,}" if record.size_bytes > 10_000_000 else str(sample_count)
                    return "Ready", f"CSV header readable ({len(header)} columns); bounded sample checked."
            if suffix == ".json":
                payload = json.loads(record.path.read_text(encoding="utf-8"))
                if isinstance(payload, list):
                    record.records = str(len(payload))
                elif isinstance(payload, dict):
                    record.records = str(len(payload))
                return "Ready", "JSON parses successfully."
            if suffix == ".jsonl":
                with record.path.open("r", encoding="utf-8") as stream:
                    checked = 0
                    for line in _take(stream, 100):
                        if line.strip():
                            json.loads(line)
                            checked += 1
                record.records = f">={checked}" if checked else "0"
                return "Ready", "JSONL sample parses successfully."
            if suffix == ".pgn":
                head = _text_preview(record.path, max_chars=4000)
                if "[Event " not in head and "[Result " not in head:
                    return "Warning", "No PGN headers detected in the file preview."
                return "Ready", "PGN header detected; full parsing is deferred."
            if suffix in {".md", ".epd"}:
                _text_preview(record.path, max_chars=300)
                return "Ready", "Text artifact is readable."
            if suffix == ".bin":
                return "Ready", "Binary artifact exists; internal schema validation is not connected."
        except (OSError, UnicodeError, json.JSONDecodeError, csv.Error) as error:
            return "Warning", str(error)
        return "Unvalidated", "No validator is registered for this file type."


def _classify(path: Path) -> str:
    value = str(path).lower().replace("-", "_")
    name = path.name.lower().replace("-", "_")
    if "annotated" in value:
        return "Annotated"
    if "loss_audit" in value or "loss_confirmation" in value or "loss_pattern" in value:
        return "Loss Audit"
    if "regression" in value or "benchmark" in value:
        return "Regression"
    if "self_play" in value or "selfplay" in value:
        return "Self-Play"
    if "nnue" in value or "dataset" in name or "shard" in value:
        return "Training"
    if "weight" in name:
        return "Weights"
    if path.suffix.lower() == ".pgn":
        return "PGN"
    if "opening" in value or "book" in name:
        return "Book"
    return "Report"


def _tags(path: Path) -> list[str]:
    text = str(path).lower()
    tags = [_classify(path).lower().replace(" ", "-")]
    for tag in ("annotated", "native", "nnue", "paired", "depth8", "depth12", "depth16"):
        if tag.replace("-", "_") in text.replace("-", "_"):
            tags.append(tag)
    return list(dict.fromkeys(tags))


def _unique_existing_roots(roots: list[Path]) -> list[Path]:
    output: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve()
        if resolved.exists() and resolved not in seen:
            output.append(resolved)
            seen.add(resolved)
    return output


def _take(iterable, count: int):
    for index, item in enumerate(iterable):
        if index >= count:
            break
        yield item


def _csv_preview(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(_take(csv.reader(stream), 7))
    if not rows:
        return "Empty CSV file."
    widths = [max(len(str(row[index])) if index < len(row) else 0 for row in rows) for index in range(len(rows[0]))]
    lines = []
    for row in rows:
        lines.append("  ".join(str(value)[:28].ljust(min(widths[index], 28)) for index, value in enumerate(row)))
    return "CSV PREVIEW (bounded first rows)\n\n" + "\n".join(lines)


def _json_preview(path: Path, suffix: str) -> str:
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as stream:
            values = [json.loads(line) for line in _take(stream, 3) if line.strip()]
        return "JSONL PREVIEW (first records)\n\n" + json.dumps(values, indent=2, ensure_ascii=False)[:1800]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        payload = payload[:3]
    return "JSON PREVIEW\n\n" + json.dumps(payload, indent=2, ensure_ascii=False)[:1800]


def _text_preview(path: Path, max_chars: int = 1800) -> str:
    with path.open("r", encoding="utf-8") as stream:
        return "TEXT PREVIEW\n\n" + stream.read(max_chars)
