"""Opening detection helpers.

The detector starts with a tiny built-in book and can be expanded by adding
TSV rows to ``assets/openings/openings.tsv``:

``eco<TAB>name<TAB>san moves separated by spaces``
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from chess_engine.chess_core.pgn import pgn_to_san_tokens


BUILTIN_OPENINGS = {
    "e4": "B00 King's Pawn Opening",
    "e4 e5": "C20 Open Game",
    "e4 e5 Nf3": "C40 King's Knight Opening",
    "e4 e5 Nf3 Nc6": "C44 Open Game: King's Knight",
    "e4 e5 Nf3 Nc6 Bb5": "C60 Ruy Lopez",
    "e4 e5 Nf3 Nc6 Bc4": "C50 Italian Game",
    "e4 c5": "B20 Sicilian Defense",
    "e4 e6": "C00 French Defense",
    "e4 c6": "B10 Caro-Kann Defense",
    "d4 d5": "D00 Closed Game",
    "d4 d5 c4": "D06 Queen's Gambit",
    "d4 Nf6 c4 g6": "E60 King's Indian Defense",
    "d4 Nf6 c4 e6": "E00 Indian Game",
    "c4": "A10 English Opening",
    "Nf3": "A04 Reti Opening",
    "Nc3": "A00 Van Geet Opening",
    "Nc3 e5": "A00 Van Geet Opening: Reversed Nimzowitsch",
    "e4 e5 Nc3": "C25 Vienna Game",
}


def opening_book(data_path: str | Path | None = None) -> dict[str, str]:
    """Return built-in openings plus optional external rows."""
    book = dict(BUILTIN_OPENINGS)
    if data_path is not None:
        book.update(_cached_opening_book(_cache_key_for_path(data_path)))
    return book


def detect_opening(san_history: list[str], data_path: str | Path | None = None) -> str:
    """Return the best matching opening name for the current SAN history."""
    moves = _normalize_history(san_history)
    if not moves:
        return "Opening: unknown"

    book = opening_book(data_path)
    best_name = "Opening: unknown"
    best_length = 0
    for sequence, name in book.items():
        parts = sequence.split()
        if len(parts) > len(moves) or len(parts) < best_length:
            continue
        if moves[: len(parts)] == parts:
            best_name = name
            best_length = len(parts)

    if best_length == len(moves):
        return best_name

    transposed_name, transposed_length = _detect_opening_by_key(moves, data_path)
    if transposed_length > best_length:
        return transposed_name
    return best_name


def next_opening_moves(
    san_history: list[str],
    data_path: str | Path | None = None,
) -> list[str]:
    """Return book SAN continuations that match the current history."""
    moves = _normalize_history(san_history)
    continuations: list[str] = []
    book = load_opening_book(data_path) if data_path is not None else {}
    if not book:
        book = dict(BUILTIN_OPENINGS)
    for sequence in book:
        parts = sequence.split()
        if len(parts) <= len(moves):
            continue
        if parts[: len(moves)] == moves:
            continuations.append(parts[len(moves)])
    return sorted(set(continuations))


def load_opening_book(path: str | Path) -> dict[str, str]:
    """Load a TSV opening book.

    Supported row formats:
    - ``name<TAB>moves``
    - ``eco<TAB>name<TAB>moves``
    """
    input_path = Path(path)
    if not input_path.exists():
        return {}

    book: dict[str, str] = {}
    for line in input_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        columns = stripped.split("\t")
        if len(columns) >= 3:
            eco = columns[0].strip()
            name = columns[1].strip()
            sequence = columns[2].strip()
            display_name = f"{eco} {name}".strip()
        elif len(columns) == 2:
            display_name = columns[0].strip()
            sequence = columns[1].strip()
        else:
            continue
        book[_normalize_sequence(sequence)] = display_name
    return book


@lru_cache(maxsize=8)
def _cached_opening_book(cache_key: tuple[str, float]) -> dict[str, str]:
    path, _ = cache_key
    if not path:
        return {}
    return load_opening_book(path)


def generate_opening_key(move_text: str | list[str] | tuple[str, ...]) -> str:
    """Return a fast transposition key from SAN moves split by side.

    This is intentionally lighter than board reconstruction. It groups White's
    and Black's moves separately, sorts each group, and joins them as
    ``white_moves|black_moves`` so simple opening transpositions share a key.
    """
    if isinstance(move_text, str):
        moves = pgn_to_san_tokens(move_text)
        if not moves:
            moves = move_text.split()
    else:
        moves = list(move_text)
    normalized = [_normalize_san(move) for move in moves if _normalize_san(move)]
    white_moves = sorted(normalized[0::2])
    black_moves = sorted(normalized[1::2])
    return f"{','.join(white_moves)}|{','.join(black_moves)}"


def _detect_opening_by_key(moves: list[str], data_path: str | Path | None) -> tuple[str, int]:
    index = _transposition_opening_index(_cache_key_for_path(data_path))
    for length in range(len(moves), 0, -1):
        match = index.get(generate_opening_key(moves[:length]))
        if match is not None:
            return match
    return "Opening: unknown", 0


def _cache_key_for_path(data_path: str | Path | None) -> tuple[str, float]:
    if data_path is None:
        return "", 0.0
    path = Path(data_path)
    if not path.exists():
        return str(path), 0.0
    return str(path), path.stat().st_mtime


@lru_cache(maxsize=8)
def _transposition_opening_index(cache_key: tuple[str, float]) -> dict[str, tuple[str, int]]:
    path, _ = cache_key
    book = opening_book(path or None)
    index: dict[str, tuple[str, int]] = {}
    for sequence, name in book.items():
        moves = tuple(sequence.split())
        key = generate_opening_key(moves)
        previous = index.get(key)
        if previous is None:
            index[key] = (name, len(moves))
    return index


def convert_opening_source_to_tsv(input_path: str | Path, output_path: str | Path) -> int:
    """Convert common ECO TSV rows into this project's opening TSV format.

    Accepted input formats:
    - ``name<TAB>SAN moves``
    - ``eco<TAB>name<TAB>PGN/SAN moves``
    """
    return convert_opening_sources_to_tsv([input_path], output_path)


def convert_opening_sources_to_tsv(input_paths: list[str | Path], output_path: str | Path) -> int:
    """Convert one or more opening source TSV files into local format."""
    rows: list[tuple[str, str, str]] = []
    for input_path in input_paths:
        rows.extend(load_opening_source_rows(input_path))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{eco}\t{name}\t{sequence}" if eco else f"{name}\t{sequence}"
        for eco, name, sequence in rows
        if sequence
    ]
    output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def load_opening_source_rows(path: str | Path) -> list[tuple[str, str, str]]:
    """Load flexible source rows as ``(eco, name, normalized_sequence)`` tuples."""
    input_path = Path(path)
    rows: list[tuple[str, str, str]] = []
    for line in input_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        columns = stripped.split("\t")
        if len(columns) >= 3:
            eco = columns[0].strip()
            name = columns[1].strip()
            move_text = columns[2].strip()
        elif len(columns) == 2:
            eco = ""
            name = columns[0].strip()
            move_text = columns[1].strip()
        else:
            continue
        tokens = pgn_to_san_tokens(move_text)
        sequence = _normalize_sequence(" ".join(tokens) if tokens else move_text)
        rows.append((eco, name, sequence))
    return rows


def _normalize_history(san_history: list[str]) -> list[str]:
    return [_normalize_san(move) for move in san_history]


def _normalize_sequence(sequence: str) -> str:
    return " ".join(_normalize_san(move) for move in sequence.split())


def _normalize_san(san: str) -> str:
    return san.strip().rstrip("+#!?")
