"""Board square representation.

The engine stores squares as zero-based coordinates:

- file: 0 to 7, where 0 is file 'a' and 7 is file 'h'
- rank: 0 to 7, where 0 is rank '1' and 7 is rank '8'

This keeps board logic independent from any UI. Any UI layer can
convert these coordinates into text or pixels when they render the board.
"""

from __future__ import annotations

from dataclasses import dataclass


FILES = "abcdefgh"
RANKS = "12345678"
BOARD_SIZE = 8


@dataclass(frozen=True, slots=True)
class Square:
    """A single square on the chess board."""

    file: int
    rank: int

    def __post_init__(self) -> None:
        if not 0 <= self.file < BOARD_SIZE:
            raise ValueError(f"file must be between 0 and 7: {self.file}")
        if not 0 <= self.rank < BOARD_SIZE:
            raise ValueError(f"rank must be between 0 and 7: {self.rank}")

    @classmethod
    def from_algebraic(cls, value: str) -> "Square":
        """Create a square from notation such as 'e4'."""
        if len(value) != 2:
            raise ValueError(f"square must have length 2: {value!r}")

        file_char = value[0].lower()
        rank_char = value[1]
        if file_char not in FILES or rank_char not in RANKS:
            raise ValueError(f"invalid square notation: {value!r}")

        return cls(file=FILES.index(file_char), rank=RANKS.index(rank_char))

    def to_algebraic(self) -> str:
        """Return notation such as 'e4'."""
        return f"{FILES[self.file]}{RANKS[self.rank]}"

    def offset(self, file_delta: int, rank_delta: int) -> "Square | None":
        """Return a nearby square, or None if the offset leaves the board."""
        file = self.file + file_delta
        rank = self.rank + rank_delta
        if 0 <= file < BOARD_SIZE and 0 <= rank < BOARD_SIZE:
            return Square(file, rank)
        return None

    def __str__(self) -> str:
        return self.to_algebraic()

