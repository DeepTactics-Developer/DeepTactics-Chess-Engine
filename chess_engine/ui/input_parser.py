"""Console input parsing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from chess_engine.chess_core.move import Move


class ConsoleCommand(Enum):
    """Commands the console UI understands."""

    HELP = "help"
    LEGAL = "legal"
    FEN = "fen"
    PGN = "pgn"
    QUIT = "quit"


@dataclass(frozen=True, slots=True)
class ParsedInput:
    """Result of parsing one console input line."""

    move: Move | None = None
    command: ConsoleCommand | None = None

    @property
    def is_move(self) -> bool:
        """Return whether this input represents a chess move."""
        return self.move is not None

    @property
    def is_command(self) -> bool:
        """Return whether this input represents a console command."""
        return self.command is not None


COMMAND_ALIASES = {
    "help": ConsoleCommand.HELP,
    "?": ConsoleCommand.HELP,
    "legal": ConsoleCommand.LEGAL,
    "moves": ConsoleCommand.LEGAL,
    "fen": ConsoleCommand.FEN,
    "pgn": ConsoleCommand.PGN,
    "quit": ConsoleCommand.QUIT,
    "exit": ConsoleCommand.QUIT,
    "q": ConsoleCommand.QUIT,
}


def parse_console_input(text: str) -> ParsedInput:
    """Parse a console line into either a command or a Move.

    Moves use UCI-style notation such as ``e2e4`` or ``a7a8q``.
    """
    normalized = text.strip().lower()
    if not normalized:
        raise ValueError("input is empty")

    command = COMMAND_ALIASES.get(normalized)
    if command is not None:
        return ParsedInput(command=command)

    return ParsedInput(move=Move.from_uci(normalized))
