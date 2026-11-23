"""Qt-facing game document state.

This object owns a game/timeline representation for GUI pages. Engines receive
positions later through UCI; they do not own or mutate this state.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QObject, Signal

from chess_engine.chess_core.fen import STANDARD_STARTING_FEN, board_from_fen, board_to_fen
from chess_engine.chess_core.game import Game
from chess_engine.chess_core.move import Move


@dataclass(frozen=True, slots=True)
class GameSnapshot:
    fen: str
    move_list: tuple[str, ...]
    current_ply: int
    pgn: str
    status: str
    variation_moves: tuple[str, ...] = ()
    variation_base_ply: int | None = None
    active_line: tuple[str, ...] = ()


class GameDocument(QObject):
    """Keep board state and move history independent from engine analysis."""

    changed = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self._initial_fen = STANDARD_STARTING_FEN
        self._mainline_moves: list[str] = []
        self._variation_moves: list[str] = []
        self._variation_base_ply: int | None = None
        self._viewing_variation = False
        self.game = Game.new()
        self.current_ply = 0

    def new_game(self) -> None:
        self._initial_fen = STANDARD_STARTING_FEN
        self._mainline_moves = []
        self._variation_moves = []
        self._variation_base_ply = None
        self._viewing_variation = False
        self.game = Game.new()
        self.current_ply = 0
        self._emit()

    def load_fen(self, fen: str) -> None:
        self._initial_fen = fen
        self._mainline_moves = []
        self._variation_moves = []
        self._variation_base_ply = None
        self._viewing_variation = False
        self.game = Game(board=board_from_fen(fen))
        self.current_ply = 0
        self._emit()

    def load_uci_line(self, moves: list[str] | tuple[str, ...], initial_fen: str = STANDARD_STARTING_FEN) -> None:
        self._initial_fen = initial_fen
        self._mainline_moves = list(moves)
        self._variation_moves = []
        self._variation_base_ply = None
        self._viewing_variation = False
        self.seek(len(self._mainline_moves))

    def apply_uci_move(self, move_text: str) -> None:
        if not self._mainline_moves and self._variation_base_ply is None:
            self.game.apply_move(Move.from_uci(move_text))
            self._mainline_moves.append(move_text)
            self.current_ply += 1
            self._emit()
            return
        if self._variation_base_ply is None or not self._viewing_variation:
            self._variation_base_ply = self.current_ply
            self._variation_moves = []
        elif self.current_ply < self._variation_base_ply:
            self._variation_base_ply = self.current_ply
            self._variation_moves = []
        else:
            variation_index = self.current_ply - self._variation_base_ply
            self._variation_moves = self._variation_moves[:variation_index]
        self.game.apply_move(Move.from_uci(move_text))
        self._variation_moves.append(move_text)
        self._viewing_variation = True
        self.current_ply += 1
        self._emit()

    def append_mainline_move(self, move_text: str) -> None:
        """Apply a live game move to the official line rather than an analysis branch."""
        if self.current_ply < len(self._mainline_moves):
            self._mainline_moves = self._mainline_moves[: self.current_ply]
        self._variation_moves = []
        self._variation_base_ply = None
        self._viewing_variation = False
        self.game.apply_move(Move.from_uci(move_text))
        self._mainline_moves.append(move_text)
        self.current_ply += 1
        self._emit()

    def undo(self) -> None:
        if self._variation_moves:
            self._variation_moves.pop()
            self.seek((self._variation_base_ply or 0) + len(self._variation_moves))
        elif self._mainline_moves and self.current_ply == len(self._mainline_moves):
            self._mainline_moves.pop()
            self.seek_mainline(len(self._mainline_moves))

    def seek(self, ply: int) -> None:
        self._viewing_variation = self._variation_base_ply is not None
        sequence = self._active_moves()
        self.current_ply = max(0, min(len(sequence), ply))
        self.game = Game(board=board_from_fen(self._initial_fen))
        for move_text in sequence[: self.current_ply]:
            self.game.apply_move(Move.from_uci(move_text))
        self._emit()

    def seek_mainline(self, ply: int) -> None:
        self._viewing_variation = False
        self.current_ply = max(0, min(len(self._mainline_moves), ply))
        self.game = Game(board=board_from_fen(self._initial_fen))
        for move_text in self._mainline_moves[: self.current_ply]:
            self.game.apply_move(Move.from_uci(move_text))
        self._emit()

    def seek_variation_end(self) -> None:
        if self._variation_base_ply is None:
            return
        self._viewing_variation = True
        self.seek(self._variation_base_ply + len(self._variation_moves))

    @property
    def viewing_variation(self) -> bool:
        return self._viewing_variation

    @property
    def total_plies(self) -> int:
        return len(self._mainline_moves)

    @property
    def mainline_uci_moves(self) -> tuple[str, ...]:
        return tuple(self._mainline_moves)

    @property
    def initial_fen(self) -> str:
        return self._initial_fen

    def mainline_positions(self) -> tuple[str, ...]:
        """Return immutable FEN snapshots before move 1 through after the last move."""
        game = Game(board=board_from_fen(self._initial_fen))
        positions = [board_to_fen(game.board)]
        for move_text in self._mainline_moves:
            game.apply_move(Move.from_uci(move_text))
            positions.append(board_to_fen(game.board))
        return tuple(positions)

    @property
    def last_move_uci(self) -> str | None:
        sequence = self._active_moves()
        return sequence[self.current_ply - 1] if self.current_ply else None

    def snapshot(self) -> GameSnapshot:
        return GameSnapshot(
            fen=board_to_fen(self.game.board),
            move_list=self._full_san_line(),
            current_ply=self.current_ply,
            pgn=self.game.to_pgn(),
            status=self.game.status().value,
            variation_moves=self._variation_san(),
            variation_base_ply=self._variation_base_ply,
            active_line=self._active_san_line(),
        )

    @property
    def fen(self) -> str:
        return board_to_fen(self.game.board)

    def _emit(self) -> None:
        self.changed.emit(self.snapshot())

    def _full_san_line(self) -> tuple[str, ...]:
        game = Game(board=board_from_fen(self._initial_fen))
        for move_text in self._mainline_moves:
            game.apply_move(Move.from_uci(move_text))
        return tuple(game.san_history)

    def _active_moves(self) -> list[str]:
        if self._variation_base_ply is None or not self._viewing_variation:
            return list(self._mainline_moves)
        return self._mainline_moves[: self._variation_base_ply] + self._variation_moves

    def _variation_san(self) -> tuple[str, ...]:
        if self._variation_base_ply is None:
            return ()
        game = Game(board=board_from_fen(self._initial_fen))
        for move_text in self._mainline_moves[: self._variation_base_ply]:
            game.apply_move(Move.from_uci(move_text))
        san: list[str] = []
        for move_text in self._variation_moves:
            game.apply_move(Move.from_uci(move_text))
            san.append(game.san_history[-1])
        return tuple(san)

    def _active_san_line(self) -> tuple[str, ...]:
        game = Game(board=board_from_fen(self._initial_fen))
        for move_text in self._active_moves()[: self.current_ply]:
            game.apply_move(Move.from_uci(move_text))
        return tuple(game.san_history)


def starting_snapshot() -> GameSnapshot:
    document = GameDocument()
    assert document.fen == STANDARD_STARTING_FEN
    return document.snapshot()
