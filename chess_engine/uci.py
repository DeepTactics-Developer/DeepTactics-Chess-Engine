"""Minimal UCI protocol loop for external chess GUIs.

The UCI layer is intentionally thin: it translates text protocol commands
into ``Game`` and ``Move`` objects, then delegates move selection to the same
backend selector used by the desktop UI.
"""

from __future__ import annotations

import sys
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import json
from pathlib import Path
from time import perf_counter
from typing import TextIO

from chess_engine.ai.backend import SearchBackend, find_backend_move, find_backend_multipv
from chess_engine.ai.evaluator import (
    get_evaluation_weights,
    linear_nnue_status,
    load_accumulator_nnue_model,
    load_linear_nnue_model,
    native_evaluator_status,
    reset_linear_nnue_model,
    score_for_color,
)
from chess_engine.ai.repetition import combined_root_move_penalties
from chess_engine.ai.search import SearchResult
from chess_engine.ai.tablebase import probe_syzygy_root, python_chess_available
from chess_engine.chess_core.enums import Color, GameStatus
from chess_engine.chess_core.fen import board_from_fen, board_to_fen
from chess_engine.chess_core.game import Game
from chess_engine.chess_core.move import Move
from chess_engine.chess_core.pgn import move_from_san
from chess_engine.openings import next_opening_moves
from chess_engine.native.native_eval import (
    DEFAULT_FORCED_LINE_EXTENSION_BUDGET,
    DEFAULT_MAJOR_PIECE_SAFETY_ORDERING,
    native_available,
    native_build_status,
    request_native_search_stop,
    set_native_hash_mb,
    set_native_search_experiments,
)


UCI_INFINITE_TIME_LIMIT_SECONDS = 24 * 60 * 60
OPENING_BOOK_SAN_PRIORITY = (
    "e4", "d4", "Nf3", "c4", "Nc3", "g3", "b3", "f4",
    "e5", "c5", "e6", "c6", "d5", "Nf6", "Nc6", "g6",
)
ENGINE_MODE_ALIASES = {
    "pure engine": "Pure Engine",
    "pure": "Pure Engine",
    "practical play": "Practical Play",
    "practical": "Practical Play",
    "analysis": "Analysis",
    "analysis mode": "Analysis",
    "development audit": "Development Audit",
    "development/audit": "Development Audit",
    "development": "Development Audit",
    "audit": "Development Audit",
}
ENGINE_MODE_OPTIONS = ("Pure Engine", "Practical Play", "Analysis", "Development Audit")
TIME_MANAGER_ALIASES = {
    "legacy": "Legacy",
    "conservative": "Conservative",
    "balanced": "Balanced",
    "aggressive": "Aggressive",
}
TIME_MANAGER_OPTIONS = ("Legacy", "Conservative", "Balanced", "Aggressive")


def normalize_engine_mode(value: str | None) -> str:
    """Return a stable public engine mode name without changing search behavior."""
    if not value:
        return "Pure Engine"
    return ENGINE_MODE_ALIASES.get(value.strip().lower(), "Pure Engine")


def normalize_time_manager(value: str | None) -> str:
    if not value:
        return "Balanced"
    return TIME_MANAGER_ALIASES.get(value.strip().lower(), "Balanced")


def position_key(fen: str) -> str:
    """Normalize a FEN to placement, side, castling and en passant (no move counters)."""
    return " ".join(fen.split()[:4])


class UCIEngine:
    """Small UCI command processor."""

    def __init__(
        self,
        *,
        depth: int = 2,
        backend: SearchBackend = "native",
        time_limit: float | None = None,
        adaptive: bool = False,
        quiescence_depth: int = 4,
        weights_path: str | Path | None = None,
        nnue_model_path: str | Path | None = None,
        nnue_mode: str = "off",
        nnue_blend: float = 0.35,
        native_forced_extension_budget: int = DEFAULT_FORCED_LINE_EXTENSION_BUDGET,
        native_mate_danger_ordering: bool = False,
        native_defensive_mate_threat_ordering: bool = False,
        native_major_piece_safety_ordering: bool = DEFAULT_MAJOR_PIECE_SAFETY_ORDERING,
        native_internal_iterative_deepening: bool = True,
        native_clustered_tt: bool = False,
        native_aspiration_windows: bool = True,
        native_pawn_endgame_extension_budget: int = 0,
        native_check_extension_budget: int = 0,
        native_singular_extension: bool = False,
        native_futility_pruning: bool = True,
        native_lmr_formula: bool = False,
        native_razoring: bool = False,
        native_continuation_history: bool = False,
        native_probcut: bool = False,
        hash_mb: int = 16,
        engine_mode: str = "Pure Engine",
        use_opening_book: bool = False,
        opening_book_path: str | Path = "assets/openings/openings.tsv",
        opening_book_max_ply: int = 2,
        use_experience_book: bool = False,
        experience_book_path: str | Path = "artifacts/experience_book.json",
        use_tablebase: bool = False,
        syzygy_path: str | Path = "",
        tablebase_max_pieces: int = 7,
        time_manager: str = "Balanced",
    ) -> None:
        self.game = Game.new()
        self.depth = depth
        self.backend = backend
        self.time_limit = time_limit
        self.adaptive = adaptive
        self.quiescence_depth = max(0, quiescence_depth)
        self.move_overhead_ms = 50
        self.multipv = 1
        self.hash_mb = max(1, min(4096, hash_mb))
        self.engine_mode = normalize_engine_mode(engine_mode)
        self.use_opening_book = bool(use_opening_book)
        self.opening_book_path = str(opening_book_path)
        self.opening_book_max_ply = max(0, min(40, opening_book_max_ply))
        self.use_experience_book = bool(use_experience_book)
        self.experience_book_path = str(experience_book_path)
        self.use_tablebase = bool(use_tablebase)
        self.syzygy_path = str(syzygy_path)
        self.tablebase_max_pieces = max(2, min(7, int(tablebase_max_pieces)))
        self.time_manager = normalize_time_manager(time_manager)
        self.last_time_manager: dict[str, object] = {
            "mode": self.time_manager,
            "budget_ms": "off",
        }
        self.weights_path = str(weights_path) if weights_path else "off"
        self.nnue_model_path = str(nnue_model_path) if nnue_model_path else ""
        self.nnue_mode = nnue_mode
        self.nnue_blend = nnue_blend
        self.native_forced_extension_budget = max(0, min(2, native_forced_extension_budget))
        self.native_mate_danger_ordering = bool(native_mate_danger_ordering)
        self.native_defensive_mate_threat_ordering = bool(native_defensive_mate_threat_ordering)
        self.native_major_piece_safety_ordering = bool(native_major_piece_safety_ordering)
        self.native_internal_iterative_deepening = bool(native_internal_iterative_deepening)
        self.native_clustered_tt = bool(native_clustered_tt)
        self.native_aspiration_windows = bool(native_aspiration_windows)
        self.native_pawn_endgame_extension_budget = max(0, min(2, native_pawn_endgame_extension_budget))
        self.native_check_extension_budget = max(0, min(2, native_check_extension_budget))
        self.native_singular_extension = bool(native_singular_extension)
        self.native_futility_pruning = bool(native_futility_pruning)
        self.native_lmr_formula = bool(native_lmr_formula)
        self.native_razoring = bool(native_razoring)
        self.native_continuation_history = bool(native_continuation_history)
        self.native_probcut = bool(native_probcut)
        set_native_search_experiments(
            forced_line_extension_budget=self.native_forced_extension_budget,
            mate_danger_ordering=self.native_mate_danger_ordering,
            defensive_mate_threat_ordering=self.native_defensive_mate_threat_ordering,
            major_piece_safety_ordering=self.native_major_piece_safety_ordering,
            internal_iterative_deepening=self.native_internal_iterative_deepening,
            clustered_tt_replacement=self.native_clustered_tt,
            aspiration_windows=self.native_aspiration_windows,
            pawn_endgame_extension_budget=self.native_pawn_endgame_extension_budget,
            check_extension_budget=self.native_check_extension_budget,
            singular_extension=self.native_singular_extension,
            futility_pruning=self.native_futility_pruning,
            lmr_formula=self.native_lmr_formula,
            razoring=self.native_razoring,
            continuation_history=self.native_continuation_history,
            probcut=self.native_probcut,
        )
        set_native_hash_mb(self.hash_mb)
        self.last_root_penalties: dict[str, int] = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="uci-search")
        self._pending_search: Future[SearchResult] | None = None

    def handle_line(self, line: str) -> list[str]:
        """Handle one UCI input line and return response lines."""
        parts = line.strip().split()
        if not parts:
            return []

        command = parts[0]
        if command == "uci":
            return [
                "id name DeepTactics Engine",
                "id author DeepTactics Project",
                "option name Depth type spin default 2 min 1 max 64",
                "option name Backend type combo default native var native",
                "option name Move Overhead type spin default 50 min 0 max 5000",
                "option name Adaptive type check default false",
                "option name Quiescence Depth type spin default 4 min 0 max 16",
                "option name MultiPV type spin default 1 min 1 max 4",
                "option name Hash type spin default 16 min 1 max 4096",
                "option name Threads type spin default 1 min 1 max 1",
                "option name NNUE Mode type combo default off var off var blend var linear",
                "option name NNUE Model type string default <empty>",
                "option name NNUE Blend Percent type spin default 35 min 0 max 100",
                (
                    "option name Engine Mode type combo default Pure Engine "
                    "var Pure Engine var Practical Play var Analysis var Development Audit"
                ),
                "option name Use Opening Book type check default false",
                "option name Opening Book Path type string default assets/openings/openings.tsv",
                "option name Opening Book Max Ply type spin default 2 min 0 max 40",
                "option name Use Experience Book type check default false",
                "option name Experience Book Path type string default artifacts/experience_book.json",
                "option name Use Tablebase type check default false",
                "option name SyzygyPath type string default <empty>",
                "option name Tablebase Max Pieces type spin default 7 min 2 max 7",
                (
                    "option name Time Manager type combo default Balanced "
                    "var Legacy var Conservative var Balanced var Aggressive"
                ),
                "uciok",
            ]
        if command == "isready":
            return ["readyok"]
        if command == "ucinewgame":
            self.game = Game.new()
            return []
        if command == "position":
            self._set_position(parts[1:])
            return []
        if command == "go":
            return self._go(parts[1:])
        if command == "setoption":
            self._set_option(parts[1:])
            return []
        if command == "ponderhit":
            return self._finish_pending_search()
        if command == "stop":
            return self._finish_pending_search(request_stop=True)
        if command == "quit":
            request_native_search_stop()
            self._executor.shutdown(wait=False, cancel_futures=True)
            return ["__quit__"]
        return []

    def _set_position(self, parts: list[str]) -> None:
        if not parts:
            return

        moves_index = parts.index("moves") if "moves" in parts else None
        position_parts = parts[:moves_index] if moves_index is not None else parts
        move_parts = parts[moves_index + 1 :] if moves_index is not None else []

        if position_parts[:1] == ["startpos"]:
            self.game = Game.new()
        elif position_parts[:1] == ["fen"]:
            fen_fields = position_parts[1:7]
            if len(fen_fields) < 6:
                raise ValueError("UCI position fen requires six FEN fields")
            self.game = Game(board=board_from_fen(" ".join(fen_fields)))
        else:
            raise ValueError(f"unsupported UCI position: {' '.join(parts)}")

        for uci_text in move_parts:
            self.game.apply_move(Move.from_uci(uci_text))

    def _go(self, parts: list[str]) -> list[str]:
        raw_go_command = "go " + " ".join(parts)
        depth = self.depth
        time_limit = self.time_limit
        adaptive = self.adaptive
        infinite = "infinite" in parts or "ponder" in parts
        if "depth" in parts:
            depth_index = parts.index("depth")
            if depth_index + 1 < len(parts):
                depth = max(1, int(parts[depth_index + 1]))
                time_limit = None
                adaptive = False
        if "movetime" in parts:
            time_index = parts.index("movetime")
            if time_index + 1 < len(parts):
                time_limit = max(1, int(parts[time_index + 1])) / 1000
                adaptive = True
        elif any(token in parts for token in ("wtime", "btime")):
            time_limit = self._time_limit_from_clock(parts)
            adaptive = True

        search_depth = max(depth, 64) if infinite and self.backend == "native" else depth
        search_time_limit = time_limit
        if infinite and self.backend == "native" and search_time_limit is None:
            search_time_limit = UCI_INFINITE_TIME_LIMIT_SECONDS
        config = self._format_search_config(
            raw_go_command=raw_go_command,
            requested_depth=search_depth,
            time_limit=time_limit,
            adaptive=adaptive,
        )
        terminal_result = self._terminal_root_result(depth=search_depth)
        if terminal_result is not None:
            result_log = self._format_search_result(terminal_result)
            return [
                config,
                result_log,
                self._format_info(terminal_result, terminal_result.time_ms, terminal_result.nodes),
                "bestmove 0000",
            ]
        tablebase_result = self._tablebase_root_result(depth=search_depth)
        if tablebase_result is not None:
            result_log = self._format_search_result(tablebase_result)
            bestmove = tablebase_result.move.to_uci() if tablebase_result.move else "0000"
            return [
                config,
                (
                    "info string tablebase selected "
                    f"move={bestmove} path={self.syzygy_path} max_pieces={self.tablebase_max_pieces}"
                ),
                result_log,
                self._format_info(tablebase_result, tablebase_result.time_ms, tablebase_result.nodes),
                f"bestmove {bestmove}",
            ]
        book_move = self._opening_book_move()
        if book_move is not None:
            self.last_root_penalties = {}
            result = SearchResult(
                move=book_move,
                score=0,
                depth=0,
                completed_depth=0,
                principal_variation=(book_move,),
                nodes=0,
                time_ms=0,
            )
            san = self.game.san_history[-1] if self.game.san_history else "-"
            return [
                config,
                f"info string opening_book selected move={book_move.to_uci()} san={san} path={self.opening_book_path}",
                self._format_info(result, 1, 1),
                f"bestmove {book_move.to_uci()}",
            ]
        experience_move = self._experience_book_move()
        if experience_move is not None:
            self.last_root_penalties = {}
            result = SearchResult(
                move=experience_move,
                score=0,
                depth=0,
                completed_depth=0,
                principal_variation=(experience_move,),
                nodes=0,
                time_ms=0,
            )
            return [
                config,
                (
                    "info string experience_book selected "
                    f"move={experience_move.to_uci()} path={self.experience_book_path}"
                ),
                self._format_info(result, 1, 1),
                f"bestmove {experience_move.to_uci()}",
            ]
        if infinite:
            self._start_pending_search(depth=search_depth, time_limit=search_time_limit, adaptive=adaptive)
            return [config, f"info string search started mode={'ponder' if 'ponder' in parts else 'infinite'}"]
        result, infos = self._search_with_info(depth=search_depth, time_limit=time_limit, adaptive=adaptive)
        result_log = self._format_search_result(result)
        if result.move is None:
            return [config] + infos + [result_log, self._format_info(result, result.time_ms, result.nodes), "bestmove 0000"]
        return [config] + infos + [result_log, f"bestmove {result.move.to_uci()}"]

    def _terminal_root_result(self, *, depth: int) -> SearchResult | None:
        status = self.game.status()
        if status is GameStatus.ONGOING:
            return None
        score = score_for_color(self.game.board, self.game.board.side_to_move) if status is GameStatus.CHECKMATE else 0
        return SearchResult(
            move=None,
            score=score,
            depth=depth,
            completed_depth=0,
            principal_variation=(),
            nodes=0,
            time_ms=0,
        )

    def _start_pending_search(self, *, depth: int, time_limit: float | None, adaptive: bool) -> None:
        if self._pending_search is not None and not self._pending_search.done():
            return
        board = self.game.board.copy()
        root_penalties = combined_root_move_penalties(
            self.game.board,
            self.game.board_history,
            self.game.move_history,
        )
        self.last_root_penalties = {
            move_uci: penalty for move_uci, penalty in root_penalties.items() if penalty > 0
        }
        self._pending_search = self._executor.submit(
            find_backend_move,
            board,
            depth=depth,
            backend=self.backend,
            time_limit=time_limit,
            adaptive=adaptive,
            quiescence_depth=self.quiescence_depth,
            root_move_penalties=root_penalties,
        )

    def _finish_pending_search(self, *, request_stop: bool = False) -> list[str]:
        if self._pending_search is None:
            return ["bestmove 0000"]
        if request_stop and not self._pending_search.cancel():
            request_native_search_stop()
        if self._pending_search.cancelled():
            self._pending_search = None
            return ["bestmove 0000"]
        if request_stop:
            while True:
                try:
                    result = self._pending_search.result(timeout=0.01)
                    break
                except FutureTimeoutError:
                    # Covers the small race where stop arrives while the
                    # worker is entering native search and resets its state.
                    request_native_search_stop()
        else:
            result = self._pending_search.result()
        self._pending_search = None
        info = self._format_info(result)
        bestmove = result.move.to_uci() if result.move is not None else "0000"
        return [info, f"bestmove {bestmove}"]

    def _set_option(self, parts: list[str]) -> None:
        if "name" not in parts:
            return
        name_index = parts.index("name")
        value_index = parts.index("value") if "value" in parts else len(parts)
        name = " ".join(parts[name_index + 1 : value_index]).lower()
        value = " ".join(parts[value_index + 1 :]) if value_index < len(parts) else ""
        if name == "depth":
            self.depth = max(1, int(value))
        elif name == "backend" and value == "native":
            self.backend = value  # type: ignore[assignment]
        elif name == "move overhead":
            self.move_overhead_ms = max(0, int(value))
        elif name == "adaptive":
            self.adaptive = value.strip().lower() in {"true", "1", "yes", "on"}
        elif name == "quiescence depth":
            self.quiescence_depth = max(0, min(16, int(value)))
        elif name == "multipv":
            self.multipv = max(1, min(4, int(value)))
        elif name == "hash":
            self.hash_mb = max(1, min(4096, int(value)))
            set_native_hash_mb(self.hash_mb)
        elif name == "nnue model":
            self.nnue_model_path = value.strip()
            self._apply_nnue_options()
        elif name == "nnue mode" and value in {"off", "blend", "linear"}:
            self.nnue_mode = value
            self._apply_nnue_options()
        elif name == "nnue blend percent":
            self.nnue_blend = max(0.0, min(1.0, int(value) / 100))
            self._apply_nnue_options()
        elif name == "engine mode":
            self.engine_mode = normalize_engine_mode(value)
        elif name == "use opening book":
            self.use_opening_book = value.strip().lower() in {"true", "1", "yes", "on"}
        elif name == "opening book path":
            self.opening_book_path = value.strip() or self.opening_book_path
        elif name == "opening book max ply":
            self.opening_book_max_ply = max(0, min(40, int(value)))
        elif name == "use experience book":
            self.use_experience_book = value.strip().lower() in {"true", "1", "yes", "on"}
        elif name == "experience book path":
            self.experience_book_path = value.strip() or self.experience_book_path
        elif name == "use tablebase":
            self.use_tablebase = value.strip().lower() in {"true", "1", "yes", "on"}
        elif name == "syzygypath":
            self.syzygy_path = "" if value.strip() == "<empty>" else value.strip()
        elif name == "tablebase max pieces":
            self.tablebase_max_pieces = max(2, min(7, int(value)))
        elif name == "time manager":
            self.time_manager = normalize_time_manager(value)

    def _opening_book_move(self) -> Move | None:
        if self._opening_book_status() != "enabled":
            return None
        try:
            candidates = next_opening_moves(self.game.san_history, self.opening_book_path)
        except OSError:
            return None
        for san in self._ordered_opening_candidates(candidates):
            try:
                return move_from_san(self.game.board, san)
            except ValueError:
                continue
        return None

    def _ordered_opening_candidates(self, candidates: list[str]) -> list[str]:
        priority = {san: index for index, san in enumerate(OPENING_BOOK_SAN_PRIORITY)}
        return sorted(candidates, key=lambda san: (priority.get(san, len(priority)), san))

    def _opening_book_status(self) -> str:
        if not self.use_opening_book:
            return "off"
        if self.engine_mode == "Pure Engine":
            return "blocked_by_pure_engine"
        if self.opening_book_max_ply <= 0:
            return "disabled_max_ply_0"
        if self._current_game_ply() >= self.opening_book_max_ply:
            return "past_max_ply"
        if not Path(self.opening_book_path).exists():
            return "missing"
        return "enabled"

    def _current_game_ply(self) -> int:
        """Return ply from either applied history or a FEN-only UCI position."""
        fen_ply = max(0, (self.game.board.fullmove_number - 1) * 2)
        if self.game.board.side_to_move is Color.BLACK:
            fen_ply += 1
        return max(len(self.game.move_history), fen_ply)

    def _experience_book_move(self) -> Move | None:
        if self._experience_book_status() != "enabled":
            return None
        try:
            payload = json.loads(Path(self.experience_book_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        entries = payload.get("entries", payload if isinstance(payload, list) else [])
        if isinstance(entries, dict):
            entries = [{"fen": fen, **data} for fen, data in entries.items() if isinstance(data, dict)]
        current_key = position_key(board_to_fen(self.game.board))
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            entry_fen = entry.get("fen")
            if not isinstance(entry_fen, str) or position_key(entry_fen) != current_key:
                continue
            move_text = str(entry.get("move") or entry.get("bestmove") or "")
            try:
                move = Move.from_uci(move_text)
            except ValueError:
                continue
            if move in self.game.legal_moves():
                return move
        return None

    def _experience_book_status(self) -> str:
        if not self.use_experience_book:
            return "off"
        if self.engine_mode == "Pure Engine":
            return "blocked_by_pure_engine"
        if not Path(self.experience_book_path).exists():
            return "missing"
        return "enabled"

    def _tablebase_status(self) -> str:
        if not self.use_tablebase:
            return "off"
        if self.engine_mode == "Pure Engine":
            return "blocked_by_pure_engine"
        if not self.syzygy_path:
            return "missing"
        path = Path(self.syzygy_path)
        if not path.exists() or not path.is_dir():
            return "missing"
        if not python_chess_available():
            return "unavailable_python_chess"
        piece_count = sum(1 for _, piece in self.game.board.iter_squares() if piece is not None)
        if piece_count > self.tablebase_max_pieces:
            return "too_many_pieces"
        return "enabled"

    def _tablebase_root_result(self, *, depth: int) -> SearchResult | None:
        if self._tablebase_status() != "enabled":
            return None
        probed = probe_syzygy_root(
            board_to_fen(self.game.board),
            self.syzygy_path,
            max_pieces=self.tablebase_max_pieces,
        )
        if probed is None:
            return None
        try:
            move = Move.from_uci(probed.move_uci)
        except ValueError:
            return None
        if move not in self.game.legal_moves():
            return None
        return SearchResult(
            move=move,
            score=probed.score_cp,
            depth=depth,
            completed_depth=0,
            tablebase_hits=1,
            table_stores=0,
            principal_variation=(move,),
            nodes=1,
            time_ms=1,
        )

    def _time_limit_from_clock(self, parts: list[str]) -> float:
        side = self.game.board.side_to_move
        remaining_key = "wtime" if side is Color.WHITE else "btime"
        increment_key = "winc" if side is Color.WHITE else "binc"
        remaining_ms = _int_after(parts, remaining_key, 1000)
        increment_ms = _int_after(parts, increment_key, 0)
        moves_to_go = max(1, _int_after(parts, "movestogo", 40))
        usable_ms = max(1, remaining_ms - self.move_overhead_ms)
        base_budget_ms = usable_ms / moves_to_go + increment_ms * 0.75
        multiplier = {
            "Legacy": 1.0,
            "Conservative": 0.75,
            "Balanced": 1.15,
            "Aggressive": 1.4,
        }[self.time_manager]
        cap_ms = 3000 if self.time_manager == "Legacy" else max(500, min(10000, usable_ms * 0.18))
        budget_ms = base_budget_ms * multiplier
        limited_ms = max(50, min(cap_ms, budget_ms))
        self.last_time_manager = {
            "mode": self.time_manager,
            "side": side.value,
            "remaining_ms": remaining_ms,
            "increment_ms": increment_ms,
            "moves_to_go": moves_to_go,
            "base_budget_ms": int(base_budget_ms),
            "budget_ms": int(limited_ms),
            "cap_ms": int(cap_ms),
        }
        return limited_ms / 1000

    def _search_with_info(self, *, depth: int, time_limit: float | None, adaptive: bool) -> tuple[SearchResult, list[str]]:
        started = perf_counter()
        deadline = started + time_limit if time_limit is not None else None
        best_result = SearchResult(move=None, score=0, depth=depth)
        infos: list[str] = []
        total_nodes = 0
        previous_score: int | None = None
        previous_bestmove = ""
        stable_bestmove = 0
        instability_bonus = 1.0
        max_depth = max(1, depth)
        if adaptive and time_limit is None:
            time_limit = self.time_limit or 2.0
            deadline = started + time_limit
        root_penalties = combined_root_move_penalties(
            self.game.board,
            self.game.board_history,
            self.game.move_history,
        )
        self.last_root_penalties = {
            move_uci: penalty for move_uci, penalty in root_penalties.items() if penalty > 0
        }
        if (
            self.backend == "native" and
            self.native_internal_iterative_deepening and
            deadline is not None and
            self.multipv == 1
        ):
            result = find_backend_move(
                self.game.board,
                depth=max_depth,
                backend=self.backend,
                time_limit=max(0.001, deadline - perf_counter()),
                adaptive=False,
                quiescence_depth=self.quiescence_depth,
                root_move_penalties=root_penalties,
            )
            elapsed_ms = max(1, int((perf_counter() - started) * 1000))
            result = SearchResult(
                move=result.move,
                score=result.score,
                depth=depth,
                completed_depth=result.completed_depth,
                table_hits=result.table_hits,
                table_stores=result.table_stores,
                tablebase_hits=result.tablebase_hits,
                principal_variation=result.principal_variation,
                mate_in=result.mate_in,
                nodes=max(1, result.nodes),
                time_ms=elapsed_ms,
            )
            return result, [self._format_info(result, elapsed_ms, max(1, result.nodes))]
        for current_depth in range(1, max_depth + 1):
            if deadline is not None and current_depth > 1 and perf_counter() >= deadline:
                break
            multipv_results = find_backend_multipv(
                self.game.board,
                depth=current_depth,
                multipv=self.multipv,
                backend=self.backend,
                quiescence_depth=self.quiescence_depth,
                root_move_penalties=root_penalties,
                native_time_limit=(
                    None if deadline is None else max(0.001, deadline - perf_counter())
                ),
            )
            if not multipv_results:
                break
            if multipv_results[0].completed_depth == 0:
                if best_result.move is None:
                    best_result = multipv_results[0]
                break
            result = multipv_results[0]
            elapsed_ms = max(1, int((perf_counter() - started) * 1000))
            depth_nodes = sum(max(1, item.nodes or item.table_hits + item.table_stores or current_depth) for item in multipv_results)
            total_nodes += depth_nodes
            current_bestmove = result.move.to_uci() if result.move is not None else ""
            if current_bestmove and current_bestmove == previous_bestmove:
                stable_bestmove += 1
            else:
                stable_bestmove = 0
            previous_bestmove = current_bestmove
            if previous_score is not None:
                score_delta = abs(result.score - previous_score)
                if score_delta > 120:
                    instability_bonus = min(1.8, instability_bonus + 0.25)
                elif stable_bestmove >= 2 and score_delta < 25:
                    instability_bonus = max(0.75, instability_bonus - 0.15)
            previous_score = result.score
            best_result = SearchResult(
                move=result.move,
                score=result.score,
                depth=depth,
                completed_depth=result.completed_depth or current_depth,
                table_hits=result.table_hits,
                table_stores=result.table_stores,
                tablebase_hits=result.tablebase_hits,
                principal_variation=result.principal_variation,
                mate_in=result.mate_in,
                nodes=total_nodes,
                time_ms=elapsed_ms,
            )
            for pv_index, pv_result in enumerate(multipv_results, start=1):
                infos.append(
                    self._format_info(
                        pv_result,
                        elapsed_ms,
                        max(1, pv_result.nodes),
                        multipv=pv_index if self.multipv > 1 else None,
                    )
                )
            if deadline is not None and perf_counter() >= deadline:
                break
            if deadline is not None and current_depth >= 2:
                remaining = deadline - perf_counter()
                elapsed = perf_counter() - started
                predicted_next = elapsed / current_depth * (current_depth + 1) * instability_bonus
                if stable_bestmove >= 2:
                    predicted_next *= 0.8
                if remaining <= 0 or predicted_next > remaining:
                    break
        return best_result, infos

    def _format_search_config(
        self,
        *,
        raw_go_command: str,
        requested_depth: int,
        time_limit: float | None,
        adaptive: bool,
    ) -> str:
        weights = get_evaluation_weights()
        see_policy = "native_ordering_qsearch" if self.backend == "native" and native_available() else "python_reference"
        return (
            "info string config "
            f"mode=UCI engine_mode=\"{self.engine_mode}\" backend={self.backend} raw_go=\"{raw_go_command}\" "
            f"requested_depth={requested_depth} quiescence_depth={self.quiescence_depth} adaptive={str(adaptive).lower()} "
            f"time_limit={time_limit if time_limit is not None else 'off'} "
            f"weights_path={self.weights_path} weights={weights} nnue={linear_nnue_status()} see_policy={see_policy} "
            f"opening_book={self._opening_book_status()} opening_book_path={self.opening_book_path} "
            f"opening_book_max_ply={self.opening_book_max_ply} "
            f"tablebase={self._tablebase_status()} syzygy_path={self.syzygy_path or '<empty>'} "
            f"tablebase_max_pieces={self.tablebase_max_pieces} experience_book={self._experience_book_status()} "
            f"experience_book_path={self.experience_book_path} analysis_cache=gui_only "
            f"time_manager=\"{self.last_time_manager}\" "
            f"native=\"{native_evaluator_status()}\" native_build={native_build_status()} "
            f"native_forced_extension_budget={self.native_forced_extension_budget} "
            f"native_mate_danger_ordering={str(self.native_mate_danger_ordering).lower()} "
            f"native_defensive_mate_threat_ordering={str(self.native_defensive_mate_threat_ordering).lower()}"
            f" native_major_piece_safety_ordering={str(self.native_major_piece_safety_ordering).lower()}"
            f" native_internal_iterative_deepening={str(self.native_internal_iterative_deepening).lower()}"
            f" native_clustered_tt={str(self.native_clustered_tt).lower()}"
            f" native_aspiration_windows={str(self.native_aspiration_windows).lower()}"
            f" native_pawn_endgame_extension_budget={self.native_pawn_endgame_extension_budget}"
            f" native_check_extension_budget={self.native_check_extension_budget}"
            f" native_singular_extension={str(self.native_singular_extension).lower()}"
            f" native_futility_pruning={str(self.native_futility_pruning).lower()}"
            f" native_lmr_formula={str(self.native_lmr_formula).lower()}"
            f" native_razoring={str(self.native_razoring).lower()}"
            f" native_continuation_history={str(self.native_continuation_history).lower()}"
            f" native_probcut={str(self.native_probcut).lower()}"
            f" hash_mb={self.hash_mb}"
        )

    def _format_search_result(self, result: SearchResult) -> str:
        pv = " ".join(move.to_uci() for move in result.principal_variation) or "-"
        bestmove = result.move.to_uci() if result.move is not None else "0000"
        static_white = score_for_color(self.game.board, Color.WHITE)
        root_score = result.score
        white_search_score = root_score if self.game.board.side_to_move is Color.WHITE else -root_score
        selected_penalty = self.last_root_penalties.get(bestmove, 0)
        nonzero_penalties = ",".join(
            f"{move}:{penalty}" for move, penalty in sorted(self.last_root_penalties.items())
        ) or "-"
        return (
            "info string result "
            f"completed_depth={result.completed_depth} nodes={result.nodes} elapsed_ms={result.time_ms} "
            f"bestmove={bestmove} pv=\"{pv}\" "
            f"static_eval_current_board_white={static_white} root_score={root_score} "
            f"searched_score_white={white_search_score} selected_root_penalty={selected_penalty} "
            f"root_penalties=\"{nonzero_penalties}\" "
            f"tt_hits={result.table_hits} tt_stores={result.table_stores} tb_hits={result.tablebase_hits}"
        )

    def _format_info(self, result: SearchResult, time_ms: int | None = None, nodes: int | None = None, multipv: int | None = None) -> str:
        depth = result.completed_depth or result.depth
        score = f"mate {result.mate_in}" if result.mate_in is not None else f"cp {result.score}"
        pv = " ".join(move.to_uci() for move in result.principal_variation)
        elapsed = max(1, time_ms if time_ms is not None else result.time_ms)
        node_count = max(1, nodes if nodes is not None else result.nodes)
        nps = max(1, int(node_count * 1000 / elapsed))
        multipv_part = f" multipv {multipv}" if multipv is not None else ""
        tbhits_part = f" tbhits {result.tablebase_hits}" if result.tablebase_hits else ""
        suffix = f" pv {pv}" if pv else ""
        return (
            f"info depth {depth}{multipv_part} score {score} nodes {node_count} "
            f"nps {nps} time {elapsed}{tbhits_part}{suffix}"
        )

    def _apply_nnue_options(self) -> None:
        if self.nnue_mode == "off":
            reset_linear_nnue_model()
            return
        if not self.nnue_model_path or not Path(self.nnue_model_path).exists():
            return
        if Path(self.nnue_model_path).suffix.lower() == ".dtnnue":
            load_accumulator_nnue_model(self.nnue_model_path, mode=self.nnue_mode, blend_weight=self.nnue_blend)
        else:
            load_linear_nnue_model(self.nnue_model_path, mode=self.nnue_mode, blend_weight=self.nnue_blend)


def run_uci_loop(
    *,
    depth: int = 2,
    backend: SearchBackend = "native",
    time_limit: float | None = None,
    adaptive: bool = False,
    quiescence_depth: int = 4,
    weights_path: str | Path | None = None,
    nnue_model_path: str | Path | None = None,
    nnue_mode: str = "off",
    nnue_blend: float = 0.35,
    native_forced_extension_budget: int = DEFAULT_FORCED_LINE_EXTENSION_BUDGET,
    native_mate_danger_ordering: bool = False,
    native_defensive_mate_threat_ordering: bool = False,
    native_major_piece_safety_ordering: bool = DEFAULT_MAJOR_PIECE_SAFETY_ORDERING,
    native_internal_iterative_deepening: bool = True,
    native_clustered_tt: bool = False,
    native_aspiration_windows: bool = True,
    native_pawn_endgame_extension_budget: int = 0,
    native_check_extension_budget: int = 0,
    native_singular_extension: bool = False,
    native_futility_pruning: bool = True,
    native_lmr_formula: bool = False,
    native_razoring: bool = False,
    native_continuation_history: bool = False,
    native_probcut: bool = False,
    hash_mb: int = 16,
    engine_mode: str = "Pure Engine",
    use_opening_book: bool = False,
    opening_book_path: str | Path = "assets/openings/openings.tsv",
    opening_book_max_ply: int = 2,
    use_experience_book: bool = False,
    experience_book_path: str | Path = "artifacts/experience_book.json",
    use_tablebase: bool = False,
    syzygy_path: str | Path = "",
    tablebase_max_pieces: int = 7,
    time_manager: str = "Balanced",
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
) -> None:
    """Run a blocking UCI loop over text streams."""
    engine = UCIEngine(
        depth=depth,
        backend=backend,
        time_limit=time_limit,
        adaptive=adaptive,
        quiescence_depth=quiescence_depth,
        weights_path=weights_path,
        nnue_model_path=nnue_model_path,
        nnue_mode=nnue_mode,
        nnue_blend=nnue_blend,
        native_forced_extension_budget=native_forced_extension_budget,
        native_mate_danger_ordering=native_mate_danger_ordering,
        native_defensive_mate_threat_ordering=native_defensive_mate_threat_ordering,
        native_major_piece_safety_ordering=native_major_piece_safety_ordering,
        native_internal_iterative_deepening=native_internal_iterative_deepening,
        native_clustered_tt=native_clustered_tt,
        native_aspiration_windows=native_aspiration_windows,
        native_pawn_endgame_extension_budget=native_pawn_endgame_extension_budget,
        native_check_extension_budget=native_check_extension_budget,
        native_singular_extension=native_singular_extension,
        native_futility_pruning=native_futility_pruning,
        native_lmr_formula=native_lmr_formula,
        native_razoring=native_razoring,
        native_continuation_history=native_continuation_history,
        native_probcut=native_probcut,
        hash_mb=hash_mb,
        engine_mode=engine_mode,
        use_opening_book=use_opening_book,
        opening_book_path=opening_book_path,
        opening_book_max_ply=opening_book_max_ply,
        use_experience_book=use_experience_book,
        experience_book_path=experience_book_path,
        use_tablebase=use_tablebase,
        syzygy_path=syzygy_path,
        tablebase_max_pieces=tablebase_max_pieces,
        time_manager=time_manager,
    )
    input_stream = input_stream or sys.stdin
    output_stream = output_stream or sys.stdout
    for line in input_stream:
        try:
            responses = engine.handle_line(line)
        except Exception as exc:  # UCI GUIs expect text, not Python tracebacks.
            responses = [f"info string error: {exc}"]
        if "__quit__" in responses:
            return
        for response in responses:
            print(response, file=output_stream, flush=True)


def _int_after(parts: list[str], key: str, default: int) -> int:
    if key not in parts:
        return default
    index = parts.index(key)
    if index + 1 >= len(parts):
        return default
    try:
        return int(parts[index + 1])
    except ValueError:
        return default
