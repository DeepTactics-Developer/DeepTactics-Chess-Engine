"""Asynchronous UCI process client for the Qt desktop interface.

The client intentionally treats the chess engine as a separate process. Qt
widgets receive parsed protocol events and never call search or evaluation
functions directly.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import sys
from typing import Any, Mapping

from PySide6.QtCore import QObject, QProcess, QTimer, Signal


@dataclass(frozen=True, slots=True)
class UCIAnalysisInfo:
    """Latest parsed UCI analysis values, normalized to White's perspective."""

    depth: int = 0
    score_cp_white: int | None = None
    mate_white: int | None = None
    nodes: int = 0
    nps: int = 0
    time_ms: int = 0
    pv: tuple[str, ...] = ()
    multipv: int = 1
    tbhits: int | None = None
    hashfull: int | None = None


def parse_info_line(line: str, side_to_move: str = "w") -> UCIAnalysisInfo | None:
    """Parse an ``info depth`` line and convert its score to White perspective."""
    fields = line.strip().split()
    if len(fields) < 3 or fields[0:2] != ["info", "depth"]:
        return None

    def int_after(name: str, default: int = 0) -> int:
        try:
            return int(fields[fields.index(name) + 1])
        except (ValueError, IndexError):
            return default

    sign = 1 if side_to_move == "w" else -1
    score_cp_white: int | None = None
    mate_white: int | None = None
    try:
        score_index = fields.index("score")
        score_kind = fields[score_index + 1]
        score_value = int(fields[score_index + 2])
        if score_kind == "cp":
            score_cp_white = score_value * sign
        elif score_kind == "mate":
            mate_white = score_value * sign
    except (ValueError, IndexError):
        pass
    try:
        pv = tuple(fields[fields.index("pv") + 1 :])
    except ValueError:
        pv = ()
    return UCIAnalysisInfo(
        depth=int_after("depth"),
        score_cp_white=score_cp_white,
        mate_white=mate_white,
        nodes=int_after("nodes"),
        nps=int_after("nps"),
        time_ms=int_after("time"),
        pv=pv,
        multipv=int_after("multipv", 1),
        tbhits=int_after("tbhits") if "tbhits" in fields else None,
        hashfull=int_after("hashfull") if "hashfull" in fields else None,
    )


def parse_result_white_score(line: str) -> int | None:
    """Read DeepTactics' diagnostic white-perspective score when it is present."""
    marker = "searched_score_white="
    if not line.startswith("info string result ") or marker not in line:
        return None
    tail = line.split(marker, 1)[1].split()[0]
    try:
        return int(tail)
    except ValueError:
        return None


class UCIClient(QObject):
    """Manage one UCI engine asynchronously through ``QProcess``."""

    state_changed = Signal(str)
    ready = Signal()
    info_received = Signal(object)
    bestmove_received = Signal(str)
    error_received = Signal(str)
    raw_line_received = Signal(str)
    command_sent = Signal(str)
    identity_received = Signal(object)

    def __init__(self, project_root: str | Path, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.project_root = Path(project_root)
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.started.connect(self._on_started)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.errorOccurred.connect(self._on_process_error)
        self.process.finished.connect(self._on_finished)
        self.timeout = QTimer(self)
        self.timeout.setSingleShot(True)
        self.timeout.timeout.connect(self._on_timeout)
        self._buffer = ""
        self._initialized = False
        self._initializing = False
        self._searching = False
        self._quitting = False
        self._queued_search: tuple[str, str, str] | None = None
        self._side_to_move = "w"
        self.latest_info = UCIAnalysisInfo()
        self._options: dict[str, str] = {}
        self.profile: dict[str, Any] = {"name": "DeepTactics", "path": "DeepTactics-UCI.cmd", "builtin": True}
        self.uci_name = "DeepTactics"
        self.supported_options: set[str] = set()

    @property
    def is_running(self) -> bool:
        return self.process.state() != QProcess.ProcessState.NotRunning

    @property
    def is_searching(self) -> bool:
        return self._searching

    @property
    def supports_nnue(self) -> bool:
        return any("nnue" in name.lower() or "evalfile" in name.lower() for name in self.supported_options)

    def configure_engine(self, profile: Mapping[str, Any] | None = None) -> None:
        """Select the engine executable used by the next session."""
        selected = dict(profile or {"name": "DeepTactics", "path": "DeepTactics-UCI.cmd", "builtin": True})
        selected["builtin"] = selected.get("name") == "DeepTactics"
        if self.is_running:
            self.quit()
        self.profile = selected
        self.uci_name = str(selected.get("name", "UCI Engine"))
        self.supported_options.clear()
        self._initialized = False

    def start(self) -> None:
        if self.is_running:
            return
        self._quitting = False
        self._initializing = True
        self._initialized = False
        self.supported_options.clear()
        self.state_changed.emit("Starting")
        if bool(self.profile.get("builtin")):
            self.process.setWorkingDirectory(str(self.project_root))
            args = ["-m", "chess_engine", "uci", "--search-backend", "native"]
            if bool(self.profile.get("use_tablebase")):
                args.append("--use-tablebase")
                if self.profile.get("syzygy_path"):
                    args.extend(["--syzygy-path", str(self.profile.get("syzygy_path"))])
                if self.profile.get("tablebase_max_pieces") is not None:
                    args.extend(["--tablebase-max-pieces", str(self.profile.get("tablebase_max_pieces"))])
            self.process.start(
                sys.executable,
                args,
            )
        else:
            executable = Path(str(self.profile.get("path", "")))
            self.process.setWorkingDirectory(str(executable.parent))
            if sys.platform == "win32" and executable.suffix.lower() in (".cmd", ".bat"):
                self.process.start("cmd.exe", ["/c", str(executable)])
            else:
                self.process.start(str(executable), [])
        self.timeout.start(5000)

    def set_option(self, name: str, value: str | int | bool) -> None:
        text = str(value).lower() if isinstance(value, bool) else str(value)
        self._options[name] = text
        if self._initialized:
            self._write(f"setoption name {name} value {text}")

    def trigger_option(self, name: str) -> bool:
        if self.supported_options and name not in self.supported_options:
            return False
        if not self._initialized:
            return False
        self._write(f"setoption name {name}")
        return True

    def set_position(
        self,
        fen: str,
        *,
        moves: tuple[str, ...] | None = None,
        initial_fen: str | None = None,
    ) -> None:
        self._side_to_move = fen.split()[1] if len(fen.split()) > 1 else "w"
        if moves is not None:
            origin = "startpos" if initial_fen is None else f"fen {initial_fen}"
            suffix = f" moves {' '.join(moves)}" if moves else ""
            self._write(f"position {origin}{suffix}")
        else:
            self._write(f"position fen {fen}")

    def analyze_fen(
        self,
        fen: str,
        *,
        depth: int | None = None,
        movetime_ms: int | None = None,
        infinite: bool = False,
        moves: tuple[str, ...] | None = None,
        initial_fen: str | None = None,
    ) -> None:
        if infinite:
            command = "go infinite"
        elif movetime_ms is not None:
            command = f"go movetime {max(1, int(movetime_ms))}"
        else:
            command = f"go depth {max(1, int(depth or 4))}"
        position = self._position_command(fen, moves=moves, initial_fen=initial_fen)
        self._queued_search = (fen, position, command)
        if not self._initialized:
            self.start()
            return
        if self._searching:
            self.stop()
            return
        self._begin_queued_search()

    def play_fen_with_clock(
        self,
        fen: str,
        *,
        white_ms: int,
        black_ms: int,
        white_increment_ms: int = 0,
        black_increment_ms: int = 0,
        moves: tuple[str, ...] | None = None,
        initial_fen: str | None = None,
    ) -> None:
        """Request one clock-managed UCI move without owning any GUI game state."""
        command = (
            f"go wtime {max(1, int(white_ms))} btime {max(1, int(black_ms))} "
            f"winc {max(0, int(white_increment_ms))} binc {max(0, int(black_increment_ms))}"
        )
        position = self._position_command(fen, moves=moves, initial_fen=initial_fen)
        self._queued_search = (fen, position, command)
        if not self._initialized:
            self.start()
            return
        if self._searching:
            self.stop()
            return
        self._begin_queued_search()

    def go_depth(self, depth: int) -> None:
        self._start_search(f"go depth {max(1, depth)}")

    def go_movetime(self, milliseconds: int) -> None:
        self._start_search(f"go movetime {max(1, milliseconds)}")

    def go_infinite(self) -> None:
        self._start_search("go infinite")

    def stop(self) -> None:
        if self._searching:
            self._write("stop")
            self.state_changed.emit("Stopping")

    def cancel_analysis(self) -> None:
        """Stop user-requested analysis and discard any queued position refresh."""
        self._queued_search = None
        self.stop()

    def quit(self) -> None:
        self._queued_search = None
        self._quitting = True
        if self.is_running:
            self._write("quit")
            self.process.waitForFinished(250)
            if self.is_running:
                self.process.kill()
                self.process.waitForFinished(250)

    def _on_started(self) -> None:
        self.state_changed.emit("Connecting")
        self._write("uci")

    def _write(self, command: str) -> None:
        if not self.is_running:
            return
        self.command_sent.emit(command)
        self.process.write((command + "\n").encode("utf-8"))

    def _read_output(self) -> None:
        self._buffer += bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        while "\n" in self._buffer:
            raw, self._buffer = self._buffer.split("\n", 1)
            self._handle_line(raw.rstrip("\r"))

    def _handle_line(self, line: str) -> None:
        if not line:
            return
        self.raw_line_received.emit(line)
        if line.startswith("id name "):
            self.uci_name = line[8:].strip() or self.uci_name
            return
        if line.startswith("option name "):
            option = line[12:].split(" type ", 1)[0].strip()
            if option:
                self.supported_options.add(option)
            return
        if line == "uciok":
            for option, value in (
                ("Threads", self.profile.get("threads")),
                ("Hash", self.profile.get("hash")),
                ("Use Tablebase", self.profile.get("use_tablebase")),
                ("SyzygyPath", self.profile.get("syzygy_path")),
                ("Tablebase Max Pieces", self.profile.get("tablebase_max_pieces")),
            ):
                if value not in (None, "") and (not self.supported_options or option in self.supported_options):
                    text = str(value).lower() if isinstance(value, bool) else str(value)
                    self._write(f"setoption name {option} value {text}")
            for name, value in self._options.items():
                if not self.supported_options or name in self.supported_options:
                    self._write(f"setoption name {name} value {value}")
            self._write("isready")
            return
        if line == "readyok":
            self.timeout.stop()
            self._initializing = False
            self._initialized = True
            self.state_changed.emit("Ready")
            self.identity_received.emit({
                "name": self.uci_name,
                "options": tuple(sorted(self.supported_options)),
                "supports_nnue": self.supports_nnue,
            })
            self.ready.emit()
            self._begin_queued_search()
            return
        info = parse_info_line(line, self._side_to_move)
        if info is not None:
            if info.multipv == 1:
                self.latest_info = info
            self.info_received.emit(info)
            return
        white_score = parse_result_white_score(line)
        if white_score is not None:
            if self.latest_info.mate_white is None:
                self.latest_info = replace(self.latest_info, score_cp_white=white_score)
            self.info_received.emit(self.latest_info)
            return
        if line.startswith("bestmove "):
            move = line.split()[1] if len(line.split()) > 1 else "0000"
            self._searching = False
            self.state_changed.emit("Ready")
            self.bestmove_received.emit(move)
            if self._queued_search is not None:
                self._begin_queued_search()

    def _begin_queued_search(self) -> None:
        if self._queued_search is None or not self._initialized or self._searching:
            return
        fen, position, command = self._queued_search
        self._queued_search = None
        self._side_to_move = fen.split()[1] if len(fen.split()) > 1 else "w"
        self._write(position)
        self._start_search(command)

    @staticmethod
    def _position_command(
        fen: str,
        *,
        moves: tuple[str, ...] | None = None,
        initial_fen: str | None = None,
    ) -> str:
        if moves is None:
            return f"position fen {fen}"
        origin = "startpos" if initial_fen is None else f"fen {initial_fen}"
        return f"position {origin}" + (f" moves {' '.join(moves)}" if moves else "")

    def _start_search(self, command: str) -> None:
        if not self._initialized:
            self.error_received.emit("UCI engine is not ready.")
            return
        self.latest_info = UCIAnalysisInfo()
        self._searching = True
        self.state_changed.emit("Analyzing")
        self._write(command)

    def _on_timeout(self) -> None:
        if self._initializing:
            self.error_received.emit("The UCI engine did not respond during startup.")
            self.state_changed.emit("Error")
            self.process.kill()

    def _on_process_error(self, error: QProcess.ProcessError) -> None:
        if self._quitting:
            return
        self.error_received.emit(f"UCI process error: {error.name}")
        self.state_changed.emit("Error")

    def _on_finished(self, _exit_code: int, _status: QProcess.ExitStatus) -> None:
        self._initialized = False
        self._searching = False
        self.timeout.stop()
        if not self._quitting:
            self.error_received.emit("The UCI engine process exited unexpectedly.")
            self.state_changed.emit("Stopped")
