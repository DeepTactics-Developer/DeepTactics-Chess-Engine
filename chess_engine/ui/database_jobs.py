"""Sequential QProcess runner for long-lived Database page CLI jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
import uuid

from PySide6.QtCore import QObject, QProcess, Signal


@dataclass(slots=True)
class DatabaseJob:
    task: str
    dataset: str
    arguments: list[str]
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: str = "Queued"
    output: str = ""


class DatabaseJobRunner(QObject):
    queued = Signal(object)
    started = Signal(object)
    output_received = Signal(object)
    finished = Signal(object)

    def __init__(self, project_root: Path, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.project_root = Path(project_root)
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(self.project_root))
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._error)
        self.pending: list[DatabaseJob] = []
        self.current: DatabaseJob | None = None
        self._cancel_requested = False

    def enqueue(self, job: DatabaseJob) -> None:
        self.pending.append(job)
        self.queued.emit(job)
        self._start_next()

    def cancel_current(self) -> None:
        if self.current is not None and self.process.state() != QProcess.ProcessState.NotRunning:
            self.current.status = "Cancelling"
            self._cancel_requested = True
            self.process.kill()

    def _start_next(self) -> None:
        if self.current is not None or not self.pending:
            return
        self.current = self.pending.pop(0)
        self._cancel_requested = False
        self.current.status = "Running"
        self.started.emit(self.current)
        self.process.start(sys.executable, ["-m", "chess_engine", *self.current.arguments])

    def _read_output(self) -> None:
        if self.current is None:
            return
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self.current.output += text
        self.output_received.emit(self.current)

    def _finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        if self.current is None:
            return
        job = self.current
        job.status = "Cancelled" if self._cancel_requested else ("Completed" if exit_code == 0 else "Failed")
        self.current = None
        self._cancel_requested = False
        self.finished.emit(job)
        self._start_next()

    def _error(self, error: QProcess.ProcessError) -> None:
        if self.current is not None:
            self.current.output += f"\nQProcess error: {error.name}\n"
        if error != QProcess.ProcessError.FailedToStart or self.current is None:
            return
        job = self.current
        job.status = "Failed"
        self.current = None
        self.finished.emit(job)
        self._start_next()
