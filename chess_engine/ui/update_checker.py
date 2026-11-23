"""Asynchronous GitHub Releases checker for the optional Qt desktop UI."""

from __future__ import annotations

import json

from PySide6.QtCore import QObject, QUrl, Signal
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest


RELEASES_API = "https://api.github.com/repos/DeepTactics-Developer/DeepTactics-Chess-Engine/releases/latest"


class UpdateChecker(QObject):
    completed = Signal(str, str, str)
    failed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.manager = QNetworkAccessManager(self)
        self.manager.finished.connect(self._finished)

    def check(self) -> None:
        request = QNetworkRequest(QUrl(RELEASES_API))
        request.setRawHeader(b"Accept", b"application/vnd.github+json")
        request.setRawHeader(b"User-Agent", b"DeepTactics-Qt")
        self.manager.get(request)

    def _finished(self, reply: QNetworkReply) -> None:
        try:
            if reply.error() != QNetworkReply.NetworkError.NoError:
                self.failed.emit(reply.errorString())
                return
            payload = json.loads(bytes(reply.readAll()).decode("utf-8"))
            self.completed.emit(
                str(payload.get("tag_name", "")),
                str(payload.get("html_url", "")),
                str(payload.get("name", "")),
            )
        except (UnicodeError, json.JSONDecodeError) as error:
            self.failed.emit(str(error))
        finally:
            reply.deleteLater()

