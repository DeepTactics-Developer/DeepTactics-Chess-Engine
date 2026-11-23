"""PySide6 desktop interface for DeepTactics.

The visual shell is now hosted in the application package. Initial functionality
connects local GUI state and user settings; UCI-backed live analysis is layered
on separately so engine computation remains isolated from Qt.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass, replace
from datetime import datetime
import json
import math
import os
import platform
import random
import re
import shutil
import sys
import time
from pathlib import Path

from PySide6.QtCore import QObject, QPointF, QRectF, QSize, Qt, QTimer, QVariantAnimation, Signal, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QFont, QIcon, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHeaderView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QTabBar,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from chess_engine.chess_core.board import Board, CastlingRights
from chess_engine.chess_core.enums import Color, GameStatus as ChessGameStatus, PieceType
from chess_engine.chess_core.fen import STANDARD_STARTING_FEN, board_from_fen, board_to_fen
from chess_engine.chess_core.game import Game
from chess_engine.chess_core.move import Move
from chess_engine.chess_core.pgn import format_san_move_list, game_from_pgn
from chess_engine.chess_core.piece import Piece
from chess_engine.chess_core.rules import generate_legal_moves
from chess_engine.chess_core.square import Square
from chess_engine.openings import detect_opening
from chess_engine.analysis import MoveClassification
from chess_engine.ui.asset_manager import AssetManager
from chess_engine.ui.database_store import DatabaseStore, DatasetRecord
from chess_engine.ui.database_jobs import DatabaseJob, DatabaseJobRunner
from chess_engine.ui.engine_registry import EngineRegistry
from chess_engine.ui.game_library import GameLibraryStore, GameRecord
from chess_engine.ui.gui_state import GameDocument, GameSnapshot
from chess_engine.ui.i18n import TranslationCatalog
from chess_engine.ui.opening_store import OpeningRecord, load_opening_records, opening_user_state, write_opening_user_state
from chess_engine.ui.settings_store import DEFAULT_GUI_SETTINGS, SettingsStore
from chess_engine.ui.sound_manager import SoundManager
from chess_engine.ui.uci_client import UCIAnalysisInfo, UCIClient
from chess_engine.ui.update_checker import UpdateChecker
from chess_engine.native.native_eval import native_build_status, native_available


ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets"
SETTINGS_PATH = ROOT / "config" / "gui_settings.json"
OPENING_BOOK_PATH = ASSETS / "openings" / "openings.tsv"
AI_MATCH_OPENING_SUITE_PATH = ASSETS / "benchmarks" / "uci_opening_lines.txt"
SETTINGS_STORE = SettingsStore(SETTINGS_PATH)
ENGINE_REGISTRY = EngineRegistry(SETTINGS_STORE)
ASSET_MANAGER = AssetManager(ASSETS)
TRANSLATIONS = TranslationCatalog(ASSETS, str(SETTINGS_STORE.load().get("language", "English")))
APP_VERSION = "0.9.0-alpha.1"
ANALYSIS_CLASSIFICATION_POLICY_VERSION = "expected-points-v2"
TERMINAL_CHECKMATE_CP = 100_000

APP_BG = "#080f17"
HEADER = "#0a1119"
CARD = "#0e1721"
CARD_HIGH = "#111c28"
BORDER = "#1a2a3b"
BORDER_BLUE = "#174574"
TEXT = "#e8edf5"
SECONDARY = "#a5afbf"
MUTED = "#728092"
BLUE = "#1977f3"
BLUE_SOFT = "#112c53"
BLUE_LINE = "#2587ff"
GREEN = "#2bd878"
WOOD_LIGHT = QColor("#cea674")
WOOD_DARK = QColor("#855d38")
WOOD_LAST_LIGHT = QColor("#c7a55a")
WOOD_LAST_DARK = QColor("#927047")


def load_ai_match_opening_lines(path: Path = AI_MATCH_OPENING_SUITE_PATH) -> tuple[tuple[str, ...], ...]:
    """Load legal paired-match prefixes without consulting them during search."""
    lines: list[tuple[str, ...]] = []
    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ()
    for raw in raw_lines:
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        moves = tuple(text.split())
        game = Game.new()
        try:
            for move in moves:
                game.apply_move(Move.from_uci(move))
        except ValueError:
            continue
        lines.append(moves)
    return tuple(lines)


def text_label(text: str, object_name: str = "") -> QLabel:
    label = QLabel(text)
    if object_name:
        label.setObjectName(object_name)
    return label


def preview_notice(text: str) -> QLabel:
    """Identify design-stage records until a persisted data source is connected."""
    label = text_label(text, "blueText")
    label.setContentsMargins(4, 0, 0, 3)
    return label


def move_badge(kind: str) -> QLabel:
    symbols = {"star": "\u2605", "good": "\u2713", "best": "\u2605"}
    badge = QLabel(symbols.get(kind, ""))
    badge.setFixedSize(16, 16)
    badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
    if kind:
        badge.setObjectName("moveBadge")
        badge.setProperty("kind", kind)
        badge.setToolTip({"star": "Excellent move", "good": "Good move", "best": "Best move"}[kind])
    return badge


DISPLAYED_CLASSIFICATIONS = {
    "brilliant", "great", "book", "forced", "best", "excellent",
    "good", "inaccuracy", "mistake", "missed_win", "blunder",
}


def set_move_badge(
    badge: QLabel,
    classification: MoveClassification | None,
    detail_tooltip: str = "",
) -> None:
    badge.clear()
    badge.setVisible(False)
    if classification is None or classification.key not in DISPLAYED_CLASSIFICATIONS:
        return
    path = ASSETS / "classifications" / f"{classification.key}.png"
    if path.exists():
        badge.setPixmap(QPixmap(str(path)).scaled(16, 16, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
    else:
        badge.setText(classification.symbol)
    badge.setToolTip(detail_tooltip or f"{classification.label} ({classification.loss_cp} cp loss)")
    badge.setVisible(True)


def format_evaluation(score_cp_white: int | None = None, mate_white: int | None = None) -> str:
    """Format a White-perspective engine evaluation for display only."""
    if mate_white is not None:
        return f"{'+' if mate_white > 0 else '-'}M{abs(mate_white)}"
    if score_cp_white is None:
        return "--"
    bounded = max(-1000, min(1000, score_cp_white))
    suffix = "+" if abs(score_cp_white) >= 1000 else ""
    return f"{bounded / 100:+.2f}{suffix}"


@dataclass(frozen=True, slots=True)
class AnalyzeClassificationPolicy:
    """Centralized Qt review thresholds for move quality labels."""

    inaccuracy_loss: float = 0.08
    mistake_loss: float = 0.16
    blunder_loss: float = 0.32
    miss_opportunity_gain: float = 0.24
    miss_expected_loss: float = 0.22
    miss_min_expected_before: float = 0.70
    logistic_scale_cp: float = 220.0


@dataclass(frozen=True, slots=True)
class AnalyzeMoveEvidence:
    """Raw and normalized review evidence retained for UI explanations."""

    before_display: str
    after_display: str
    cp_loss: int | None
    expected_before: float
    expected_after: float
    expected_loss: float
    opportunity_gain: float
    label: str
    reason: str

    def tooltip(self) -> str:
        cp_text = "--" if self.cp_loss is None else f"{self.cp_loss} cp"
        return (
            f"{self.label}: {self.reason}\n"
            f"Evaluation {self.before_display} -> {self.after_display}; raw loss {cp_text}\n"
            f"Expected points {self.expected_before:.3f} -> {self.expected_after:.3f}; "
            f"loss {self.expected_loss:.3f}"
        )


ANALYZE_CLASSIFICATION_POLICY = AnalyzeClassificationPolicy()
QT_CLASSIFICATION_LABELS = {
    "book": ("Book", "B"),
    "forced": ("Forced", "F"),
    "best": ("Best", "*"),
    "inaccuracy": ("Inaccuracy", "?!"),
    "mistake": ("Mistake", "?"),
    "missed_win": ("Miss", "x"),
    "blunder": ("Blunder", "??"),
}


def cp_to_expected_points(cp: int, *, scale_cp: float = ANALYZE_CLASSIFICATION_POLICY.logistic_scale_cp) -> float:
    """Convert a White-perspective centipawn score into saturated expected points."""
    bounded = max(-4000, min(4000, int(cp)))
    return 1.0 / (1.0 + math.exp(-bounded / scale_cp))


def _analysis_expected_points(info: UCIAnalysisInfo) -> float | None:
    if info.mate_white is not None:
        if info.mate_white > 0:
            return 1.0
        if info.mate_white < 0:
            return 0.0
        return 0.5
    if info.score_cp_white is None:
        return None
    return cp_to_expected_points(info.score_cp_white)


def _terminal_display_info(fen: str, info: UCIAnalysisInfo) -> UCIAnalysisInfo:
    """Keep completed checkmates directional when a UCI engine returns a neutral terminal score."""
    board = board_from_fen(fen)
    if Game(board=board).status() is not ChessGameStatus.CHECKMATE:
        return info
    white_score = TERMINAL_CHECKMATE_CP if board.side_to_move is Color.BLACK else -TERMINAL_CHECKMATE_CP
    return replace(info, score_cp_white=white_score, mate_white=None)


def _mover_expected_points(white_expected_points: float, player: Color) -> float:
    return white_expected_points if player is Color.WHITE else 1.0 - white_expected_points


def _qt_move_classification(
    *,
    player: Color,
    before: UCIAnalysisInfo,
    after: UCIAnalysisInfo,
    previous: UCIAnalysisInfo | None,
    played_move: Move,
    best_move: Move | None,
    is_forced: bool,
    is_book: bool,
    policy: AnalyzeClassificationPolicy = ANALYZE_CLASSIFICATION_POLICY,
) -> tuple[MoveClassification | None, AnalyzeMoveEvidence | None]:
    """Classify one completed review result without changing engine evaluation."""
    white_before = _analysis_expected_points(before)
    white_after = _analysis_expected_points(after)
    if white_before is None or white_after is None:
        return None, None
    mover_before = _mover_expected_points(white_before, player)
    mover_after = _mover_expected_points(white_after, player)
    expected_loss = max(0.0, mover_before - mover_after)
    cp_before = before.score_cp_white
    cp_after = after.score_cp_white
    cp_loss = None
    if cp_before is not None and cp_after is not None:
        cp_loss = max(0, cp_before - cp_after) if player is Color.WHITE else max(0, cp_after - cp_before)
    opportunity_gain = 0.0
    previous_expected = _analysis_expected_points(previous) if previous is not None else None
    if previous_expected is not None:
        opportunity_gain = max(0.0, mover_before - _mover_expected_points(previous_expected, player))

    key = ""
    reason = "Ordinary move below the display threshold"
    if is_book:
        key, reason = "book", "Opening-book line"
    elif is_forced:
        key, reason = "forced", "Only one legal move"
    elif best_move is not None and played_move.to_uci() == best_move.to_uci():
        key, reason = "best", "Matches the reviewed best move"
    elif (
        mover_before >= policy.miss_min_expected_before
        and opportunity_gain >= policy.miss_opportunity_gain
        and expected_loss >= policy.miss_expected_loss
    ):
        key, reason = "missed_win", "Missed a significant opportunity created in the previous position"
    elif expected_loss >= policy.blunder_loss:
        key, reason = "blunder", "Large expected-points loss"
    elif expected_loss >= policy.mistake_loss:
        key, reason = "mistake", "Meaningful expected-points loss"
    elif expected_loss >= policy.inaccuracy_loss:
        key, reason = "inaccuracy", "Small but visible expected-points loss"

    evidence = AnalyzeMoveEvidence(
        before_display=format_evaluation(before.score_cp_white, before.mate_white),
        after_display=format_evaluation(after.score_cp_white, after.mate_white),
        cp_loss=cp_loss,
        expected_before=mover_before,
        expected_after=mover_after,
        expected_loss=expected_loss,
        opportunity_gain=opportunity_gain,
        label=QT_CLASSIFICATION_LABELS.get(key, ("No icon", ""))[0],
        reason=reason,
    )
    if not key:
        return None, evidence
    label, symbol = QT_CLASSIFICATION_LABELS[key]
    return MoveClassification(key, label, symbol, cp_loss or 0), evidence


def pgn_with_metadata(pgn: str, **tags: str) -> str:
    """Update standard GUI-created PGN headers while preserving the move text."""
    result = pgn
    additions: list[str] = []
    for key, value in tags.items():
        safe_value = value.replace('"', "'")
        pattern = rf'(?m)^\[{re.escape(key)}\s+"[^"]*"\]\s*$'
        replacement = f'[{key} "{safe_value}"]'
        if re.search(pattern, result):
            result = re.sub(pattern, replacement, result, count=1)
        else:
            additions.append(replacement)
    if additions:
        result = "\n".join(additions) + "\n" + result
    return result


def default_pgn_filename(kind: str, white: str = "", black: str = "") -> str:
    date = datetime.now().strftime("%Y-%m-%d")
    players = f"_{white}_vs_{black}" if white and black else ""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{kind}{players}_{date}")
    return f"{safe}.pgn"


def divider(vertical: bool = False) -> QFrame:
    rule = QFrame()
    rule.setFrameShape(QFrame.Shape.VLine if vertical else QFrame.Shape.HLine)
    rule.setObjectName("divider")
    return rule


def button(text: str, *, primary: bool = False, compact: bool = False) -> QPushButton:
    control = QPushButton(text)
    control.setProperty("primary", primary)
    control.setProperty("compact", compact)
    control.setCursor(Qt.CursorShape.PointingHandCursor)
    return control


class Card(QFrame):
    def __init__(self, object_name: str = "card") -> None:
        super().__init__()
        self.setObjectName(object_name)


class Header(QWidget):
    tab_selected = Signal(str)

    def __init__(self, active_tab: str = "Analyze") -> None:
        super().__init__()
        self._drag_origin = None
        self.tabs: dict[str, QPushButton] = {}
        self.setObjectName("header")
        self.setFixedHeight(78)
        row = QHBoxLayout(self)
        row.setContentsMargins(30, 0, 23, 0)
        row.setSpacing(0)

        identity = QWidget()
        identity.setObjectName("transparent")
        identity.setFixedWidth(345)
        self.identity = identity
        identity_row = QHBoxLayout(identity)
        identity_row.setContentsMargins(0, 0, 0, 0)
        identity_row.setSpacing(17)
        logo = text_label("\u265e", "logoMark")
        brand = text_label("DeepTactics", "brand")
        version = text_label(f"v{APP_VERSION}", "version")
        self.version_label = version
        identity_row.addWidget(logo)
        identity_row.addWidget(brand)
        identity_row.addWidget(version)
        identity_row.addStretch()
        row.addWidget(identity)
        row.addStretch()

        navigation = QHBoxLayout()
        navigation.setSpacing(10)
        for name, icon in (
            ("Play", "\u25b6"),
            ("Analyze", "\u2302"),
            ("AI Match", "\u2694"),
            ("Games", "\u25a6"),
            ("Openings", "\u2658"),
            ("Database", "\u25a4"),
            ("Setting", "\u2699"),
        ):
            translation_key = f"tab.{name.lower().replace(' ', '_')}"
            nav = QPushButton(f"{icon}  {TRANSLATIONS.text(translation_key, name)}")
            nav.setProperty("nav", True)
            nav.setProperty("selected", name == active_tab)
            nav.clicked.connect(lambda _checked=False, tab=name: self.tab_selected.emit(tab))
            self.tabs[name] = nav
            navigation.addWidget(nav)
        row.addLayout(navigation)
        row.addStretch()

        controls = QWidget()
        controls.setObjectName("transparent")
        controls.setFixedWidth(345)
        controls_row = QHBoxLayout(controls)
        controls_row.setContentsMargins(0, 0, 0, 0)
        controls_row.setSpacing(8)
        controls_row.addStretch()
        for glyph, handler in (("\u2014", lambda: self.window().showMinimized()), ("\u25a1", self.toggle_maximize), ("\u00d7", lambda: self.window().close())):
            window_control = QToolButton()
            window_control.setText(glyph)
            window_control.setObjectName("windowControl")
            window_control.clicked.connect(handler)
            controls_row.addWidget(window_control)
        row.addWidget(controls)

    def set_active(self, active_tab: str) -> None:
        for name, nav in self.tabs.items():
            nav.setProperty("selected", name == active_tab)
            nav.style().unpolish(nav)
            nav.style().polish(nav)

    def toggle_maximize(self) -> None:
        if self.window().isMaximized():
            self.window().showNormal()
        else:
            self.window().showMaximized()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_origin = event.globalPosition().toPoint() - self.window().frameGeometry().topLeft()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_origin is not None and event.buttons() & Qt.MouseButton.LeftButton and not self.window().isMaximized():
            self.window().move(event.globalPosition().toPoint() - self._drag_origin)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._drag_origin = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.toggle_maximize()
        super().mouseDoubleClickEvent(event)


class WoodenBoard(QWidget):
    """Static wooden board with existing DeepTactics piece bitmaps."""
    move_requested = Signal(str)
    left_click_completed = Signal(str)

    pieces = {
        "a8": "black_rook",
        "b8": "black_knight",
        "c8": "black_bishop",
        "d8": "black_queen",
        "e8": "black_king",
        "f8": "black_bishop",
        "g8": "black_knight",
        "h8": "black_rook",
        "a7": "black_pawn",
        "b7": "black_pawn",
        "c7": "black_pawn",
        "d7": "black_pawn",
        "e7": "black_pawn",
        "f7": "black_pawn",
        "g7": "black_pawn",
        "h7": "black_pawn",
        "e5": "black_pawn",
        "e4": "white_pawn",
        "a2": "white_pawn",
        "b2": "white_pawn",
        "c2": "white_pawn",
        "d2": "white_pawn",
        "f2": "white_pawn",
        "g2": "white_pawn",
        "h2": "white_pawn",
        "a1": "white_rook",
        "b1": "white_knight",
        "c1": "white_bishop",
        "d1": "white_queen",
        "e1": "white_king",
        "f1": "white_bishop",
        "g1": "white_knight",
        "h1": "white_rook",
    }

    def __init__(
        self,
        starting_position: bool = False,
        minimum_size: int = 660,
        position: dict[str, str] | None = None,
        visual_theme: str = "Wood",
    ) -> None:
        super().__init__()
        self.setMinimumSize(minimum_size, minimum_size)
        self.starting_position = starting_position
        self.flipped = False
        self.display_pieces = dict(self.pieces)
        if starting_position:
            self.display_pieces.pop("e4", None)
            self.display_pieces.pop("e5", None)
            self.display_pieces["e2"] = "white_pawn"
        if position is not None:
            self.display_pieces = dict(position)
        self.suggested_arrow: tuple[str, str] | None = None
        self.last_move: tuple[str, str] | None = None
        self.interactive = False
        self.selected_square: str | None = None
        self.legal_moves: dict[str, list[str]] = {}
        self.marked_squares: set[str] = set()
        self.annotation_arrows: list[tuple[str, str]] = []
        self._right_drag_start: str | None = None
        self._left_press_square: str | None = None
        self._left_press_position: QPointF | None = None
        self._drag_square: str | None = None
        self._drag_position: QPointF | None = None
        self.last_input_was_drag = False
        self.move_classification: MoveClassification | None = None
        self._animation: tuple[str, str, str, float] | None = None
        self._animation_timer = QTimer(self)
        self._animation_timer.setInterval(16)
        self._animation_timer.timeout.connect(self.update)
        self.images = ASSET_MANAGER.piece_images()
        self.visual_theme = visual_theme
        self.show_coordinates = True

    def set_piece_folder(self, folder: str | Path) -> None:
        self.images = ASSET_MANAGER.piece_images(folder)
        self.update()

    def set_visual_theme(self, theme: str) -> None:
        self.visual_theme = theme if theme in {"Blue", "Wood", "Classic"} else "Wood"
        self.update()

    def set_coordinates_visible(self, visible: bool) -> None:
        self.show_coordinates = visible
        self.update()

    def _square_colors(self) -> tuple[QColor, QColor, QColor, QColor]:
        if self.visual_theme == "Blue":
            return (QColor("#c9d5df"), QColor("#54718c"), QColor("#9cbcc9"), QColor("#527f93"))
        if self.visual_theme == "Classic":
            return (QColor("#f0f0e8"), QColor("#59616b"), QColor("#d8d2a1"), QColor("#788050"))
        return (WOOD_LIGHT, WOOD_DARK, WOOD_LAST_LIGHT, WOOD_LAST_DARK)

    def set_position(self, position: dict[str, str]) -> None:
        self.starting_position = True
        self.display_pieces = dict(position)
        self.update()

    def set_fen(self, fen: str, last_move_uci: str | None = None) -> None:
        board = board_from_fen(fen)
        self.display_pieces = {
            square.to_algebraic(): f"{piece.color.value}_{piece.piece_type.value}"
            for square, piece in board.pieces()
        }
        self.starting_position = False
        self.last_move = (last_move_uci[:2], last_move_uci[2:4]) if last_move_uci else None
        self.suggested_arrow = None
        self.update()

    def set_move_classification(self, classification: MoveClassification | None) -> None:
        self.move_classification = classification
        self.update()

    def animate_move(self, move_uci: str) -> None:
        """Animate a completed click move while the authoritative FEN is already loaded."""
        if len(move_uci) < 4:
            return
        destination = move_uci[2:4]
        piece_name = self.display_pieces.get(destination)
        if piece_name is None:
            return
        self._animation = (move_uci[:2], destination, piece_name, time.monotonic())
        self._animation_timer.start()
        self.update()

    def set_interactive(self, interactive: bool) -> None:
        self.interactive = interactive

    def set_legal_moves(self, moves) -> None:
        legal: dict[str, list[str]] = {}
        for move in moves:
            legal.setdefault(move.from_square.to_algebraic(), []).append(move.to_uci())
        self.legal_moves = legal
        if self.selected_square not in legal:
            self.selected_square = None
        self.update()

    def clear_annotations(self) -> None:
        self.marked_squares.clear()
        self.annotation_arrows.clear()
        self.suggested_arrow = None
        self.update()

    def flip(self) -> None:
        self.flipped = not self.flipped
        self.update()

    def show_suggestion(self, start: str = "e2", end: str = "e4") -> None:
        self.suggested_arrow = (start, end)
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(672, 672)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        size = min(self.width(), self.height())
        cell = size / 8.0
        ox = (self.width() - size) / 2
        oy = (self.height() - size) / 2

        light, dark, last_light, last_dark = self._square_colors()
        for visual_rank in range(8):
            rank = visual_rank + 1 if self.flipped else 8 - visual_rank
            for file_index in range(8):
                file_name = chr(104 - file_index) if self.flipped else chr(97 + file_index)
                square = f"{file_name}{rank}"
                is_dark = (file_index + visual_rank) % 2 == 1
                base = dark if is_dark else light
                if self.last_move and square == self.last_move[0]:
                    base = last_light if not is_dark else last_dark
                elif self.last_move and square == self.last_move[1]:
                    base = last_dark if is_dark else last_light
                rect = QRectF(ox + file_index * cell, oy + visual_rank * cell, cell, cell)
                self._draw_wood_square(painter, rect, base, file_index, visual_rank)
                if square in self.marked_squares:
                    painter.fillRect(rect, QColor(188, 54, 52, 124))
                piece_name = self.display_pieces.get(square)
                hidden_piece = (
                    (self._drag_square is not None and square == self._drag_square)
                    or (self._animation is not None and square == self._animation[1])
                )
                if piece_name and piece_name in self.images and not hidden_piece:
                    margin = cell * 0.09
                    painter.drawPixmap(rect.adjusted(margin, margin, -margin, -margin).toRect(), self.images[piece_name])
                if (
                    self.last_move
                    and square == self.last_move[1]
                    and self.move_classification is not None
                    and self.move_classification.key in DISPLAYED_CLASSIFICATIONS
                ):
                    badge = QPixmap(str(ASSETS / "classifications" / f"{self.move_classification.key}.png"))
                    if not badge.isNull():
                        badge_size = cell * 0.34
                        badge_rect = QRectF(rect.right() - badge_size - cell * 0.04, rect.top() + cell * 0.04, badge_size, badge_size)
                        painter.drawPixmap(
                            badge_rect.toRect(),
                            badge.scaled(
                                int(badge_size),
                                int(badge_size),
                                Qt.AspectRatioMode.KeepAspectRatio,
                                Qt.TransformationMode.SmoothTransformation,
                            ),
                        )
                if self.selected_square and square in {move[2:4] for move in self.legal_moves.get(self.selected_square, [])}:
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(QColor(110, 112, 116, 145))
                    painter.drawEllipse(rect.center(), cell * 0.12, cell * 0.12)

        self._draw_moving_piece(painter, ox, oy, cell)
        for start, end in self.annotation_arrows:
            self._draw_arrow(painter, ox, oy, cell, start, end, QColor(242, 184, 26, 190))
        if self.suggested_arrow:
            self._draw_arrow(painter, ox, oy, cell, *self.suggested_arrow, QColor(60, 168, 255, 190))

        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        if not self.show_coordinates:
            return
        for index in range(8):
            bottom_dark = (index + 7) % 2 == 1
            left_dark = index % 2 == 1
            painter.setPen(QColor("#edcf9f") if bottom_dark else QColor("#674529"))
            painter.drawText(
                QRectF(ox + index * cell + 4, oy + size - 20, cell - 8, 16),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom,
                chr(104 - index) if self.flipped else chr(97 + index),
            )
            painter.setPen(QColor("#edcf9f") if left_dark else QColor("#674529"))
            painter.drawText(
                QRectF(ox + 5, oy + index * cell + 5, 18, 18),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                str(index + 1) if self.flipped else str(8 - index),
            )

    def _draw_moving_piece(self, painter: QPainter, ox: float, oy: float, cell: float) -> None:
        piece_name: str | None = None
        center: QPointF | None = None
        if self._drag_square is not None and self._drag_position is not None:
            piece_name = self.display_pieces.get(self._drag_square)
            center = self._drag_position
        elif self._animation is not None:
            start, end, piece_name, started = self._animation
            progress = min(1.0, (time.monotonic() - started) / 0.12)
            source = self._square_center(start, ox, oy, cell)
            target = self._square_center(end, ox, oy, cell)
            eased = 1.0 - (1.0 - progress) ** 3
            center = QPointF(
                source.x() + (target.x() - source.x()) * eased,
                source.y() + (target.y() - source.y()) * eased,
            )
            if progress >= 1.0:
                self._animation = None
                self._animation_timer.stop()
        image = self.images.get(piece_name or "")
        if image is None or center is None:
            return
        size = cell * 0.82
        painter.drawPixmap(
            QRectF(center.x() - size / 2, center.y() - size / 2, size, size).toRect(),
            image,
        )

    def _square_center(self, square: str, ox: float, oy: float, cell: float) -> QPointF:
        file_index = ord(square[0]) - ord("a")
        rank = int(square[1])
        visual_file = 7 - file_index if self.flipped else file_index
        visual_rank = rank - 1 if self.flipped else 8 - rank
        return QPointF(ox + (visual_file + 0.5) * cell, oy + (visual_rank + 0.5) * cell)

    def _draw_arrow(self, painter: QPainter, ox: float, oy: float, cell: float, start: str, end: str, color: QColor) -> None:
        route = self._arrow_route(start, end, ox, oy, cell)
        begin, finish = route[0], route[-1]
        previous = route[-2]
        dx, dy = finish.x() - previous.x(), finish.y() - previous.y()
        length = max(1.0, math.hypot(dx, dy))
        ux, uy = dx / length, dy / length
        first_dx, first_dy = route[1].x() - begin.x(), route[1].y() - begin.y()
        first_length = max(1.0, math.hypot(first_dx, first_dy))
        tail = QPointF(
            begin.x() + first_dx / first_length * cell * 0.20,
            begin.y() + first_dy / first_length * cell * 0.20,
        )
        tip = QPointF(finish.x(), finish.y())
        neck = QPointF(finish.x() - ux * cell * 0.40, finish.y() - uy * cell * 0.40)
        perp_x, perp_y = -uy, ux
        head_half = cell * 0.27
        shaft = QPainterPath(tail)
        for point in route[1:-1]:
            shaft.lineTo(point)
        shaft.lineTo(neck)
        head = QPainterPath()
        head.moveTo(neck.x() + perp_x * head_half, neck.y() + perp_y * head_half)
        head.lineTo(tip)
        head.lineTo(neck.x() - perp_x * head_half, neck.y() - perp_y * head_half)
        head.closeSubpath()
        painter.save()
        painter.setPen(QPen(color, cell * 0.21, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap, Qt.PenJoinStyle.RoundJoin))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(shaft)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawPath(head)
        painter.restore()

    def _arrow_route(self, start: str, end: str, ox: float, oy: float, cell: float) -> tuple[QPointF, ...]:
        begin = self._square_center(start, ox, oy, cell)
        finish = self._square_center(end, ox, oy, cell)
        file_delta = ord(end[0]) - ord(start[0])
        rank_delta = int(end[1]) - int(start[1])
        if sorted((abs(file_delta), abs(rank_delta))) != [1, 2]:
            return (begin, finish)
        # Knight guides make the long leg first, then turn toward the landing square.
        bend_square = f"{end[0]}{start[1]}" if abs(file_delta) == 2 else f"{start[0]}{end[1]}"
        return (begin, self._square_center(bend_square, ox, oy, cell), finish)

    def _square_at(self, position: QPointF) -> str | None:
        size = min(self.width(), self.height())
        cell = size / 8.0
        ox = (self.width() - size) / 2
        oy = (self.height() - size) / 2
        file_index = int((position.x() - ox) // cell)
        visual_rank = int((position.y() - oy) // cell)
        if not (0 <= file_index < 8 and 0 <= visual_rank < 8):
            return None
        file_name = chr(104 - file_index) if self.flipped else chr(97 + file_index)
        rank = visual_rank + 1 if self.flipped else 8 - visual_rank
        return f"{file_name}{rank}"

    def mousePressEvent(self, event) -> None:
        square = self._square_at(event.position())
        if square is None:
            return
        if event.button() == Qt.MouseButton.RightButton:
            self._right_drag_start = square
            return
        if event.button() != Qt.MouseButton.LeftButton or not self.interactive:
            return
        self.clear_annotations()
        self._left_press_square = square
        self._left_press_position = event.position()
        self._drag_square = None
        self._drag_position = None
        self.last_input_was_drag = False

    def mouseMoveEvent(self, event) -> None:
        if (
            not self.interactive
            or not (event.buttons() & Qt.MouseButton.LeftButton)
            or self._left_press_square is None
            or self._left_press_position is None
            or self._left_press_square not in self.legal_moves
        ):
            return
        distance = event.position() - self._left_press_position
        if self._drag_square is None and abs(distance.x()) + abs(distance.y()) < 8:
            return
        self._drag_square = self._left_press_square
        self._drag_position = event.position()
        self.last_input_was_drag = True
        self.update()

    def _handle_click_move(self, square: str) -> None:
        if self.selected_square:
            matching = [move for move in self.legal_moves.get(self.selected_square, []) if move[2:4] == square]
            if matching:
                self.selected_square = None
                self.move_requested.emit(self._choose_promotion_move(matching))
                return
        self.selected_square = square if square in self.legal_moves else None
        self.update()

    def _choose_promotion_move(self, matching: list[str]) -> str:
        promotion_moves = [move for move in matching if len(move) == 5]
        if len(promotion_moves) <= 1:
            return matching[0]
        queen = next((move for move in promotion_moves if move.endswith("q")), promotion_moves[0])
        if bool(SETTINGS_STORE.load().get("auto_promote_queen", False)):
            return queen
        options = {
            "Queen": queen,
            "Rook": next((move for move in promotion_moves if move.endswith("r")), queen),
            "Bishop": next((move for move in promotion_moves if move.endswith("b")), queen),
            "Knight": next((move for move in promotion_moves if move.endswith("n")), queen),
        }
        selected, accepted = QInputDialog.getItem(self, "Promotion", "Promote to:", list(options), editable=False)
        return options.get(selected, queen) if accepted else queen

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.interactive and self._left_press_square is not None:
            end = self._square_at(event.position())
            start = self._left_press_square
            dragged = self._drag_square is not None
            self._left_press_square = None
            self._left_press_position = None
            self._drag_square = None
            self._drag_position = None
            if end is not None and dragged:
                matching = [move for move in self.legal_moves.get(start, []) if move[2:4] == end]
                if matching:
                    self.selected_square = None
                    self.move_requested.emit(self._choose_promotion_move(matching))
                self.update()
                return
            if end is not None:
                self._handle_click_move(end)
                self.left_click_completed.emit(end)
            return
        if event.button() != Qt.MouseButton.RightButton or self._right_drag_start is None:
            return
        end = self._square_at(event.position())
        start = self._right_drag_start
        self._right_drag_start = None
        if end is None:
            return
        if start == end:
            if start in self.marked_squares:
                self.marked_squares.remove(start)
            else:
                self.marked_squares.add(start)
        else:
            arrow = (start, end)
            if arrow in self.annotation_arrows:
                self.annotation_arrows.remove(arrow)
            else:
                self.annotation_arrows.append(arrow)
        self.update()

    def _draw_wood_square(self, painter: QPainter, rect: QRectF, base: QColor, file_index: int, rank: int) -> None:
        del file_index, rank
        painter.fillRect(rect, base)


class CustomPositionDialog(QDialog):
    """Small position builder used only to create an Analyze FEN."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Custom Analysis Position")
        self.board = Board.empty()
        self.result_fen: str | None = None
        layout = QHBoxLayout(self)
        grid = QGridLayout()
        grid.setSpacing(1)
        self.squares: dict[str, QPushButton] = {}
        for visual_rank in range(8):
            rank = 8 - visual_rank
            for file_index in range(8):
                square = f"{chr(97 + file_index)}{rank}"
                cell = QPushButton()
                cell.setFixedSize(58, 58)
                cell.setProperty("dark", (visual_rank + file_index) % 2 == 1)
                cell.clicked.connect(lambda _checked=False, target=square: self.place_piece(target))
                grid.addWidget(cell, visual_rank, file_index)
                self.squares[square] = cell
        layout.addLayout(grid)

        controls = QVBoxLayout()
        controls.addWidget(text_label("CUSTOM POSITION", "sectionHeader"))
        self.piece_choice = QComboBox()
        self.piece_choice.addItem("Erase", None)
        for color, color_text in ((Color.WHITE, "White"), (Color.BLACK, "Black")):
            for piece_type, name in (
                (PieceType.KING, "King"), (PieceType.QUEEN, "Queen"), (PieceType.ROOK, "Rook"),
                (PieceType.BISHOP, "Bishop"), (PieceType.KNIGHT, "Knight"), (PieceType.PAWN, "Pawn"),
            ):
                self.piece_choice.addItem(f"{color_text} {name}", Piece(color, piece_type))
        controls.addWidget(self.piece_choice)
        self.side_to_move = QComboBox()
        self.side_to_move.addItems(["White to move", "Black to move"])
        controls.addWidget(self.side_to_move)
        controls.addWidget(text_label("Castling rights", "caps"))
        self.castling: dict[str, QCheckBox] = {}
        for key, title in (("K", "White O-O"), ("Q", "White O-O-O"), ("k", "Black O-O"), ("q", "Black O-O-O")):
            control = QCheckBox(title)
            self.castling[key] = control
            controls.addWidget(control)
        controls.addStretch()
        clear = button("Clear Board", compact=True)
        clear.clicked.connect(self.clear_board)
        controls.addWidget(clear)
        actions = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel)
        actions.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.apply_position)
        actions.rejected.connect(self.reject)
        controls.addWidget(actions)
        layout.addLayout(controls)

    def place_piece(self, square_name: str) -> None:
        piece = self.piece_choice.currentData()
        self.board.set_piece(Square.from_algebraic(square_name), piece)
        self._refresh_square(square_name)

    def clear_board(self) -> None:
        self.board = Board.empty()
        for square_name in self.squares:
            self._refresh_square(square_name)

    def _refresh_square(self, square_name: str) -> None:
        control = self.squares[square_name]
        piece = self.board.get_piece(Square.from_algebraic(square_name))
        control.setIcon(QIcon())
        if piece is not None:
            key = f"{piece.color.value}_{piece.piece_type.value}"
            image = ASSET_MANAGER.piece_images().get(key)
            if image is not None:
                control.setIcon(QIcon(image))
                control.setIconSize(QSize(50, 50))

    def apply_position(self) -> None:
        pieces = list(self.board.pieces())
        for color, name in ((Color.WHITE, "white"), (Color.BLACK, "black")):
            kings = [piece for _square, piece in pieces if piece.color is color and piece.piece_type is PieceType.KING]
            if len(kings) != 1:
                QMessageBox.warning(self, "Invalid Position", f"The custom position needs exactly one {name} king.")
                return
        self.board.side_to_move = Color.WHITE if self.side_to_move.currentIndex() == 0 else Color.BLACK
        self.board.castling_rights = CastlingRights(
            self.castling["K"].isChecked(),
            self.castling["Q"].isChecked(),
            self.castling["k"].isChecked(),
            self.castling["q"].isChecked(),
        )
        self.result_fen = board_to_fen(self.board)
        self.accept()


class EvalMeter(QWidget):
    def __init__(self, neutral: bool = False, minimum_height: int = 660) -> None:
        super().__init__()
        self.neutral = neutral
        self.active = not neutral
        self.white_percentage = 50.0
        self._animation = QVariantAnimation(self)
        self._animation.setDuration(180)
        self._animation.valueChanged.connect(self._set_animated_percentage)
        self.setFixedWidth(35)
        self.setMinimumHeight(minimum_height)

    def set_score(self, score_cp_white: int | None = None, mate_white: int | None = None) -> None:
        if not self.active:
            self._animate_to(50.0)
            return
        if mate_white is not None:
            target = 100.0 if mate_white > 0 else 0.0
        elif score_cp_white is None:
            target = 50.0
        elif abs(score_cp_white) >= 90_000:
            target = 100.0 if score_cp_white > 0 else 0.0
        else:
            bounded_cp = max(-1000, min(1000, score_cp_white))
            target = max(
                0.0,
                min(100.0, 50 + 50 * (2 / (1 + math.exp(-0.004 * bounded_cp)) - 1)),
            )
        self._animate_to(target)

    def set_active(self, active: bool) -> None:
        self.active = active
        if not active:
            self._animate_to(50.0)
        else:
            self.update()

    def _animate_to(self, target: float) -> None:
        self._animation.stop()
        self._animation.setStartValue(self.white_percentage)
        self._animation.setEndValue(target)
        self._animation.start()

    def _set_animated_percentage(self, value) -> None:
        self.white_percentage = float(value)
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        bar = QRectF(6, 0, self.width() - 12, self.height())
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#25303b") if self.active else QColor("#1c2630"))
        painter.drawRoundedRect(bar, 3, 3)
        split_y = bar.top() + bar.height() * (1.0 - self.white_percentage / 100.0)
        painter.setBrush(QColor("#eff1ee") if self.active else QColor("#697481"))
        painter.drawRoundedRect(QRectF(bar.left(), split_y, bar.width(), bar.bottom() - split_y), 3, 3)
        painter.setPen(QPen(QColor("#536477"), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(bar, 3, 3)
        painter.setPen(QPen(QColor("#96a3b2") if self.active else QColor("#536171"), 1))
        painter.drawLine(QPointF(bar.left() + 1, split_y), QPointF(bar.right() - 1, split_y))
        painter.drawLine(QPointF(bar.left() + 1, bar.center().y()), QPointF(bar.right() - 1, bar.center().y()))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(BLUE_LINE))
        marker_y = split_y
        painter.drawPolygon(
            [
                QPointF(bar.right() + 1, marker_y),
                QPointF(bar.right() + 8, marker_y - 5),
                QPointF(bar.right() + 8, marker_y + 5),
            ]
        )
        if not self.active:
            painter.setBrush(QColor(86, 101, 118))
            painter.drawPolygon(
                [
                    QPointF(bar.right() + 1, marker_y),
                    QPointF(bar.right() + 8, marker_y - 5),
                    QPointF(bar.right() + 8, marker_y + 5),
                ]
            )


class BoardWorkspace(Card):
    new_requested = Signal()
    load_pgn_requested = Signal()
    save_pgn_requested = Signal()
    undo_requested = Signal()
    custom_requested = Signal()
    start_play_requested = Signal()
    pause_play_requested = Signal(bool)
    eval_toggled = Signal(bool)

    def __init__(self, play_screen: bool = False) -> None:
        super().__init__("transparent")
        self.play_screen = play_screen
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(12)
        board_card = Card("boardCard")
        board_layout = QHBoxLayout(board_card)
        board_layout.setContentsMargins(22, 21, 16, 21)
        board_layout.setSpacing(14)
        center = QHBoxLayout()
        center.setSpacing(14)
        self.board = WoodenBoard(starting_position=play_screen)
        center.addWidget(self.board, 1)
        self.eval_meter = EvalMeter(neutral=play_screen)
        center.addWidget(self.eval_meter)
        board_layout.addLayout(center, 1)
        outer.addWidget(board_card, 1)

        action_card = Card("toolbarCard")
        actions = QHBoxLayout()
        actions.setContentsMargins(10, 8, 10, 8)
        actions.setSpacing(8)
        new_game = button("\u25b6  Start Play" if play_screen else "\u2693  New Game")
        self.new_button = new_game
        if play_screen:
            new_game.setToolTip("Apply the selected players and time control, then begin a new game.")
            new_game.clicked.connect(self.start_play_requested.emit)
        else:
            new_game.clicked.connect(self.new_requested.emit)
        actions.addWidget(new_game)
        load_pgn = button("\u25a1  Load PGN")
        self.load_button = load_pgn
        load_pgn.clicked.connect(self.load_pgn_requested.emit)
        actions.addWidget(load_pgn)
        save_pgn = button("\u25a3  Save PGN")
        self.save_button = save_pgn
        save_pgn.clicked.connect(self.save_pgn_requested.emit)
        actions.addWidget(save_pgn)
        flip = button("\u21bb  Flip Board")
        flip.clicked.connect(self.board.flip)
        actions.addWidget(flip)
        undo = button("\u21b6  Undo")
        self.undo_button = undo
        undo.clicked.connect(self.undo_requested.emit)
        if play_screen:
            undo.setEnabled(False)
            undo.setToolTip("Enabled after the Play game controller is connected.")
        else:
            undo.setToolTip("Undo the most recent exploratory analysis move.")
        actions.addWidget(undo)
        tail = button("\u23f8  Pause" if play_screen else "\u2699  Custom")
        if play_screen:
            tail.setToolTip("Pause or resume the game.")
            tail.setCheckable(True)
            tail.setEnabled(False)
            tail.toggled.connect(self.pause_play_requested.emit)
        else:
            tail.setToolTip("Build a custom legal analysis position.")
            tail.clicked.connect(self.custom_requested.emit)
        self.tail_action = tail
        actions.addWidget(tail)
        action_card.setLayout(actions)
        outer.addWidget(action_card)


class EngineStatus(Card):
    engine_changed = Signal(str)
    analyze_requested = Signal()
    stop_requested = Signal()
    best_move_requested = Signal()
    depth_changed = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        self.setFixedHeight(220)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 12, 20, 10)
        layout.setSpacing(10)

        overview = QHBoxLayout()
        overview.setSpacing(20)
        engine_block = QVBoxLayout()
        head = QHBoxLayout()
        head.setSpacing(9)
        head.addWidget(text_label("ENGINE", "caps"))
        self.indicator = QLabel()
        self.indicator.setFixedSize(8, 8)
        self.indicator.setObjectName("greenDot")
        self.indicator.setProperty("error", False)
        head.addWidget(self.indicator)
        self.connection_state = text_label("Not connected", "secondary")
        head.addWidget(self.connection_state)
        head.addStretch()
        engine_block.addLayout(head)
        name_line = QHBoxLayout()
        name_line.setSpacing(5)
        self.engine_name = text_label("DeepTactics", "engineName")
        self.engine_name.setFixedHeight(27)
        self.engine_name.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        name_line.addWidget(self.engine_name)
        self.version = text_label(APP_VERSION, "engineVersion")
        self.version.setFixedHeight(27)
        self.version.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        name_line.addWidget(self.version)
        self.nnue = text_label("NNUE", "nnueBadge")
        self.nnue.setProperty("active", False)
        self.nnue.setFixedSize(55, 27)
        self.nnue.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_line.addWidget(self.nnue)
        name_line.addStretch()
        engine_block.addLayout(name_line)
        self.ready_state = text_label("Ready for UCI connection", "secondary")
        state_row = QHBoxLayout()
        state_row.addWidget(self.ready_state, 1)
        engine_block.addLayout(state_row)
        overview.addLayout(engine_block, 1)
        actions = QGridLayout()
        actions.setHorizontalSpacing(10)
        actions.setVerticalSpacing(10)
        self.analyze_button = button("\u2315  Analyze", compact=True)
        self.analyze_button.setCheckable(True)
        self.analyze_button.clicked.connect(self.analyze_requested.emit)
        self.stop_button = button("\u25a0  Stop", compact=True)
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_requested.emit)
        best = button("\u2726  Find Best Move", compact=True)
        best.clicked.connect(self.best_move_requested.emit)
        change = button("\u2699  Change Engine", compact=True)
        change.clicked.connect(self.choose_engine)
        depth_box = QWidget()
        depth_layout = QHBoxLayout(depth_box)
        depth_layout.setContentsMargins(7, 0, 5, 0)
        depth_layout.setSpacing(4)
        depth_layout.addWidget(text_label("Depth", "muted"))
        self.depth_selector = QSpinBox()
        self.depth_selector.setObjectName("interactiveField")
        self.depth_selector.setRange(1, 30)
        self.depth_selector.setValue(int(SETTINGS_STORE.load().get("default_analysis_depth", 4)))
        self.depth_selector.setFixedSize(62, 34)
        self.depth_selector.setToolTip("Analysis depth used for review and live position analysis.")
        self.depth_selector.valueChanged.connect(self.depth_changed.emit)
        depth_layout.addWidget(self.depth_selector)
        actions.addWidget(self.analyze_button, 0, 0)
        actions.addWidget(self.stop_button, 0, 1)
        actions.addWidget(best, 1, 0)
        actions.addWidget(depth_box, 1, 1)
        actions.addWidget(change, 2, 0, 1, 2)
        overview.addLayout(actions)
        layout.addLayout(overview)

        stats = QGridLayout()
        stats.setHorizontalSpacing(25)
        stats.setVerticalSpacing(2)
        entries = [
            ("Depth", "--"), ("Nodes", "--"), ("NPS", "--"), ("Time", "--"),
            ("Hash", "--"), ("Threads", "--"), ("TB Hits", "--"), ("Score", "--"),
        ]
        self.stat_values: dict[str, QLabel] = {}
        for index, (key, value) in enumerate(entries):
            col = index % 4
            row = index // 4 * 2
            stats.addWidget(text_label(key, "muted"), row, col)
            value_label = text_label(value, "statValue")
            self.stat_values[key] = value_label
            stats.addWidget(value_label, row + 1, col)
        self.stat_values["Hash"].setToolTip("Configured UCI Hash value, when provided by this engine profile.")
        self.stat_values["Threads"].setToolTip("Configured UCI Threads value, not a live UCI info metric.")
        self.stat_values["TB Hits"].setToolTip("Shown only when the engine reports UCI tbhits.")
        layout.addLayout(stats)
        line = QFrame()
        line.setObjectName("progress")
        line.setFixedHeight(3)
        layout.addWidget(line)
        self.set_engine("DeepTactics", notify=False)

    def choose_engine(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Change Analysis Engine")
        layout = QVBoxLayout(dialog)
        layout.addWidget(text_label("Installed engines", "sectionHeader"))
        engines = QListWidget()
        engines.addItems(ENGINE_REGISTRY.names())
        engines.setCurrentRow(max(0, ENGINE_REGISTRY.names().index(self.engine_name.text()) if self.engine_name.text() in ENGINE_REGISTRY.names() else 0))
        layout.addWidget(engines)
        controls = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        controls.accepted.connect(dialog.accept)
        controls.rejected.connect(dialog.reject)
        layout.addWidget(controls)
        if dialog.exec() == QDialog.DialogCode.Accepted and engines.currentItem():
            selected = engines.currentItem().text()
            self.set_engine(selected)

    def set_engine(self, name: str, *, notify: bool = True) -> None:
        self.engine_name.setText(name)
        native = name == "DeepTactics"
        self.version.setText(APP_VERSION if native else "UCI")
        self.nnue.setProperty("active", ENGINE_REGISTRY.supports_nnue(name))
        self.nnue.setToolTip("NNUE available" if ENGINE_REGISTRY.supports_nnue(name) else "NNUE not reported by this engine")
        self.nnue.style().unpolish(self.nnue)
        self.nnue.style().polish(self.nnue)
        entry = ENGINE_REGISTRY.entry(name) or {}
        self.stat_values["Hash"].setText(f'{entry.get("hash", "--")} MB' if entry.get("hash") is not None else "--")
        self.stat_values["Threads"].setText(str(entry.get("threads", "--")))
        self.stat_values["TB Hits"].setText("--")
        if notify:
            self.engine_changed.emit(name)

    def set_state(self, state: str, session_active: bool = False) -> None:
        self.connection_state.setText(state)
        error = state == "Error"
        self.indicator.setProperty("error", error)
        self.indicator.style().unpolish(self.indicator)
        self.indicator.style().polish(self.indicator)
        if error:
            detail = "Engine error"
        elif session_active and state.startswith("Reviewing"):
            detail = "Analyzing original game for graph and move review"
        elif session_active and state in {"Starting", "Connecting", "Analyzing", "Stopping"}:
            detail = "Analyzing current position"
        elif session_active:
            detail = "Analysis active - monitoring position"
        else:
            detail = "Ready"
        self.ready_state.setText(detail)
        self.analyze_button.setChecked(session_active)
        self.analyze_button.setEnabled(not session_active and not error)
        self.stop_button.setEnabled(session_active)

    def update_analysis(self, info: UCIAnalysisInfo) -> None:
        score = format_evaluation(info.score_cp_white, info.mate_white)
        self.stat_values["Depth"].setText(str(info.depth or "--"))
        self.stat_values["Nodes"].setText(f"{info.nodes:,}" if info.nodes else "--")
        self.stat_values["NPS"].setText(f"{info.nps:,}" if info.nps else "--")
        self.stat_values["Time"].setText(f"{info.time_ms / 1000:.2f}s" if info.time_ms else "--")
        self.stat_values["TB Hits"].setText(f"{info.tbhits:,}" if info.tbhits is not None else "--")
        self.stat_values["Score"].setText(score)

    def begin_position_analysis(self) -> None:
        for key in ("Depth", "Nodes", "NPS", "Time", "Score"):
            self.stat_values[key].setText("--")


class MoveLedger(Card):
    ply_selected = Signal(int)
    variation_selected = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.current_ply = 0
        self.total_plies = 0
        self.moves: tuple[str, ...] = ()
        self.variation_moves: tuple[str, ...] = ()
        self.variation_base_ply: int | None = None
        self.classifications: dict[int, MoveClassification] = {}
        self.classification_tooltips: dict[int, str] = {}
        self._pending_scroll_value: int | None = None
        self.move_buttons: list[QPushButton] = []
        self.badges: list[QWidget] = []
        self.variation_slots: list[QPushButton] = []
        # Match the right-hand INFO/PV/GRAPH stack height so the moves card
        # leaves the same breathing room above the position metadata card.
        self.setFixedHeight(388)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 13, 15, 13)
        layout.setSpacing(4)
        layout.addWidget(text_label("MOVES", "caps"))
        self.scroll = QScrollArea()
        self.scroll.setObjectName("movesScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll.verticalScrollBar().rangeChanged.connect(self._restore_pending_scroll)
        self.moves_container = QWidget()
        self.moves_grid = QGridLayout(self.moves_container)
        self.moves_grid.setContentsMargins(0, 2, 4, 2)
        self.moves_grid.setHorizontalSpacing(5)
        self.moves_grid.setVerticalSpacing(4)
        self.moves_grid.setColumnMinimumWidth(0, 27)
        self.moves_grid.setColumnMinimumWidth(1, 48)
        self.moves_grid.setColumnMinimumWidth(2, 17)
        self.moves_grid.setColumnMinimumWidth(3, 49)
        self.moves_grid.setColumnMinimumWidth(4, 17)
        self.moves_grid.setColumnStretch(5, 1)
        self.scroll.setWidget(self.moves_container)
        layout.addWidget(self.scroll, 1)
        self.variation: QPushButton | None = None
        controls = QHBoxLayout()
        controls.setSpacing(6)
        for glyph, tooltip, target in (
            ("\u23ee", "First move", lambda: self.select_ply(0)),
            ("\u2039", "Previous move", lambda: self.select_ply(self.current_ply - 1)),
            ("\u203a", "Next move", lambda: self.select_ply(self.current_ply + 1)),
            ("\u23ed", "Last move", lambda: self.select_ply(self.total_plies)),
        ):
            control = button(glyph, compact=True)
            control.setFixedHeight(39)
            control.setToolTip(tooltip)
            control.clicked.connect(target)
            controls.addWidget(control)
        layout.addLayout(controls)
        self._rebuild_rows()

    def select_ply(self, ply: int) -> None:
        self.current_ply = max(0, min(self.total_plies, ply))
        self.ply_selected.emit(self.current_ply)

    def set_timeline(
        self,
        moves: tuple[str, ...],
        current_ply: int,
        variation_moves: tuple[str, ...] = (),
        variation_base_ply: int | None = None,
    ) -> None:
        if moves == self.moves:
            self.variation_moves = variation_moves
            self.variation_base_ply = variation_base_ply
            self._update_variation()
            self.set_current_ply(current_ply)
            return
        scroll_value = self.scroll.verticalScrollBar().value()
        self.moves = moves
        self.total_plies = len(moves)
        self.variation_moves = variation_moves
        self.variation_base_ply = variation_base_ply
        self._rebuild_rows()
        self._update_variation()
        self.set_current_ply(current_ply, ensure_visible=False)
        self._queue_scroll_restore(scroll_value)

    def set_classifications(
        self,
        classifications: dict[int, MoveClassification],
        tooltips: dict[int, str] | None = None,
    ) -> None:
        self.classifications = dict(classifications)
        self.classification_tooltips = dict(tooltips or {})
        for ply, badge in enumerate(self.badges, start=1):
            set_move_badge(badge, self.classifications.get(ply), self.classification_tooltips.get(ply, ""))
        for ply, control in enumerate(self.move_buttons, start=1):
            control.setToolTip(self.classification_tooltips.get(ply, ""))

    def set_current_ply(self, ply: int, *, ensure_visible: bool = True) -> None:
        self.current_ply = max(0, min(self.total_plies, ply))
        for index, control in enumerate(self.move_buttons, start=1):
            control.setProperty("selected", index == self.current_ply)
            control.style().unpolish(control)
            control.style().polish(control)
        if ensure_visible and 1 <= self.current_ply <= len(self.move_buttons):
            self.scroll.ensureWidgetVisible(self.move_buttons[self.current_ply - 1], 4, 8)

    def _rebuild_rows(self) -> None:
        self.moves_container.setUpdatesEnabled(False)
        while self.moves_grid.count():
            item = self.moves_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.deleteLater()
        self.move_buttons.clear()
        self.badges.clear()
        self.variation_slots.clear()
        full_move_rows = max(1, (len(self.moves) + 1) // 2)
        grid_row = 0
        for slot_index in range(full_move_rows + 1):
            variation_slot = QPushButton()
            variation_slot.setObjectName("ledgerVariation")
            variation_slot.setToolTip("View the exploratory line")
            variation_slot.setVisible(False)
            variation_slot.clicked.connect(lambda _checked=False: self.variation_selected.emit())
            self.variation_slots.append(variation_slot)
            self.moves_grid.addWidget(variation_slot, grid_row, 0, 1, 5)
            grid_row += 1
            if slot_index == full_move_rows:
                break
            move_index = slot_index
            number = text_label(f"{move_index + 1}.", "muted")
            number.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.moves_grid.addWidget(number, grid_row, 0)
            for column, ply in ((1, move_index * 2 + 1), (3, move_index * 2 + 2)):
                move = self.moves[ply - 1] if ply <= len(self.moves) else ""
                control = QPushButton(move)
                control.setObjectName("ledgerMove")
                control.setEnabled(bool(move))
                control.setToolTip(self.classification_tooltips.get(ply, ""))
                control.clicked.connect(lambda _checked=False, selected=ply: self.select_ply(selected))
                self.move_buttons.append(control)
                self.moves_grid.addWidget(control, grid_row, column)
                badge = move_badge("")
                self.badges.append(badge)
                set_move_badge(badge, self.classifications.get(ply), self.classification_tooltips.get(ply, ""))
                self.moves_grid.addWidget(badge, grid_row, column + 1)
            grid_row += 1
        self.moves_grid.setRowStretch(grid_row, 1)
        self.moves_container.setUpdatesEnabled(True)
        self._update_variation()
        self.moves_container.update()

    def _update_variation(self) -> None:
        for slot in self.variation_slots:
            slot.setVisible(False)
        if not self.variation_moves or not self.variation_slots:
            return
        base_ply = self.variation_base_ply or 0
        prefix = f"{base_ply // 2 + 1}. " if base_ply % 2 == 0 else f"{base_ply // 2 + 1}... "
        branch_after = min((base_ply + 1) // 2, len(self.variation_slots) - 1)
        self.variation = self.variation_slots[branch_after]
        self.variation.setText(f"Try: {prefix}{'  '.join(self.variation_moves)}")
        self.variation.setVisible(True)

    def _queue_scroll_restore(self, value: int) -> None:
        self._pending_scroll_value = value if value > 0 else None
        if value == 0:
            self.scroll.verticalScrollBar().setValue(0)
            return
        QTimer.singleShot(0, self._restore_pending_scroll)

    def _restore_pending_scroll(self, *_range) -> None:
        if self._pending_scroll_value is None:
            return
        bar = self.scroll.verticalScrollBar()
        if bar.maximum() <= 0:
            return
        bar.setValue(min(self._pending_scroll_value, bar.maximum()))
        self._pending_scroll_value = None
        self.scroll.viewport().update()


class InfoSummary(Card):
    def __init__(self) -> None:
        super().__init__()
        self.candidate_moves: list[QLabel] = []
        self.candidate_scores: list[QLabel] = []
        self.setFixedHeight(126)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 7)
        layout.setSpacing(3)
        self.tabs = QTabBar()
        for title in ("INFO", "BEST LINE (PV)", "CANDIDATES"):
            self.tabs.addTab(title)
        layout.addWidget(self.tabs)
        self.stack = QStackedWidget()
        self.stack.addWidget(self._info())
        self.stack.addWidget(self._line())
        self.stack.addWidget(self._candidates())
        self.tabs.currentChanged.connect(self.stack.setCurrentIndex)
        layout.addWidget(self.stack, 1)

    def _info(self) -> QWidget:
        panel = QWidget()
        grid = QGridLayout(panel)
        grid.setContentsMargins(20, 6, 18, 0)
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(2)
        grid.addWidget(text_label("Evaluation", "muted"), 0, 0)
        self.description = text_label("No analysis loaded", "analysisText")
        self.score = text_label("--", "evalScore")
        self.depth = text_label("Depth\n--", "detail")
        self.mate = text_label("Mate\n\u2014", "detail")
        self.time = text_label("Time\n--", "detail")
        grid.addWidget(self.description, 0, 1, 1, 3)
        grid.addWidget(self.score, 1, 0)
        grid.addWidget(self.depth, 1, 1)
        grid.addWidget(self.mate, 1, 2)
        grid.addWidget(self.time, 1, 3)
        self.score_unit = text_label("Evaluation", "muted")
        grid.addWidget(self.score_unit, 2, 0)
        return panel

    def _line(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 6, 18, 0)
        layout.addWidget(text_label("Principal variation at current position", "muted"))
        self.line = text_label("Waiting for UCI principal variation.", "pv")
        self.line.setWordWrap(True)
        layout.addWidget(self.line)
        return panel

    def update_analysis(self, info: UCIAnalysisInfo) -> None:
        if info.mate_white is not None:
            score = format_evaluation(mate_white=info.mate_white)
            description = "Forced mate for White" if info.mate_white > 0 else "Forced mate for Black"
            self.score_unit.setText("Forced mate")
        elif info.score_cp_white is not None:
            score = format_evaluation(score_cp_white=info.score_cp_white)
            description = "White is better" if info.score_cp_white > 35 else (
                "Black is better" if info.score_cp_white < -35 else "Position is approximately equal"
            )
            self.score_unit.setText("Evaluation")
        else:
            score = "--"
            description = "Analyzing..."
            self.score_unit.setText("Evaluation")
        self.score.setText(score)
        self.description.setText(description)
        self.depth.setText(f"Depth\n{info.depth or '--'}")
        self.mate.setText(f"Mate\n{format_evaluation(mate_white=info.mate_white)}" if info.mate_white is not None else "Mate\n\u2014")
        self.time.setText(f"Time\n{info.time_ms / 1000:.2f}s" if info.time_ms else "Time\n--")
        self.line.setText(" ".join(info.pv) if info.pv else "Waiting for UCI principal variation.")

    def _candidates(self) -> QWidget:
        panel = QWidget()
        grid = QGridLayout(panel)
        grid.setContentsMargins(20, 5, 18, 0)
        for row in range(3):
            move = text_label(f"{row + 1}.  --", "statValue")
            score = text_label("--", "blueTextLarge")
            self.candidate_moves.append(move)
            self.candidate_scores.append(score)
            grid.addWidget(move, row, 0)
            grid.addWidget(score, row, 1)
            grid.addWidget(text_label("Principal line" if row == 0 else "Alternative", "muted"), row, 2)
        return panel

    def reset_candidates(self) -> None:
        for row, (move, score) in enumerate(zip(self.candidate_moves, self.candidate_scores), start=1):
            move.setText(f"{row}.  --")
            score.setText("--")

    def begin_position_analysis(self) -> None:
        self.score.setText("--")
        self.description.setText("Analyzing current position...")
        self.depth.setText("Depth\n--")
        self.mate.setText("Mate\n\u2014")
        self.time.setText("Time\n--")
        self.score_unit.setText("Evaluation")
        self.line.setText("Waiting for UCI principal variation.")
        self.reset_candidates()

    def update_candidate(self, info: UCIAnalysisInfo) -> None:
        index = info.multipv - 1
        if not (0 <= index < len(self.candidate_moves)):
            return
        move = info.pv[0] if info.pv else "--"
        score = format_evaluation(info.score_cp_white, info.mate_white)
        self.candidate_moves[index].setText(f"{info.multipv}.  {move}")
        self.candidate_scores[index].setText(score)


class PrincipalVariation(Card):
    suggestion_toggled = Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        self.setFixedHeight(78)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 11, 18, 12)
        row = QHBoxLayout()
        row.addWidget(text_label("BEST LINE (PV)", "caps"))
        row.addStretch()
        self.suggestion = QToolButton()
        self.suggestion.setText("Show Arrow")
        self.suggestion.setToolTip("Show the engine's current best move on the board")
        self.suggestion.setCheckable(True)
        self.suggestion.setObjectName("pvArrowToggle")
        self.suggestion.toggled.connect(self._toggle_suggestion)
        row.addWidget(self.suggestion)
        layout.addLayout(row)
        self.pv = text_label("Waiting for UCI analysis result.", "pv")
        self.pv.setWordWrap(True)
        self.pv.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.pv, 1)

    def set_pv(self, moves: tuple[str, ...]) -> None:
        self.pv.setText(" ".join(moves) if moves else "Waiting for UCI analysis result.")

    def _toggle_suggestion(self, checked: bool) -> None:
        self.suggestion.setText("Arrow On" if checked else "Show Arrow")
        self.suggestion_toggled.emit(checked)


class EvaluationGraph(Card):
    def __init__(self) -> None:
        super().__init__()
        self.setFixedHeight(160)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 10, 18, 6)
        heading = QHBoxLayout()
        heading.addWidget(text_label("EVALUATION GRAPH", "caps"))
        heading.addStretch()
        self.selected = text_label("Ply 24  |  Click to view", "muted")
        heading.addWidget(self.selected)
        layout.addLayout(heading)
        self.canvas = GraphCanvas()
        self.canvas.ply_selected.connect(lambda ply: self.selected.setText(f"Ply {ply}  |  Selected position"))
        layout.addWidget(self.canvas, 1)

    def select_ply(self, ply: int) -> None:
        self.canvas.selected_ply = max(0, min(self.canvas.total_plies, ply))
        self.selected.setText(f"Ply {self.canvas.selected_ply}  |  Selected position")
        self.canvas.update()

    def set_total_plies(self, total_plies: int) -> None:
        self.canvas.total_plies = max(1, total_plies)
        self.select_ply(min(self.canvas.selected_ply, total_plies))

    def set_score(self, ply: int, score_cp_white: int | None, mate_white: int | None) -> None:
        if mate_white is not None:
            value = 10.0 if mate_white > 0 else -10.0
        elif score_cp_white is not None:
            value = max(-10.0, min(10.0, score_cp_white / 100.0))
        else:
            return
        self.canvas.scores[ply] = value
        self.canvas.update()

    def clear_scores(self) -> None:
        self.canvas.scores.clear()
        self.canvas.update()


class GraphCanvas(QWidget):
    ply_selected = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        self.total_plies = 90
        self.selected_ply = 24
        self.hovered_ply: int | None = None
        self.scores: dict[int, float] = {}
        self.setMouseTracking(True)

    def _plot(self):
        return self.rect().adjusted(42, 8, -6, -27)

    def _ply_at(self, x: float) -> int:
        plot = self._plot()
        ratio = max(0.0, min(1.0, (x - plot.left()) / max(1, plot.width())))
        return int(round(ratio * self.total_plies))

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        plot = self._plot()
        painter.setFont(QFont("Segoe UI", 9))
        painter.setPen(QPen(QColor("#243344"), 1))
        for i in range(5):
            y = plot.top() + plot.height() * i / 4
            painter.drawLine(plot.left(), int(y), plot.right(), int(y))
            painter.setPen(QColor(MUTED))
            painter.drawText(QRectF(0, y - 8, 35, 16), Qt.AlignmentFlag.AlignRight, f"{10 - i * 5:+.0f}")
            painter.setPen(QPen(QColor("#243344"), 1))
        for i in range(10):
            x = plot.left() + plot.width() * i / 9
            painter.drawLine(int(x), plot.top(), int(x), plot.bottom())

        center = plot.top() + plot.height() / 2
        points = []
        for ply, value in sorted(self.scores.items()):
            x = plot.left() + plot.width() * ply / self.total_plies
            y = center - value / 10 * plot.height() / 2
            points.append(QPointF(x, y))
        if points:
            path = QPainterPath(points[0])
            fill = QPainterPath(QPointF(points[0].x(), center))
            fill.lineTo(points[0])
            for point in points[1:]:
                path.lineTo(point)
                fill.lineTo(point)
            fill.lineTo(points[-1].x(), center)
            fill.closeSubpath()
            painter.fillPath(fill, QColor(24, 111, 233, 90))
            painter.setPen(QPen(QColor(BLUE_LINE), 2))
            painter.drawPath(path)
            painter.setBrush(QColor(BLUE_LINE))
            painter.drawEllipse(points[-1], 3, 3)
        marker_ply = self.hovered_ply if self.hovered_ply is not None else self.selected_ply
        selected_x = plot.left() + plot.width() * marker_ply / self.total_plies
        painter.setPen(QPen(QColor(BLUE_LINE), 2))
        painter.drawLine(int(selected_x), plot.top(), int(selected_x), plot.bottom() + 10)
        painter.setBrush(QColor(BLUE))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(QRectF(selected_x - 13, plot.bottom() + 2, 27, 18), 3, 3)
        painter.setPen(QColor("#ffffff"))
        painter.drawText(QRectF(selected_x - 13, plot.bottom() + 3, 27, 16), Qt.AlignmentFlag.AlignCenter, str(marker_ply))

    def mouseMoveEvent(self, event) -> None:
        self.hovered_ply = self._ply_at(event.position().x())
        self.update()

    def leaveEvent(self, _event) -> None:
        self.hovered_ply = None
        self.update()

    def mousePressEvent(self, event) -> None:
        self.selected_ply = self._ply_at(event.position().x())
        self.hovered_ply = None
        self.ply_selected.emit(self.selected_ply)
        self.update()


def _format_san_line(moves: tuple[str, ...]) -> str:
    return "  ".join(
        f"{index // 2 + 1}. {moves[index]}" + (f" {moves[index + 1]}" if index + 1 < len(moves) else "")
        for index in range(0, len(moves), 2)
    )


def _opening_label(moves: tuple[str, ...]) -> str:
    if not moves:
        return "Starting Position"
    name = detect_opening(list(moves), OPENING_BOOK_PATH if OPENING_BOOK_PATH.exists() else None)
    return name.replace("Opening: unknown", "Opening not identified")


def _pv_to_san(fen: str, moves: tuple[str, ...]) -> tuple[str, ...]:
    game = Game(board=board_from_fen(fen))
    san: list[str] = []
    for move_text in moves:
        try:
            game.apply_move(Move.from_uci(move_text))
        except ValueError:
            return moves
        san.append(game.san_history[-1])
    return tuple(san)


class PositionInfo(Card):
    def __init__(self) -> None:
        super().__init__()
        self.setFixedHeight(150)
        row = QHBoxLayout(self)
        row.setContentsMargins(20, 12, 20, 12)
        row.setSpacing(18)
        heading = QVBoxLayout()
        heading.addWidget(text_label("POSITION METADATA", "caps"))
        self.metadata = text_label("", "detailMeta")
        self.metadata.setWordWrap(True)
        self.metadata.setFixedWidth(145)
        heading.addWidget(self.metadata)
        heading.addStretch()
        row.addLayout(heading)
        row.addWidget(divider(True))
        self.values: dict[str, QLabel] = {}
        self.fen = ""
        for title, value, sub, width in (
            (
                "OPENING",
                "C50 Italian Game: Giuoco Piano, Main Line",
                "",
                220,
            ),
            (
                "OPENING LINE",
                "1. e4 e5  2. Nf3 Nc6  3. Bc4 Bc5",
                "",
                245,
            ),
            (
                "FEN",
                "r1bq1rk1/1ppnbppp/p1np4/4p3/4P3/1BPP1N1P/PP3PP1/RNBQR1K1",
                "",
                258,
            ),
        ):
            box = QVBoxLayout()
            box.addWidget(text_label(title, "caps"))
            value_label = text_label(value, "statValue")
            if title == "FEN":
                value_label.setObjectName("fenValue")
            value_label.setWordWrap(True)
            value_label.setMaximumWidth(width)
            self.values[title] = value_label
            box.addWidget(value_label)
            if title == "FEN":
                copy_fen = button("Copy FEN", compact=True)
                copy_fen.setObjectName("linkButton")
                copy_fen.setFixedHeight(24)
                copy_fen.setToolTip("Copy the current position in FEN notation")
                copy_fen.clicked.connect(self.copy_fen)
                box.addWidget(copy_fen)
            elif sub:
                sub_label = text_label(sub, "blueText" if sub == "Copy FEN" else "muted")
                sub_label.setWordWrap(True)
                sub_label.setMaximumWidth(width)
                box.addWidget(sub_label)
            box.addStretch()
            row.addLayout(box, 1)
            if title != "FEN":
                row.addWidget(divider(True))
        self.set_ply(24)

    def copy_fen(self) -> None:
        QApplication.clipboard().setText(self.fen)
        QMessageBox.information(self, "FEN Copied", "The current FEN has been copied to the clipboard.")

    def set_ply(self, ply: int) -> None:
        positions = {
            0: ("Starting Position", "-", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
            1: ("King's Pawn Opening", "1. e4", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"),
            6: ("C60 Ruy Lopez", "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6", "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4"),
            24: ("C50 Italian Game: Giuoco Piano, Main Line", "1. e4 e5  2. Nf3 Nc6  3. Bc4 Bc5\n4. c3 Nf6  5. d4 exd4", "r1bq1rk1/1ppnbppp/p1np4/4p3/4P3/1BPP1N1P/PP3PP1/RNBQR1K1 w - - 0 13"),
        }
        key = max((item for item in positions if item <= ply), default=0)
        opening, line, fen = positions[key]
        self.values["OPENING"].setText(opening)
        self.values["OPENING LINE"].setText(line)
        self.values["FEN"].setText(fen)
        self.fen = fen
        fields = fen.split()
        turn = "White" if fields[1] == "w" else "Black"
        castling = fields[2] if fields[2] != "-" else "None"
        en_passant = fields[3] if fields[3] != "-" else "None"
        self.metadata.setText(f"Turn: {turn}\nCastling: {castling}\nEn passant: {en_passant}\nMove: {fields[5]}")

    def set_snapshot(self, snapshot: GameSnapshot) -> None:
        visible_moves = snapshot.active_line
        opening = (
            "Custom Position"
            if not visible_moves and snapshot.fen != STANDARD_STARTING_FEN
            else _opening_label(visible_moves)
        )
        line = _format_san_line(visible_moves)
        self.values["OPENING"].setText(opening)
        self.values["OPENING LINE"].setText(line or "-")
        self.values["FEN"].setText(snapshot.fen)
        self.fen = snapshot.fen
        fields = snapshot.fen.split()
        turn = "White" if fields[1] == "w" else "Black"
        castling = fields[2] if fields[2] != "-" else "None"
        en_passant = fields[3] if fields[3] != "-" else "None"
        self.metadata.setText(
            f"Turn: {turn}\nCastling: {castling}\nEn passant: {en_passant}\nPly: {snapshot.current_ply}"
        )


class RightConsole(QWidget):
    ply_requested = Signal(int)
    variation_requested = Signal()
    suggestion_toggled = Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        self.engine = EngineStatus()
        layout.addWidget(self.engine)
        middle = QHBoxLayout()
        middle.setSpacing(12)
        middle.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.ledger = MoveLedger()
        self.ledger.setFixedWidth(242)
        middle.addWidget(self.ledger)
        summary_stack = QVBoxLayout()
        summary_stack.setSpacing(12)
        self.summary = InfoSummary()
        summary_stack.addWidget(self.summary)
        self.principal_variation = PrincipalVariation()
        summary_stack.addWidget(self.principal_variation)
        self.graph = EvaluationGraph()
        summary_stack.addWidget(self.graph)
        middle.addLayout(summary_stack, 1)
        layout.addLayout(middle, 1)
        self.position_info = PositionInfo()
        layout.addWidget(self.position_info)
        self.ledger.ply_selected.connect(self.select_ply)
        self.ledger.variation_selected.connect(self.variation_requested.emit)
        self.graph.canvas.ply_selected.connect(self.select_ply)
        self.principal_variation.suggestion_toggled.connect(self.suggestion_toggled.emit)

    def select_ply(self, ply: int) -> None:
        self.graph.select_ply(ply)
        self.ledger.set_current_ply(ply)
        self.ply_requested.emit(ply)

    def set_snapshot(self, snapshot: GameSnapshot) -> None:
        self.ledger.set_timeline(
            snapshot.move_list,
            snapshot.current_ply,
            snapshot.variation_moves,
            snapshot.variation_base_ply,
        )
        self.graph.set_total_plies(len(snapshot.move_list))
        self.graph.select_ply(snapshot.current_ply)
        self.position_info.set_snapshot(snapshot)

    def load_play_position(self) -> None:
        self.select_ply(0)
        QMessageBox.information(self, "Analyze Position", "The current Play position has been copied into Analyze. The ongoing game remains unchanged.")

    def update_analysis(self, info: UCIAnalysisInfo) -> None:
        self.summary.update_candidate(info)
        if info.multipv == 1:
            self.engine.update_analysis(info)
            self.summary.update_analysis(info)
            self.principal_variation.set_pv(info.pv)

    def begin_position_analysis(self) -> None:
        self.engine.begin_position_analysis()
        self.summary.begin_position_analysis()
        self.principal_variation.set_pv(())


def play_field(icon: str, value: str) -> QFrame:
    field = QFrame()
    field.setObjectName("playField")
    row = QHBoxLayout(field)
    row.setContentsMargins(14, 0, 12, 0)
    row.setSpacing(10)
    row.addWidget(text_label(icon, "fieldIcon"))
    row.addWidget(text_label(value, "fieldValue"))
    row.addStretch()
    row.addWidget(text_label("\u2304", "muted"))
    return field


@dataclass(frozen=True, slots=True)
class TimeControlConfig:
    label: str
    base_ms: int | None
    increment_ms: int = 0


def format_play_clock(milliseconds: int | None) -> str:
    if milliseconds is None:
        return "\u221e"
    milliseconds = max(0, milliseconds)
    if milliseconds >= 3_600_000:
        seconds = milliseconds // 1000
        return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"
    if milliseconds >= 60_000:
        seconds = milliseconds // 1000
        return f"{seconds // 60:02d}:{seconds % 60:02d}"
    tenths = milliseconds // 100
    return f"{tenths // 600:02d}:{tenths % 600 // 10:02d}.{tenths % 10}"


class PlayerClock(Card):
    def __init__(self) -> None:
        super().__init__()
        self.setFixedHeight(292)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 12)
        layout.setSpacing(9)
        layout.addWidget(text_label("PLAYER / CLOCK", "sectionHeader"))
        self.names: dict[str, QLabel] = {}
        self.messages: dict[str, QLabel] = {}
        self.clocks: dict[str, QLabel] = {}
        self.avatars: dict[str, QLabel] = {}
        for side, icon, player in (
            ("WHITE", "\u2659", "Human"),
            ("BLACK", "\u2699", "DeepTactics"),
        ):
            row_card = QFrame()
            row_card.setObjectName("clockPlayer")
            row = QHBoxLayout(row_card)
            row.setContentsMargins(15, 11, 13, 11)
            row.setSpacing(13)
            avatar = text_label(icon, "avatarHuman" if side == "WHITE" else "avatarEngine")
            avatar.setFixedWidth(28)
            self.avatars[side] = avatar
            row.addWidget(avatar)
            identity = QVBoxLayout()
            identity.setSpacing(3)
            identity.addWidget(text_label(side, "caps"))
            name = text_label(player, "clockName")
            self.names[side] = name
            identity.addWidget(name)
            row.addLayout(identity, 1)
            row.addWidget(divider(True))
            time_box = QVBoxLayout()
            time_box.setSpacing(3)
            clock_label = text_label("\u221e", "clockValue")
            clock_label.setFixedWidth(90)
            clock_label.setAlignment(Qt.AlignmentFlag.AlignRight)
            message_label = text_label("Waiting  \u25cf", "secondary")
            message_label.setFixedWidth(104)
            message_label.setAlignment(Qt.AlignmentFlag.AlignRight)
            self.clocks[side] = clock_label
            self.messages[side] = message_label
            time_box.addWidget(clock_label)
            time_box.addWidget(message_label)
            row.addLayout(time_box)
            layout.addWidget(row_card)
        status = QHBoxLayout()
        self.turn = text_label("Turn:  White", "statValue")
        self.move_number = text_label("Move: 0 ply", "secondary")
        status.addWidget(self.turn)
        status.addWidget(text_label("|", "muted"))
        status.addWidget(self.move_number)
        status.addStretch()
        layout.addLayout(status)
        layout.addWidget(text_label("Opening:", "secondary"))
        self.opening = text_label("Starting Position", "openingValue")
        self.opening.setWordWrap(True)
        layout.addWidget(self.opening)

    def set_players(self, white: str, black: str) -> None:
        self.names["WHITE"].setText(white)
        self.names["BLACK"].setText(black)
        for side, player in (("WHITE", white), ("BLACK", black)):
            human = player == "Human"
            self.avatars[side].setText("\u2659" if human else "\u2699")
            self.avatars[side].setObjectName("avatarHuman" if human else "avatarEngine")
            self.avatars[side].style().unpolish(self.avatars[side])
            self.avatars[side].style().polish(self.avatars[side])

    def set_clock_values(self, white_ms: int | None, black_ms: int | None) -> None:
        self.clocks["WHITE"].setText(format_play_clock(white_ms))
        self.clocks["BLACK"].setText(format_play_clock(black_ms))

    def update_snapshot(self, snapshot: GameSnapshot, *, active: bool = False) -> None:
        fields = snapshot.fen.split()
        turn = "White" if fields[1] == "w" else "Black"
        self.turn.setText(f"Turn:  {turn}")
        self.move_number.setText(f"Move: {snapshot.current_ply} ply")
        self.opening.setText(_opening_label(snapshot.active_line))
        for side in ("WHITE", "BLACK"):
            is_turn = side.title() == turn and active
            self.messages[side].setText("To Move  \u25cf" if is_turn else "Waiting  \u25cf")
            self.messages[side].setObjectName("greenText" if is_turn else "secondary")
            self.messages[side].style().unpolish(self.messages[side])
            self.messages[side].style().polish(self.messages[side])


class PlayerSetup(Card):
    def __init__(self) -> None:
        super().__init__()
        self.selections: list[QComboBox] = []
        self.setFixedHeight(156)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 14)
        layout.setSpacing(10)
        layout.addWidget(text_label("PLAYERS", "sectionHeader"))
        for side, player in (("White", "Human"), ("Black", "DeepTactics")):
            row = QHBoxLayout()
            side_label = text_label(side, "statValue")
            side_label.setFixedWidth(71)
            row.addWidget(side_label)
            selection = QComboBox()
            selection.setObjectName("interactiveField")
            selection.addItems(["Human", *ENGINE_REGISTRY.names()])
            selection.setCurrentText(player)
            self.selections.append(selection)
            row.addWidget(selection, 1)
            layout.addLayout(row)
        ENGINE_REGISTRY.changed.connect(self.refresh_engines)

    def refresh_engines(self, names: list[str]) -> None:
        for selection in self.selections:
            current = selection.currentText()
            selection.clear()
            selection.addItems(["Human", *names])
            selection.setCurrentText(current if current in ["Human", *names] else "Human")

    def selected_players(self) -> tuple[str, str]:
        return self.selections[0].currentText(), self.selections[1].currentText()

class TimeControl(Card):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 13)
        layout.setSpacing(11)
        layout.addWidget(text_label("TIME CONTROL", "sectionHeader"))
        presets = QGridLayout()
        presets.setHorizontalSpacing(10)
        presets.setVerticalSpacing(10)
        self.time_group = QButtonGroup(self)
        self.time_group.setExclusive(True)
        self.last_regular: QPushButton | None = None
        self.controls: dict[QPushButton, TimeControlConfig] = {}
        for index, config in enumerate((
            TimeControlConfig("Unlimited", None),
            TimeControlConfig("Bullet 2+1", 120_000, 1_000),
            TimeControlConfig("Blitz 5+3", 300_000, 3_000),
            TimeControlConfig("Rapid 10+5", 600_000, 5_000),
            TimeControlConfig("Standard 15+10", 900_000, 10_000),
            TimeControlConfig("Classical 30+20", 1_800_000, 20_000),
        )):
            value = f"\u221e  {config.label}" if config.base_ms is None else f"\u25f4  {config.label}"
            choice = button(value, compact=True)
            choice.setCheckable(True)
            choice.setChecked(index == 0)
            self.time_group.addButton(choice)
            choice.clicked.connect(lambda checked=False, selected=choice: self._remember_regular(selected, checked))
            if index == 0:
                self.last_regular = choice
            self.controls[choice] = config
            choice.setFixedHeight(47)
            presets.addWidget(choice, index // 2, index % 2)
        layout.addLayout(presets)
        self.custom = button("\u2637   Custom", compact=True)
        self.custom.setCheckable(True)
        self.custom.setFixedHeight(46)
        self.custom.clicked.connect(self.open_custom_dialog)
        self.time_group.addButton(self.custom)
        self.custom_config = TimeControlConfig("Custom 15+10", 900_000, 10_000)
        layout.addWidget(self.custom)

    def open_custom_dialog(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Custom Time Control")
        form = QFormLayout(dialog)
        hours = QSpinBox()
        hours.setRange(0, 24)
        minutes = QSpinBox()
        minutes.setRange(0, 59)
        minutes.setValue(15)
        seconds = QSpinBox()
        seconds.setRange(0, 59)
        increment = QSpinBox()
        increment.setRange(0, 120)
        increment.setValue(10)
        form.addRow("Base hours", hours)
        form.addRow("Base minutes", minutes)
        form.addRow("Base seconds", seconds)
        form.addRow("Increment (seconds / move)", increment)
        accept = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        accept.accepted.connect(dialog.accept)
        accept.rejected.connect(dialog.reject)
        form.addRow(accept)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            base_seconds = hours.value() * 3600 + minutes.value() * 60 + seconds.value()
            if base_seconds <= 0:
                QMessageBox.warning(self, "Custom Time Control", "Base time must be greater than zero.")
                if self.last_regular:
                    self.last_regular.setChecked(True)
                return
            self.custom_config = TimeControlConfig(
                f"Custom {hours.value():02d}:{minutes.value():02d}:{seconds.value():02d}+{increment.value()}",
                base_seconds * 1000,
                increment.value() * 1000,
            )
            self.custom.setText(f"\u2637   {self.custom_config.label}")
            self.custom.setChecked(True)
        elif self.last_regular:
            self.last_regular.setChecked(True)

    def _remember_regular(self, selected: QPushButton, checked: bool) -> None:
        if checked:
            self.last_regular = selected

    def selected_config(self) -> TimeControlConfig:
        selected = self.time_group.checkedButton()
        return self.custom_config if selected is self.custom else self.controls.get(selected, TimeControlConfig("Unlimited", None))


class GameStatus(Card):
    def __init__(self) -> None:
        super().__init__()
        self.setFixedHeight(215)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(19, 17, 19, 17)
        layout.setSpacing(13)
        layout.addWidget(text_label("GAME STATUS", "sectionHeader"))
        self.values: dict[str, QLabel] = {}
        for title, value, style in (
            ("Time Control", "Unlimited", "statValue"),
            ("Rule Set", "Standard", "statValue"),
            ("Players", "Human vs DeepTactics", "statValue"),
            ("Status", "In progress", "greenText"),
        ):
            row = QHBoxLayout()
            label = text_label(title, "secondary")
            label.setFixedWidth(112)
            row.addWidget(label)
            value_label = text_label(value, style)
            self.values[title] = value_label
            row.addWidget(value_label)
            row.addStretch()
            layout.addLayout(row)

    def start_session(self, white: str, black: str, control: TimeControlConfig) -> None:
        self.values["Time Control"].setText(control.label)
        self.values["Rule Set"].setText("Standard")
        self.values["Players"].setText(f"{white} vs {black}")
        self.values["Status"].setText("In progress")

    def update_snapshot(self, snapshot: GameSnapshot) -> None:
        fields = snapshot.fen.split()
        turn = "White" if fields[1] == "w" else "Black"
        statuses = {
            ChessGameStatus.ONGOING.value: "In progress",
            ChessGameStatus.CHECKMATE.value: f"Checkmate - {'Black' if turn == 'White' else 'White'} wins",
            ChessGameStatus.STALEMATE.value: "Draw - stalemate",
            ChessGameStatus.THREEFOLD_REPETITION.value: "Draw - repetition",
            ChessGameStatus.FIFTY_MOVE_RULE.value: "Draw - fifty-move rule",
            ChessGameStatus.INSUFFICIENT_MATERIAL.value: "Draw - insufficient material",
        }
        self.values["Status"].setText(statuses.get(snapshot.status, snapshot.status.title()))


class PlayMoves(Card):
    ply_selected = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 15)
        layout.setSpacing(10)
        header = QHBoxLayout()
        header.addWidget(text_label("MOVES", "sectionHeader"))
        header.addStretch()
        layout.addLayout(header)
        self.table = QTableWidget(1, 3)
        self.table.setObjectName("gamesTable")
        self.table.setHorizontalHeaderLabels(["#", "White", "Black"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.cellClicked.connect(self._select_move)
        self.set_moves(())
        layout.addWidget(self.table, 1)

    def set_moves(self, moves: tuple[str, ...], current_ply: int | None = None) -> None:
        scroll_value = self.table.verticalScrollBar().value()
        rows = max(1, (len(moves) + 1) // 2)
        self.table.setRowCount(rows)
        for row in range(rows):
            self.table.setItem(row, 0, QTableWidgetItem(f"{row + 1}."))
            self.table.setItem(row, 1, QTableWidgetItem(moves[row * 2] if row * 2 < len(moves) else "\u2014"))
            self.table.setItem(row, 2, QTableWidgetItem(moves[row * 2 + 1] if row * 2 + 1 < len(moves) else "\u2014"))
        if current_ply:
            row, column = (current_ply - 1) // 2, 1 if current_ply % 2 else 2
            self.table.setCurrentCell(row, column)
        self.table.verticalScrollBar().setValue(scroll_value)

    def _select_move(self, row: int, column: int) -> None:
        if column not in (1, 2):
            return
        ply = row * 2 + column
        item = self.table.item(row, column)
        if item is not None and item.text() != "\u2014":
            self.ply_selected.emit(ply)


class AssistPanel(Card):
    analyze_requested = Signal()
    suggest_requested = Signal()
    evaluation_toggled = Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        self.setFixedHeight(195)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 15)
        layout.setSpacing(15)
        layout.addWidget(text_label("ASSIST", "sectionHeader"))
        actions = QHBoxLayout()
        actions.setSpacing(10)
        analyze = button("\u2315  Analyze Position", compact=True)
        suggest = button("\u25cc  Suggest Move", compact=True)
        analyze.clicked.connect(self.analyze_requested.emit)
        suggest.clicked.connect(self.suggest_requested.emit)
        self.suggest = suggest
        suggest.setEnabled(False)
        suggest.setToolTip("Available during an active game.")
        actions.addWidget(analyze)
        actions.addWidget(suggest)
        layout.addLayout(actions)
        self.eval_toggle = button("Evaluation Meter", compact=True)
        self.eval_toggle.setCheckable(True)
        self.eval_toggle.setToolTip("Show live engine evaluation beside the board.")
        self.eval_toggle.toggled.connect(self.evaluation_toggled.emit)
        layout.addWidget(self.eval_toggle)
        layout.addStretch()


class PlayConsole(QWidget):
    analyze_requested = Signal()
    suggest_requested = Signal()
    evaluation_toggled = Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        setup = QVBoxLayout()
        setup.setSpacing(10)
        self.player_clock = PlayerClock()
        setup.addWidget(self.player_clock)
        self.player_setup = PlayerSetup()
        setup.addWidget(self.player_setup)
        self.time_control = TimeControl()
        setup.addWidget(self.time_control, 1)
        layout.addLayout(setup, 49)
        game = QVBoxLayout()
        game.setSpacing(10)
        self.game_status = GameStatus()
        game.addWidget(self.game_status)
        self.moves = PlayMoves()
        game.addWidget(self.moves, 1)
        self.assist = AssistPanel()
        self.assist.analyze_requested.connect(self.analyze_requested.emit)
        self.assist.suggest_requested.connect(self.suggest_requested.emit)
        self.assist.evaluation_toggled.connect(self.evaluation_toggled.emit)
        game.addWidget(self.assist)
        layout.addLayout(game, 51)

    def start_session(self) -> tuple[str, str, TimeControlConfig]:
        white, black = self.player_setup.selected_players()
        control = self.time_control.selected_config()
        self.player_clock.set_players(white, black)
        self.game_status.start_session(white, black, control)
        self.assist.suggest.setEnabled(True)
        return white, black, control

    def update_snapshot(self, snapshot: GameSnapshot, *, active: bool = False) -> None:
        self.player_clock.update_snapshot(snapshot, active=active)
        self.game_status.update_snapshot(snapshot)
        self.moves.set_moves(snapshot.move_list, snapshot.current_ply)


class MatchGraph(Card):
    def __init__(self) -> None:
        super().__init__()
        self.setFixedHeight(216)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 10)
        heading = QHBoxLayout()
        heading.addWidget(text_label("EVALUATION GRAPH", "sectionHeader"))
        heading.addStretch()
        layout.addLayout(heading)
        self.canvas = GraphCanvas()
        self.canvas.total_plies = 1
        self.canvas.selected_ply = 0
        layout.addWidget(self.canvas, 1)

    def set_score(self, ply: int, score_cp_white: int | None, mate_white: int | None) -> None:
        value = 10.0 if mate_white and mate_white > 0 else (-10.0 if mate_white else None)
        if value is None and score_cp_white is not None:
            value = max(-10.0, min(10.0, score_cp_white / 100.0))
        if value is None:
            return
        self.canvas.total_plies = max(1, ply)
        self.canvas.selected_ply = ply
        self.canvas.scores[ply] = value
        self.canvas.update()

    def clear_scores(self) -> None:
        self.canvas.scores.clear()
        self.canvas.total_plies = 1
        self.canvas.selected_ply = 0
        self.canvas.update()


class AIMatchBoard(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        board_card = Card("boardCard")
        board_layout = QHBoxLayout(board_card)
        board_layout.setContentsMargins(20, 17, 14, 17)
        board_layout.setSpacing(12)
        self.board = WoodenBoard(starting_position=True, minimum_size=500)
        self.board.set_interactive(False)
        self.eval_meter = EvalMeter(neutral=True, minimum_height=500)
        board_layout.addWidget(self.board, 1)
        board_layout.addWidget(self.eval_meter)
        layout.addWidget(board_card, 1)
        self.graph = MatchGraph()
        layout.addWidget(self.graph)


class MatchMoveList(Card):
    ply_selected = Signal(int)
    live_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setFixedHeight(367)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 10)
        layout.setSpacing(6)
        title = QHBoxLayout()
        title.addWidget(text_label("MOVE LIST", "sectionHeader"))
        title.addStretch()
        live = button("Live", compact=True)
        live.clicked.connect(self.live_requested.emit)
        title.addWidget(live)
        layout.addLayout(title)
        self.table = QTableWidget(1, 3)
        self.table.setObjectName("gamesTable")
        self.table.setHorizontalHeaderLabels(["#", "White", "Black"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.cellClicked.connect(self._select_move)
        layout.addWidget(self.table, 1)
        self.set_moves(())

    def set_moves(self, moves: tuple[str, ...], current_ply: int | None = None) -> None:
        rows = max(1, (len(moves) + 1) // 2)
        self.table.setRowCount(rows)
        for row in range(rows):
            self.table.setItem(row, 0, QTableWidgetItem(f"{row + 1}."))
            self.table.setItem(row, 1, QTableWidgetItem(moves[row * 2] if row * 2 < len(moves) else "\u2014"))
            self.table.setItem(row, 2, QTableWidgetItem(moves[row * 2 + 1] if row * 2 + 1 < len(moves) else "\u2014"))
        if current_ply:
            self.table.setCurrentCell((current_ply - 1) // 2, 1 if current_ply % 2 else 2)

    def _select_move(self, row: int, column: int) -> None:
        item = self.table.item(row, column)
        if column in (1, 2) and item is not None and item.text() != "\u2014":
            self.ply_selected.emit(row * 2 + column)


class MatchSummary(Card):
    def __init__(self) -> None:
        super().__init__()
        self.setFixedHeight(168)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 13, 16, 13)
        layout.setSpacing(10)
        layout.addWidget(text_label("MATCH SUMMARY", "sectionHeader"))
        self.values: dict[str, QLabel] = {}
        for title, value, style in (
            ("Trend", "No evaluation yet", "greenText"),
            ("Opening", "Starting Position", "statValue"),
            ("Moves", "0 ply / 0 moves", "statValue"),
            ("Current", "\u25cf   Ready to start", "statValue"),
        ):
            line = QHBoxLayout()
            label = text_label(title, "secondary")
            label.setFixedWidth(95)
            line.addWidget(label)
            value_label = text_label(value, style)
            self.values[title] = value_label
            line.addWidget(value_label)
            line.addStretch()
            layout.addLayout(line)

    def set_status(self, status: str) -> None:
        descriptions = {
            "Running": "\u25cf   Black to move - running",
            "Paused": "\u25cf   Match paused",
            "Stopped": "\u25cf   Match stopped",
            "Ready": "\u25cf   Ready to start",
        }
        self.values["Current"].setText(descriptions.get(status, status))

    def reset(self) -> None:
        self.values["Moves"].setText("0")
        self.values["Opening"].setText("Starting Position")
        self.values["Trend"].setText("No evaluation yet")
        self.values["Current"].setText("\u25cf   Ready to start")

    def update_snapshot(self, snapshot: GameSnapshot, *, live: bool = True) -> None:
        self.values["Opening"].setText(_opening_label(snapshot.active_line))
        self.values["Moves"].setText(f"{snapshot.current_ply} ply / {(snapshot.current_ply + 1) // 2} moves")
        turn = "White" if " w " in snapshot.fen else "Black"
        self.values["Current"].setText(f"\u25cf   {turn} to move" if live else f"Viewing ply {snapshot.current_ply} / Live")

    def update_score(self, score_cp_white: int | None, mate_white: int | None) -> None:
        if mate_white is not None:
            self.values["Trend"].setText(f"Mate {format_evaluation(mate_white=mate_white)}")
        elif score_cp_white is None:
            self.values["Trend"].setText("No evaluation yet")
        elif abs(score_cp_white) < 30:
            self.values["Trend"].setText("Equal")
        else:
            self.values["Trend"].setText("Better for White" if score_cp_white > 0 else "Better for Black")


class ScoreBoard(Card):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 12)
        layout.setSpacing(10)
        top = QHBoxLayout()
        top.addWidget(text_label("SCOREBOARD", "sectionHeader"))
        top.addStretch()
        top.addWidget(text_label("Performance", "secondary"))
        layout.addLayout(top)
        score = QHBoxLayout()
        left = QVBoxLayout()
        self.matchup = text_label("White Engine / Black Engine", "blueTextLarge")
        left.addWidget(self.matchup)
        points = QHBoxLayout()
        self.results: list[QLabel] = []
        for number, caption, style in (("0", "White Wins", "scoreWin"), ("0", "Draws", "scoreDim"), ("0", "Black Wins", "scoreDim")):
            box = QVBoxLayout()
            value = text_label(number, style)
            self.results.append(value)
            box.addWidget(value)
            box.addWidget(text_label(caption, "secondary"))
            points.addLayout(box)
        left.addLayout(points)
        score.addLayout(left, 1)
        score.addWidget(divider(True))
        performance = QVBoxLayout()
        self.performance = text_label("Awaiting games", "secondary")
        performance.addWidget(self.performance)
        score.addLayout(performance)
        layout.addLayout(score)
        layout.addStretch()
        self.progress_text = text_label("Game 0 / 0", "statValue")
        layout.addWidget(self.progress_text)
        bar = QFrame()
        bar.setObjectName("matchProgress")
        bar.setFixedHeight(7)
        layout.addWidget(bar)

    def set_engines(self, white: str, black: str) -> None:
        self.matchup.setText(f"{white} W / {black} B")

    def set_score(self, white: int, draws: int, black: int, current_game: int, total_games: int) -> None:
        for label, value in zip(self.results, (white, draws, black)):
            label.setText(str(value))
        self.progress_text.setText(f"Game {current_game} / {total_games}")

    def set_performance(self, depth: int, nps: int) -> None:
        self.performance.setText(f"Depth {depth or '--'}\nNPS {nps:,}" if nps else f"Depth {depth or '--'}")


class AIMatchCenter(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        self.move_list = MatchMoveList()
        self.summary = MatchSummary()
        self.scoreboard = ScoreBoard()
        layout.addWidget(self.move_list)
        layout.addWidget(self.summary)
        layout.addWidget(self.scoreboard, 1)


class EngineDuelCard(Card):
    clear_hash_requested = Signal(str)

    def __init__(self, title: str, engine: str, score: str, positive: bool) -> None:
        super().__init__()
        self.setFixedHeight(226)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 11, 18, 11)
        layout.setSpacing(7)
        header = QHBoxLayout()
        header.addWidget(text_label(title, "sectionHeader"))
        header.addStretch()
        self.running = text_label("\u25cf  Ready", "greenText")
        header.addWidget(self.running)
        layout.addLayout(header)
        selection = QHBoxLayout()
        self.engine_select = QComboBox()
        self.engine_select.addItems(ENGINE_REGISTRY.names())
        self.engine_select.setCurrentText(engine)
        self.engine_select.setObjectName("interactiveField")
        selection.addWidget(self.engine_select, 1)
        load = button("Load Engine", compact=True)
        clear = button("Clear Hash", compact=True)
        load.clicked.connect(self.choose_engine)
        clear.clicked.connect(lambda: self.clear_hash_requested.emit(self.engine_select.currentText()))
        selection.addWidget(load)
        selection.addWidget(clear)
        layout.addLayout(selection)
        stats = QHBoxLayout()
        self.stat_values: dict[str, QLabel] = {}
        for title_text, value in (
            ("Depth", "--"),
            ("Nodes", "--"),
            ("NPS", "--"),
            ("Time Used", "--"),
            ("Score", "--"),
        ):
            column = QVBoxLayout()
            column.addWidget(text_label(title_text, "muted"))
            shown = text_label(value, "greenText" if title_text == "Score" else "statValue")
            self.stat_values[title_text] = shown
            column.addWidget(shown)
            stats.addLayout(column)
        layout.addLayout(stats)
        layout.addWidget(text_label("PV (Principal Variation)", "secondary"))
        pv = QHBoxLayout()
        self.pv_score = text_label("--", "greenText")
        self.pv_line = text_label("--", "greenText")
        pv.addWidget(self.pv_score)
        pv.addWidget(self.pv_line)
        pv.addStretch()
        layout.addLayout(pv)
        ENGINE_REGISTRY.changed.connect(self.refresh_engines)

    def refresh_engines(self, names: list[str]) -> None:
        current = self.engine_select.currentText()
        self.engine_select.clear()
        self.engine_select.addItems(names)
        self.engine_select.setCurrentText(current if current in names else "DeepTactics")

    def choose_engine(self) -> None:
        choices = ENGINE_REGISTRY.names()
        selected, accepted = QInputDialog.getItem(self, "Load Engine", "Registered UCI engine:", choices, editable=False)
        if accepted and selected:
            self.engine_select.setCurrentText(selected)

    def set_running(self, status: str) -> None:
        self.running.setText(f"\u25cf  {status}")

    def update_analysis(self, info: UCIAnalysisInfo, fen: str) -> None:
        score = format_evaluation(info.score_cp_white, info.mate_white)
        score_style = "lossText" if (
            (info.mate_white is not None and info.mate_white < 0)
            or (info.mate_white is None and info.score_cp_white is not None and info.score_cp_white < 0)
        ) else "greenText"
        for label in (self.stat_values["Score"], self.pv_score):
            label.setObjectName(score_style)
            label.style().unpolish(label)
            label.style().polish(label)
        self.stat_values["Depth"].setText(str(info.depth or "--"))
        self.stat_values["Nodes"].setText(f"{info.nodes:,}" if info.nodes else "--")
        self.stat_values["NPS"].setText(f"{info.nps:,}" if info.nps else "--")
        self.stat_values["Time Used"].setText(f"{info.time_ms / 1000:.2f}s" if info.time_ms else "--")
        self.stat_values["Score"].setText(score)
        self.pv_score.setText(score)
        self.pv_line.setText("  ".join(_pv_to_san(fen, info.pv)) if info.pv else "--")

    def reset_runtime(self) -> None:
        self.set_running("Ready")
        for value in self.stat_values.values():
            value.setText("--")
        for label in (self.stat_values["Score"], self.pv_score):
            label.setObjectName("greenText")
            label.style().unpolish(label)
            label.style().polish(label)
        self.pv_score.setText("--")
        self.pv_line.setText("--")


class MatchControls(Card):
    start_requested = Signal()
    pause_requested = Signal(bool)
    stop_requested = Signal()
    reset_requested = Signal()
    save_requested = Signal()
    export_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(17, 14, 17, 12)
        layout.setSpacing(9)
        layout.addWidget(text_label("AI MATCH CONTROLS", "sectionHeader"))
        actions = QHBoxLayout()
        actions.setSpacing(8)
        self.start_button = button("\u25b6  Start Match", primary=True, compact=True)
        self.pause_button = button("\u23f8  Pause", compact=True)
        self.stop_button = button("\u25a0  Stop", compact=True)
        for action in (self.start_button, self.pause_button, self.stop_button):
            action.setFixedHeight(39)
            actions.addWidget(action, 1)
        self.pause_button.setCheckable(True)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_requested.emit)
        self.pause_button.toggled.connect(self.pause_requested.emit)
        self.stop_button.clicked.connect(self.stop_requested.emit)
        self.swap_colors = QCheckBox("Swap each game")
        self.swap_colors.setToolTip("Exchange White and Black engine assignments between games.")
        actions.addWidget(self.swap_colors, 1)
        layout.addLayout(actions)

        form = QGridLayout()
        form.setHorizontalSpacing(20)
        form.setVerticalSpacing(8)
        form.setColumnStretch(1, 1)
        form.setColumnStretch(3, 1)
        form.addWidget(text_label("Time Control", "secondary"), 0, 0)
        self.time_control = QComboBox()
        self.time_control.addItems(["Unlimited", "Bullet 2+1", "Blitz 5+3", "Rapid 10+5", "Standard 15+10", "Classical 30+20"])
        self.time_control.setCurrentText("Blitz 5+3")
        self.time_control.setObjectName("interactiveField")
        form.addWidget(self.time_control, 0, 1)
        form.addWidget(text_label("Search Mode", "secondary"), 0, 2)
        self.search_mode = QComboBox()
        self.search_mode.addItems(["Fixed Depth", "Move Time", "Clock"])
        self.search_mode.setCurrentText(str(SETTINGS_STORE.load().get("ai_match_search_mode", "Fixed Depth")))
        self.search_mode.setObjectName("interactiveField")
        self.search_mode.setToolTip("Clock uses UCI wtime/btime. Fixed Depth is deterministic and useful for engine comparisons.")
        form.addWidget(self.search_mode, 0, 3)

        form.addWidget(text_label("Depth", "secondary"), 1, 0)
        self.depth = QSpinBox()
        self.depth.setRange(1, 30)
        self.depth.setValue(int(SETTINGS_STORE.load().get("ai_match_depth", 5)))
        self.depth.setObjectName("interactiveField")
        self.depth.setToolTip("Depth used when AI Match Search Mode is Fixed Depth.")
        form.addWidget(self.depth, 1, 1)
        form.addWidget(text_label("Move Time", "secondary"), 1, 2)
        self.movetime_ms = QSpinBox()
        self.movetime_ms.setRange(10, 600_000)
        self.movetime_ms.setSingleStep(50)
        self.movetime_ms.setSuffix(" ms")
        self.movetime_ms.setValue(int(SETTINGS_STORE.load().get("ai_match_movetime_ms", 500)))
        self.movetime_ms.setObjectName("interactiveField")
        self.movetime_ms.setToolTip("Per-move time used when AI Match Search Mode is Move Time.")
        form.addWidget(self.movetime_ms, 1, 3)

        form.addWidget(text_label("Opening Suite", "secondary"), 2, 0)
        self.opening_suite = QComboBox()
        self.opening_suite.addItems(["Start Position", "Balanced (Default)"])
        self.opening_suite.setCurrentText("Balanced (Default)")
        self.opening_suite.setToolTip("Balanced uses the bundled paired opening prefixes.")
        form.addWidget(self.opening_suite, 2, 1)
        form.addWidget(text_label("Games", "secondary"), 2, 2)
        self.games = QSpinBox()
        self.games.setRange(1, 10000)
        self.games.setValue(10)
        self.games.setObjectName("interactiveField")
        form.addWidget(self.games, 2, 3)
        self.randomize_opening = QCheckBox("Randomize opening line")
        form.addWidget(self.randomize_opening, 3, 0, 1, 2)
        self.auto_next = QCheckBox("Auto-next game")
        self.auto_next.setChecked(True)
        form.addWidget(self.auto_next, 3, 2, 1, 2)
        vary_order = QCheckBox("Vary move order")
        vary_order.setEnabled(False)
        form.addWidget(vary_order, 4, 0, 1, 2)
        self.use_opening_suite = QCheckBox("Use opening suite")
        self.use_opening_suite.setChecked(True)
        form.addWidget(self.use_opening_suite, 4, 2, 1, 2)
        form.addWidget(text_label("Multi-PV", "secondary"), 5, 0)
        multipv = QSpinBox()
        multipv.setRange(1, 1)
        multipv.setValue(1)
        multipv.setEnabled(False)
        form.addWidget(multipv, 5, 1)
        form.addWidget(text_label("Auto Resign", "secondary"), 5, 2)
        resign = QDoubleSpinBox()
        resign.setRange(-99.0, 0.0)
        resign.setValue(0.0)
        resign.setEnabled(False)
        resign.setToolTip("Automatic adjudication remains off until verified.")
        form.addWidget(resign, 5, 3)
        layout.addLayout(form)

        bottom = QHBoxLayout()
        bottom.setSpacing(8)
        for value, handler in (
            ("\u25a3  Save PGN", self.save_requested.emit), ("\u21e9  Export Match", self.export_requested.emit),
            ("\u2295  New Match", self.new_match), ("...", self.more_actions),
        ):
            action = button(value, compact=True)
            action.setFixedHeight(38)
            action.clicked.connect(handler)
            bottom.addWidget(action, 1)
        layout.addLayout(bottom)

    def new_match(self) -> None:
        if QMessageBox.question(self, "New Match", "Reset the current AI match setup?") == QMessageBox.StandardButton.Yes:
            self.reset_requested.emit()

    def more_actions(self) -> None:
        QMessageBox.information(self, "Additional Actions", "Use Settings > Advanced for engine logs. Custom opening-suite import remains a future extension.")

    def set_running(self, running: bool, paused: bool = False) -> None:
        self.start_button.setEnabled(not running)
        self.pause_button.setEnabled(running)
        self.stop_button.setEnabled(running)
        self.swap_colors.setEnabled(not running)
        self.time_control.setEnabled(not running)
        self.search_mode.setEnabled(not running)
        self.depth.setEnabled(not running)
        self.movetime_ms.setEnabled(not running)
        self.opening_suite.setEnabled(not running)
        self.randomize_opening.setEnabled(not running)
        self.use_opening_suite.setEnabled(not running)
        self.games.setEnabled(not running)
        self.pause_button.blockSignals(True)
        self.pause_button.setChecked(paused)
        self.pause_button.setText("\u25b6  Resume" if paused else "\u23f8  Pause")
        self.pause_button.blockSignals(False)


class AIMatchRight(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        self.white = EngineDuelCard("WHITE ENGINE", "DeepTactics", "--", True)
        self.black = EngineDuelCard("BLACK ENGINE", "DeepTactics", "--", False)
        self.controls = MatchControls()
        layout.addWidget(self.white)
        layout.addWidget(self.black)
        layout.addWidget(self.controls, 1)

    def set_status(self, status: str) -> None:
        self.white.set_running(status)
        self.black.set_running(status)

    def reset(self) -> None:
        self.white.engine_select.setCurrentText("DeepTactics")
        self.black.engine_select.setCurrentText("DeepTactics")
        self.white.reset_runtime()
        self.black.reset_runtime()


class GamesLibrary(Card):
    category_selected = Signal(str)
    files_imported = Signal(list)
    folder_imported = Signal(str)
    pgn_pasted = Signal(str)

    def __init__(self, store: GameLibraryStore) -> None:
        super().__init__()
        self.store = store
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(8)
        layout.addWidget(text_label("LIBRARY", "sectionHeader"))
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.category_definitions = [
            ("\u25a6  All Games", "All"),
            ("\u25f4  Recent PGN", "Recent"),
            ("\u25a1  Saved Games", "Saved"),
            ("\u2694  Match Records", "Match"),
            ("\u2699  Human vs AI", "Human vs AI"),
            ("\u2315  Analyzed Games", "Analyzed"),
            ("\u2606  Favorites", "Favorites"),
        ]
        self.category_buttons: dict[str, QPushButton] = {}
        for index, (title, key) in enumerate(self.category_definitions):
            control = QPushButton()
            control.setObjectName("libraryButton")
            control.setCheckable(True)
            control.setChecked(index == 0)
            control.clicked.connect(lambda _checked=False, category=key: self.category_selected.emit(category))
            self.group.addButton(control)
            self.category_buttons[key] = control
            layout.addWidget(control)
        layout.addWidget(divider())
        collections = QHBoxLayout()
        collections.addWidget(text_label("COLLECTIONS", "caps"))
        collections.addStretch()
        add_collection = button("+", compact=True)
        add_collection.clicked.connect(self.create_collection)
        collections.addWidget(add_collection)
        layout.addLayout(collections)
        self.collection_box = QVBoxLayout()
        layout.addLayout(self.collection_box)
        self.update_counts([])
        self.refresh_collections()
        layout.addStretch()

        import_card = Card()
        imports = QVBoxLayout(import_card)
        imports.setContentsMargins(15, 14, 15, 14)
        imports.setSpacing(8)
        imports.addWidget(text_label("IMPORT / ADD GAMES", "sectionHeader"))
        import_file = button("\u21e7  Import PGN Files", compact=True)
        import_folder = button("\u25a1  Import Folder", compact=True)
        paste = button("\u2398  Paste PGN", compact=True)
        import_file.clicked.connect(self._import_files)
        import_folder.clicked.connect(self._import_folder)
        paste.clicked.connect(self._paste_pgn)
        imports.addWidget(import_file)
        imports.addWidget(import_folder)
        imports.addWidget(paste)
        layout.addWidget(import_card)

    def update_counts(self, games: list[GameRecord]) -> None:
        counts = {
            "All": len(games),
            "Recent": len([game for game in games if game.source_type == "Imported"]),
            "Saved": len([game for game in games if game.source_type == "Saved"]),
            "Match": len([game for game in games if game.game_type == "AI Match"]),
            "Human vs AI": len([game for game in games if game.game_type == "Human vs AI"]),
            "Analyzed": len([game for game in games if game.analyzed]),
            "Favorites": len([game for game in games if game.favorite]),
        }
        for title, key in self.category_definitions:
            self.category_buttons[key].setText(f"{title:<23}  {counts[key]}")
        self.refresh_collections()

    def create_collection(self) -> None:
        name, accepted = QInputDialog.getText(self, "New Collection", "Collection name:")
        if not accepted or not name.strip():
            return
        self.store.create_collection(name)
        self.refresh_collections()

    def refresh_collections(self) -> None:
        while self.collection_box.count():
            item = self.collection_box.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        if not self.store.collections:
            self.collection_box.addWidget(text_label("No custom collections yet", "muted"))
            return
        for collection_id, name in sorted(self.store.collections.items(), key=lambda item: item[1].lower()):
            count = sum(collection_id in game.collections for game in self.store.records)
            control = QPushButton(f"\u25a1  {name}  {count}")
            control.setObjectName("libraryButton")
            control.clicked.connect(lambda _checked=False, key=collection_id: self.category_selected.emit(f"Collection:{key}"))
            self.collection_box.addWidget(control)

    def _import_files(self) -> None:
        paths, _filter = QFileDialog.getOpenFileNames(self, "Import PGN Files", "", "PGN Files (*.pgn)")
        if paths:
            self.files_imported.emit(paths)

    def _import_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Import Folder")
        if folder:
            self.folder_imported.emit(folder)

    def _paste_pgn(self) -> None:
        content, accepted = QInputDialog.getMultiLineText(self, "Paste PGN", "PGN text:")
        if accepted and content.strip():
            self.pgn_pasted.emit(content)


class GamesTable(Card):
    game_selected = Signal(object)
    open_requested = Signal(object)
    games_changed = Signal(list)
    selection_cleared = Signal()

    def __init__(self, store: GameLibraryStore) -> None:
        super().__init__()
        self.store = store
        self.category = "All"
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 14, 0, 10)
        layout.setSpacing(10)
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(16, 0, 16, 0)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search games, players, events, tags...")
        self.search.setObjectName("searchField")
        self.search.textChanged.connect(self.refresh)
        toolbar.addWidget(self.search, 1)
        refine = button("Filter", compact=True)
        refine.clicked.connect(lambda: QMessageBox.information(self, "Game Filters", "Use the game-type filters below or search by player and event."))
        toolbar.addWidget(refine)
        toolbar.addWidget(text_label("Sort by:", "secondary"))
        self.sort = QComboBox()
        self.sort.setObjectName("interactiveField")
        self.sort.addItems(["Date (Newest)", "Date (Oldest)", "White Player", "Black Player", "Event", "Result", "Opening", "Recently Opened", "Favorites First"])
        self.sort.currentIndexChanged.connect(self.refresh)
        toolbar.addWidget(self.sort)
        view_group = QButtonGroup(self)
        view_group.setExclusive(True)
        list_view = button("List", compact=True)
        comfortable = button("Comfort", compact=True)
        for choice in (list_view, comfortable):
            choice.setCheckable(True)
            view_group.addButton(choice)
            toolbar.addWidget(choice)
        list_view.setChecked(True)
        list_view.clicked.connect(lambda: self.table.verticalHeader().setDefaultSectionSize(37))
        comfortable.clicked.connect(lambda: self.table.verticalHeader().setDefaultSectionSize(52))
        columns = button("Columns", compact=True)
        columns.setToolTip("Choose visible game columns")
        columns.clicked.connect(self.choose_columns)
        toolbar.addWidget(columns)
        layout.addLayout(toolbar)

        filters = QHBoxLayout()
        filters.setContentsMargins(16, 0, 16, 0)
        group = QButtonGroup(self)
        group.setExclusive(True)
        for index, title in enumerate(("All", "Recent", "Annotated", "Imported", "Tournament", "Training")):
            control = button(title, compact=True)
            control.setCheckable(True)
            control.setChecked(index == 0)
            control.clicked.connect(lambda _checked=False, value=title: self.set_category(value))
            group.addButton(control)
            filters.addWidget(control)
        add_filter = button("+", compact=True)
        add_filter.setEnabled(False)
        add_filter.setToolTip("Reserved for saved filters.")
        filters.addWidget(add_filter)
        filters.addStretch()
        layout.addLayout(filters)

        self.table = QTableWidget(0, 8)
        self.table.setObjectName("gamesTable")
        self.table.setHorizontalHeaderLabels(["", "Name", "White", "Black", "Result", "Date", "Type", "Event / Tags"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        for column, width in enumerate((32, 165, 125, 130, 64, 102, 112)):
            self.table.setColumnWidth(column, width)
        self.table.horizontalHeader().setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(37)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._emit_selection)
        self.table.cellClicked.connect(self._cell_clicked)
        self.table.itemDoubleClicked.connect(lambda _item: self._emit_selection(open_requested=True))
        layout.addWidget(self.table, 1)
        self.summary = text_label("", "secondary")
        layout.addWidget(self.summary)
        self.refresh()

    def set_category(self, category: str) -> None:
        self.category = category
        self.refresh()

    def refresh(self) -> None:
        selected = self.current_game()
        selected_id = selected.id if selected is not None else ""
        query = self.search.text().strip().lower()
        games = list(self.store.records)
        if self.category != "All":
            games = [game for game in games if self._matches_filter(game, self.category)]
        if query:
            games = [game for game in games if query in self._search_text(game)]
        sort_name = self.sort.currentText()
        key = {
            "Date (Newest)": lambda game: game.date,
            "Date (Oldest)": lambda game: game.date,
            "White Player": lambda game: game.white.lower(),
            "Black Player": lambda game: game.black.lower(),
            "Event": lambda game: game.event.lower(),
            "Result": lambda game: game.result,
            "Opening": lambda game: game.opening.lower(),
            "Recently Opened": lambda game: game.last_opened_at,
            "Favorites First": lambda game: (game.favorite, game.date),
        }[sort_name]
        games.sort(key=key, reverse=sort_name in ("Date (Newest)", "Recently Opened", "Favorites First"))
        self.visible_games = games
        self.table.setRowCount(len(games))
        for row, game in enumerate(games):
            tags = ", ".join(game.tags)
            values = ["\u2605" if game.favorite else "\u2606", game.name, game.white, game.black, game.result, game.date, game.game_type, f"{game.event}  |  {tags}"]
            for col, value in enumerate(values):
                self.table.setItem(row, col, QTableWidgetItem(value))
        self.summary.setText(f"Showing 1 - {len(games)} of {len(self.store.records)} games" if games else "No games match this filter.")
        self.games_changed.emit(self.store.records)
        if games:
            target = next((row for row, game in enumerate(games) if game.id == selected_id), 0)
            self.table.selectRow(target)
        else:
            self.selection_cleared.emit()

    def delete_selected(self) -> None:
        selected = self.current_game()
        if selected:
            self.store.remove_from_library(selected)
            self.refresh()

    def current_game(self) -> GameRecord | None:
        row = self.table.currentRow()
        return self.visible_games[row] if row >= 0 and row < len(self.visible_games) else None

    def _emit_selection(self, *, open_requested: bool = False) -> None:
        game = self.current_game()
        if game:
            self.game_selected.emit(game)
            if open_requested:
                self.open_requested.emit(game)

    def _cell_clicked(self, row: int, column: int) -> None:
        if column != 0 or row >= len(self.visible_games):
            return
        self.store.toggle_favorite(self.visible_games[row])
        self.refresh()

    @staticmethod
    def _search_text(game: GameRecord) -> str:
        return " ".join(
            [game.name, game.white, game.black, game.result, game.date, game.event, game.site, game.opening, game.eco, " ".join(game.tags), game.notes, game.source_path]
        ).lower()

    @staticmethod
    def _matches_filter(game: GameRecord, value: str) -> bool:
        if value.startswith("Collection:"):
            return value.split(":", 1)[1] in game.collections
        return {
            "Recent": game.source_type == "Imported",
            "Saved": game.source_type == "Saved",
            "Match": game.game_type == "AI Match",
            "Human vs AI": game.game_type == "Human vs AI",
            "Analyzed": game.analyzed,
            "Favorites": game.favorite,
            "Annotated": bool(game.notes),
            "Imported": game.source_type == "Imported",
            "Tournament": game.game_type == "Tournament",
            "Training": "Training" in game.tags,
        }.get(value, True)

    def choose_columns(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Visible Columns")
        layout = QVBoxLayout(dialog)
        checks = []
        for column, name in enumerate(("Marker", "Name", "White", "Black", "Result", "Date", "Type", "Event / Tags")):
            checked = QCheckBox(name)
            checked.setChecked(not self.table.isColumnHidden(column))
            checks.append((column, checked))
            layout.addWidget(checked)
        controls = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        controls.accepted.connect(dialog.accept)
        controls.rejected.connect(dialog.reject)
        layout.addWidget(controls)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            for column, checked in checks:
                self.table.setColumnHidden(column, not checked.isChecked())


class GameDetail(Card):
    open_analyze = Signal(object)
    load_play = Signal(object)
    delete_requested = Signal(object)
    game_changed = Signal()

    def __init__(self, store: GameLibraryStore) -> None:
        super().__init__()
        self.store = store
        self.current: GameRecord | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)
        self.title = text_label("Select a game", "detailTitle")
        layout.addWidget(self.title)
        summary = QHBoxLayout()
        self.preview = WoodenBoard(starting_position=True, minimum_size=155)
        self.preview.setFixedSize(180, 180)
        summary.addWidget(self.preview)
        self.metadata = text_label("", "detailMeta")
        self.metadata.setWordWrap(True)
        summary.addWidget(self.metadata, 1)
        layout.addLayout(summary)
        open_button = button("\u2197  Open in Analyze", primary=True, compact=True)
        open_button.clicked.connect(lambda: self.current and self.open_analyze.emit(self.current))
        layout.addWidget(open_button)
        commands = QGridLayout()
        commands.setHorizontalSpacing(9)
        commands.setVerticalSpacing(9)
        load = button("Load Game", compact=True)
        load.clicked.connect(lambda: self.current and self.load_play.emit(self.current))
        commands.addWidget(load, 0, 0)
        view = button("View PGN", compact=True)
        view.clicked.connect(self.view_pgn)
        commands.addWidget(view, 0, 1)
        self.favorite = button("Add to Favorites", compact=True)
        self.favorite.setCheckable(True)
        self.favorite.clicked.connect(self.toggle_favorite)
        commands.addWidget(self.favorite, 1, 0)
        collection = button("Add to Collection", compact=True)
        collection.clicked.connect(self.add_to_collection)
        commands.addWidget(collection, 2, 0)
        delete = button("Delete", compact=True)
        delete.clicked.connect(self._delete)
        commands.addWidget(delete, 1, 1)
        layout.addLayout(commands)
        layout.addWidget(text_label("NOTES", "sectionHeader"))
        self.notes = QTextEdit()
        self.notes.setObjectName("notesEditor")
        self.notes.setPlaceholderText("Write analysis notes...")
        self.notes.textChanged.connect(self._save_notes)
        layout.addWidget(self.notes, 1)
        layout.addWidget(text_label("RECENT COMMENTS", "sectionHeader"))
        layout.addWidget(text_label("No comments saved for this game.", "secondary"))

    def set_game(self, game: GameRecord) -> None:
        self.current = game
        marked = "\u2605" if game.favorite else "\u2606"
        self.title.setText(f"{marked}  " + game.name)
        self.metadata.setText(
            f"Event    {game.event}\nDate     {game.date}\nWhite    {game.white}\nBlack    {game.black}\n"
            f"Result   {game.result}\nOpening  {game.opening}\nECO      {game.eco or '-'}\nTags     {', '.join(game.tags) or '-'}"
        )
        if game.final_fen:
            self.preview.set_fen(game.final_fen)
        self.notes.blockSignals(True)
        self.notes.setPlainText(game.notes)
        self.notes.blockSignals(False)
        self.favorite.blockSignals(True)
        self.favorite.setChecked(game.favorite)
        self.favorite.setText("Remove Favorite" if game.favorite else "Add to Favorites")
        self.favorite.blockSignals(False)

    def clear_game(self) -> None:
        self.current = None
        self.title.setText("Select a game")
        self.metadata.setText("Import or save a PGN to view its details.")
        self.preview.set_fen(STANDARD_STARTING_FEN)
        self.notes.blockSignals(True)
        self.notes.clear()
        self.notes.blockSignals(False)
        self.favorite.setChecked(False)
        self.favorite.setText("Add to Favorites")

    def _save_notes(self) -> None:
        if self.current is not None:
            self.store.set_notes(self.current, self.notes.toPlainText())
            self.game_changed.emit()

    def _delete(self) -> None:
        if self.current and QMessageBox.question(
            self,
            "Remove Game",
            f'Remove "{self.current.name}" from the library?\n\nThe original and managed PGN files will not be deleted.',
        ) == QMessageBox.StandardButton.Yes:
            self.delete_requested.emit(self.current)

    def toggle_favorite(self, checked: bool) -> None:
        if self.current is not None:
            self.current.favorite = not checked
            self.store.toggle_favorite(self.current)
            self.favorite.setText("Remove Favorite" if checked else "Add to Favorites")
            self.set_game(self.current)
            self.game_changed.emit()

    def add_to_collection(self) -> None:
        if self.current is None:
            return
        if not self.store.collections:
            QMessageBox.information(self, "Collections", "Create a collection with the + button in the Library panel first.")
            return
        names = sorted(self.store.collections.values(), key=str.lower)
        name, accepted = QInputDialog.getItem(self, "Add to Collection", "Collection:", names, 0, False)
        if accepted and name:
            collection_id = next(key for key, value in self.store.collections.items() if value == name)
            self.store.add_to_collection(self.current, collection_id)
            self.game_changed.emit()

    def view_pgn(self) -> None:
        if self.current:
            dialog = QDialog(self)
            dialog.setWindowTitle(f"PGN - {self.current.name}")
            dialog.resize(640, 460)
            layout = QVBoxLayout(dialog)
            text = QTextEdit()
            text.setReadOnly(True)
            text.setPlainText(self.store.pgn_text(self.current))
            layout.addWidget(text)
            actions = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Close)
            copy = actions.addButton("Copy PGN", QDialogButtonBox.ButtonRole.ActionRole)
            copy.clicked.connect(lambda: QApplication.clipboard().setText(text.toPlainText()))
            actions.rejected.connect(dialog.reject)
            actions.button(QDialogButtonBox.StandardButton.Save).clicked.connect(lambda: self._save_pgn_as(text.toPlainText()))
            layout.addWidget(actions)
            dialog.exec()

    def _save_pgn_as(self, content: str) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save PGN As", f"{self.current.name if self.current else 'game'}.pgn", "PGN Files (*.pgn)")
        if path:
            Path(path).write_text(content, encoding="utf-8")


class GamesPage(QWidget):
    open_analyze = Signal(object)
    load_play = Signal(object)

    def __init__(self, store: GameLibraryStore) -> None:
        super().__init__()
        self.store = store
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 10, 18, 14)
        layout.setSpacing(6)
        self.notice = preview_notice("")
        layout.addWidget(self.notice)
        body = QHBoxLayout()
        body.setSpacing(10)
        self.library = GamesLibrary(store)
        self.library.setFixedWidth(258)
        self.center = GamesTable(store)
        self.detail = GameDetail(store)
        self.detail.setFixedWidth(378)
        self.library.category_selected.connect(self.center.set_category)
        self.library.files_imported.connect(self.import_files)
        self.library.folder_imported.connect(self.import_folder)
        self.library.pgn_pasted.connect(self.import_text)
        self.center.games_changed.connect(self.library.update_counts)
        self.center.game_selected.connect(self.detail.set_game)
        self.center.selection_cleared.connect(self.detail.clear_game)
        self.center.open_requested.connect(self.open_analyze.emit)
        self.detail.delete_requested.connect(lambda _record: self.center.delete_selected())
        self.detail.game_changed.connect(self.center.refresh)
        self.detail.open_analyze.connect(self.open_analyze.emit)
        self.detail.load_play.connect(self.load_play.emit)
        selected = self.center.current_game()
        if selected:
            self.detail.set_game(selected)
        else:
            self.detail.clear_game()
        body.addWidget(self.library)
        body.addWidget(self.center, 1)
        body.addWidget(self.detail)
        layout.addLayout(body, 1)
        self.refresh_status()

    def import_files(self, paths: list[str]) -> None:
        added: list[GameRecord] = []
        warnings: list[str] = []
        for path in paths:
            records, errors = self.store.import_file(path)
            added.extend(records)
            warnings.extend(errors)
        self._after_import(added, warnings)

    def import_folder(self, folder: str) -> None:
        added, warnings = self.store.import_folder(folder)
        self._after_import(added, warnings)

    def import_text(self, content: str) -> None:
        added, warnings = self.store.import_text(content, source_type="Imported")
        self._after_import(added, warnings)

    def _after_import(self, added: list[GameRecord], warnings: list[str]) -> None:
        self.center.refresh()
        self.refresh_status()
        message = f"Import completed: {len(added)} game(s) added."
        if warnings:
            message += "\n\nWarnings:\n" + "\n".join(warnings[:8])
            QMessageBox.warning(self, "Import PGN", message)
        else:
            QMessageBox.information(self, "Import PGN", message)

    def refresh_status(self) -> None:
        count = len(self.store.records)
        self.notice.setText(
            f"PGN LIBRARY READY - {count} game(s) indexed"
            if count
            else "NO GAMES IN YOUR LIBRARY - import PGN files or save a game from Play / AI Match."
        )
        self.library.update_counts(self.store.records)
        self.center.refresh()


class OpeningLibrary(Card):
    opening_selected = Signal(object)

    def __init__(self, openings: list[OpeningRecord], user_state: dict, persist) -> None:
        super().__init__()
        self.openings = openings
        self.state = user_state
        self.persist = persist
        self.favorite_names = self.state["favorites"]
        self.filter_name = "All"
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(9)
        layout.addWidget(text_label("OPENING LIBRARY", "sectionHeader"))
        search_row = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setObjectName("searchField")
        self.search.setPlaceholderText("Search openings by name or ECO...")
        self.search.textChanged.connect(self.refresh)
        search_row.addWidget(self.search, 1)
        library_filter = button("Filter", compact=True)
        library_filter.clicked.connect(lambda: QMessageBox.information(self, "Opening Filters", "Filter by side, favorites, popularity, or recent views below."))
        search_row.addWidget(library_filter)
        layout.addLayout(search_row)
        filters = QHBoxLayout()
        self.filters = QButtonGroup(self)
        self.filters.setExclusive(True)
        for index, title in enumerate(("All", "White", "Black", "Favorites", "Popular", "Recent")):
            control = button(title, compact=True)
            control.setObjectName("openingFilter")
            control.setCheckable(True)
            control.setChecked(index == 0)
            control.clicked.connect(lambda _checked=False, value=title: self.set_filter(value))
            if title == "Popular":
                control.setEnabled(False)
                control.setToolTip("Popularity statistics are not present in openings.tsv.")
            self.filters.addButton(control)
            filters.addWidget(control)
        layout.addLayout(filters)
        sort_row = QHBoxLayout()
        sort_row.addWidget(text_label("Sort by:", "secondary"))
        self.sort = QComboBox()
        self.sort.setObjectName("interactiveField")
        self.sort.addItems(["Book Order", "ECO Code", "Name", "Recently Viewed"])
        self.sort.currentIndexChanged.connect(self.refresh)
        sort_row.addWidget(self.sort)
        sort_row.addStretch()
        layout.addLayout(sort_row)
        self.list = QListWidget()
        self.list.setObjectName("openingList")
        self.list.currentRowChanged.connect(self.select_row)
        layout.addWidget(self.list, 1)
        self.list_status = text_label("", "muted")
        layout.addWidget(self.list_status)
        layout.addWidget(divider())
        recent_title = QHBoxLayout()
        recent_title.addWidget(text_label("RECENTLY VIEWED", "sectionHeader"))
        recent_title.addStretch()
        clear = QPushButton("Clear")
        clear.setObjectName("linkButton")
        clear.clicked.connect(self.clear_recent)
        recent_title.addWidget(clear)
        layout.addLayout(recent_title)
        self.recent = QVBoxLayout()
        self.recent.setSpacing(4)
        layout.addLayout(self.recent)
        self.recent_names = self.state["recent"]
        self.visible: list[OpeningRecord] = []
        self.refresh()
        self.refresh_recent()

    def set_filter(self, name: str) -> None:
        self.filter_name = name
        self.refresh()

    def refresh(self) -> None:
        query = self.search.text().lower().strip()
        visible = list(self.openings)
        if self.filter_name in ("White", "Black"):
            visible = [opening for opening in visible if opening.side == self.filter_name]
        elif self.filter_name == "Favorites":
            visible = [opening for opening in visible if opening.record_id in self.favorite_names]
        elif self.filter_name == "Popular":
            visible = visible[: min(200, len(visible))]
        elif self.filter_name == "Recent":
            visible = [opening for opening in visible if opening.record_id in self.recent_names]
        if query:
            visible = [opening for opening in visible if query in f"{opening.name} {opening.eco} {' '.join(opening.san_moves)}".lower()]
        if self.sort.currentText() == "Name":
            visible.sort(key=lambda opening: opening.name)
        elif self.sort.currentText() == "ECO Code":
            visible.sort(key=lambda opening: (opening.eco, opening.name))
        elif self.sort.currentText() == "Recently Viewed":
            ordering = {identifier: index for index, identifier in enumerate(self.recent_names)}
            visible.sort(key=lambda opening: ordering.get(opening.record_id, len(ordering)))
        else:
            visible.sort(key=lambda opening: (len(opening.san_moves), opening.eco, opening.name))
        available_count = len(visible)
        self.visible = visible[:250]
        self.list.blockSignals(True)
        self.list.clear()
        for opening in self.visible:
            favorite_mark = "\u2605" if opening.record_id in self.favorite_names else "\u2606"
            self.list.addItem(f"{favorite_mark}  {opening.name}\n     {opening.eco}    {' '.join(opening.san_moves[:5])}")
        if self.visible:
            self.list.setCurrentRow(0)
        self.list.blockSignals(False)
        self.list_status.setText(
            f"Showing {len(self.visible):,} of {available_count:,} matching lines"
            if available_count > len(self.visible)
            else f"{available_count:,} matching line(s)"
        )

    def select_row(self, row: int) -> None:
        if 0 <= row < len(self.visible):
            self.select_opening(self.visible[row])

    def select_opening(self, opening: OpeningRecord) -> None:
        if opening.record_id in self.recent_names:
            self.recent_names.remove(opening.record_id)
        self.recent_names.insert(0, opening.record_id)
        self.recent_names[:] = self.recent_names[:3]
        self.persist()
        self.refresh_recent()
        self.opening_selected.emit(opening)

    def refresh_recent(self) -> None:
        while self.recent.count():
            widget = self.recent.takeAt(0).widget()
            if widget:
                widget.deleteLater()
        for index, identifier in enumerate(self.recent_names):
            opening = next((item for item in self.openings if item.record_id == identifier), None)
            if opening is None:
                continue
            control = QPushButton(f'\u25f4  {opening.name} ({opening.eco})       {("Just now", "Recently", "Earlier")[index]}')
            control.setObjectName("recentOpening")
            control.clicked.connect(lambda _checked=False, value=opening: self.select_opening(value))
            self.recent.addWidget(control)

    def clear_recent(self) -> None:
        if QMessageBox.question(self, "Clear Recent Openings", "Clear the recently viewed opening list?") == QMessageBox.StandardButton.Yes:
            self.recent_names.clear()
            self.persist()
            self.refresh_recent()


class OpeningStudyBoard(Card):
    open_analyze = Signal(object)
    favorite_changed = Signal(str, bool)

    def __init__(self, openings: list[OpeningRecord], user_state: dict, persist) -> None:
        super().__init__()
        self.openings = openings
        self.current = openings[0] if openings else None
        self.positions = (STANDARD_STARTING_FEN,)
        self.ply = 0
        self.state = user_state
        self.persist = persist
        self.favorite_names = self.state["favorites"]
        self.repertoire_names = self.state["repertoire"]
        self.note_store = self.state["notes"]
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 12, 15, 10)
        layout.setSpacing(8)
        top = QHBoxLayout()
        top.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.name = text_label("", "openingTitle")
        self.name.setMinimumSize(225, 55)
        self.name.setMaximumWidth(285)
        self.name.setWordWrap(True)
        self.eco = text_label("", "ecoBadge")
        top.addWidget(self.name)
        top.addWidget(self.eco)
        top.addStretch()
        self.favorite_button = button("\u2605  Favorite", compact=True)
        self.favorite_button.setCheckable(True)
        self.favorite_button.setChecked(True)
        self.favorite_button.clicked.connect(self.toggle_favorite)
        self.repertoire = button("\u25a1  In Repertoire", compact=True)
        self.repertoire.setCheckable(True)
        self.repertoire.clicked.connect(self.toggle_repertoire)
        self.notes = button("Notes (0)", compact=True)
        self.notes.clicked.connect(self.open_notes)
        top.addWidget(self.favorite_button)
        top.addWidget(self.repertoire)
        top.addWidget(self.notes)
        layout.addLayout(top)
        self.tags = text_label("", "openingTags")
        layout.addWidget(self.tags)
        board_row = QHBoxLayout()
        self.board = WoodenBoard(minimum_size=332)
        self.board.setFixedSize(346, 346)
        board_row.addWidget(self.board, 1)
        tools = QVBoxLayout()
        tools.setSpacing(8)
        flip = button("\u21bb\nFlip Board", compact=True)
        flip.clicked.connect(self.board.flip)
        tools.addWidget(flip)
        settings = button("\u2699\nSettings", compact=True)
        settings.clicked.connect(lambda: QMessageBox.information(self, "Board Settings", "Board theme and coordinate settings will be configured here."))
        tools.addWidget(settings)
        tools.addStretch()
        board_row.addLayout(tools)
        layout.addLayout(board_row, 1)
        nav = QHBoxLayout()
        for glyph, action in (("\u23ee", self.first), ("\u2039", self.previous)):
            control = button(glyph, compact=True)
            control.clicked.connect(action)
            nav.addWidget(control)
        self.moves_view = QScrollArea()
        self.moves_view.setObjectName("openingLineScroll")
        self.moves_view.setWidgetResizable(True)
        self.moves_view.setFrameShape(QFrame.Shape.NoFrame)
        self.moves_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.moves_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.moves_widget = QWidget()
        self.moves = QHBoxLayout(self.moves_widget)
        self.moves.setContentsMargins(0, 0, 0, 0)
        self.moves.setSpacing(2)
        self.moves_view.setWidget(self.moves_widget)
        nav.addWidget(self.moves_view, 1)
        for glyph, action in (("\u203a", self.next), ("\u23ed", self.last)):
            control = button(glyph, compact=True)
            control.clicked.connect(action)
            nav.addWidget(control)
        layout.addLayout(nav)
        eval_row = QHBoxLayout()
        eval_row.addWidget(text_label("--", "evalBadge"))
        eval_row.addWidget(text_label("Source: openings.tsv  |  Eval not available", "secondary"))
        eval_row.addStretch()
        analyze = button("\u2197  Open in Analyze", primary=True, compact=True)
        analyze.clicked.connect(self.send_to_analyze)
        eval_row.addWidget(analyze)
        layout.addLayout(eval_row)
        if self.current is not None:
            self.load_opening(self.current)

    def load_opening(self, opening: OpeningRecord) -> None:
        if not opening.materialize():
            QMessageBox.warning(self, "Opening Line", "This line could not be reconstructed from the opening source.")
            return
        self.current = opening
        self.name.setText(opening.name)
        self.eco.setText(opening.eco)
        self.tags.setText(f"Book line     Source: {Path(opening.source).name}     Side: {opening.side}")
        favorite = opening.record_id in self.favorite_names
        self.favorite_button.setChecked(favorite)
        self.favorite_button.setText(("\u2605" if favorite else "\u2606") + "  Favorite")
        stored = opening.record_id in self.repertoire_names
        self.repertoire.setChecked(stored)
        self.repertoire.setText(("\u2713" if stored else "\u25a1") + "  In Repertoire")
        self.notes.setText(f"Notes ({len(self.note_store.get(opening.record_id, []))})")
        self.positions = opening.positions
        self.ply = len(self.positions) - 1
        self._update_line()

    def _update_line(self) -> None:
        previous_move = None
        if self.current is not None and self.ply:
            previous_move = self.current.uci_moves[self.ply - 1]
        self.board.set_fen(self.positions[self.ply], previous_move)
        while self.moves.count():
            widget = self.moves.takeAt(0).widget()
            if widget:
                widget.deleteLater()
        if self.current is None:
            return
        for index, san in enumerate(self.current.san_moves):
            prefix = f"{index // 2 + 1}. " if index % 2 == 0 else "..."
            move = QPushButton(prefix + san)
            move.setObjectName("lineMove")
            move.setProperty("selected", index + 1 == self.ply)
            move.clicked.connect(lambda _checked=False, target=index + 1: self.go_to(target))
            self.moves.addWidget(move)
        self.moves.addStretch()

    def go_to(self, ply: int) -> None:
        self.ply = max(0, min(ply, len(self.positions) - 1))
        self._update_line()

    def first(self) -> None:
        self.go_to(0)

    def previous(self) -> None:
        self.go_to(self.ply - 1)

    def next(self) -> None:
        self.go_to(self.ply + 1)

    def last(self) -> None:
        self.go_to(len(self.positions) - 1)

    def toggle_favorite(self, checked: bool) -> None:
        self.favorite_button.setText(("\u2605" if checked else "\u2606") + "  Favorite")
        if checked:
            self.favorite_names.add(self.current.record_id)
        else:
            self.favorite_names.discard(self.current.record_id)
        self.persist()
        self.favorite_changed.emit(self.current.record_id, checked)

    def toggle_repertoire(self, checked: bool) -> None:
        if checked:
            self.repertoire_names.add(self.current.record_id)
        else:
            self.repertoire_names.discard(self.current.record_id)
        self.repertoire.setText(("\u2713" if checked else "\u25a1") + "  In Repertoire")
        self.persist()

    def open_notes(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Notes - {self.current.name}")
        dialog.setMinimumSize(430, 300)
        layout = QVBoxLayout(dialog)
        notes = QListWidget()
        layout.addWidget(notes, 1)
        stored = self.note_store.setdefault(self.current.record_id, [])
        notes.addItems(stored)
        controls = QHBoxLayout()
        add = button("+  Add Note", primary=True, compact=True)
        remove = button("Delete Selected", compact=True)
        close = button("Close", compact=True)
        controls.addWidget(add)
        controls.addWidget(remove)
        controls.addStretch()
        controls.addWidget(close)
        layout.addLayout(controls)

        def add_note() -> None:
            note, accepted = QInputDialog.getMultiLineText(dialog, "Add Note", "Opening note:")
            if accepted and note.strip():
                stored.append(note.strip())
                notes.addItem(note.strip())
                self.notes.setText(f"Notes ({len(stored)})")
                self.persist()

        add.clicked.connect(add_note)

        def remove_note() -> None:
            row = notes.currentRow()
            if row >= 0 and QMessageBox.question(dialog, "Delete Note", "Delete the selected opening note?") == QMessageBox.StandardButton.Yes:
                stored.pop(row)
                notes.takeItem(row)
                self.notes.setText(f"Notes ({len(stored)})")
                self.persist()

        remove.clicked.connect(remove_note)
        close.clicked.connect(dialog.accept)
        dialog.exec()

    def send_to_analyze(self) -> None:
        if self.current is not None:
            self.open_analyze.emit(self.current.uci_moves[: self.ply])

    def load_related(self, opening: OpeningRecord) -> None:
        self.load_opening(opening)


class Continuations(Card):
    line_selected = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.setFixedHeight(145)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(14, 11, 14, 11)
        self.layout.setSpacing(6)
        self.layout.addWidget(text_label("COMMON CONTINUATIONS", "sectionHeader"))
        self.scroll = QScrollArea()
        self.scroll.setObjectName("continuationScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.rows = QWidget()
        self.row_layout = QVBoxLayout(self.rows)
        self.row_layout.setContentsMargins(0, 0, 0, 0)
        self.row_layout.setSpacing(5)
        self.scroll.setWidget(self.rows)
        self.layout.addWidget(self.scroll, 1)
        self.empty = text_label("No longer book continuation is available for this line.", "secondary")
        self.row_layout.addWidget(self.empty)

    def set_opening(self, selected: OpeningRecord, openings: list[OpeningRecord]) -> None:
        while self.row_layout.count():
            item = self.row_layout.takeAt(0)
            if item.widget() and item.widget() is not self.empty:
                item.widget().deleteLater()
        candidates = [
            opening for opening in openings
            if len(opening.san_moves) > len(selected.san_moves)
            and opening.san_moves[: len(selected.san_moves)] == selected.san_moves
        ][:5]
        self.empty.setVisible(not candidates)
        self.row_layout.addWidget(self.empty)
        for opening in candidates:
            row = QHBoxLayout()
            next_text = " ".join(opening.san_moves[len(selected.san_moves): len(selected.san_moves) + 2])
            choose = QPushButton(next_text)
            choose.setObjectName("continuation")
            choose.clicked.connect(lambda _checked=False, value=opening: self.line_selected.emit(value))
            row.addWidget(choose, 2)
            row.addWidget(text_label(opening.eco, "blueTextLarge"))
            row.addWidget(text_label("Stats --", "statValue"), 2)
            row.addWidget(text_label("Games --", "secondary"))
            container = QWidget()
            container.setLayout(row)
            self.row_layout.addWidget(container)
        self.row_layout.addStretch()


class OpeningInfo(Card):

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 10)
        self.tabs = QTabBar()
        for title in ("Overview", "Key Ideas", "Traps", "Transpositions", "Stats"):
            self.tabs.addTab(title)
        layout.addWidget(self.tabs)
        self.stack = QStackedWidget()
        self.stack.addWidget(self._overview())
        self.pages: dict[str, QLabel] = {}
        for title, text in (
            ("KEY IDEAS", "No curated strategic notes are stored for this opening line."),
            ("TRAPS", "No verified trap data is available in the current opening source."),
            ("TRANSPOSITIONS", "Related book lines with the same ECO family are listed below."),
            ("STATS", "White wins --\nDraws --\nBlack wins --\nAverage evaluation --\nGames analyzed --"),
        ):
            panel = QWidget()
            panel_layout = QVBoxLayout(panel)
            panel_layout.addWidget(text_label(title, "sectionHeader"))
            body = text_label(text, "detailMeta")
            body.setWordWrap(True)
            self.pages[title] = body
            panel_layout.addWidget(body)
            panel_layout.addStretch()
            self.stack.addWidget(panel)
        self.tabs.currentChanged.connect(self.stack.setCurrentIndex)
        layout.addWidget(self.stack, 1)

    def set_opening(self, opening: OpeningRecord) -> None:
        self.overview.setText(
            f"{opening.name} ({opening.eco}) is available as a verified move line from the local ECO opening book.\n\n"
            f"Current representative line: {format_san_move_list(list(opening.san_moves), '').strip()}"
        )
        self.plans.setText("Strategic plans: Not available in the current TSV source.")
        self.statistics.setText("White --      Draw --      Black --\nAvg. Eval --       Games Analyzed --       GM --")
        self.pages["TRANSPOSITIONS"].setText(
            f"Look for related variations in ECO family {opening.eco[:1] or '--'}.\n"
            "Position transposition data is not yet indexed for this library view."
        )

    def _overview(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(15, 9, 15, 5)
        layout.addWidget(text_label("OPENING OVERVIEW", "sectionHeader"))
        self.overview = text_label("Select an opening line.", "detailMeta")
        self.overview.setWordWrap(True)
        layout.addWidget(self.overview)
        layout.addWidget(divider())
        layout.addWidget(text_label("COMMON PLANS", "sectionHeader"))
        self.plans = text_label("Strategic plans: Not available in the current TSV source.", "statValue")
        self.plans.setWordWrap(True)
        layout.addWidget(self.plans)
        layout.addWidget(divider())
        layout.addWidget(text_label("STATISTICS", "sectionHeader"))
        self.statistics = text_label("White --      Draw --      Black --\nAvg. Eval --       Games Analyzed --       GM --", "statValue")
        layout.addWidget(self.statistics)
        layout.addStretch()
        return panel


class OpeningRelated(Card):
    opening_requested = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 11, 14, 11)
        layout.setSpacing(7)
        title = QHBoxLayout()
        title.addWidget(text_label("RELATED OPENINGS", "sectionHeader"))
        title.addStretch()
        view_all = QPushButton("View All")
        view_all.setObjectName("linkButton")
        view_all.clicked.connect(self.show_all)
        title.addWidget(view_all)
        layout.addLayout(title)
        self.rows = QVBoxLayout()
        layout.addLayout(self.rows)
        self.empty = text_label("No related lines selected.", "secondary")
        layout.addWidget(self.empty)
        self.related: list[OpeningRecord] = []

    def show_all(self) -> None:
        text = "\n".join(f"{line.eco}  {line.name}" for line in self.related) or "No related book lines available."
        QMessageBox.information(self, "Related Openings", text)

    def set_opening(self, opening: OpeningRecord, openings: list[OpeningRecord]) -> None:
        while self.rows.count():
            item = self.rows.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        family = opening.eco[:1]
        self.related = [
            candidate for candidate in openings
            if candidate.record_id != opening.record_id and candidate.eco.startswith(family)
            and (candidate.name.startswith(opening.name.split(":")[0]) or candidate.eco == opening.eco)
        ][:3]
        self.empty.setVisible(not self.related)
        for candidate in self.related:
            control = QPushButton(f"{candidate.name}     {candidate.eco}")
            control.setObjectName("relatedOpening")
            control.clicked.connect(lambda _checked=False, value=candidate: self.opening_requested.emit(value))
            self.rows.addWidget(control)


class ModelGames(Card):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 11, 14, 11)
        layout.setSpacing(8)
        layout.addWidget(text_label("MODEL GAMES", "sectionHeader"))
        message = text_label("No model-game PGNs are linked to this opening source.", "secondary")
        message.setWordWrap(True)
        layout.addWidget(message)


class OpeningPage(QWidget):
    open_analyze = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.openings = load_opening_records(OPENING_BOOK_PATH)
        self.user_state = opening_user_state(SETTINGS_STORE.load())
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 7, 14, 10)
        layout.setSpacing(5)
        self.notice = preview_notice(
            f"OPENING BOOK READY - {len(self.openings):,} line(s) loaded from assets/openings/openings.tsv"
            if self.openings else
            "OPENING BOOK NOT FOUND - add assets/openings/openings.tsv to browse opening lines"
        )
        layout.addWidget(self.notice)
        body = QHBoxLayout()
        body.setSpacing(10)
        library = OpeningLibrary(self.openings, self.user_state, self.persist_state)
        self.library = library
        library.setFixedWidth(420)
        center = QVBoxLayout()
        center.setSpacing(9)
        study = OpeningStudyBoard(self.openings, self.user_state, self.persist_state)
        self.study = study
        continuations = Continuations()
        self.continuations = continuations
        actions = QHBoxLayout()
        for title in ("Add to Repertoire", "Copy PGN", "Export Line", "Import Book", "Sync Book", "..."):
            action = button(title, compact=True)
            if title == "Add to Repertoire":
                action.clicked.connect(study.repertoire.click)
            elif title == "Copy PGN":
                action.clicked.connect(self.copy_line)
            elif title == "Export Line":
                action.clicked.connect(self.export_line)
            elif title == "Import Book":
                action.setEnabled(False)
                action.setToolTip("User opening-book import will be enabled after an isolated book index is added.")
            elif title == "Sync Book":
                action.clicked.connect(self.sync_book)
            elif title == "...":
                action.setToolTip("Reserved for future opening actions")
                action.setEnabled(False)
            actions.addWidget(action)
        center.addWidget(study, 1)
        center.addWidget(continuations)
        center.addLayout(actions)
        right = QVBoxLayout()
        right.setSpacing(9)
        info = OpeningInfo()
        self.info = info
        right.addWidget(info, 1)
        related = OpeningRelated()
        related.opening_requested.connect(library.select_opening)
        self.related = related
        right.addWidget(related)
        model_games = ModelGames()
        right.addWidget(model_games)
        library.opening_selected.connect(self.selected_opening_changed)
        study.open_analyze.connect(self.open_analyze.emit)
        study.favorite_changed.connect(self._update_favorite(library))
        continuations.line_selected.connect(self.selected_opening_changed)
        body.addWidget(library)
        body.addLayout(center, 1)
        body.addLayout(right, 46)
        layout.addLayout(body, 1)
        if self.openings:
            self.selected_opening_changed(self.openings[0])

    def persist_state(self) -> None:
        SETTINGS_STORE.save(write_opening_user_state(SETTINGS_STORE.load(), self.user_state))

    def selected_opening_changed(self, opening: OpeningRecord) -> None:
        self.study.load_opening(opening)
        self.info.set_opening(opening)
        self.related.set_opening(opening, self.openings)
        self.continuations.set_opening(opening, self.openings)

    def copy_line(self) -> None:
        if self.study.current is None:
            return
        text = format_san_move_list(list(self.study.current.san_moves[: self.study.ply]), "").strip()
        QApplication.clipboard().setText(text)
        QMessageBox.information(self, "Copy PGN", "The current opening line was copied to the clipboard.")

    def export_line(self) -> None:
        if self.study.current is None:
            return
        path, _filter = QFileDialog.getSaveFileName(
            self, "Export Opening Line", f"{self.study.current.name.replace(':', '')}.pgn", "PGN Files (*.pgn)"
        )
        if path:
            text = format_san_move_list(list(self.study.current.san_moves[: self.study.ply]), "*")
            Path(path).write_text(text + "\n", encoding="utf-8")

    def sync_book(self) -> None:
        self.openings[:] = load_opening_records(OPENING_BOOK_PATH)
        self.library.openings = self.openings
        self.library.refresh()
        self.notice.setText(f"OPENING BOOK READY - {len(self.openings):,} line(s) loaded from assets/openings/openings.tsv")
        QMessageBox.information(self, "Sync Book", f"Reloaded {len(self.openings):,} opening lines.")

    @staticmethod
    def _update_favorite(library: OpeningLibrary):
        def update(name: str, checked: bool) -> None:
            if checked:
                library.favorite_names.add(name)
            else:
                library.favorite_names.discard(name)
            library.refresh()
        return update


class DatabaseSources(Card):
    source_selected = Signal(str)

    def __init__(self, datasets: list[DatasetRecord]) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(7)
        header = QHBoxLayout()
        header.addWidget(text_label("DATA SOURCES", "sectionHeader"))
        header.addStretch()
        add_source = button("+", compact=True)
        add_source.clicked.connect(lambda: QMessageBox.information(self, "Data Sources", "Imported books and datasets are added through the actions panel."))
        header.addWidget(add_source)
        layout.addLayout(header)
        counts: dict[str, int] = {}
        for item in datasets:
            counts[item.data_type] = counts.get(item.data_type, 0) + 1
            if item.archived:
                counts["Archived"] = counts.get("Archived", 0) + 1
        self.group = QButtonGroup(self)
        self.source_buttons: dict[str, QPushButton] = {}
        self.group.setExclusive(True)
        for index, (label, data_type) in enumerate((
            ("Self-Play Results", "Self-Play"), ("Annotated Analysis", "Annotated"),
            ("Loss Audit", "Loss Audit"), ("Regression Positions", "Regression"),
            ("Training Datasets", "Training"), ("Analyzed PGNs", "PGN"),
            ("Imported Books", "Book"), ("Archived Runs", "Archived"),
        )):
            choice = QPushButton(f"{label:<26} {counts.get(data_type, 0)}")
            choice.setObjectName("libraryButton")
            choice.setCheckable(True)
            choice.setChecked(index == 0)
            choice.clicked.connect(lambda _checked=False, value=data_type: self.source_selected.emit(value))
            self.group.addButton(choice)
            self.source_buttons[data_type] = choice
            layout.addWidget(choice)
        layout.addWidget(divider())
        layout.addWidget(text_label("QUICK FILTERS", "sectionHeader"))
        for label in ("Recently Updated", "Failed / Warnings", "Large Files (> 1 GB)", "Unvalidated", "Favorites"):
            choice = button(label, compact=True)
            choice.clicked.connect(lambda _checked=False, value=label: self.source_selected.emit(value))
            layout.addWidget(choice)
        layout.addStretch()

    def refresh_counts(self, datasets: list[DatasetRecord]) -> None:
        counts: dict[str, int] = {}
        for item in datasets:
            counts[item.data_type] = counts.get(item.data_type, 0) + 1
            if item.archived:
                counts["Archived"] = counts.get("Archived", 0) + 1
        labels = {
            "Self-Play": "Self-Play Results", "Annotated": "Annotated Analysis",
            "Loss Audit": "Loss Audit", "Regression": "Regression Positions",
            "Training": "Training Datasets", "PGN": "Analyzed PGNs",
            "Book": "Imported Books", "Archived": "Archived Runs",
        }
        for data_type, choice in self.source_buttons.items():
            choice.setText(f"{labels.get(data_type, data_type):<26} {counts.get(data_type, 0)}")


class DatasetExplorer(Card):
    dataset_selected = Signal(object)
    datasets_changed = Signal()

    def __init__(self, store: DatabaseStore) -> None:
        super().__init__()
        self.store = store
        self.datasets = self.store.records
        self.filter_name = "All"
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 13, 0, 10)
        layout.setSpacing(9)
        layout.addWidget(text_label("   DATASET EXPLORER", "sectionHeader"))
        search_row = QHBoxLayout()
        search_row.setContentsMargins(16, 0, 16, 0)
        self.search = QLineEdit()
        self.search.setObjectName("searchField")
        self.search.setPlaceholderText("Search datasets...")
        self.search.textChanged.connect(self.refresh)
        search_row.addWidget(self.search, 1)
        filters = button("Filters", compact=True)
        filters.clicked.connect(lambda: QMessageBox.information(self, "Dataset Filters", "Use the filter chips below or search by experiment tag."))
        search_row.addWidget(filters)
        layout.addLayout(search_row)
        chips = QHBoxLayout()
        chips.setContentsMargins(16, 0, 16, 0)
        group = QButtonGroup(self)
        group.setExclusive(True)
        for index, name in enumerate(("All", "Recent", "Annotated", "Regression", "PGN", "Training", "Failed")):
            chip = button(name, compact=True)
            chip.setCheckable(True)
            chip.setChecked(index == 0)
            chip.clicked.connect(lambda _checked=False, value=name: self.set_filter(value))
            group.addButton(chip)
            chips.addWidget(chip)
        chips.addStretch()
        layout.addLayout(chips)
        self.table = QTableWidget(0, 6)
        self.table.setObjectName("gamesTable")
        self.table.setHorizontalHeaderLabels(["Name", "Type", "Records", "Size", "Updated", "Status"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        for column, width in enumerate((244, 104, 101, 81, 108, 88)):
            self.table.setColumnWidth(column, width)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(41)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self.emit_selection)
        layout.addWidget(self.table, 1)
        self.summary = text_label("", "muted")
        layout.addWidget(self.summary)
        self.refresh()
        self.table.selectRow(0)

    def set_filter(self, value: str) -> None:
        self.filter_name = value
        self.refresh()

    def set_source(self, source: str) -> None:
        source_map = {"Recently Updated": "Recent", "Failed / Warnings": "Failed"}
        self.filter_name = source_map.get(source, source)
        self.refresh()

    def refresh(self) -> None:
        query = self.search.text().strip().lower()
        records = list(self.datasets)
        mapped = {
            "Self-Play": "Self-Play", "Loss Audit": "Loss Audit", "Regression": "Regression",
            "Annotated": "Annotated", "PGN": "PGN", "Training": "Training", "Book": "Book",
        }.get(self.filter_name)
        if mapped:
            records = [row for row in records if row.data_type == mapped]
        elif self.filter_name == "Failed":
            records = [row for row in records if row.status in {"Warning", "Failed", "Missing"}]
        elif self.filter_name == "Recent":
            records = records[:20]
        elif self.filter_name == "Large Files (> 1 GB)":
            records = [row for row in records if row.size_bytes >= 1024 ** 3]
        elif self.filter_name == "Unvalidated":
            records = [row for row in records if row.status == "Unvalidated"]
        elif self.filter_name == "Favorites":
            records = [row for row in records if row.favorite]
        elif self.filter_name == "Archived":
            records = [row for row in records if row.archived]
        if query:
            records = [
                row for row in records
                if query in " ".join((row.name, row.data_type, row.relative_path, row.status, " ".join(row.tags), row.notes)).lower()
            ]
        if self.filter_name != "Archived":
            records = [row for row in records if not row.archived]
        self.visible = records
        self.table.setRowCount(len(records))
        for row, data in enumerate(records):
            for col, value in enumerate((data.name, data.data_type, data.records, data.size_text, data.updated_text, data.status)):
                self.table.setItem(row, col, QTableWidgetItem(value))
        self.summary.setText(f"Showing {len(records):,} of {len(self.datasets):,} scanned datasets")
        if records:
            self.table.selectRow(0)
        self.datasets_changed.emit()

    def current(self) -> DatasetRecord | None:
        row = self.table.currentRow()
        return self.visible[row] if 0 <= row < len(self.visible) else None

    def emit_selection(self) -> None:
        selected = self.current()
        if selected:
            self.dataset_selected.emit(selected)

    def reload(self) -> None:
        self.datasets = self.store.scan()
        self.filter_name = "All"
        self.refresh()


class DatasetDetail(Card):
    notes_saved = Signal(object, str)

    def __init__(self, store: DatabaseStore) -> None:
        super().__init__()
        self.store = store
        self.current: DatasetRecord | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 12, 15, 10)
        layout.setSpacing(8)
        layout.addWidget(text_label("SELECTED DATASET", "sectionHeader"))
        self.name = text_label("", "detailTitle")
        layout.addWidget(self.name)
        self.description = text_label("", "detailMeta")
        layout.addWidget(self.description)
        details = QHBoxLayout()
        self.metadata = text_label("", "detailMeta")
        self.metadata.setWordWrap(True)
        details.addWidget(self.metadata, 1)
        self.stats = text_label("", "statValue")
        details.addWidget(self.stats, 1)
        layout.addLayout(details)
        self.tabs = QTabBar()
        for title in ("Preview", "Stats", "Lineage", "Notes"):
            self.tabs.addTab(title)
        layout.addWidget(self.tabs)
        self.stack = QStackedWidget()
        self.preview = text_label("", "detailMeta")
        self.preview.setWordWrap(True)
        self.stats_page = text_label("No calculated statistics. Validate or run the appropriate CLI workflow to create evidence.", "detailMeta")
        self.stats_page.setWordWrap(True)
        self.lineage = text_label("", "detailMeta")
        self.lineage.setWordWrap(True)
        self.notes = QTextEdit()
        self.notes.setObjectName("notesEditor")
        for widget in (self.preview, self.stats_page, self.lineage, self.notes):
            container = QWidget()
            holder = QVBoxLayout(container)
            holder.addWidget(widget, 1)
            self.stack.addWidget(container)
        self.tabs.currentChanged.connect(self.stack.setCurrentIndex)
        layout.addWidget(self.stack, 1)
        save_notes = button("Save Notes", compact=True)
        save_notes.clicked.connect(self.save_notes)
        layout.addWidget(save_notes, 0, Qt.AlignmentFlag.AlignRight)

    def set_dataset(self, data: DatasetRecord) -> None:
        self.current = data
        self.name.setText(data.name)
        self.description.setText(data.description)
        self.metadata.setText(
            f"Type       {data.data_type}\nUpdated    {data.updated_text}\nStatus     {data.status}\n"
            f"Tags       {', '.join(data.tags) or '--'}\nPath       {data.relative_path}"
        )
        self.stats.setText(f"Files        1\nRecords      {data.records}\nStorage      {data.size_text}\nLast Sync    {data.updated_text}")
        self.preview.setText(self.store.preview(data))
        self.stats_page.setText(
            f"Type: {data.data_type}\nRecords: {data.records}\nSize: {data.size_text}\nStatus: {data.status}\n\n"
            "Calculated distributions are displayed only after a validated report provides them."
        )
        self.lineage.setText(
            f"Source file: {data.relative_path}\nRoot: {data.root}\n\n"
            "Generating command and parent-run metadata are not embedded in this file index."
        )
        self.notes.blockSignals(True)
        self.notes.setPlainText(data.notes)
        self.notes.blockSignals(False)

    def save_notes(self) -> None:
        if self.current is not None:
            self.notes_saved.emit(self.current, self.notes.toPlainText().strip())


class DatabaseJobs(Card):
    def __init__(self, title: str) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 11, 14, 10)
        layout.addWidget(text_label(title, "sectionHeader"))
        self.table = QTableWidget(0, 4)
        self.table.setObjectName("gamesTable")
        self.table.setHorizontalHeaderLabels(["Task", "Dataset", "Status", "Progress"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table, 1)
        self.rows: dict[str, int] = {}

    def add_job(self, task: str, dataset: str, status: str = "Queued", *, job_id: str = "") -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        for column, value in enumerate((task, dataset, status, "0%" if status == "Queued" else "100%")):
            self.table.setItem(row, column, QTableWidgetItem(value))
        if job_id:
            self.rows[job_id] = row

    def update_job(self, job: DatabaseJob) -> None:
        row = self.rows.get(job.job_id)
        if row is None:
            self.add_job(job.task, job.dataset, job.status, job_id=job.job_id)
            row = self.rows[job.job_id]
        self.table.item(row, 2).setText(job.status)
        self.table.item(row, 3).setText("100%" if job.status == "Completed" else ("--" if job.status == "Failed" else "Running"))


class DatabasePage(QWidget):
    open_analyze = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        settings = SETTINGS_STORE.load()
        database_root = Path(str(settings.get("database", "artifacts/database")))
        if not database_root.is_absolute():
            database_root = ROOT / database_root
        self.store = DatabaseStore([ROOT / "artifacts", ASSETS / "benchmarks", database_root], settings)
        self.runner = DatabaseJobRunner(ROOT, self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 10, 18, 10)
        layout.setSpacing(6)
        self.notice = preview_notice(f"DATABASE INDEX READY - {len(self.store.records):,} real development artifacts scanned")
        layout.addWidget(self.notice)
        main = QHBoxLayout()
        main.setSpacing(10)
        self.sources = DatabaseSources(self.store.records)
        sources = self.sources
        sources.setFixedWidth(282)
        self.explorer = DatasetExplorer(self.store)
        self.detail = DatasetDetail(self.store)
        self.detail.setFixedWidth(650)
        sources.source_selected.connect(self.explorer.set_source)
        self.explorer.dataset_selected.connect(self.detail.set_dataset)
        self.detail.notes_saved.connect(self.save_notes)
        main.addWidget(sources)
        main.addWidget(self.explorer, 1)
        main.addWidget(self.detail)
        layout.addLayout(main, 1)
        actions = QHBoxLayout()
        self.action_buttons: dict[str, QPushButton] = {}
        for title, handler in (
            ("Open", self.open_selected), ("Send to Analyze", self.send_to_analyze),
            ("Export", self.export_selected), ("Compare", self.queue_compare),
            ("Validate", self.validate_selected),
            ("Train Candidate Weights", self.queue_training),
            ("Create Dataset", self.create_dataset), ("Import PGN", self.import_pgn),
            ("Archive", self.archive_selected), ("Cancel Job", self.cancel_current_job),
        ):
            action = button(title, compact=True)
            action.clicked.connect(handler)
            self.action_buttons[title] = action
            actions.addWidget(action)
        layout.addLayout(actions)
        jobs_row = QHBoxLayout()
        jobs_row.setSpacing(10)
        self.recent_jobs = DatabaseJobs("RECENT JOBS")
        self.queue = DatabaseJobs("TASK QUEUE")
        jobs_row.addWidget(self.recent_jobs, 1)
        jobs_row.addWidget(self.queue, 1)
        layout.addLayout(jobs_row, 34)
        self.runner.queued.connect(lambda job: self.queue.update_job(job))
        self.runner.started.connect(lambda job: self.queue.update_job(job))
        self.runner.finished.connect(self.job_finished)
        current = self.explorer.current()
        if current:
            self.detail.set_dataset(current)

    def selected_name(self) -> str:
        current = self.explorer.current()
        return current.name if current else "No dataset selected"

    def open_selected(self) -> None:
        self.detail.tabs.setCurrentIndex(0)

    def export_selected(self) -> None:
        current = self.explorer.current()
        if current is None:
            return
        destination, _filter = QFileDialog.getSaveFileName(self, "Export Dataset", current.name, "Data files (*.*)")
        if destination:
            shutil.copy2(current.path, destination)

    def validate_selected(self) -> None:
        current = self.explorer.current()
        if current is None:
            return
        status, detail = self.store.validate(current)
        self.store.update_validation(current, status)
        self.persist_state()
        self.explorer.refresh()
        self.detail.set_dataset(current)
        self.recent_jobs.add_job("Dataset validation", current.name, status)
        QMessageBox.information(self, "Dataset Validation", detail)

    def archive_selected(self) -> None:
        if QMessageBox.question(self, "Archive Dataset", f'Archive "{self.selected_name()}"?') == QMessageBox.StandardButton.Yes:
            current = self.explorer.current()
            if current is None:
                return
            name = current.name
            self.store.archive(current)
            self.persist_state()
            self.explorer.refresh()
            self.recent_jobs.add_job("Archive", name, "Completed")

    def queue_training(self) -> None:
        current = self.explorer.current()
        if current is None or current.path.suffix.lower() not in {".csv", ".json"}:
            QMessageBox.information(self, "Train Candidate Weights", "Select a self-play CSV/JSON dataset first.")
            return
        mode = "move-loss" if current.data_type == "Annotated" else "result"
        output = ROOT / "artifacts" / "database" / "weights" / f"{current.path.stem}_{mode}_candidate.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        self._enqueue("Train candidate weights", current.name, [
            "learn", "--input", str(current.path), "--output", str(output), "--target-mode", mode,
        ])

    def queue_compare(self) -> None:
        current = self.explorer.current()
        if current is None or current.data_type != "Weights":
            QMessageBox.information(self, "Compare Weights", "Select a candidate weights JSON file first.")
            return
        output = ROOT / "artifacts" / "database" / "comparisons" / f"{current.path.stem}_compare.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        self._enqueue("Compare weights", current.name, [
            "compare-weights", "--candidate", str(current.path), "--games", "10", "--depth", "2",
            "--search-backend", "native", "--output", str(output),
        ])

    def create_dataset(self) -> None:
        paths, _filter = QFileDialog.getOpenFileNames(self, "Create Dataset Index", "", "Data files (*.csv *.json *.jsonl *.pgn *.epd *.bin)")
        if not paths:
            return
        output, _filter = QFileDialog.getSaveFileName(self, "Save Dataset Index", "dataset_index.json", "JSON Files (*.json)")
        if not output:
            return
        Path(output).write_text(
            json.dumps({"created_at": datetime.now().isoformat(timespec="seconds"), "files": paths}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.explorer.reload()
        self.sources.refresh_counts(self.store.records)
        self.recent_jobs.add_job("Create dataset index", Path(output).name, "Completed")

    def _enqueue(self, task: str, dataset: str, arguments: list[str]) -> None:
        self.runner.enqueue(DatabaseJob(task, dataset, arguments))

    def cancel_current_job(self) -> None:
        if self.runner.current is None:
            QMessageBox.information(self, "Task Queue", "There is no running Database job.")
            return
        self.runner.cancel_current()

    def job_finished(self, job: DatabaseJob) -> None:
        self.queue.update_job(job)
        self.recent_jobs.add_job(job.task, job.dataset, job.status)
        self.explorer.reload()
        self.sources.refresh_counts(self.store.records)
        self.notice.setText(f"DATABASE INDEX READY - {len(self.store.records):,} real development artifacts scanned")
        if job.status == "Failed":
            QMessageBox.warning(self, job.task, f"Job failed.\n\n{job.output[-1600:]}")

    def import_pgn(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(self, "Import PGN", "", "PGN Files (*.pgn)")
        if path:
            target_root = ROOT / "artifacts" / "database" / "imported_pgn"
            target_root.mkdir(parents=True, exist_ok=True)
            target = target_root / Path(path).name
            if Path(path).resolve() != target.resolve():
                shutil.copy2(path, target)
            self.explorer.reload()
            self.sources.refresh_counts(self.store.records)
            self.notice.setText(f"DATABASE INDEX READY - {len(self.store.records):,} real development artifacts scanned")
            self.recent_jobs.add_job("PGN import", Path(path).name, "Completed")

    def send_to_analyze(self) -> None:
        current = self.explorer.current()
        if current is None:
            return
        if current.path.suffix.lower() != ".pgn":
            QMessageBox.information(self, "Send to Analyze", "Only PGN datasets can be sent directly to Analyze in this stage.")
            return
        self.open_analyze.emit(current)

    def save_notes(self, record: DatasetRecord, notes: str) -> None:
        self.store.set_notes(record, notes)
        self.persist_state()
        QMessageBox.information(self, "Dataset Notes", "Notes saved.")

    def persist_state(self) -> None:
        settings = SETTINGS_STORE.load()
        settings.update(self.store.user_state())
        SETTINGS_STORE.save(settings)


class SettingsSidebar(Card):
    category_selected = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 16, 14, 16)
        layout.setSpacing(6)
        layout.addWidget(text_label(TRANSLATIONS.text("settings.title", "SETTINGS"), "sectionHeader"))
        self.buttons: dict[str, QPushButton] = {}
        group = QButtonGroup(self)
        group.setExclusive(True)
        for index, (title, icon, key) in enumerate((
            ("General", "\u2699", "settings.general"), ("Engine Management", "\u2692", "settings.engines"),
            ("Board & Pieces", "\u265e", "settings.board"), ("Gameplay", "\u265f", "settings.gameplay"),
            ("Folders & Files", "\u25a1", "settings.folders"), ("Analysis", "\u2315", "settings.analysis"),
            ("Display", "\u25a4", "settings.display"), ("Sound", "\u266b", "settings.sound"),
            ("Advanced", "\u2692", "settings.advanced"), ("About", "\u24d8", "settings.about"),
        )):
            control = QPushButton(f"{icon}   {TRANSLATIONS.text(key, title)}")
            control.setObjectName("settingsNav")
            control.setCheckable(True)
            control.setChecked(index == 0)
            control.clicked.connect(lambda _checked=False, value=title: self.category_selected.emit(value))
            group.addButton(control)
            self.buttons[title] = control
            layout.addWidget(control)
        layout.addStretch()
        layout.addWidget(divider())
        note = text_label("Configuration stored locally after Save Changes.", "muted")
        note.setWordWrap(True)
        layout.addWidget(note)


class EngineManagementPanel(Card):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(17, 14, 17, 14)
        layout.setSpacing(10)
        title = QHBoxLayout()
        title.addWidget(text_label("ENGINE MANAGEMENT", "sectionHeader"))
        title.addStretch()
        add = button("+  Add Engine", primary=True, compact=True)
        unload = button("Stop All Engines", compact=True)
        unload.setToolTip("Stops external runtime sessions when available; registered profiles remain saved.")
        add.clicked.connect(self.add_engine)
        unload.clicked.connect(self.unload_all)
        title.addWidget(add)
        title.addWidget(unload)
        layout.addLayout(title)
        self.table = QTableWidget(0, 4)
        self.table.setObjectName("gamesTable")
        self.table.setHorizontalHeaderLabels(["Engine Name", "Type / Path", "Status", "Actions"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(3, 206)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(52)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table, 1)
        ENGINE_REGISTRY.changed.connect(lambda _names: self.refresh())
        self.refresh()

    def refresh(self) -> None:
        self.table.setRowCount(len(ENGINE_REGISTRY.entries))
        for row, entry in enumerate(ENGINE_REGISTRY.entries):
            values = (entry["name"], f'{entry["type"]}  |  {entry["path"]}', entry["status"])
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(value))
            controls = QWidget()
            actions = QHBoxLayout(controls)
            actions.setContentsMargins(4, 2, 4, 2)
            actions.setSpacing(7)
            configure = button("Configure", compact=True)
            configure.setMinimumWidth(92)
            configure.setFixedHeight(36)
            configure.clicked.connect(lambda _checked=False, name=entry["name"]: self.configure(name))
            remove = button("Remove", compact=True)
            remove.setMinimumWidth(86)
            remove.setFixedHeight(36)
            remove.setEnabled(entry["name"] != "DeepTactics")
            remove.clicked.connect(lambda _checked=False, name=entry["name"]: self.remove(name))
            actions.addWidget(configure)
            actions.addWidget(remove)
            self.table.setCellWidget(row, 3, controls)
        if ENGINE_REGISTRY.entries:
            self.table.selectRow(0)

    def selected_name(self) -> str:
        row = self.table.currentRow()
        return ENGINE_REGISTRY.entries[row]["name"] if 0 <= row < len(ENGINE_REGISTRY.entries) else "DeepTactics"

    def add_engine(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(self, "Add UCI Engine", "", "Engines (*.exe *.cmd *.bat);;All files (*.*)")
        if path:
            name = ENGINE_REGISTRY.add_engine(path)
            if name is None:
                QMessageBox.information(self, "Engine Registered", "This engine profile is already registered.")
                return
            self.test_engine(name)

    def unload_all(self) -> None:
        ENGINE_REGISTRY.unload_external()
        self.refresh()

    def configure_selected(self) -> None:
        self.configure(self.selected_name())

    def configure(self, name: str) -> None:
        entry = ENGINE_REGISTRY.entry(name) or {}
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Engine Settings - {name}")
        form = QFormLayout(dialog)
        form.addRow("Engine", text_label(name, "statValue"))
        path = QLineEdit(str(entry.get("path", "")))
        path.setObjectName("interactiveField")
        path.setReadOnly(True)
        form.addRow("Path", path)
        threads = QSpinBox()
        threads.setRange(1, 128)
        threads.setValue(int(entry.get("threads", 1)))
        hash_size = QSpinBox()
        hash_size.setRange(16, 65536)
        hash_size.setValue(int(entry.get("hash", 256)))
        multipv = QSpinBox()
        multipv.setRange(1, 10)
        multipv.setValue(int(entry.get("multipv", 1)))
        use_tablebase = QCheckBox("Use Syzygy tablebase for this profile")
        use_tablebase.setChecked(bool(entry.get("use_tablebase", False)))
        syzygy = QLineEdit(str(entry.get("syzygy_path", "")))
        syzygy.setObjectName("interactiveField")
        browse_syzygy = button("Browse Syzygy Folder", compact=True)
        browse_syzygy.clicked.connect(lambda: self._browse_engine_syzygy(syzygy))
        tb_max = QSpinBox()
        tb_max.setRange(2, 7)
        tb_max.setValue(int(entry.get("tablebase_max_pieces", 7)))
        form.addRow("Threads", threads)
        form.addRow("Hash (MB)", hash_size)
        form.addRow("MultiPV", multipv)
        form.addRow(use_tablebase)
        form.addRow("SyzygyPath", syzygy)
        form.addRow(browse_syzygy)
        form.addRow("Tablebase Max Pieces", tb_max)
        scope = text_label(
            "Hash and Threads are applied when a new UCI session starts.\n"
            "Analyze uses its Analysis MultiPV setting; match play uses MultiPV 1.\n"
            "Syzygy requires real tablebase files and is ignored by engines that do not support the option.",
            "muted",
        )
        scope.setWordWrap(True)
        form.addRow(scope)
        test = button("Test Engine", compact=True)
        test.clicked.connect(lambda: self.test_engine(name))
        form.addRow(test)
        restore = button("Restore Profile Defaults", compact=True)
        restore.clicked.connect(lambda: (
            threads.setValue(1),
            hash_size.setValue(256),
            multipv.setValue(1),
            use_tablebase.setChecked(False),
            syzygy.setText(""),
            tb_max.setValue(7),
        ))
        form.addRow(restore)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        def save_profile() -> None:
            ENGINE_REGISTRY.update_profile(
                name,
                threads=threads.value(),
                hash=hash_size.value(),
                multipv=multipv.value(),
                use_tablebase=use_tablebase.isChecked(),
                syzygy_path=syzygy.text().strip(),
                tablebase_max_pieces=tb_max.value(),
            )
            dialog.accept()
        buttons.accepted.connect(save_profile)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        dialog.exec()

    def _browse_engine_syzygy(self, target: QLineEdit) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Syzygy Tablebase Folder")
        if path:
            target.setText(path)

    def test_engine(self, name: str) -> None:
        entry = ENGINE_REGISTRY.entry(name)
        if entry is None:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Test UCI Engine - {name}")
        layout = QVBoxLayout(dialog)
        status = text_label("Starting UCI handshake...", "secondary")
        layout.addWidget(status)
        close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close.setEnabled(False)
        close.rejected.connect(dialog.reject)
        layout.addWidget(close)
        client = UCIClient(ROOT, dialog)
        client.configure_engine(entry)

        def succeeded(identity: dict) -> None:
            engine_name = str(identity.get("name") or name)
            ENGINE_REGISTRY.update_profile(
                name,
                status="Ready",
                detected_name=engine_name,
                nnue=False if name == "DeepTactics" else bool(identity.get("supports_nnue", False)),
            )
            status.setText(
                f"Ready: {engine_name}\n"
                f"NNUE capability detected: {'No' if name == 'DeepTactics' else ('Yes' if identity.get('supports_nnue') else 'No')}"
            )
            client.quit()
            close.setEnabled(True)
            self.refresh()

        def failed(message: str) -> None:
            ENGINE_REGISTRY.update_profile(name, status="Error")
            status.setText(f"UCI test failed:\n{message}")
            close.setEnabled(True)
            self.refresh()

        client.identity_received.connect(succeeded)
        client.error_received.connect(failed)
        client.start()
        dialog.exec()
        client.quit()

    def remove_selected(self) -> None:
        self.remove(self.selected_name())

    def remove(self, name: str) -> None:
        if name == "DeepTactics":
            QMessageBox.information(self, "Built-in Engine", "DeepTactics is the built-in engine and cannot be removed.")
            return
        if QMessageBox.question(self, "Remove Engine", f'Remove "{name}" from registered engines?') == QMessageBox.StandardButton.Yes:
            ENGINE_REGISTRY.remove_engine(name)

    def mouseDoubleClickEvent(self, _event) -> None:
        self.configure_selected()


class UciLogDialog(QDialog):
    """Bounded live view of commands and replies from Qt-managed UCI sessions."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("UCI Log")
        self.resize(900, 560)
        layout = QVBoxLayout(self)
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setObjectName("notesEditor")
        layout.addWidget(self.text, 1)
        actions = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        clear = actions.addButton("Clear", QDialogButtonBox.ButtonRole.ActionRole)
        clear.clicked.connect(self.text.clear)
        actions.rejected.connect(self.reject)
        layout.addWidget(actions)

    def append_line(self, session: str, direction: str, line: str) -> None:
        self.text.append(f"[{session}] {direction} {line}")
        document = self.text.document()
        while document.blockCount() > 3000:
            cursor = self.text.textCursor()
            cursor.movePosition(cursor.MoveOperation.Start)
            cursor.select(cursor.SelectionType.BlockUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()


class SettingsPage(QWidget):
    theme_changed = Signal(str)
    settings_applied = Signal(object)
    clear_analysis_cache_requested = Signal()
    show_uci_log_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.defaults = {key: value for key, value in DEFAULT_GUI_SETTINGS.items() if key != "engines"}
        self.state = SETTINGS_STORE.load()
        if self.state.get("piece_set_path") == "assets/pieces/default" and (ASSETS / "pieces" / "white_king.png").exists():
            self.state["piece_set_path"] = "assets/pieces"
        self.category_indices: dict[str, int] = {}
        self.choice_controls: dict[str, dict[str, QPushButton]] = {}
        self.checkbox_controls: dict[str, list[QCheckBox]] = {}
        self.combo_controls: dict[str, list[QComboBox]] = {}
        self.spin_controls: dict[str, list[QSpinBox]] = {}
        self.slider_controls: dict[str, list[QSlider]] = {}
        outer = QHBoxLayout(self)
        outer.setContentsMargins(18, 12, 18, 12)
        outer.setSpacing(12)
        self.sidebar = SettingsSidebar()
        self.sidebar.setFixedWidth(255)
        self.sidebar.category_selected.connect(self.show_category)
        outer.addWidget(self.sidebar)
        body = QVBoxLayout()
        body.setSpacing(10)
        heading = QHBoxLayout()
        title = QVBoxLayout()
        title.addWidget(text_label("Settings", "settingsTitle"))
        title.addWidget(text_label("Configure DeepTactics, installed engines, and development tools.", "secondary"))
        heading.addLayout(title)
        heading.addStretch()
        reset = button("Reset to Default", compact=True)
        save = button("Save Changes", primary=True, compact=True)
        reset.clicked.connect(self.reset_defaults)
        save.clicked.connect(self.save_settings)
        heading.addWidget(reset)
        heading.addWidget(save)
        body.addLayout(heading)
        self.stack = QStackedWidget()
        pages = {
            "General": self.general_page(),
            "Engine Management": self.engine_page(),
            "Board & Pieces": self.board_page(),
            "Gameplay": self.gameplay_page(),
            "Folders & Files": self.folders_page(),
            "Analysis": self.analysis_page(),
            "Display": self.display_page(),
            "Sound": self.sound_page(),
            "Advanced": self.advanced_page(),
            "About": self.about_page(),
        }
        for name, page in pages.items():
            self.category_indices[name] = self.stack.count()
            self.stack.addWidget(page)
        body.addWidget(self.stack, 1)
        outer.addLayout(body, 1)
        ENGINE_REGISTRY.changed.connect(self.refresh_engine_choices)

    def show_category(self, name: str) -> None:
        self.stack.setCurrentIndex(self.category_indices[name])
        if name in self.sidebar.buttons:
            self.sidebar.buttons[name].setChecked(True)

    def refresh_engine_choices(self, names: list[str]) -> None:
        for combo in self.combo_controls.get("default_analysis_engine", []):
            current = combo.currentText()
            blocked = combo.blockSignals(True)
            combo.clear()
            combo.addItems(names)
            combo.setCurrentText(current if current in names else "DeepTactics")
            combo.blockSignals(blocked)

    def section(self, title: str) -> tuple[Card, QVBoxLayout]:
        card = Card()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(17, 15, 17, 15)
        layout.setSpacing(10)
        header = QHBoxLayout()
        header.addWidget(text_label(title, "sectionHeader"))
        header.addStretch()
        reset = button("Reset", compact=True)
        reset.setObjectName("miniReset")
        reset.clicked.connect(lambda: self.reset_section(title))
        header.addWidget(reset)
        layout.addLayout(header)
        return card, layout

    def choices(self, values: tuple[str, ...], current: str, key: str, on_change=None) -> QHBoxLayout:
        row = QHBoxLayout()
        group = QButtonGroup(self)
        group.setExclusive(True)
        self.choice_controls.setdefault(key, {})
        for value in values:
            control = button(value, compact=True)
            control.setCheckable(True)
            control.setChecked(value == current)
            control.clicked.connect(lambda checked=False, choice=value: self.select_setting(key, choice, checked, on_change))
            group.addButton(control)
            self.choice_controls[key][value] = control
            row.addWidget(control)
        row.addStretch()
        return row

    def select_setting(self, key: str, value, checked: bool = True, callback=None) -> None:
        if checked:
            self.state[key] = value
            if callback:
                callback(value)

    def checkbox(self, label: str, key: str, *, enabled: bool = True, tooltip: str = "") -> QCheckBox:
        control = QCheckBox(label)
        control.setChecked(bool(self.state.get(key, self.defaults.get(key, False))))
        control.setEnabled(enabled)
        if tooltip:
            control.setToolTip(tooltip)
        control.toggled.connect(lambda checked, setting=key: self.select_setting(setting, checked))
        self.checkbox_controls.setdefault(key, []).append(control)
        return control

    def combo(self, values: list[str], key: str) -> QComboBox:
        control = QComboBox()
        control.setObjectName("interactiveField")
        control.addItems(values)
        current = str(self.state.get(key, self.defaults.get(key, values[0])))
        control.setCurrentText(current if current in values else values[0])
        control.currentTextChanged.connect(lambda value, setting=key: self.select_setting(setting, value))
        self.combo_controls.setdefault(key, []).append(control)
        return control

    def spin(self, key: str, minimum: int, maximum: int, suffix: str = "") -> QSpinBox:
        control = QSpinBox()
        control.setObjectName("interactiveField")
        control.setRange(minimum, maximum)
        control.setValue(int(self.state.get(key, self.defaults.get(key, minimum))))
        control.setSuffix(suffix)
        control.valueChanged.connect(lambda value, setting=key: self.select_setting(setting, value))
        self.spin_controls.setdefault(key, []).append(control)
        return control

    def refresh_controls(self) -> None:
        for key, controls in self.choice_controls.items():
            current = self.state.get(key)
            if current in controls:
                controls[current].setChecked(True)
        for key, controls in self.checkbox_controls.items():
            for control in controls:
                blocked = control.blockSignals(True)
                control.setChecked(bool(self.state.get(key, self.defaults.get(key, False))))
                control.blockSignals(blocked)
        for key, controls in self.combo_controls.items():
            for control in controls:
                blocked = control.blockSignals(True)
                control.setCurrentText(str(self.state.get(key, self.defaults.get(key, ""))))
                control.blockSignals(blocked)
        for key, controls in self.spin_controls.items():
            for control in controls:
                blocked = control.blockSignals(True)
                control.setValue(int(self.state.get(key, self.defaults.get(key, control.minimum()))))
                control.blockSignals(blocked)
        for key, controls in self.slider_controls.items():
            for control in controls:
                blocked = control.blockSignals(True)
                control.setValue(int(self.state.get(key, self.defaults.get(key, control.minimum()))))
                control.blockSignals(blocked)
        if hasattr(self, "interface_scale_value"):
            self.interface_scale_value.setText(f'{int(self.state.get("interface_scale", 100))}%')

    def general_page(self) -> QWidget:
        page = QWidget()
        layout = QGridLayout(page)
        layout.setSpacing(10)
        appearance, box = self.section("APPEARANCE")
        box.addWidget(text_label("GUI Theme", "secondary"))
        box.addLayout(self.choices(("Light", "Dark", "System"), self.state["theme"], "theme", self.theme_changed.emit))
        box.addWidget(text_label("Interface Language", "secondary"))
        self.general_language = self.combo(["English", "Korean"], "language")
        box.addWidget(self.general_language)
        box.addWidget(text_label("Language changes are applied after saving and restarting.", "muted"))
        layout.addWidget(appearance, 0, 0)
        engine, box = self.section("ENGINE STATUS")
        box.addWidget(text_label("DeepTactics                      Built-in / Available", "statValue"))
        box.addWidget(text_label("Registered UCI engines", "secondary"))
        self.engine_count = text_label(f"{len(ENGINE_REGISTRY.entries)} installed engines", "blueTextLarge")
        ENGINE_REGISTRY.changed.connect(lambda names: self.engine_count.setText(f"{len(names)} installed engines"))
        box.addWidget(self.engine_count)
        manage = button("Open Engine Management", compact=True)
        manage.clicked.connect(lambda: self.show_category("Engine Management"))
        box.addWidget(manage)
        layout.addWidget(engine, 0, 1)
        board, box = self.section("BOARD DESIGN")
        box.addWidget(text_label("Current appearance", "secondary"))
        appearance = QHBoxLayout()
        preview = WoodenBoard(starting_position=True, minimum_size=85, visual_theme=str(self.state["board_theme"]))
        preview.setFixedSize(108, 108)
        preview.set_piece_folder(self.resolve_setting_path(str(self.state["piece_set_path"])))
        self.general_board_preview = preview
        appearance.addWidget(preview)
        current = QVBoxLayout()
        self.general_board_value = text_label(f'Board:  {self.state["board_theme"]}', "statValue")
        current.addWidget(self.general_board_value)
        self.general_piece_value = text_label(f'Pieces:  {self.state["piece_set_path"]}', "secondary")
        current.addWidget(self.general_piece_value)
        current.addStretch()
        appearance.addLayout(current, 1)
        box.addLayout(appearance)
        open_board = button("Board & Pieces Settings", compact=True)
        open_board.clicked.connect(lambda: self.show_category("Board & Pieces"))
        box.addWidget(open_board)
        layout.addWidget(board, 1, 0)
        options, box = self.section("COMMON OPTIONS")
        for label, key in (("Auto Save PGN", "auto_save_pgn"), ("Confirm Resign", "confirm_resign"), ("Show Tooltips", "show_tooltips")):
            box.addWidget(self.checkbox(label, key))
        box.addWidget(text_label("Start with", "secondary"))
        start = self.combo(["Play", "Analyze", "Games", "AI Match", "Openings", "Database", "Settings"], "start_with")
        box.addWidget(start)
        layout.addWidget(options, 1, 1)
        return page

    def engine_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        panel = EngineManagementPanel()
        self.engine_panel = panel
        layout.addWidget(panel, 1)
        controls = QHBoxLayout()
        configure = button("Engine Settings", compact=True)
        remove = button("Remove Selected", compact=True)
        configure.clicked.connect(panel.configure_selected)
        remove.clicked.connect(panel.remove_selected)
        controls.addWidget(configure)
        controls.addWidget(remove)
        controls.addStretch()
        layout.addLayout(controls)
        return page

    def board_page(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        themes, box = self.section("BOARD DESIGN")
        box.addWidget(text_label("Select Board Style", "secondary"))
        box.addLayout(self.choices(("Blue", "Wood", "Classic"), self.state["board_theme"], "board_theme", self.update_board_theme))
        box.addWidget(text_label("Preview of selected board and current pieces", "secondary"))
        preview = WoodenBoard(starting_position=True, minimum_size=260, visual_theme=str(self.state["board_theme"]))
        preview.setFixedSize(300, 300)
        preview.set_piece_folder(self.resolve_setting_path(str(self.state["piece_set_path"])))
        self.board_preview = preview
        box.addWidget(preview, 0, Qt.AlignmentFlag.AlignHCenter)
        box.addStretch()
        layout.addWidget(themes, 1)
        pieces, box = self.section("PIECE SET")
        box.addWidget(text_label("Select piece images folder", "secondary"))
        self.piece_path = QLineEdit(self.state["piece_set_path"])
        self.piece_path.setObjectName("interactiveField")
        self.piece_path.textChanged.connect(lambda value: self.select_setting("piece_set_path", value))
        box.addWidget(self.piece_path)
        browse = button("Browse", compact=True)
        test = button("Test Pieces", compact=True)
        browse.clicked.connect(self.browse_pieces)
        test.clicked.connect(self.test_pieces)
        action = QHBoxLayout()
        action.addWidget(browse)
        action.addWidget(test)
        box.addLayout(action)
        piece_preview = WoodenBoard(starting_position=True, minimum_size=180, visual_theme=str(self.state["board_theme"]))
        piece_preview.setFixedSize(210, 210)
        piece_preview.set_piece_folder(self.resolve_setting_path(str(self.state["piece_set_path"])))
        self.piece_preview = piece_preview
        box.addWidget(piece_preview, 0, Qt.AlignmentFlag.AlignHCenter)
        box.addWidget(text_label("Required: white/black king, queen, rook, bishop, knight, pawn PNG files.", "muted"))
        box.addStretch()
        layout.addWidget(pieces, 1)
        return page

    def gameplay_page(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        display, box = self.section("GAMEPLAY DISPLAY OPTIONS")
        for label, key in (
            ("Show Arrows", "show_arrows"), ("Enable Premoves", "enable_premoves"),
            ("Highlight Last Move", "highlight_last_move"), ("Show Move Numbers", "show_move_numbers"),
            ("Confirm Resign", "confirm_resign"),
        ):
            box.addWidget(self.checkbox(label, key))
        box.addWidget(self.checkbox(
            "Restore clock on Undo",
            "restore_clock_on_undo",
            enabled=False,
            tooltip="Clock snapshot restoration is not connected to Play yet.",
        ))
        box.addWidget(self.checkbox(
            "Auto promote to queen",
            "auto_promote_queen",
            tooltip="When off, Play and Analyze ask which piece to promote to.",
        ))
        box.addWidget(text_label("Board coordinate visibility is controlled under Display.", "muted"))
        box.addStretch()
        layout.addWidget(display, 1)
        other, box = self.section("OTHER SETTINGS")
        box.addWidget(self.checkbox("Auto Save PGN", "auto_save_pgn"))
        updates = self.checkbox(
            "Check for Updates",
            "check_for_updates",
            tooltip="Check GitHub Releases for a newer DeepTactics tag after startup.",
        )
        box.addWidget(updates)
        check_now = button("Check Now", compact=True)
        check_now.clicked.connect(self.check_for_updates)
        box.addWidget(check_now, 0, Qt.AlignmentFlag.AlignLeft)
        box.addWidget(text_label("UI Engine Response Timeout", "secondary"))
        max_time = self.spin("max_think_time", 1, 3600, " sec")
        max_time.setToolTip("GUI timeout for engine responses; this does not alter chess clock controls.")
        box.addWidget(max_time)
        box.addWidget(text_label("Update checks read GitHub Releases only. Installation remains an explicit user action.", "muted"))
        box.addStretch()
        layout.addWidget(other, 1)
        return page

    def folders_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        card, box = self.section("DEFAULT FOLDERS")
        self.folder_fields: dict[str, QLineEdit] = {}
        for label, key, default in (
            ("PGN Auto Save", "pgn_auto_save", "artifacts/pgn"),
            ("Analysis Export", "analysis_export", "artifacts/analysis"),
            ("Database", "database", "artifacts/database"),
            ("Engine Files", "engine_files", "engines"),
            ("Screenshots", "screenshots", "artifacts/screenshots"),
            ("Logs", "logs", "artifacts/logs"),
            ("Opening Books / Lines", "opening_books", "assets/openings"),
            ("Syzygy Tablebases", "syzygy_path", ""),
            ("Training Outputs", "training_outputs", "artifacts/training"),
        ):
            row = QHBoxLayout()
            title = text_label(label, "secondary")
            title.setFixedWidth(175)
            row.addWidget(title)
            field = QLineEdit(str(self.state.get(key, default)))
            field.setObjectName("interactiveField")
            self.folder_fields[label] = field
            field.textChanged.connect(lambda value, setting=key: self.select_setting(setting, value))
            row.addWidget(field, 1)
            browse = button("Browse", compact=True)
            browse.clicked.connect(lambda _checked=False, target=field, setting=key: self.browse_folder(target, setting))
            row.addWidget(browse)
            box.addLayout(row)
        box.addWidget(text_label("Folders are stored as defaults; data is only created when its feature writes output.", "muted"))
        layout.addWidget(card)
        layout.addStretch()
        return page

    def analysis_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        card, form = self.section("ANALYSIS DEFAULTS")
        row = QHBoxLayout()
        row.addWidget(text_label("Default Analysis Engine", "secondary"))
        self.analysis_engine_combo = self.combo(ENGINE_REGISTRY.names(), "default_analysis_engine")
        self.analysis_engine_combo.setToolTip("Default UCI engine profile used by Analyze.")
        row.addWidget(self.analysis_engine_combo, 1)
        form.addLayout(row)
        form.addWidget(text_label("External engines must pass the Engine Management UCI test before reliable use.", "muted"))
        row = QHBoxLayout()
        row.addWidget(text_label("Score Perspective", "secondary"))
        score = self.combo(["White perspective"], "score_perspective")
        score.setToolTip("The connected GUI evaluation display is normalized to White perspective.")
        row.addWidget(score, 1)
        form.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(text_label("NNUE Mode", "secondary"))
        nnue = self.combo(["Off"], "nnue_mode")
        nnue.setToolTip("DeepTactics does not currently expose supported NNUE evaluation in the GUI.")
        row.addWidget(nnue, 1)
        form.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(text_label("Analysis Mode", "secondary"))
        mode = self.combo(["Fixed Depth", "Move Time", "Infinite until Stop"], "analysis_mode")
        mode.setToolTip("Infinite mode applies to the current position only; full-game review falls back to fixed depth.")
        row.addWidget(mode, 1)
        form.addLayout(row)
        for label, key, maximum in (
            ("Default Depth", "default_analysis_depth", 30),
            ("Default Move Time (sec)", "default_move_time", 120),
            ("MultiPV Count", "analysis_multipv", 10),
        ):
            row = QHBoxLayout()
            row.addWidget(text_label(label, "secondary"))
            spin = self.spin(key, 1, maximum)
            if key == "default_analysis_depth":
                self.analysis_depth_default = spin
            if key == "default_move_time":
                spin.setToolTip("Used by Move Time analysis mode.")
            if key == "analysis_multipv":
                spin.setToolTip("Used for live Candidates after Save Changes; full-game review still uses one primary PV.")
            row.addWidget(spin)
            form.addLayout(row)
        form.addWidget(self.checkbox("Auto Analyze After Move", "auto_analyze_after_move"))
        form.addWidget(self.checkbox("Show Move Classification", "show_move_classification"))
        form.addWidget(self.checkbox("Use analysis cache", "use_analysis_cache"))
        form.addWidget(self.checkbox("Use Syzygy tablebase for DeepTactics", "use_tablebase"))
        row = QHBoxLayout()
        row.addWidget(text_label("Tablebase Max Pieces", "secondary"))
        row.addWidget(self.spin("tablebase_max_pieces", 2, 7))
        form.addLayout(row)
        form.addWidget(self.checkbox("Review full game on Analyze start", "review_full_game"))
        form.addWidget(self.checkbox("Live analysis after review", "live_analysis_after_review"))
        clear_cache = button("Clear Analysis Cache", compact=True)
        clear_cache.clicked.connect(self.clear_analysis_cache_requested.emit)
        form.addWidget(clear_cache)
        layout.addWidget(card)
        layout.addStretch()
        return page

    def display_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        card, box = self.section("DISPLAY")
        scale_header = QHBoxLayout()
        scale_header.addWidget(text_label("Interface Scale", "secondary"))
        scale_header.addStretch()
        self.interface_scale_value = text_label(f'{int(self.state.get("interface_scale", 100))}%', "statValue")
        scale_header.addWidget(self.interface_scale_value)
        box.addLayout(scale_header)
        scale = QSlider(Qt.Orientation.Horizontal)
        scale.setRange(80, 125)
        scale.setValue(int(self.state.get("interface_scale", 100)))
        scale.valueChanged.connect(lambda value: (self.select_setting("interface_scale", value), self.interface_scale_value.setText(f"{value}%")))
        box.addWidget(scale)
        self.interface_scale = scale
        self.slider_controls.setdefault("interface_scale", []).append(scale)
        box.addWidget(text_label("Applied when Save Changes is selected; restrained limits protect fixed panels.", "muted"))
        box.addWidget(self.checkbox("Show coordinates on boards", "show_coordinates"))
        box.addWidget(self.checkbox("Show hover tooltips", "show_tooltips"))
        box.addWidget(self.checkbox(
            "Compact data tables",
            "compact_tables",
            enabled=False,
            tooltip="Compact table sizing has not been connected yet.",
        ))
        box.addWidget(self.checkbox("Show raw UCI score in tooltip", "show_raw_score_tooltip"))
        layout.addWidget(card)
        layout.addStretch()
        return page

    def sound_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        card, box = self.section("SOUND")
        box.addWidget(text_label("Uses assets/sounds/*.wav when present and a system beep fallback otherwise.", "muted"))
        for title, key in (
            ("Move Sound", "sound_move"), ("Capture Sound", "sound_capture"),
            ("Check Sound", "sound_check"), ("Game End Sound", "sound_game_end"),
            ("Engine Ready Sound", "sound_engine_ready"),
        ):
            box.addWidget(self.checkbox(title, key))
        box.addWidget(text_label("Volume", "secondary"))
        volume = QSlider(Qt.Orientation.Horizontal)
        volume.setRange(0, 100)
        volume.setValue(int(self.state.get("sound_volume", 70)))
        volume.valueChanged.connect(lambda value: self.select_setting("sound_volume", value))
        box.addWidget(volume)
        self.slider_controls.setdefault("sound_volume", []).append(volume)
        box.addWidget(self.checkbox("Mute All", "sound_mute_all"))
        test = button("Test Sound", compact=True)
        test.clicked.connect(self.test_sound)
        box.addWidget(test, 0, Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(card)
        layout.addStretch()
        return page

    def advanced_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        card, box = self.section("ADVANCED / DIAGNOSTICS")
        box.addWidget(text_label("UCI logs are bounded in memory and can optionally be appended to artifacts/logs.", "muted"))
        for title, key in (
            ("Enable Debug Mode", "debug_mode"), ("Show UCI Log", "show_uci_log"),
            ("Save UCI Logs", "save_uci_logs"), ("Show Engine Raw Output", "show_raw_engine_output"),
        ):
            box.addWidget(self.checkbox(title, key))
        checks = QHBoxLayout()
        native_check = button("Native Build Check", compact=True)
        native_check.clicked.connect(self.show_native_build_status)
        checks.addWidget(native_check)
        tools = button("Open UCI Log", compact=True)
        tools.clicked.connect(self.show_uci_log_requested.emit)
        checks.addWidget(tools)
        clear = button("Clear Analysis Cache", compact=True)
        clear.clicked.connect(self.clear_analysis_cache_requested.emit)
        checks.addWidget(clear)
        box.addLayout(checks)
        layout.addWidget(card)
        layout.addStretch()
        return page

    def test_sound(self) -> None:
        settings = dict(self.state)
        settings["sound_mute_all"] = False
        settings["sound_move"] = True
        SoundManager(ASSETS, settings, self).play("move")

    def check_for_updates(self) -> None:
        checker = UpdateChecker(self)
        self._update_checker = checker
        checker.completed.connect(self._update_check_completed)
        checker.failed.connect(lambda message: QMessageBox.warning(self, "Update Check", f"Could not check GitHub Releases:\n{message}"))
        checker.check()

    def _update_check_completed(self, tag: str, url: str, name: str) -> None:
        latest = tag or name or "No published release tag"
        message = f"Current version: v{APP_VERSION}\nLatest published release: {latest}"
        if url and QMessageBox.question(self, "Update Check", message + "\n\nOpen the release page?") == QMessageBox.StandardButton.Yes:
            QDesktopServices.openUrl(QUrl(url))
        elif not url:
            QMessageBox.information(self, "Update Check", message)

    def about_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        card, box = self.section("ABOUT DEEPTACTICS")
        box.addWidget(text_label("DeepTactics Engine", "settingsTitle"))
        backend = "Native C++ available" if native_available() else "Native C++ unavailable"
        box.addWidget(text_label(
            f"App Version         v{APP_VERSION}\n"
            f"Engine backend      {backend}\n"
            f"Native DLL status   {native_build_status()}\n"
            f"Python UI           PySide6 / Python {platform.python_version()}\n"
            f"Platform            {platform.system()} {platform.release()}\n"
            "Build purpose       Chess analysis, play, and engine development",
            "detailMeta",
        ))
        box.addWidget(text_label("Version information is read from the same application version shown in the title bar.", "muted"))
        layout.addWidget(card)
        layout.addStretch()
        return page

    def browse_pieces(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Piece Images Folder")
        if path:
            missing = self.missing_piece_images(Path(path))
            if missing:
                QMessageBox.warning(self, "Invalid Piece Set", "This folder is missing required piece files:\n" + "\n".join(missing))
                return
            self.piece_path.setText(path)
            self.state["piece_set_path"] = path
            if hasattr(self, "general_piece_value"):
                self.general_piece_value.setText(f"Pieces:  {path}")
            for preview_name in ("general_board_preview", "board_preview", "piece_preview"):
                preview = getattr(self, preview_name, None)
                if preview is not None:
                    preview.set_piece_folder(path)

    def test_pieces(self) -> None:
        folder = self.resolve_setting_path(self.piece_path.text())
        missing = self.missing_piece_images(folder)
        message = "All 12 piece images are available." if not missing else "Missing files:\n" + "\n".join(missing)
        QMessageBox.information(self, "Test Pieces", message)

    @staticmethod
    def resolve_setting_path(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else ROOT / path

    @staticmethod
    def missing_piece_images(folder: Path) -> list[str]:
        required = (
            "white_king.png", "white_queen.png", "white_rook.png", "white_bishop.png",
            "white_knight.png", "white_pawn.png", "black_king.png", "black_queen.png",
            "black_rook.png", "black_bishop.png", "black_knight.png", "black_pawn.png",
        )
        return [name for name in required if not (folder / name).exists()]

    def browse_folder(self, target: QLineEdit, key: str | None = None) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if path:
            target.setText(path)
            if key is not None:
                self.state[key] = path

    def update_board_theme(self, value: str) -> None:
        if hasattr(self, "general_board_value"):
            self.general_board_value.setText(f"Board:  {value}")
        for preview_name in ("general_board_preview", "board_preview", "piece_preview"):
            preview = getattr(self, preview_name, None)
            if preview is not None:
                preview.set_visual_theme(value)

    def show_native_build_status(self) -> None:
        status = native_build_status()
        available = "available" if native_available() else "unavailable"
        QMessageBox.information(
            self,
            "Native Build Status",
            f"Native backend: {available}\nNative DLL build: {status}",
        )

    def save_settings(self) -> None:
        piece_path = self.resolve_setting_path(str(self.state.get("piece_set_path", "")))
        missing = self.missing_piece_images(piece_path)
        if missing:
            QMessageBox.warning(
                self,
                "Settings Not Saved",
                "The selected piece set is incomplete. Previous settings were kept.\n\nMissing:\n" + "\n".join(missing),
            )
            return
        payload = dict(self.state)
        deep = ENGINE_REGISTRY.entry("DeepTactics")
        if deep is not None:
            ENGINE_REGISTRY.update_profile(
                "DeepTactics",
                use_tablebase=bool(payload.get("use_tablebase", False)),
                syzygy_path=str(payload.get("syzygy_path", "")),
                tablebase_max_pieces=int(payload.get("tablebase_max_pieces", 7)),
            )
        payload["engines"] = [dict(entry) for entry in ENGINE_REGISTRY.entries]
        path = SETTINGS_STORE.save(payload)
        if hasattr(self, "general_piece_value"):
            self.general_piece_value.setText(f'Pieces:  {payload["piece_set_path"]}')
        self.settings_applied.emit(payload)
        suffix = "\n\nLanguage navigation resources will be applied after restart." if payload.get("language") != TRANSLATIONS.language else ""
        QMessageBox.information(self, "Settings Saved", f"Settings saved to:\n{path}{suffix}")

    def reset_section(self, title: str) -> None:
        if QMessageBox.question(self, "Reset Section", f'Reset "{title}" to its defaults?') != QMessageBox.StandardButton.Yes:
            return
        key_map = {
            "APPEARANCE": ("theme", "language"),
            "BOARD DESIGN": ("board_theme",),
            "PIECE SET": ("piece_set_path",),
            "COMMON OPTIONS": ("auto_save_pgn", "confirm_resign", "show_tooltips", "start_with"),
            "GAMEPLAY DISPLAY OPTIONS": ("show_arrows", "enable_premoves", "highlight_last_move", "show_move_numbers", "confirm_resign", "restore_clock_on_undo", "auto_promote_queen"),
            "OTHER SETTINGS": ("auto_save_pgn", "max_think_time", "check_for_updates"),
            "DEFAULT FOLDERS": ("pgn_auto_save", "analysis_export", "database", "engine_files", "screenshots", "logs", "opening_books", "training_outputs"),
            "ANALYSIS DEFAULTS": (
                "default_analysis_engine", "score_perspective", "nnue_mode", "analysis_mode",
                "default_analysis_depth", "default_move_time", "analysis_multipv", "auto_analyze_after_move",
                "show_move_classification", "use_analysis_cache",
                "review_full_game", "live_analysis_after_review",
            ),
            "DISPLAY": ("interface_scale", "show_coordinates", "show_tooltips", "compact_tables", "show_raw_score_tooltip"),
            "SOUND": ("sound_move", "sound_capture", "sound_check", "sound_game_end", "sound_engine_ready", "sound_mute_all", "sound_volume"),
            "ADVANCED / DIAGNOSTICS": ("debug_mode", "show_uci_log", "save_uci_logs", "show_raw_engine_output"),
        }
        for key in key_map.get(title, ()):
            self.state[key] = self.defaults[key]
            if key in self.choice_controls and self.defaults[key] in self.choice_controls[key]:
                self.choice_controls[key][self.defaults[key]].setChecked(True)
        if title == "APPEARANCE":
            self.general_language.setCurrentText(self.defaults["language"])
            self.theme_changed.emit(self.defaults["theme"])
        if title in {"BOARD DESIGN", "PIECE SET"} and hasattr(self, "general_board_value"):
            self.update_board_theme(str(self.state.get("board_theme", self.defaults["board_theme"])))
            if title == "PIECE SET":
                self.piece_path.setText(str(self.defaults["piece_set_path"]))
                self.general_piece_value.setText(f'Pieces:  {self.defaults["piece_set_path"]}')
                for preview_name in ("general_board_preview", "board_preview", "piece_preview"):
                    preview = getattr(self, preview_name, None)
                    if preview is not None:
                        preview.set_piece_folder(self.resolve_setting_path(str(self.defaults["piece_set_path"])))
        if title == "DEFAULT FOLDERS" and hasattr(self, "folder_fields"):
            labels = {
                "PGN Auto Save": "pgn_auto_save", "Analysis Export": "analysis_export",
                "Database": "database", "Engine Files": "engine_files",
                "Screenshots": "screenshots", "Logs": "logs",
                "Opening Books / Lines": "opening_books", "Training Outputs": "training_outputs",
            }
            for label, key in labels.items():
                self.folder_fields[label].setText(str(self.defaults[key]))
        self.refresh_controls()
        QMessageBox.information(self, "Section Reset", f"{title} defaults restored.")

    def reset_defaults(self) -> None:
        if QMessageBox.question(self, "Reset Settings", "Reset all settings to their defaults?") == QMessageBox.StandardButton.Yes:
            preserved = {
                key: value for key, value in self.state.items()
                if key.startswith("opening_") or key.startswith("database_")
            }
            self.state = dict(self.defaults)
            self.state.update(preserved)
            ENGINE_REGISTRY.entries = [dict(entry) for entry in DEFAULT_GUI_SETTINGS["engines"]]
            ENGINE_REGISTRY.changed.emit(ENGINE_REGISTRY.names())
            for key, controls in self.choice_controls.items():
                default = self.defaults.get(key)
                if default in controls:
                    controls[default].setChecked(True)
            self.general_language.setCurrentText(self.defaults["language"])
            self.theme_changed.emit("Dark")
            self.piece_path.setText(str(self.defaults["piece_set_path"]))
            self.update_board_theme(str(self.defaults["board_theme"]))
            self.general_piece_value.setText(f'Pieces:  {self.defaults["piece_set_path"]}')
            self.refresh_controls()
            QMessageBox.information(self, "Settings Reset", "Defaults restored.")


class Footer(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("footer")
        self.setFixedHeight(58)
        self.engine_name = "DeepTactics"
        self.status = "Ready"
        self.hash_size = "--"
        self.nodes = "--"
        self.nps = "--"
        self.started_at = time.monotonic()
        self.last_sample_at = self.started_at
        self.last_cpu_time = time.process_time()
        row = QHBoxLayout(self)
        row.setContentsMargins(45, 0, 42, 0)
        self.entries = [
            ("\u25cf", "DeepTactics  |  Ready", GREEN),
            ("\u25a6", "Hash:  2,048 MB", SECONDARY),
            ("\u2318", "Nodes:  --", SECONDARY),
            ("\u26a1", "NPS:  --", "#f1be24"),
            ("\u25f4", "Time:  00:00", SECONDARY),
            ("", "UI CPU:  --", SECONDARY),
            ("", "RAM:  --", SECONDARY),
        ]
        self.metrics: list[QLabel] = []
        for index, (glyph, content, tint) in enumerate(self.entries):
            item = QLabel(f"{glyph}  {content}" if glyph else content)
            item.setStyleSheet(f"color: {tint};")
            row.addWidget(item)
            self.metrics.append(item)
            if index < len(self.entries) - 1:
                row.addStretch()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_runtime_metrics)
        self.timer.start(1000)
        self.update_runtime_metrics()

    def apply_theme(self, theme: str) -> None:
        light_tints = ("#11854c", "#a86900", "#41536b", "#41536b", "#41536b", "#41536b", "#41536b")
        dark_tints = (GREEN, "#f1be24", SECONDARY, SECONDARY, SECONDARY, SECONDARY, SECONDARY)
        for label, tint in zip(self.metrics, light_tints if theme == "Light" else dark_tints):
            label.setStyleSheet(f"color: {tint};")

    def set_context(self, engine_name: str, status: str = "Ready", hash_size: str = "--") -> None:
        self.engine_name = engine_name
        self.status = status
        self.hash_size = hash_size
        self.update_runtime_metrics()

    def set_search_metrics(self, nodes: str = "--", nps: str = "--", status: str | None = None) -> None:
        self.nodes = nodes
        self.nps = nps
        if status is not None:
            self.status = status
        self.update_runtime_metrics()

    def update_runtime_metrics(self) -> None:
        now = time.monotonic()
        elapsed = max(0, int(now - self.started_at))
        elapsed_text = f"{elapsed // 60:02d}:{elapsed % 60:02d}"
        interval = max(now - self.last_sample_at, 0.001)
        cpu_time = time.process_time()
        ui_cpu = max(0.0, min(999.0, (cpu_time - self.last_cpu_time) / interval * 100.0))
        self.last_sample_at = now
        self.last_cpu_time = cpu_time
        values = (
            f"\u25cf  {self.engine_name}  |  {self.status}",
            f"\u25a6  Hash:  {self.hash_size}",
            f"\u2318  Nodes:  {self.nodes}",
            f"\u26a1  NPS:  {self.nps}",
            f"\u25f4  Time:  {elapsed_text}",
            f"UI CPU:  {ui_cpu:.1f}%",
            f"RAM:  {process_memory_mb():.1f} MB",
        )
        for label, value in zip(self.metrics, values):
            label.setText(value)


def process_memory_mb() -> float:
    """Return this UI process working set on Windows without extra packages."""
    if os.name != "nt":
        return 0.0

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    return counters.WorkingSetSize / (1024 * 1024) if ok else 0.0


class DeepTacticsReferenceWindow(QMainWindow):
    def __init__(self, screen: str = "analyze") -> None:
        super().__init__()
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        screen_title = {"play": "Play", "ai_match": "AI Match", "games": "Games", "opening": "Openings", "database": "Database", "setting": "Setting"}.get(screen, "Analyze")
        self.setWindowTitle(f"DeepTactics - {screen_title} GUI Preview")
        icon_path = ASSETS / "icon" / "program_icon.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        self.resize(1680, 944)
        self.setMinimumSize(1440, 840)
        self.analysis_document = GameDocument()
        self.play_document = GameDocument()
        self.play_started = False
        self.play_players: tuple[str, str] = ("Human", "Human")
        self.play_time_control = TimeControlConfig("Unlimited", None)
        self.play_clock_ms: dict[Color, int | None] = {Color.WHITE: None, Color.BLACK: None}
        self.play_clock_history: list[dict[Color, int | None]] = []
        self.play_turn_started_at: float | None = None
        self.play_live_ply = 0
        self.play_paused = False
        self._play_search_purpose: str | None = None
        self._play_discard_result = False
        self._play_hint_move: str | None = None
        self.play_premoves: list[str] = []
        self.play_premove_color: Color | None = None
        self.match_document = GameDocument()
        self.match_running = False
        self.match_paused = False
        self.match_search_color: Color | None = None
        self.match_live_ply = 0
        self.match_game_number = 0
        self.match_total_games = 0
        self.match_results = [0, 0, 0]
        self.match_pgns: list[str] = []
        self.match_time_control = TimeControlConfig("Blitz 5+3", 300_000, 3_000)
        self.match_clock_ms: dict[Color, int | None] = {Color.WHITE: 300_000, Color.BLACK: 300_000}
        self.match_turn_started_at: float | None = None
        self.match_opening_lines = load_ai_match_opening_lines()
        self.analysis_active = False
        self.analysis_phase = "idle"
        self._baseline_positions: tuple[str, ...] = ()
        self._baseline_ply = 0
        self._baseline_results: dict[int, UCIAnalysisInfo] = {}
        self._baseline_bestmoves: dict[int, str] = {}
        self._analysis_classifications: dict[int, MoveClassification] = {}
        self._analysis_classification_evidence: dict[int, AnalyzeMoveEvidence] = {}
        self._analysis_cache: dict[tuple[object, ...], dict[int, UCIAnalysisInfo]] = {}
        self._analysis_requested_fen = ""
        self._analysis_request_started_at = 0.0
        self._analysis_review_started_at = 0.0
        self._changing_analysis_source = False
        self._discard_replaced_result = False
        self._analysis_suggestion_enabled = False
        self._latest_live_bestmove: str | None = None
        library_root = Path(str(SETTINGS_STORE.load().get("pgn_auto_save", "artifacts/pgn")))
        if not library_root.is_absolute():
            library_root = ROOT / library_root
        self.game_library = GameLibraryStore(library_root)

        body = QWidget()
        main = QVBoxLayout(body)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)
        self.header = Header(screen_title)
        self.header.tab_selected.connect(self.show_screen)
        main.addWidget(self.header)
        self.stack = QStackedWidget()
        self.pages: dict[str, QWidget] = {
            "Play": self._board_page("play"),
            "Analyze": self._board_page("analyze"),
            "AI Match": self._match_page(),
            "Games": GamesPage(self.game_library),
            "Openings": OpeningPage(),
            "Database": DatabasePage(),
            "Setting": SettingsPage(),
        }
        self.pages["Games"].open_analyze.connect(self.open_library_game_in_analyze)
        self.pages["Games"].load_play.connect(self.load_library_game_in_play)
        self.pages["Openings"].open_analyze.connect(self.open_opening_in_analyze)
        self.pages["Database"].open_analyze.connect(self.open_database_pgn_in_analyze)
        self.pages["Setting"].theme_changed.connect(self.apply_theme)
        self.pages["Setting"].settings_applied.connect(self.apply_saved_settings)
        self.pages["Setting"].clear_analysis_cache_requested.connect(self.clear_analysis_cache_from_settings)
        self.play_console.analyze_requested.connect(self.open_play_in_analyze)
        self.analysis_console.ply_requested.connect(self.navigate_analysis_ply)
        self.analysis_console.variation_requested.connect(self.analysis_document.seek_variation_end)
        self.analysis_document.changed.connect(self.apply_analysis_snapshot)
        self.play_document.changed.connect(self.apply_play_snapshot)
        for page in self.pages.values():
            self.stack.addWidget(page)
        main.addWidget(self.stack, 1)
        self.footer = Footer()
        self.analysis_console.engine.engine_changed.connect(lambda name: self.footer.set_context(name, "Ready"))
        self.analysis_console.engine.engine_changed.connect(self.analysis_engine_changed)
        main.addWidget(self.footer)
        TRANSLATIONS.apply_widgets(body)
        self.setCentralWidget(body)
        self.setStyleSheet(STYLE)
        initial_settings = SETTINGS_STORE.load()
        self._runtime_settings = dict(initial_settings)
        self.sound_manager = SoundManager(ASSETS, initial_settings, self)
        self.uci_log_dialog = UciLogDialog(self)
        initial_piece_folder = SettingsPage.resolve_setting_path(str(initial_settings.get("piece_set_path", "assets/pieces")))
        for board in self.findChildren(WoodenBoard):
            board.set_visual_theme(str(initial_settings.get("board_theme", "Wood")))
            board.set_coordinates_visible(bool(initial_settings.get("show_coordinates", True)))
            if not SettingsPage.missing_piece_images(initial_piece_folder):
                board.set_piece_folder(initial_piece_folder)
        self.uci_client = UCIClient(ROOT, self)
        self.play_uci_client = UCIClient(ROOT, self)
        self.match_white_client = UCIClient(ROOT, self)
        self.match_black_client = UCIClient(ROOT, self)
        for session, client in (
            ("Analyze", self.uci_client),
            ("Play", self.play_uci_client),
            ("AI Match White", self.match_white_client),
            ("AI Match Black", self.match_black_client),
        ):
            client.command_sent.connect(lambda line, name=session: self._record_uci_log(name, ">", line))
            client.raw_line_received.connect(lambda line, name=session: self._record_uci_log(name, "<", line))
        initial_engine = str(initial_settings.get("default_analysis_engine", "DeepTactics"))
        if ENGINE_REGISTRY.entry(initial_engine) is None:
            initial_engine = "DeepTactics"
        self.analysis_console.engine.set_engine(initial_engine, notify=False)
        self.uci_client.configure_engine(ENGINE_REGISTRY.entry(initial_engine))
        self.analysis_console.engine.analyze_requested.connect(self.start_analysis)
        self.analysis_console.engine.stop_requested.connect(self.stop_analysis)
        self.analysis_console.engine.best_move_requested.connect(self.find_best_analysis_move)
        self.analysis_console.engine.depth_changed.connect(self.change_analysis_depth)
        self.analysis_console.suggestion_toggled.connect(self.toggle_analysis_suggestion)
        self.uci_client.state_changed.connect(self.analysis_state_changed)
        self.uci_client.info_received.connect(self.analysis_info_received)
        self.uci_client.bestmove_received.connect(self.analysis_bestmove_received)
        self.uci_client.error_received.connect(self.analysis_error_received)
        self.uci_client.identity_received.connect(self.analysis_identity_received)
        self.play_uci_client.info_received.connect(self.play_info_received)
        self.play_uci_client.bestmove_received.connect(self.play_bestmove_received)
        self.play_uci_client.error_received.connect(self.play_error_received)
        self.match_white_client.info_received.connect(lambda info: self.match_info_received(Color.WHITE, info))
        self.match_black_client.info_received.connect(lambda info: self.match_info_received(Color.BLACK, info))
        self.match_white_client.bestmove_received.connect(lambda move: self.match_bestmove_received(Color.WHITE, move))
        self.match_black_client.bestmove_received.connect(lambda move: self.match_bestmove_received(Color.BLACK, move))
        self.match_white_client.error_received.connect(lambda message: self.match_error_received(Color.WHITE, message))
        self.match_black_client.error_received.connect(lambda message: self.match_error_received(Color.BLACK, message))
        self.pages["Setting"].show_uci_log_requested.connect(self.show_uci_log)
        self.play_clock_timer = QTimer(self)
        self.play_clock_timer.setInterval(100)
        self.play_clock_timer.timeout.connect(self.update_play_clocks)
        self.analysis_document.new_game()
        self.play_document.new_game()
        self.match_document.new_game()
        self.show_screen(screen_title)
        if bool(initial_settings.get("show_uci_log", False)):
            QTimer.singleShot(0, self.show_uci_log)
        if bool(initial_settings.get("check_for_updates", False)):
            QTimer.singleShot(0, self.pages["Setting"].check_for_updates)

    def _board_page(self, screen: str) -> QWidget:
        play_screen = screen == "play"
        content = QWidget()
        shell = QVBoxLayout(content)
        shell.setContentsMargins(24, 10, 24, 18)
        shell.setSpacing(5)
        if play_screen:
            shell.addWidget(preview_notice("PLAY - Human games and one selected UCI engine profile are active. Use AI Match for two independent engine sessions."))
        horizontal = QHBoxLayout()
        horizontal.setSpacing(20)
        board = BoardWorkspace(play_screen=play_screen)
        board.setMinimumWidth(780)
        horizontal.addWidget(board, 49)
        console = PlayConsole() if play_screen else RightConsole()
        if play_screen:
            self.play_board = board
            self.play_console = console
            self.play_board.start_play_requested.connect(self.start_play_session)
            self.play_board.pause_play_requested.connect(self.toggle_play_pause)
            self.play_board.load_pgn_requested.connect(self.load_play_pgn)
            self.play_board.save_pgn_requested.connect(self.save_play_pgn)
            self.play_board.undo_requested.connect(self.undo_play_move)
            self.play_board.board.move_requested.connect(self.apply_play_move)
            self.play_board.board.left_click_completed.connect(self.handle_play_board_click)
            self.play_console.moves.ply_selected.connect(self.view_play_ply)
            self.play_console.suggest_requested.connect(self.request_play_suggestion)
            self.play_console.evaluation_toggled.connect(self.toggle_play_evaluation)
        else:
            self.analysis_board = board
            self.analysis_console = console
            self.analysis_board.new_requested.connect(self.new_analysis_document)
            self.analysis_board.load_pgn_requested.connect(self.load_analysis_pgn)
            self.analysis_board.save_pgn_requested.connect(self.save_analysis_pgn)
            self.analysis_board.undo_requested.connect(self.analysis_document.undo)
            self.analysis_board.custom_requested.connect(self.open_custom_position)
            self.analysis_board.board.set_interactive(True)
            self.analysis_board.board.move_requested.connect(self.apply_analysis_move)
        horizontal.addWidget(console, 51)
        shell.addLayout(horizontal, 1)
        return content

    def _match_page(self) -> QWidget:
        content = QWidget()
        shell = QVBoxLayout(content)
        shell.setContentsMargins(20, 10, 20, 18)
        shell.setSpacing(5)
        self.match_notice = preview_notice("MATCH READY - select two registered engines and start a match")
        shell.addWidget(self.match_notice)
        horizontal = QHBoxLayout()
        horizontal.setSpacing(12)
        self.match_board = AIMatchBoard()
        horizontal.addWidget(self.match_board, 43)
        self.match_center = AIMatchCenter()
        self.match_right = AIMatchRight()
        self.match_document.changed.connect(self.apply_match_snapshot)
        self.match_center.move_list.ply_selected.connect(self.view_match_ply)
        self.match_center.move_list.live_requested.connect(lambda: self.view_match_ply(self.match_live_ply))
        self.match_board.graph.canvas.ply_selected.connect(self.view_match_ply)
        self.match_right.controls.start_requested.connect(self.start_ai_match)
        self.match_right.controls.pause_requested.connect(self.toggle_ai_match_pause)
        self.match_right.controls.stop_requested.connect(self.stop_ai_match)
        self.match_right.controls.reset_requested.connect(self.reset_ai_match)
        self.match_right.controls.save_requested.connect(self.save_ai_match_pgn)
        self.match_right.controls.export_requested.connect(self.export_ai_match)
        self.match_right.white.clear_hash_requested.connect(lambda _name: self.clear_match_hash(Color.WHITE))
        self.match_right.black.clear_hash_requested.connect(lambda _name: self.clear_match_hash(Color.BLACK))
        horizontal.addWidget(self.match_center, 23)
        horizontal.addWidget(self.match_right, 34)
        shell.addLayout(horizontal, 1)
        return content

    def show_screen(self, screen_name: str) -> None:
        page = self.pages.get(screen_name)
        if page is None:
            return
        self.stack.setCurrentWidget(page)
        self.header.set_active(screen_name)
        self.setWindowTitle(f"DeepTactics - {screen_name} GUI Preview")
        if hasattr(self, "footer"):
            if screen_name == "Analyze":
                state = "Analysis active" if self.analysis_active else "Ready"
                self.footer.set_context(self.analysis_console.engine.engine_name.text(), state)
            elif screen_name == "AI Match":
                self.footer.set_context("UCI Match", "Ready")
            else:
                self.footer.set_context("DeepTactics", "Ready")

    def open_play_in_analyze(self) -> None:
        self.show_screen("Analyze")
        moves = self.play_document.mainline_uci_moves[: self.play_live_ply]
        self._replace_analysis_source(lambda: self.analysis_document.load_uci_line(moves))
        QMessageBox.information(self, "Analyze Position", "The current Play position has been copied into Analyze. The ongoing game remains unchanged.")

    def open_library_game_in_analyze(self, record: GameRecord) -> None:
        try:
            moves = self.game_library.uci_moves(record)
            self.game_library.mark_opened(record)
        except (OSError, UnicodeError, ValueError) as error:
            QMessageBox.warning(self, "Open in Analyze", f"Could not load this PGN:\n{error}")
            return
        self.show_screen("Analyze")
        self._replace_analysis_source(lambda: self.analysis_document.load_uci_line(moves))
        self.pages["Games"].refresh_status()

    def open_opening_in_analyze(self, moves: tuple[str, ...]) -> None:
        self.show_screen("Analyze")
        self._replace_analysis_source(lambda: self.analysis_document.load_uci_line(moves))
        self.footer.set_context("DeepTactics", "Opening line loaded for analysis", "--")

    def open_database_pgn_in_analyze(self, record: DatasetRecord) -> None:
        try:
            game = game_from_pgn(record.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as error:
            QMessageBox.warning(self, "Database PGN", f"Unable to open this PGN for analysis:\n{error}")
            return
        moves = tuple(move.to_uci() for move in game.move_history)
        self.show_screen("Analyze")
        self._replace_analysis_source(lambda: self.analysis_document.load_uci_line(moves))
        self.footer.set_context("DeepTactics", f"Database PGN loaded: {record.name}", "--")

    def load_library_game_in_play(self, record: GameRecord) -> None:
        try:
            game = game_from_pgn(self.game_library.pgn_text(record))
            if (game.is_over() or record.result != "*") and QMessageBox.question(
                self,
                "Finished Game",
                "This PGN is already finished. Load its final position for review or analysis anyway?",
            ) != QMessageBox.StandardButton.Yes:
                return
            moves = tuple(move.to_uci() for move in game.move_history)
            self.game_library.mark_opened(record)
        except (OSError, UnicodeError, ValueError) as error:
            QMessageBox.warning(self, "Load Game", f"Could not load this PGN:\n{error}")
            return
        if self.play_uci_client.is_searching:
            self._play_discard_result = True
            self._play_search_purpose = None
            self.play_uci_client.cancel_analysis()
        self.play_started = True
        self.play_premoves.clear()
        self.play_premove_color = None
        self.play_players = self.play_console.player_setup.selected_players()
        self.play_time_control = self.play_console.time_control.selected_config()
        self.play_console.start_session()
        self.play_clock_ms = {Color.WHITE: self.play_time_control.base_ms, Color.BLACK: self.play_time_control.base_ms}
        self.play_clock_history = []
        self.play_document.load_uci_line(moves)
        self.play_live_ply = len(moves)
        self.play_turn_started_at = time.monotonic()
        self.play_clock_timer.start()
        self.show_screen("Play")
        self.footer.set_context("DeepTactics", "PGN loaded for continued play", "--")
        self.pages["Games"].refresh_status()

    def _register_saved_pgn(self, text: str, source_path: str, source_type: str) -> None:
        _added, warnings = self.game_library.import_text(text, source_path=source_path, source_type=source_type)
        self.pages["Games"].refresh_status()
        if warnings:
            QMessageBox.warning(self, "Games Library", "The PGN was saved but could not be fully indexed:\n" + "\n".join(warnings))

    def apply_analysis_snapshot(self, snapshot: GameSnapshot) -> None:
        update_started_at = time.perf_counter()
        self.analysis_board.board.set_fen(snapshot.fen, self.analysis_document.last_move_uci)
        shown_classification = None
        if not self.analysis_document.viewing_variation:
            shown_classification = self._analysis_classifications.get(snapshot.current_ply)
        self.analysis_board.board.set_move_classification(shown_classification)
        self.analysis_board.board.set_legal_moves(self.analysis_document.game.legal_moves())
        self.analysis_console.set_snapshot(snapshot)
        if self.analysis_active and self.analysis_phase == "live" and not self._changing_analysis_source:
            if not self._restore_cached_position(snapshot.fen, cancel_running=True):
                should_analyze = (
                    not self.analysis_document.viewing_variation
                    or bool(SETTINGS_STORE.load().get("auto_analyze_after_move", True))
                )
                if should_analyze:
                    self._request_current_analysis()
        elif not self.analysis_active:
            self._restore_cached_position(snapshot.fen)
        ui_ms = (time.perf_counter() - update_started_at) * 1000
        if ui_ms >= 10:
            print(f"Qt Analyze timing ui_snapshot_ms={ui_ms:.1f} ply={snapshot.current_ply}")

    def navigate_analysis_ply(self, ply: int) -> None:
        self.analysis_document.seek_mainline(ply)

    def apply_play_snapshot(self, snapshot: GameSnapshot) -> None:
        is_live = snapshot.current_ply == len(self.play_document.mainline_uci_moves)
        if is_live:
            self.play_live_ply = snapshot.current_ply
        self.play_console.update_snapshot(snapshot, active=self.play_started and not self.play_paused and is_live)
        self.update_play_clocks()
        self.play_board.undo_button.setEnabled(self.play_started and self.play_live_ply > 0)
        human_turn = self._play_player_to_move() == "Human"
        allow_input = self.play_started and not self.play_paused and is_live and snapshot.status == ChessGameStatus.ONGOING.value
        display_board = self._premove_virtual_board() if allow_input and self.play_premoves else None
        self.play_board.board.set_fen(
            board_to_fen(display_board) if display_board is not None else snapshot.fen,
            self.play_document.last_move_uci,
        )
        if self.play_premoves:
            self.play_board.board.marked_squares = {square for move in self.play_premoves for square in (move[:2], move[2:4])}
        self.play_board.board.set_interactive(allow_input)
        legal_moves: list[Move] = []
        if allow_input and human_turn:
            legal_moves.extend(self.play_document.game.legal_moves())
        if allow_input and bool(SETTINGS_STORE.load().get("enable_premoves", True)):
            virtual = display_board or self.play_document.game.board
            for color, player in ((Color.WHITE, self.play_players[0]), (Color.BLACK, self.play_players[1])):
                if color is not self.play_document.game.board.side_to_move and player == "Human":
                    legal_moves.extend(generate_legal_moves(virtual, color))
        self.play_board.board.set_legal_moves(legal_moves)
        if self.play_started and is_live and snapshot.status != ChessGameStatus.ONGOING.value:
            self.play_clock_timer.stop()
            self.play_board.board.set_interactive(False)
        elif self.play_started and is_live and not human_turn:
            QTimer.singleShot(0, self._request_engine_play_move)
        elif (
            self.play_started
            and is_live
            and human_turn
            and self.play_console.assist.eval_toggle.isChecked()
            and self._play_search_purpose is None
        ):
            QTimer.singleShot(0, lambda: self._request_play_search("evaluation"))

    def open_custom_position(self) -> None:
        dialog = CustomPositionDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.result_fen:
            self._replace_analysis_source(lambda: self.analysis_document.load_fen(dialog.result_fen))
            self.footer.set_context("DeepTactics", "Custom analysis position loaded", "--")

    def apply_analysis_move(self, move_text: str) -> None:
        try:
            self.analysis_document.apply_uci_move(move_text)
        except ValueError:
            QMessageBox.warning(self, "Illegal move", "This move is not legal in the current position.")
            return
        self.analysis_board.board.clear_annotations()

    def new_analysis_document(self) -> None:
        self._replace_analysis_source(self.analysis_document.new_game)
        self.analysis_board.board.clear_annotations()

    def load_analysis_pgn(self) -> None:
        path, _selected_filter = QFileDialog.getOpenFileName(self, "Load Analysis PGN", "", "PGN Files (*.pgn)")
        if not path:
            return
        try:
            loaded_game = game_from_pgn(Path(path).read_text(encoding="utf-8"))
            moves = tuple(move.to_uci() for move in loaded_game.move_history)
            self._replace_analysis_source(lambda: self.analysis_document.load_uci_line(moves))
        except (OSError, UnicodeError, ValueError) as error:
            QMessageBox.warning(self, "Load PGN", f"Could not load this PGN:\n{error}")

    def save_analysis_pgn(self) -> None:
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Analysis PGN",
            default_pgn_filename("DeepTactics_Analysis"),
            "PGN Files (*.pgn)",
        )
        if not path:
            return
        try:
            text = pgn_with_metadata(
                self.analysis_document.snapshot().pgn,
                Event="DeepTactics Analysis",
                Date=datetime.now().strftime("%Y.%m.%d"),
                Type="Analyzed",
            )
            Path(path).write_text(text, encoding="utf-8")
            self._register_saved_pgn(text, path, "Analyzed")
        except OSError as error:
            QMessageBox.warning(self, "Save PGN", f"Could not save this PGN:\n{error}")

    def start_play_session(self) -> None:
        white, black = self.play_console.player_setup.selected_players()
        engine_players = [name for name in (white, black) if name != "Human"]
        if len(set(engine_players)) > 1:
            QMessageBox.information(
                self,
                "Use AI Match",
                "Play supports human games or one engine profile. For two independent engine sessions, use AI Match.",
            )
            return
        if engine_players:
            self.play_uci_client.configure_engine(ENGINE_REGISTRY.entry(engine_players[0]))
        if self.play_uci_client.is_searching:
            self._play_discard_result = True
            self.play_uci_client.cancel_analysis()
        self.play_players = (white, black)
        _white, _black, self.play_time_control = self.play_console.start_session()
        self.play_clock_ms = {
            Color.WHITE: self.play_time_control.base_ms,
            Color.BLACK: self.play_time_control.base_ms,
        }
        self.play_clock_history = []
        self.play_live_ply = 0
        self.play_paused = False
        self._play_search_purpose = None
        self.play_uci_client.cancel_analysis()
        self.play_started = True
        self.play_premoves.clear()
        self.play_premove_color = None
        self.play_board.board.clear_annotations()
        self.play_board.eval_meter.set_active(self.play_console.assist.eval_toggle.isChecked())
        self.play_board.tail_action.blockSignals(True)
        self.play_board.tail_action.setChecked(False)
        self.play_board.tail_action.setText("\u23f8  Pause")
        self.play_board.tail_action.setEnabled(True)
        self.play_board.tail_action.blockSignals(False)
        self.play_document.new_game()
        self.play_turn_started_at = time.monotonic()
        self.play_clock_timer.start()
        self.footer.set_context("DeepTactics", f"Game active | {white} vs {black}", "--")

    def apply_play_move(self, move_text: str) -> None:
        if not self.play_started:
            return
        virtual_board = self._premove_virtual_board()
        piece = virtual_board.get_piece(Square.from_algebraic(move_text[:2]))
        if (
            piece is not None
            and piece.color is not self.play_document.game.board.side_to_move
            and self._player_for_color(piece.color) == "Human"
            and bool(SETTINGS_STORE.load().get("enable_premoves", True))
        ):
            if self.play_board.board.last_input_was_drag:
                self._queue_play_premove(move_text, piece.color)
            return
        if self._play_player_to_move() != "Human":
            return
        if (
            self.play_premoves
            and self.play_premove_color is self.play_document.game.board.side_to_move
            and not self.play_board.board.last_input_was_drag
        ):
            self.clear_play_premoves()
        if self.play_uci_client.is_searching:
            self._play_discard_result = True
            self._play_search_purpose = None
            self.play_uci_client.cancel_analysis()
        if self.play_document.current_ply != self.play_live_ply:
            self.play_document.seek_mainline(self.play_live_ply)
        dragged = self.play_board.board.last_input_was_drag
        self._commit_play_move(move_text, animate=not dragged)

    def _commit_play_move(self, move_text: str, *, animate: bool = False, premove: bool = False) -> None:
        moving_color = self.play_document.game.board.side_to_move
        remaining = self._clock_value(moving_color)
        clock_snapshot = dict(self.play_clock_ms)
        clock_snapshot[moving_color] = remaining
        self.play_board.board.clear_annotations()
        try:
            self.play_document.append_mainline_move(move_text)
        except ValueError:
            QMessageBox.warning(self, "Illegal move", "This move is not legal in the current game.")
            return
        self.play_clock_history.append(clock_snapshot)
        captured = self.play_document.game.captured_history[-1] is not None
        self.sound_manager.play("capture" if captured else "move")
        if self.play_document.game.is_in_check():
            self.sound_manager.play("check")
        if self.play_document.game.is_over():
            self.sound_manager.play("game_end")
        self._apply_clock_after_move(moving_color, remaining, premove=premove)
        if animate:
            self.play_board.board.animate_move(move_text)
        self._try_apply_play_premove()

    def _play_player_to_move(self) -> str:
        return self._player_for_color(self.play_document.game.board.side_to_move)

    def _player_for_color(self, color: Color) -> str:
        return self.play_players[0 if color is Color.WHITE else 1]

    def _premove_virtual_board(self) -> Board:
        board = self.play_document.game.board.copy()
        for move_text in self.play_premoves:
            piece = board.get_piece(Square.from_algebraic(move_text[:2]))
            if piece is None:
                continue
            candidate_board = board.copy()
            candidate_board.side_to_move = piece.color
            legal = next((move for move in generate_legal_moves(candidate_board) if move.to_uci() == move_text), None)
            if legal is not None:
                board.side_to_move = piece.color
                board.apply_move_unchecked(legal)
        return board

    def _queue_play_premove(self, move_text: str, color: Color) -> None:
        if self.play_premoves and self.play_premove_color is not color:
            return
        virtual = self._premove_virtual_board()
        virtual.side_to_move = color
        legal = next((move for move in generate_legal_moves(virtual) if move.to_uci() == move_text), None)
        if legal is None:
            return
        self.play_premove_color = color
        self.play_premoves.append(legal.to_uci())
        self.apply_play_snapshot(self.play_document.snapshot())
        self.footer.set_context("DeepTactics", f"Premove queued ({len(self.play_premoves)})", "--")

    def clear_play_premoves(self) -> None:
        if not self.play_premoves:
            return
        self.play_premoves.clear()
        self.play_premove_color = None
        self.apply_play_snapshot(self.play_document.snapshot())

    def handle_play_board_click(self, square: str) -> None:
        """Only the player owning queued premoves may cancel them by clicking their piece."""
        if not self.play_premoves or self.play_premove_color is None:
            return
        piece = self._premove_virtual_board().get_piece(Square.from_algebraic(square))
        if piece is not None and piece.color is self.play_premove_color:
            self.clear_play_premoves()

    def _try_apply_play_premove(self) -> None:
        if not self.play_premoves or not self.play_started:
            return
        next_move = self.play_premoves[0]
        piece = self.play_document.game.board.get_piece(Square.from_algebraic(next_move[:2]))
        if piece is None or piece.color is not self.play_document.game.board.side_to_move:
            return
        if self._play_player_to_move() != "Human":
            return
        legal = next((move for move in self.play_document.game.legal_moves() if move.to_uci() == next_move), None)
        if legal is None:
            self.play_premoves.pop(0)
            if not self.play_premoves:
                self.play_premove_color = None
            self.apply_play_snapshot(self.play_document.snapshot())
            return
        self.play_premoves.pop(0)
        if not self.play_premoves:
            self.play_premove_color = None
        self._commit_play_move(legal.to_uci(), premove=True)

    def _clock_value(self, color: Color) -> int | None:
        value = self.play_clock_ms[color]
        if value is None:
            return None
        if (
            self.play_started
            and not self.play_paused
            and self.play_document.game.board.side_to_move is color
            and self.play_turn_started_at is not None
            and self.play_document.current_ply == self.play_live_ply
        ):
            return max(0, value - int((time.monotonic() - self.play_turn_started_at) * 1000))
        return max(0, value)

    def update_play_clocks(self) -> None:
        if not hasattr(self, "play_console"):
            return
        white_ms = self._clock_value(Color.WHITE)
        black_ms = self._clock_value(Color.BLACK)
        self.play_console.player_clock.set_clock_values(white_ms, black_ms)
        if not self.play_started:
            return
        active = self.play_document.game.board.side_to_move
        if self._clock_value(active) == 0 and self.play_clock_ms[active] is not None:
            winner = "Black" if active is Color.WHITE else "White"
            self.play_started = False
            self.play_paused = False
            self.play_clock_timer.stop()
            self.play_board.board.set_interactive(False)
            self.play_console.game_status.values["Status"].setText(f"Timeout - {winner} wins")
            self.footer.set_context("DeepTactics", f"Timeout | {winner} wins", "--")
            self.play_uci_client.cancel_analysis()
            self.play_board.tail_action.setEnabled(False)

    def _apply_clock_after_move(self, moving_color: Color, current: int | None, *, premove: bool = False) -> None:
        if current is not None:
            if premove:
                current = max(0, current - 100)
            self.play_clock_ms[moving_color] = current + self.play_time_control.increment_ms
        self.play_turn_started_at = time.monotonic()
        self.update_play_clocks()

    def toggle_play_pause(self, paused: bool) -> None:
        if not self.play_started:
            self.play_board.tail_action.blockSignals(True)
            self.play_board.tail_action.setChecked(False)
            self.play_board.tail_action.blockSignals(False)
            return
        if paused:
            active = self.play_document.game.board.side_to_move
            self.play_clock_ms[active] = self._clock_value(active)
            self.play_paused = True
            self.play_turn_started_at = None
            self.play_clock_timer.stop()
            self.play_board.tail_action.setText("\u25b6  Resume")
            if self.play_uci_client.is_searching:
                self._play_discard_result = True
                self._play_search_purpose = None
                self.play_uci_client.cancel_analysis()
            self.apply_play_snapshot(self.play_document.snapshot())
            self.play_console.game_status.values["Status"].setText("Paused")
            self.footer.set_context("DeepTactics", "Game paused", "--")
            return
        self.play_paused = False
        self.play_turn_started_at = time.monotonic()
        self.play_clock_timer.start()
        self.play_board.tail_action.setText("\u23f8  Pause")
        self.apply_play_snapshot(self.play_document.snapshot())
        self.footer.set_context("DeepTactics", "Game active", "--")

    def view_play_ply(self, ply: int) -> None:
        self.play_document.seek_mainline(ply)
        reviewing = ply != self.play_live_ply
        if reviewing:
            self.play_board.board.set_interactive(False)
            self.footer.set_context("DeepTactics", f"Reviewing ply {ply} | game remains live", "--")
        else:
            self.footer.set_context("DeepTactics", "Game active", "--")

    def undo_play_move(self) -> None:
        if not self.play_started:
            return
        if self.play_uci_client.is_searching:
            self._play_discard_result = True
            self._play_search_purpose = None
            self.play_uci_client.cancel_analysis()
        if self.play_document.current_ply != self.play_live_ply:
            self.play_document.seek_mainline(self.play_live_ply)
        count = 1 if self.play_players == ("Human", "Human") else 2
        restored_clock: dict[Color, int | None] | None = None
        for _index in range(min(count, self.play_live_ply)):
            if self.play_clock_history:
                restored_clock = self.play_clock_history.pop()
            self.play_document.undo()
        if bool(SETTINGS_STORE.load().get("restore_clock_on_undo", False)) and restored_clock is not None:
            self.play_clock_ms = restored_clock
        self.play_live_ply = len(self.play_document.mainline_uci_moves)
        self.play_board.board.clear_annotations()
        self.play_premoves.clear()
        self.play_premove_color = None
        self.play_turn_started_at = time.monotonic()

    def load_play_pgn(self) -> None:
        path, _selected_filter = QFileDialog.getOpenFileName(self, "Load Play PGN", "", "PGN Files (*.pgn)")
        if not path:
            return
        try:
            if self.play_uci_client.is_searching:
                self._play_discard_result = True
                self._play_search_purpose = None
                self.play_uci_client.cancel_analysis()
            loaded_game = game_from_pgn(Path(path).read_text(encoding="utf-8"))
            moves = tuple(move.to_uci() for move in loaded_game.move_history)
            self.play_started = True
            self.play_premoves.clear()
            self.play_premove_color = None
            self.play_players = self.play_console.player_setup.selected_players()
            self.play_time_control = self.play_console.time_control.selected_config()
            self.play_console.start_session()
            self.play_clock_ms = {Color.WHITE: self.play_time_control.base_ms, Color.BLACK: self.play_time_control.base_ms}
            self.play_clock_history = []
            self.play_document.load_uci_line(moves)
            self.play_live_ply = len(moves)
            self.play_turn_started_at = time.monotonic()
            self.play_clock_timer.start()
            self.footer.set_context("DeepTactics", "PGN loaded for play review", "--")
        except (OSError, UnicodeError, ValueError) as error:
            QMessageBox.warning(self, "Load PGN", f"Could not load this PGN:\n{error}")

    def save_play_pgn(self) -> None:
        white, black = self.play_players
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Game PGN",
            default_pgn_filename("DeepTactics_Play", white, black),
            "PGN Files (*.pgn)",
        )
        if not path:
            return
        try:
            source_type = "Human vs AI" if any(player != "Human" for player in self.play_players) else "Saved"
            text = pgn_with_metadata(
                self.play_document.snapshot().pgn,
                Event="DeepTactics Play",
                Date=datetime.now().strftime("%Y.%m.%d"),
                White=white,
                Black=black,
                Type=source_type,
            )
            Path(path).write_text(text, encoding="utf-8")
            self._register_saved_pgn(text, path, source_type)
        except OSError as error:
            QMessageBox.warning(self, "Save PGN", f"Could not save this PGN:\n{error}")

    def toggle_play_evaluation(self, enabled: bool) -> None:
        self.play_board.eval_meter.set_active(enabled)
        if enabled and self.play_started and not self.play_paused and self._play_search_purpose is None:
            self._request_play_search("evaluation")

    def request_play_suggestion(self) -> None:
        if self.play_started and not self.play_paused:
            self._request_play_search("hint")

    def _request_engine_play_move(self) -> None:
        if self._play_search_purpose is not None or self.play_uci_client.is_searching or not self.play_started or self.play_paused:
            return
        if self._play_player_to_move() == "Human":
            return
        self._request_play_search("move")

    def _request_play_search(self, purpose: str) -> None:
        if self._play_search_purpose is not None or self.play_uci_client.is_searching:
            return
        self._play_search_purpose = purpose
        self.play_board.board.suggested_arrow = None
        movetime = min(int(SETTINGS_STORE.load().get("max_think_time", 60)) * 1000, 750)
        self.play_uci_client.analyze_fen(
            self.play_document.fen,
            movetime_ms=movetime,
            moves=self.play_document.mainline_uci_moves,
            initial_fen=None if self.play_document.initial_fen == STANDARD_STARTING_FEN else self.play_document.initial_fen,
        )

    def play_info_received(self, info: UCIAnalysisInfo) -> None:
        if self._play_search_purpose is None or info.multipv != 1:
            return
        if self.play_console.assist.eval_toggle.isChecked():
            self.play_board.eval_meter.set_score(info.score_cp_white, info.mate_white)
        self.footer.set_search_metrics(f"{info.nodes:,}", f"{info.nps:,}", "Play engine thinking")

    def play_bestmove_received(self, move: str) -> None:
        if self._play_discard_result:
            self._play_discard_result = False
            QTimer.singleShot(0, self._request_engine_play_move)
            if self._play_player_to_move() == "Human" and self.play_console.assist.eval_toggle.isChecked():
                QTimer.singleShot(0, lambda: self._request_play_search("evaluation"))
            return
        purpose = self._play_search_purpose
        self._play_search_purpose = None
        if move == "0000" or purpose is None:
            return
        if purpose == "hint":
            self.play_board.board.show_suggestion(move[:2], move[2:4])
        elif purpose == "move" and self.play_started and self._play_player_to_move() != "Human":
            self._commit_play_move(move, animate=True)
        self.footer.set_context("DeepTactics", "Game active", "--")

    def play_error_received(self, message: str) -> None:
        self._play_search_purpose = None
        QMessageBox.warning(self, "Play Engine", message)

    def _match_control_config(self) -> TimeControlConfig:
        controls = {
            "Unlimited": TimeControlConfig("Unlimited", None),
            "Bullet 2+1": TimeControlConfig("Bullet 2+1", 120_000, 1_000),
            "Blitz 5+3": TimeControlConfig("Blitz 5+3", 300_000, 3_000),
            "Rapid 10+5": TimeControlConfig("Rapid 10+5", 600_000, 5_000),
            "Standard 15+10": TimeControlConfig("Standard 15+10", 900_000, 10_000),
            "Classical 30+20": TimeControlConfig("Classical 30+20", 1_800_000, 20_000),
        }
        return controls[self.match_right.controls.time_control.currentText()]

    def start_ai_match(self) -> None:
        chosen = (self.match_right.white.engine_select.currentText(), self.match_right.black.engine_select.currentText())
        self.match_white_client.configure_engine(ENGINE_REGISTRY.entry(chosen[0]))
        self.match_black_client.configure_engine(ENGINE_REGISTRY.entry(chosen[1]))
        settings = SETTINGS_STORE.load()
        settings["ai_match_search_mode"] = self.match_right.controls.search_mode.currentText()
        settings["ai_match_depth"] = self.match_right.controls.depth.value()
        settings["ai_match_movetime_ms"] = self.match_right.controls.movetime_ms.value()
        SETTINGS_STORE.save(settings)
        self.match_running = True
        self.match_paused = False
        self.match_total_games = self.match_right.controls.games.value()
        self.match_game_number = 1
        self.match_results = [0, 0, 0]
        self.match_pgns = []
        self.match_time_control = self._match_control_config()
        self.match_right.controls.set_running(True)
        self.match_right.set_status("Ready")
        self.match_center.scoreboard.set_engines(*chosen)
        self.match_center.scoreboard.set_score(0, 0, 0, 1, self.match_total_games)
        self._start_match_game()

    def _start_match_game(self) -> None:
        if self.match_right.controls.swap_colors.isChecked() and self.match_game_number > 1:
            white = self.match_right.white.engine_select.currentText()
            black = self.match_right.black.engine_select.currentText()
            self.match_right.white.engine_select.setCurrentText(black)
            self.match_right.black.engine_select.setCurrentText(white)
        line: tuple[str, ...] = ()
        if (
            self.match_right.controls.use_opening_suite.isChecked()
            and self.match_right.controls.opening_suite.currentText() == "Balanced (Default)"
            and self.match_opening_lines
        ):
            if self.match_right.controls.randomize_opening.isChecked():
                line = random.choice(self.match_opening_lines)
            else:
                line = self.match_opening_lines[((self.match_game_number - 1) // 2) % len(self.match_opening_lines)]
        if line:
            self.match_document.load_uci_line(line)
        else:
            self.match_document.new_game()
        self.match_live_ply = len(line)
        self.match_clock_ms = {
            Color.WHITE: self.match_time_control.base_ms,
            Color.BLACK: self.match_time_control.base_ms,
        }
        self.match_turn_started_at = time.monotonic()
        self.match_board.graph.clear_scores()
        self.match_board.eval_meter.set_active(True)
        self.match_notice.setText(f"MATCH RUNNING - game {self.match_game_number} / {self.match_total_games}")
        self.match_center.scoreboard.set_score(*self.match_results, self.match_game_number, self.match_total_games)
        self._request_match_move()

    def apply_match_snapshot(self, snapshot: GameSnapshot) -> None:
        is_live = snapshot.current_ply == len(self.match_document.mainline_uci_moves)
        if is_live:
            self.match_live_ply = snapshot.current_ply
        self.match_board.board.set_fen(snapshot.fen, self.match_document.last_move_uci)
        self.match_center.move_list.set_moves(snapshot.move_list, snapshot.current_ply)
        self.match_center.summary.update_snapshot(snapshot, live=is_live)

    def view_match_ply(self, ply: int) -> None:
        self.match_document.seek_mainline(ply)

    def _request_match_move(self) -> None:
        if not self.match_running or self.match_paused:
            return
        color = self.match_document.game.board.side_to_move
        self.match_search_color = color
        card = self.match_right.white if color is Color.WHITE else self.match_right.black
        card.set_running("Thinking")
        client = self.match_white_client if color is Color.WHITE else self.match_black_client
        search_mode = self.match_right.controls.search_mode.currentText()
        if search_mode == "Fixed Depth":
            client.analyze_fen(
                self.match_document.fen,
                depth=self.match_right.controls.depth.value(),
                moves=self.match_document.mainline_uci_moves,
                initial_fen=None if self.match_document.initial_fen == STANDARD_STARTING_FEN else self.match_document.initial_fen,
            )
        elif search_mode == "Move Time":
            client.analyze_fen(
                self.match_document.fen,
                movetime_ms=self.match_right.controls.movetime_ms.value(),
                moves=self.match_document.mainline_uci_moves,
                initial_fen=None if self.match_document.initial_fen == STANDARD_STARTING_FEN else self.match_document.initial_fen,
            )
        elif self.match_time_control.base_ms is None:
            client.analyze_fen(
                self.match_document.fen,
                movetime_ms=self.match_right.controls.movetime_ms.value(),
                moves=self.match_document.mainline_uci_moves,
                initial_fen=None if self.match_document.initial_fen == STANDARD_STARTING_FEN else self.match_document.initial_fen,
            )
        else:
            client.play_fen_with_clock(
                self.match_document.fen,
                white_ms=int(self.match_clock_ms[Color.WHITE] or 1),
                black_ms=int(self.match_clock_ms[Color.BLACK] or 1),
                white_increment_ms=self.match_time_control.increment_ms,
                black_increment_ms=self.match_time_control.increment_ms,
                moves=self.match_document.mainline_uci_moves,
                initial_fen=None if self.match_document.initial_fen == STANDARD_STARTING_FEN else self.match_document.initial_fen,
            )

    def clear_match_hash(self, color: Color) -> None:
        client = self.match_white_client if color is Color.WHITE else self.match_black_client
        card = self.match_right.white if color is Color.WHITE else self.match_right.black
        if not client.trigger_option("Clear Hash"):
            QMessageBox.information(self, "Clear Hash", f'{card.engine_select.currentText()} does not report a UCI Clear Hash option.')
            return
        QMessageBox.information(self, "Clear Hash", f'Clear Hash was sent to {card.engine_select.currentText()}.')

    def match_info_received(self, color: Color, info: UCIAnalysisInfo) -> None:
        if not self.match_running or self.match_search_color is not color or info.multipv != 1:
            return
        card = self.match_right.white if color is Color.WHITE else self.match_right.black
        card.update_analysis(info, self.match_document.fen)
        self.match_board.eval_meter.set_score(info.score_cp_white, info.mate_white)
        self.match_center.summary.update_score(info.score_cp_white, info.mate_white)
        self.match_center.scoreboard.set_performance(info.depth, info.nps)
        self.footer.set_search_metrics(f"{info.nodes:,}", f"{info.nps:,}", f"AI Match | {color.value.title()} thinking")

    def match_bestmove_received(self, color: Color, move: str) -> None:
        if not self.match_running or self.match_search_color is not color or self.match_paused:
            return
        card = self.match_right.white if color is Color.WHITE else self.match_right.black
        card.set_running("Waiting")
        if move == "0000":
            self._finish_match_game()
            return
        if self.match_document.current_ply != self.match_live_ply:
            self.match_document.seek_mainline(self.match_live_ply)
        elapsed = 0 if self.match_turn_started_at is None else int((time.monotonic() - self.match_turn_started_at) * 1000)
        if self.match_clock_ms[color] is not None:
            remaining = max(0, int(self.match_clock_ms[color] or 0) - elapsed)
            self.match_clock_ms[color] = remaining + self.match_time_control.increment_ms
            if remaining <= 0:
                self._finish_match_game(timeout_color=color)
                return
        try:
            self.match_document.append_mainline_move(move)
        except ValueError:
            self.match_error_received(color, f"Engine returned illegal move: {move}")
            return
        captured = self.match_document.game.captured_history[-1] is not None
        self.sound_manager.play("capture" if captured else "move")
        if self.match_document.game.is_in_check():
            self.sound_manager.play("check")
        latest = card.stat_values["Score"].text()
        info = (self.match_white_client if color is Color.WHITE else self.match_black_client).latest_info
        self.match_board.graph.set_score(self.match_live_ply, info.score_cp_white, info.mate_white)
        self.match_turn_started_at = time.monotonic()
        status = self.match_document.game.status()
        if status != ChessGameStatus.ONGOING:
            self._finish_match_game()
            return
        self.footer.set_context("UCI Match", f"Running | last score {latest}", "--")
        QTimer.singleShot(0, self._request_match_move)

    def _finish_match_game(self, timeout_color: Color | None = None) -> None:
        self.sound_manager.play("game_end")
        if timeout_color is Color.WHITE:
            result_index, result = 2, "Black wins on time"
        elif timeout_color is Color.BLACK:
            result_index, result = 0, "White wins on time"
        else:
            status = self.match_document.game.status()
            if status == ChessGameStatus.CHECKMATE:
                winner = self.match_document.game.board.side_to_move.opponent
                result_index = 0 if winner is Color.WHITE else 2
                result = f"{winner.value.title()} wins by checkmate"
            else:
                result_index, result = 1, "Draw"
        self.match_results[result_index] += 1
        white = self.match_right.white.engine_select.currentText()
        black = self.match_right.black.engine_select.currentText()
        self.match_pgns.append(
            pgn_with_metadata(
                self.match_document.snapshot().pgn,
                Event="DeepTactics AI Match",
                Date=datetime.now().strftime("%Y.%m.%d"),
                White=white,
                Black=black,
                Type="AI Match",
            )
        )
        self.match_center.scoreboard.set_score(*self.match_results, self.match_game_number, self.match_total_games)
        if self.match_game_number < self.match_total_games and self.match_right.controls.auto_next.isChecked():
            self.match_game_number += 1
            QTimer.singleShot(0, self._start_match_game)
            return
        self.match_running = False
        self.match_right.controls.set_running(False)
        self.match_right.set_status("Finished")
        self.match_notice.setText(f"MATCH FINISHED - {result}")
        self.footer.set_context("UCI Match", result, "--")

    def toggle_ai_match_pause(self, paused: bool) -> None:
        if not self.match_running:
            return
        self.match_paused = paused
        self.match_right.controls.set_running(True, paused)
        if paused:
            active_client = self.match_white_client if self.match_search_color is Color.WHITE else self.match_black_client
            active_client.stop()
            self.match_notice.setText("MATCH PAUSED")
            self.footer.set_context("UCI Match", "Paused", "--")
        else:
            self.match_notice.setText(f"MATCH RUNNING - game {self.match_game_number} / {self.match_total_games}")
            self._request_match_move()

    def stop_ai_match(self) -> None:
        self.match_running = False
        self.match_paused = False
        self.match_white_client.cancel_analysis()
        self.match_black_client.cancel_analysis()
        self.match_right.controls.set_running(False)
        self.match_right.set_status("Stopped")
        self.match_notice.setText("MATCH STOPPED")
        self.footer.set_context("UCI Match", "Stopped", "--")

    def reset_ai_match(self) -> None:
        self.stop_ai_match()
        self.match_results = [0, 0, 0]
        self.match_game_number = 0
        self.match_total_games = 0
        self.match_pgns = []
        self.match_document.new_game()
        self.match_board.graph.clear_scores()
        self.match_board.eval_meter.set_active(False)
        self.match_center.summary.reset()
        self.match_center.scoreboard.set_score(0, 0, 0, 0, 0)
        self.match_right.reset()
        self.match_notice.setText("MATCH READY - select two registered engines and start a match")

    def save_ai_match_pgn(self) -> None:
        white = self.match_right.white.engine_select.currentText()
        black = self.match_right.black.engine_select.currentText()
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Match PGN",
            default_pgn_filename("DeepTactics_AI_Match", white, black),
            "PGN Files (*.pgn)",
        )
        if path:
            text = "\n\n".join(
                self.match_pgns
                or [
                    pgn_with_metadata(
                        self.match_document.snapshot().pgn,
                        Event="DeepTactics AI Match",
                        Date=datetime.now().strftime("%Y.%m.%d"),
                        White=white,
                        Black=black,
                        Type="AI Match",
                    )
                ]
            )
            Path(path).write_text(text, encoding="utf-8")
            self._register_saved_pgn(text, path, "AI Match")

    def export_ai_match(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export Match Report", "ai_match_report.txt", "Text Files (*.txt)")
        if path:
            summary = (
                f"White wins: {self.match_results[0]}\nDraws: {self.match_results[1]}\n"
                f"Black wins: {self.match_results[2]}\nGames: {self.match_game_number}/{self.match_total_games}\n"
            )
            Path(path).write_text(summary, encoding="utf-8")

    def match_error_received(self, color: Color, message: str) -> None:
        self.match_running = False
        self.match_right.controls.set_running(False)
        self.match_right.set_status("Error")
        self.match_notice.setText(f"MATCH ERROR - {color.value.title()} engine failed")
        QMessageBox.warning(self, "AI Match Engine", message)

    def start_analysis(self) -> None:
        if self.analysis_active:
            return
        self.analysis_active = True
        self.analysis_console.engine.set_state("Analyzing", session_active=True)
        self._start_baseline_analysis()

    def apply_saved_settings(self, settings: dict) -> None:
        self._runtime_settings = dict(settings)
        self.sound_manager.configure(settings)
        self.apply_theme(str(settings.get("theme", "Dark")), int(settings.get("interface_scale", 100)))
        piece_folder = SettingsPage.resolve_setting_path(str(settings.get("piece_set_path", "assets/pieces")))
        board_theme = str(settings.get("board_theme", "Wood"))
        for board in self.findChildren(WoodenBoard):
            board.set_visual_theme(board_theme)
            board.set_coordinates_visible(bool(settings.get("show_coordinates", True)))
            board.set_piece_folder(piece_folder)
        depth = int(settings.get("default_analysis_depth", self._analysis_depth()))
        if self.analysis_console.engine.depth_selector.value() != depth:
            blocked = self.analysis_console.engine.depth_selector.blockSignals(True)
            self.analysis_console.engine.depth_selector.setValue(depth)
            self.analysis_console.engine.depth_selector.blockSignals(blocked)
            self._clear_analysis_cache()
        engine_name = str(settings.get("default_analysis_engine", self.analysis_console.engine.engine_name.text()))
        if ENGINE_REGISTRY.entry(engine_name) is not None and engine_name != self.analysis_console.engine.engine_name.text():
            if self.uci_client.is_searching:
                self._discard_replaced_result = True
                self.uci_client.cancel_analysis()
            self.analysis_console.engine.set_engine(engine_name)
        self.footer.set_context("DeepTactics", "Settings applied for new sessions", "--")

    def show_uci_log(self) -> None:
        self.uci_log_dialog.show()
        self.uci_log_dialog.raise_()
        self.uci_log_dialog.activateWindow()

    def _record_uci_log(self, session: str, direction: str, line: str) -> None:
        self.uci_log_dialog.append_line(session, direction, line)
        if not bool(self._runtime_settings.get("save_uci_logs", False)):
            return
        logs = Path(str(self._runtime_settings.get("logs", "artifacts/logs")))
        if not logs.is_absolute():
            logs = ROOT / logs
        try:
            logs.mkdir(parents=True, exist_ok=True)
            with (logs / "qt_uci.log").open("a", encoding="utf-8") as handle:
                stamp = datetime.now().isoformat(timespec="seconds")
                handle.write(f"{stamp} [{session}] {direction} {line}\n")
        except OSError:
            pass

    def clear_analysis_cache_from_settings(self) -> None:
        self._clear_analysis_cache()
        self.analysis_console.graph.clear_scores()
        self.analysis_console.ledger.set_classifications({})
        self.analysis_board.board.set_move_classification(None)
        self.footer.set_context("DeepTactics", "Analysis cache cleared", "--")

    def _analysis_depth(self) -> int:
        return self.analysis_console.engine.depth_selector.value()

    def _analysis_cache_key(self, fen: str) -> tuple[object, ...]:
        settings = SETTINGS_STORE.load()
        name = self.analysis_console.engine.engine_name.text()
        profile = ENGINE_REGISTRY.entry(name) or {}
        side_to_move = fen.split()[1] if len(fen.split()) > 1 else "?"
        return (
            ANALYSIS_CLASSIFICATION_POLICY_VERSION,
            name,
            str(profile.get("path", "")),
            str(profile.get("detected_name", "")),
            int(profile.get("threads", 1)),
            int(profile.get("hash", 256)),
            self._analysis_depth(),
            str(settings.get("analysis_mode", "Fixed Depth")),
            int(settings.get("default_move_time", 5)),
            int(settings.get("analysis_multipv", 3)),
            str(settings.get("score_perspective", "White perspective")),
            fen,
            side_to_move,
        )

    def _clear_analysis_cache(self) -> None:
        self._analysis_cache.clear()
        self._baseline_results.clear()
        self._baseline_bestmoves.clear()
        self._analysis_classifications.clear()
        self._analysis_classification_evidence.clear()

    def change_analysis_depth(self, depth: int) -> None:
        settings = SETTINGS_STORE.load()
        settings["default_analysis_depth"] = depth
        SETTINGS_STORE.save(settings)
        self._clear_analysis_cache()
        self.analysis_console.graph.clear_scores()
        self.analysis_console.ledger.set_classifications({})
        self.analysis_board.board.set_move_classification(None)
        self.footer.set_context(self.analysis_console.engine.engine_name.text(), f"Depth changed to {depth}; review refresh required", "--")
        if self.analysis_active:
            if self.uci_client.is_searching:
                self._discard_replaced_result = True
                self.uci_client.cancel_analysis()
            self._start_baseline_analysis()

    def analysis_engine_changed(self, name: str) -> None:
        entry = ENGINE_REGISTRY.entry(name) or {}
        self.uci_client.configure_engine(entry)
        self._clear_analysis_cache()
        self.analysis_console.graph.clear_scores()
        self.analysis_console.ledger.set_classifications({})
        self.analysis_board.board.set_move_classification(None)
        self.footer.set_context(name, "Engine changed; analysis cache cleared", "--")

    def _replace_analysis_source(self, loader) -> None:
        self._changing_analysis_source = True
        try:
            loader()
        finally:
            self._changing_analysis_source = False
        self.analysis_console.graph.clear_scores()
        self.analysis_console.ledger.set_classifications({})
        self._clear_analysis_cache()
        self.analysis_board.board.set_move_classification(None)
        self._latest_live_bestmove = None
        self._refresh_analysis_suggestion()
        if self.analysis_active:
            self._start_baseline_analysis()

    def _start_baseline_analysis(self) -> None:
        settings = SETTINGS_STORE.load()
        single_position = (
            not self.analysis_document.mainline_uci_moves
            and self.analysis_document.initial_fen != STANDARD_STARTING_FEN
        )
        if (
            single_position
            or not bool(settings.get("review_full_game", True))
            or settings.get("analysis_mode") == "Infinite until Stop"
        ):
            self.analysis_phase = "live"
            self._clear_analysis_cache()
            self.analysis_console.graph.clear_scores()
            self.analysis_console.ledger.set_classifications({})
            self.uci_client.set_option("MultiPV", int(settings.get("analysis_multipv", 3)))
            self._request_current_analysis()
            return
        self.analysis_phase = "baseline"
        self._analysis_review_started_at = time.perf_counter()
        self._baseline_positions = self.analysis_document.mainline_positions()
        self._baseline_ply = 0
        self._clear_analysis_cache()
        self.analysis_console.graph.clear_scores()
        self.analysis_console.ledger.set_classifications({})
        self.analysis_board.board.set_move_classification(None)
        self.analysis_console.summary.reset_candidates()
        self.uci_client.set_option("MultiPV", 1)
        print(
            "Qt Analyze review start "
            f"engine={self.analysis_console.engine.engine_name.text()} depth={self._analysis_depth()} "
            f"positions={len(self._baseline_positions)} backend=UCI/native multipv=1"
        )
        self._request_baseline_position()

    def _request_baseline_position(self) -> None:
        if not self.analysis_active or self.analysis_phase != "baseline":
            return
        total = max(0, len(self._baseline_positions) - 1)
        state = f"Reviewing game {self._baseline_ply}/{total}"
        self.analysis_console.engine.set_state(state, session_active=True)
        self.footer.set_context(self.analysis_console.engine.engine_name.text(), state, "--")
        self._submit_analysis(
            self._baseline_positions[self._baseline_ply],
            self._analysis_depth(),
            moves=self.analysis_document.mainline_uci_moves[: self._baseline_ply],
        )

    def _finish_baseline_analysis(self) -> None:
        self._apply_move_classifications()
        self.analysis_phase = "live"
        settings = SETTINGS_STORE.load()
        self.uci_client.set_option("MultiPV", int(settings.get("analysis_multipv", 3)))
        total_ms = (time.perf_counter() - self._analysis_review_started_at) * 1000
        print(f"Qt Analyze review complete positions={len(self._baseline_positions)} total_ms={total_ms:.1f}")
        self.sound_manager.play("engine_ready")
        if bool(settings.get("live_analysis_after_review", True)):
            self._request_current_analysis()
        else:
            self.analysis_active = False
            self.analysis_console.engine.set_state("Ready", session_active=False)
            self.footer.set_context(self.analysis_console.engine.engine_name.text(), "Review complete", "--")

    def _apply_move_classifications(self) -> None:
        if not bool(SETTINGS_STORE.load().get("show_move_classification", True)):
            self._analysis_classifications = {}
            self._analysis_classification_evidence = {}
            self.analysis_console.ledger.set_classifications({})
            self.analysis_board.board.set_move_classification(None)
            return
        classifications: dict[int, MoveClassification] = {}
        evidence_by_ply: dict[int, AnalyzeMoveEvidence] = {}
        moves = self.analysis_document.mainline_uci_moves
        san_moves = self.analysis_document.snapshot().move_list
        positions = self._baseline_positions
        for ply, move_text in enumerate(moves, start=1):
            before = self._baseline_results.get(ply - 1)
            after = self._baseline_results.get(ply)
            if before is None or after is None:
                continue
            board = board_from_fen(positions[ply - 1])
            best_text = self._baseline_bestmoves.get(ply - 1)
            best_move = Move.from_uci(best_text) if best_text and best_text != "0000" else None
            played_move = Move.from_uci(move_text)
            opening = detect_opening(
                list(san_moves[:ply]),
                OPENING_BOOK_PATH if OPENING_BOOK_PATH.exists() else None,
            )
            classification, evidence = _qt_move_classification(
                player=board.side_to_move,
                before=before,
                after=after,
                previous=self._baseline_results.get(ply - 2),
                played_move=played_move,
                best_move=best_move,
                is_forced=len(generate_legal_moves(board)) == 1,
                is_book=ply <= 12 and opening != "Opening: unknown",
            )
            if classification is not None:
                classifications[ply] = classification
            if evidence is not None:
                evidence_by_ply[ply] = evidence
        self._analysis_classifications = classifications
        self._analysis_classification_evidence = evidence_by_ply
        self.analysis_console.ledger.set_classifications(
            classifications,
            {ply: evidence.tooltip() for ply, evidence in evidence_by_ply.items()},
        )
        self.analysis_board.board.set_move_classification(classifications.get(self.analysis_document.snapshot().current_ply))

    def _request_current_analysis(self) -> None:
        self._latest_live_bestmove = None
        self._refresh_analysis_suggestion()
        self.analysis_console.begin_position_analysis()
        moves = None
        if not self.analysis_document.viewing_variation:
            moves = self.analysis_document.mainline_uci_moves[: self.analysis_document.snapshot().current_ply]
        self._submit_analysis(self.analysis_document.fen, self._analysis_depth(), moves=moves)

    def _submit_analysis(self, fen: str, depth: int, *, moves: tuple[str, ...] | None = None) -> None:
        if self.uci_client.is_searching:
            self._discard_replaced_result = True
        self._analysis_requested_fen = fen
        self._analysis_request_started_at = time.perf_counter()
        settings = SETTINGS_STORE.load()
        mode = str(settings.get("analysis_mode", "Fixed Depth"))
        kwargs: dict[str, object] = {"depth": depth}
        if mode == "Move Time":
            kwargs = {"movetime_ms": max(1, int(settings.get("default_move_time", 5))) * 1000}
        elif mode == "Infinite until Stop" and self.analysis_phase != "baseline":
            kwargs = {"infinite": True}
        self.uci_client.analyze_fen(
            fen,
            **kwargs,
            moves=moves,
            initial_fen=None if self.analysis_document.initial_fen == STANDARD_STARTING_FEN else self.analysis_document.initial_fen,
        )

    def stop_analysis(self) -> None:
        self.analysis_active = False
        self.analysis_phase = "idle"
        self._discard_replaced_result = False
        self.uci_client.cancel_analysis()
        self._latest_live_bestmove = None
        self._refresh_analysis_suggestion()
        self.analysis_console.engine.set_state("Ready", session_active=False)
        self.footer.set_context(self.analysis_console.engine.engine_name.text(), "Stopped", "--")
        self._restore_cached_position(self.analysis_document.fen)

    def _restore_cached_position(self, fen: str, *, cancel_running: bool = False) -> bool:
        if not bool(SETTINGS_STORE.load().get("use_analysis_cache", True)):
            return False
        cached = self._analysis_cache.get(self._analysis_cache_key(fen))
        if not cached and self._baseline_positions:
            try:
                ply = self._baseline_positions.index(fen)
            except ValueError:
                ply = -1
            baseline = self._baseline_results.get(ply)
            if baseline is not None:
                cached = {1: baseline}
        if not cached:
            return False
        if cancel_running and self.uci_client.is_searching:
            self._discard_replaced_result = True
            self.uci_client.cancel_analysis()
        self.analysis_console.summary.reset_candidates()
        for _multipv, info in sorted(cached.items()):
            displayed = replace(info, pv=_pv_to_san(fen, info.pv))
            self.analysis_console.update_analysis(displayed)
        primary = cached.get(1)
        if primary is not None:
            self._latest_live_bestmove = primary.pv[0] if primary.pv else None
            self.analysis_board.eval_meter.set_score(primary.score_cp_white, primary.mate_white)
            self._refresh_analysis_suggestion()
        self.footer.set_context(self.analysis_console.engine.engine_name.text(), "Cached analysis", "--")
        return True

    def analysis_state_changed(self, state: str) -> None:
        shown_state = "Analysis active" if self.analysis_active and state == "Ready" else state
        self.analysis_console.engine.set_state(shown_state, session_active=self.analysis_active)
        self.footer.set_context(self.analysis_console.engine.engine_name.text(), shown_state, "--")

    def analysis_identity_received(self, identity: dict) -> None:
        name = self.analysis_console.engine.engine_name.text()
        ENGINE_REGISTRY.update_profile(
            name,
            status="Ready",
            detected_name=str(identity.get("name") or name),
            nnue=False if name == "DeepTactics" else bool(identity.get("supports_nnue", False)),
        )
        self.analysis_console.engine.set_engine(name, notify=False)

    def find_best_analysis_move(self) -> None:
        move = None
        cached = self._analysis_cache.get(self._analysis_cache_key(self.analysis_document.fen), {})
        primary = cached.get(1)
        if primary and primary.pv:
            move = primary.pv[0]
        elif self._latest_live_bestmove:
            move = self._latest_live_bestmove
        if self._show_analysis_bestmove_arrow(move):
            self.footer.set_context(self.analysis_console.engine.engine_name.text(), f"Best move: {move}", "--")
            return
        QMessageBox.information(self, "Find Best Move", "Start Analyze first to calculate a best move for this position.")

    def _show_analysis_bestmove_arrow(self, move: str | None) -> bool:
        if not move or move == "0000" or len(move) < 4:
            self.analysis_board.board.suggested_arrow = None
            self.analysis_board.board.update()
            return False
        try:
            parsed = Move.from_uci(move)
        except ValueError:
            self.analysis_board.board.suggested_arrow = None
            self.analysis_board.board.update()
            return False
        if parsed not in self.analysis_document.game.legal_moves():
            self.analysis_board.board.suggested_arrow = None
            self.analysis_board.board.update()
            return False
        self.analysis_board.board.show_suggestion(move[:2], move[2:4])
        return True

    def analysis_info_received(self, info: UCIAnalysisInfo) -> None:
        if self._discard_replaced_result:
            return
        if self.analysis_phase == "baseline":
            if info.multipv == 1:
                info = _terminal_display_info(self._baseline_positions[self._baseline_ply], info)
                self._baseline_results[self._baseline_ply] = info
                fen = self._baseline_positions[self._baseline_ply]
                self._analysis_cache.setdefault(self._analysis_cache_key(fen), {})[1] = info
                self.footer.set_search_metrics(
                    f"{info.nodes:,}" if info.nodes else "--",
                    f"{info.nps:,}" if info.nps else "--",
                    f"Review {self._baseline_ply}/{max(0, len(self._baseline_positions) - 1)}",
                )
            return
        fen = self._analysis_requested_fen or self.analysis_document.fen
        info = _terminal_display_info(fen, info)
        self._analysis_cache.setdefault(self._analysis_cache_key(fen), {})[info.multipv] = info
        displayed = replace(info, pv=_pv_to_san(fen, info.pv))
        self.analysis_console.update_analysis(displayed)
        if info.multipv == 1:
            self._latest_live_bestmove = info.pv[0] if info.pv else None
            self._refresh_analysis_suggestion()
            self.analysis_board.eval_meter.set_score(info.score_cp_white, info.mate_white)
            self.footer.set_search_metrics(
                f"{info.nodes:,}" if info.nodes else "--",
                f"{info.nps:,}" if info.nps else "--",
                "Analyzing",
            )

    def analysis_bestmove_received(self, move: str) -> None:
        if self._discard_replaced_result:
            self._discard_replaced_result = False
            return
        if self.analysis_phase == "baseline":
            self._baseline_bestmoves[self._baseline_ply] = move
            info = self._baseline_results.get(self._baseline_ply)
            if info is not None:
                self.analysis_console.graph.set_score(self._baseline_ply, info.score_cp_white, info.mate_white)
            self._baseline_ply += 1
            if self._baseline_ply < len(self._baseline_positions):
                self._request_baseline_position()
            else:
                self._finish_baseline_analysis()
            return
        elapsed_ms = (time.perf_counter() - self._analysis_request_started_at) * 1000
        print(f"Qt Analyze live result depth={self._analysis_depth()} search_ms={elapsed_ms:.1f} bestmove={move}")
        state = "Analysis active" if self.analysis_active else "Ready"
        self.analysis_console.engine.set_state(state, session_active=self.analysis_active)
        self.footer.set_context(self.analysis_console.engine.engine_name.text(), f"{state} | bestmove {move}", "--")

    def analysis_error_received(self, message: str) -> None:
        self.analysis_active = False
        self.analysis_phase = "idle"
        self._discard_replaced_result = False
        self._latest_live_bestmove = None
        self._refresh_analysis_suggestion()
        self.analysis_console.engine.set_state("Error")
        self.footer.set_context(self.analysis_console.engine.engine_name.text(), "Error", "--")
        QMessageBox.warning(self, "UCI Engine", message)

    def toggle_analysis_suggestion(self, enabled: bool) -> None:
        self._analysis_suggestion_enabled = enabled
        self._refresh_analysis_suggestion()

    def _refresh_analysis_suggestion(self) -> None:
        move = self._latest_live_bestmove
        if self._analysis_suggestion_enabled and self._show_analysis_bestmove_arrow(move):
            return
        self.analysis_board.board.suggested_arrow = None
        self.analysis_board.board.update()

    def closeEvent(self, event) -> None:
        if hasattr(self, "uci_client"):
            self.uci_client.quit()
        if hasattr(self, "play_uci_client"):
            self.play_uci_client.quit()
        if hasattr(self, "match_white_client"):
            self.match_white_client.quit()
        if hasattr(self, "match_black_client"):
            self.match_black_client.quit()
        super().closeEvent(event)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if os.name != "nt":
            return
        try:
            preference = ctypes.c_int(2)  # DWMWCP_ROUND
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                int(self.winId()), 33, ctypes.byref(preference), ctypes.sizeof(preference)
            )
        except (AttributeError, OSError):
            pass

    def apply_theme(self, theme: str, interface_scale: int | None = None) -> None:
        scale = int(interface_scale if interface_scale is not None else SETTINGS_STORE.load().get("interface_scale", 100))
        css = STYLE + LIGHT_OVERRIDE if theme == "Light" or (theme == "System" and QApplication.palette().window().color().lightness() > 128) else STYLE
        if scale != 100:
            ratio = scale / 100.0
            css = re.sub(
                r"font-size:\s*(\d+)px",
                lambda match: f"font-size: {max(9, round(int(match.group(1)) * ratio))}px",
                css,
            )
        self.setStyleSheet(css)
        if hasattr(self, "footer"):
            self.footer.apply_theme("Light" if theme == "Light" or (theme == "System" and QApplication.palette().window().color().lightness() > 128) else "Dark")


STYLE = f"""
* {{
    font-family: "Segoe UI";
    font-size: 13px;
    color: {TEXT};
}}
QMainWindow, QWidget {{
    background: {APP_BG};
}}
QLabel, #transparent {{
    background: transparent;
}}
#header {{
    background: {HEADER};
    border-bottom: 1px solid {BORDER};
}}
#logoMark {{
    color: #20a7ff;
    font-size: 37px;
    font-weight: 700;
}}
#brand {{
    font-size: 29px;
    font-weight: 650;
}}
#version {{
    color: #3191ff;
    font-size: 12px;
    font-weight: 600;
    margin-top: 17px;
}}
QPushButton[nav="true"] {{
    color: #b5c0d1;
    border: none;
    background: transparent;
    border-radius: 21px;
    padding: 15px 20px;
    font-size: 15px;
}}
QPushButton[nav="true"][selected="true"] {{
    background: #0c1827;
    color: {BLUE_LINE};
    border-bottom: 2px solid {BLUE_LINE};
}}
#windowControl {{
    background: transparent;
    border: none;
    color: #d9e0e9;
    font-size: 20px;
    padding: 10px;
}}
#card, #boardCard, #toolbarCard {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 9px;
}}
#boardCard {{
    border-color: {BORDER_BLUE};
}}
#caps {{
    color: {SECONDARY};
    font-size: 11px;
    font-weight: 600;
}}
#secondary {{
    color: {SECONDARY};
}}
#muted {{
    color: {MUTED};
    font-size: 11px;
}}
#greenDot {{
    background: {GREEN};
    border-radius: 4px;
}}
#greenDot[error="true"] {{
    background: #f25465;
}}
#engineName {{
    font-size: 17px;
    font-weight: 600;
    padding: 0px 2px 0px 0px;
}}
#engineVersion {{
    color: #8997ab;
    font-size: 12px;
    font-weight: 500;
    padding: 1px 6px 0px 0px;
}}
#nnueBadge {{
    background: #17212e;
    border: 1px solid #29384a;
    color: #7b8796;
    padding: 3px 8px;
    border-radius: 7px;
    font-size: 11px;
    font-weight: 600;
}}
#nnueBadge[active="true"] {{
    background: #0f3a72;
    color: {BLUE_LINE};
    border-color: #1954a3;
}}
#statValue {{
    color: #edf2fb;
    font-size: 13px;
}}
#fenValue {{
    color: #d2dbe8;
    font-size: 11px;
}}
#progress {{
    background: {BLUE_LINE};
    border-radius: 1px;
}}
QPushButton {{
    background: #111d2a;
    border: 1px solid #253649;
    border-radius: 7px;
    padding: 12px 17px;
    color: #eef3fb;
}}
QPushButton:hover {{
    background: #18283b;
}}
QPushButton:disabled {{
    background: #0c141e;
    border-color: #182432;
    color: #556477;
}}
QPushButton[primary="true"] {{
    background: #155ed1;
    border-color: #1d6df2;
}}
QPushButton:checked {{
    background: #155ed1;
    border-color: #1d6df2;
    color: #ffffff;
}}
QPushButton[compact="true"] {{
    padding: 8px 14px;
}}
QToolTip {{
    background: #172331;
    color: #e6edf6;
    border: 1px solid #314458;
    border-radius: 4px;
    padding: 5px 8px;
}}
#interactiveField {{
    background: #101b27;
    color: #eef3fb;
    border: 1px solid #253649;
    border-radius: 6px;
    padding: 5px 9px;
    min-height: 27px;
}}
QComboBox#interactiveField::drop-down {{
    border: none;
    width: 20px;
}}
QSpinBox#interactiveField {{
    font-size: 12px;
    padding: 3px 5px;
}}
QCheckBox {{
    background-color: transparent;
    border: none;
    color: #d6dfeb;
    spacing: 7px;
    font-size: 12px;
}}
QCheckBox::indicator {{
    width: 13px;
    height: 13px;
    border: 1px solid #526275;
    border-radius: 3px;
    background: #0d1722;
}}
QCheckBox::indicator:checked {{
    background: {BLUE};
    border-color: {BLUE_LINE};
}}
#libraryButton {{
    background: transparent;
    color: #d9e2ef;
    border: none;
    border-radius: 6px;
    padding: 10px 12px;
    text-align: left;
}}
#settingsNav {{
    background: transparent;
    color: #d9e2ef;
    border: none;
    border-radius: 7px;
    padding: 13px 13px;
    text-align: left;
    font-size: 13px;
}}
#settingsNav:checked {{
    background: #103264;
    color: #eaf2ff;
    border-left: 2px solid {BLUE_LINE};
}}
#settingsTitle {{
    color: #eef3fb;
    font-size: 25px;
    font-weight: 600;
}}
#miniReset {{
    padding: 4px 9px;
    color: {SECONDARY};
    font-size: 11px;
}}
#libraryButton:checked {{
    background: #103264;
    color: #eaf2ff;
}}
#collectionItem {{
    color: #c4cddb;
    padding: 5px 6px;
}}
#searchField, #notesEditor {{
    background: #101b27;
    color: #eef3fb;
    border: 1px solid #253649;
    border-radius: 6px;
    padding: 8px 12px;
}}
#searchField {{
    min-height: 29px;
}}
#notesEditor {{
    font-size: 12px;
}}
#gamesTable {{
    background: {CARD};
    alternate-background-color: #0d1620;
    border: none;
    gridline-color: #1b2a3a;
    selection-background-color: #103264;
    selection-color: #eef5ff;
}}
#gamesTable QHeaderView::section {{
    background: #101b27;
    color: #b4bfce;
    border: none;
    border-bottom: 1px solid #263747;
    padding: 10px 6px;
}}
#gamesTable::item {{
    padding: 7px 5px;
    border-bottom: 1px solid #162534;
}}
#detailTitle {{
    color: #eef3fb;
    font-size: 16px;
    font-weight: 600;
}}
#detailMeta {{
    color: #c7d0de;
    line-height: 1.3;
}}
#openingItem, #recentOpening, #relatedOpening {{
    background: transparent;
    border: none;
    color: #d8e1ee;
    text-align: left;
    padding: 7px 9px;
}}
#openingList, #openingLineScroll, #continuationScroll {{
    background: transparent;
    border: none;
}}
#openingList::item {{
    color: #d8e1ee;
    padding: 7px 8px;
    border-radius: 5px;
}}
#openingList::item:selected {{
    background: #103264;
    color: #eef5ff;
}}
#openingLineScroll QScrollBar:horizontal {{
    background: #0b141d;
    height: 6px;
}}
#openingLineScroll QScrollBar::handle:horizontal {{
    background: #516273;
    border-radius: 3px;
    min-width: 30px;
}}
#continuationScroll QScrollBar:vertical {{
    background: #0b141d;
    width: 6px;
}}
#continuationScroll QScrollBar::handle:vertical {{
    background: #516273;
    border-radius: 3px;
    min-height: 24px;
}}
#openingFilter {{
    padding: 8px 8px;
    font-size: 11px;
}}
#openingItem:hover, #recentOpening:hover, #relatedOpening:hover {{
    background: #101d2b;
}}
#openingItem:focus {{
    background: #103264;
    border: 1px solid {BLUE_LINE};
}}
#linkButton {{
    background: transparent;
    border: none;
    color: {BLUE_LINE};
    padding: 3px 7px;
}}
#openingTitle {{
    color: #edf2fb;
    font-size: 22px;
    font-weight: 600;
    padding-bottom: 3px;
}}
#ecoBadge {{
    background: #0f3a72;
    border: 1px solid #1954a3;
    border-radius: 6px;
    color: {BLUE_LINE};
    font-size: 12px;
    padding: 4px 8px;
}}
#openingTags {{
    background: #111d2a;
    color: #d0d9e6;
    border-radius: 5px;
    font-size: 11px;
    padding: 6px 8px;
}}
#lineMove {{
    background: transparent;
    border: none;
    color: #cbd5e3;
    padding: 6px 6px;
}}
#lineMove[selected="true"] {{
    background: #155ed1;
    color: #ffffff;
}}
#evalBadge {{
    background: #e7e9ea;
    color: #121b26;
    border-radius: 4px;
    font-size: 13px;
    padding: 8px 13px;
}}
#continuation {{
    background: transparent;
    border: none;
    color: #d9e2ee;
    text-align: left;
    padding: 3px 4px;
}}
#continuation:hover {{
    color: {BLUE_LINE};
}}
QTabBar {{
    background: {CARD};
}}
QTabBar::tab {{
    color: {SECONDARY};
    padding: 13px 26px 11px 26px;
    border-bottom: 2px solid transparent;
}}
QTabBar::tab:selected {{
    color: {BLUE_LINE};
    border-bottom: 2px solid {BLUE_LINE};
}}
#evalScore {{
    font-size: 27px;
    font-weight: 600;
    min-height: 34px;
}}
#analysisText {{
    font-size: 13px;
}}
#detail {{
    color: #bec7d4;
}}
#pv {{
    font-size: 13px;
    color: #d7dfeb;
}}
#selectedMove {{
    background: #103264;
    color: {BLUE_LINE};
    padding: 3px 16px 3px 8px;
}}
#ledgerMove {{
    background: transparent;
    border: none;
    color: #edf2fb;
    padding: 3px 5px;
    text-align: left;
}}
#ledgerMove:hover {{
    background: #162b46;
}}
#ledgerMove[selected="true"] {{
    background: #103264;
    color: {BLUE_LINE};
}}
#ledgerVariation {{
    text-align: left;
    color: #ffcf5a;
    font-size: 12px;
    padding: 5px 8px;
    background: #15202b;
    border: 1px solid #2c3d4f;
    border-radius: 4px;
}}
#ledgerVariation:hover {{
    border-color: #4b637c;
}}
#movesScroll {{
    background: transparent;
    border: none;
}}
#movesScroll QWidget {{
    background: transparent;
}}
#movesScroll QScrollBar:vertical {{
    background: #0b141d;
    width: 8px;
    margin: 0px;
    border-radius: 4px;
}}
#movesScroll QScrollBar::handle:vertical {{
    background: #eef3fb;
    border-radius: 4px;
    min-height: 32px;
}}
#movesScroll QScrollBar::add-line:vertical, #movesScroll QScrollBar::sub-line:vertical {{
    height: 0px;
}}
#pvArrowToggle {{
    background: #101c28;
    border: 1px solid #293a4d;
    border-radius: 5px;
    color: #aab7c9;
    font-size: 10px;
    padding: 4px 8px;
}}
#pvArrowToggle:checked {{
    background: #103264;
    border-color: {BLUE_LINE};
    color: #dceaff;
}}
#moveBadge {{
    color: #ffffff;
    background: #4fae67;
    border-radius: 8px;
    font-size: 10px;
    font-weight: 700;
}}
#moveBadge[kind="best"] {{
    background: #1b79ed;
}}
#moveBadge[kind="good"] {{
    background: #519b70;
}}
#blueText {{
    color: {BLUE_LINE};
    font-size: 11px;
}}
#blueTextLarge {{
    color: {BLUE_LINE};
    font-size: 12px;
}}
#greenText {{
    color: {GREEN};
    font-size: 12px;
}}
#openingValue {{
    color: #edf2fb;
    font-size: 13px;
    padding: 0px 0px 2px 0px;
}}
#sectionHeader {{
    color: #cad3e0;
    font-size: 13px;
    font-weight: 600;
}}
#clockPlayer, #playField {{
    background: #101b27;
    border: 1px solid #203347;
    border-radius: 7px;
}}
#clockPlayer {{
    min-height: 68px;
}}
#playField {{
    min-height: 40px;
}}
#avatarHuman {{
    color: #d7e1ef;
    font-size: 29px;
}}
#avatarEngine {{
    color: {BLUE_LINE};
    font-size: 25px;
}}
#clockName {{
    color: #f2f5fb;
    font-size: 14px;
}}
#clockValue {{
    color: #f1f5fb;
    font-size: 21px;
}}
#fieldIcon {{
    color: #cbd7e6;
    font-size: 16px;
}}
#fieldValue {{
    color: #edf2fa;
    font-size: 14px;
}}
#movesHeader {{
    background: #101b27;
    border-top: 1px solid #1c3042;
    border-bottom: 1px solid #1c3042;
    min-height: 40px;
}}
#matchSelectedCell {{
    background: #112d50;
    color: #edf2fb;
    padding: 4px 5px;
}}
#matchSelectedCell[last="true"] {{
    color: {BLUE_LINE};
}}
#lossText {{
    color: #ff4b55;
    font-size: 12px;
}}
#scoreWin {{
    color: {BLUE_LINE};
    font-size: 32px;
    font-weight: 600;
}}
#scoreDim {{
    color: #aeb7c4;
    font-size: 32px;
    font-weight: 500;
}}
#matchProgress {{
    background: {BLUE_LINE};
    border-radius: 3px;
    max-width: 156px;
}}
#divider {{
    background: {BORDER};
    color: {BORDER};
    max-width: 1px;
    max-height: 1px;
}}
#footer {{
    background: #0a121c;
    border-top: 1px solid {BORDER};
}}
"""

LIGHT_OVERRIDE = """
QMainWindow, QWidget {
    background: #f4f7fb;
    color: #172235;
}
#header, #footer {
    background: #ffffff;
    border-color: #d7e0eb;
}
#card, #boardCard, #toolbarCard {
    background: #ffffff;
    border-color: #d8e2ee;
}
QLabel#statValue, QLabel#detailTitle, QLabel#sectionHeader {
    color: #172235;
}
#settingsTitle {
    color: #172235;
}
QLabel#secondary, QLabel#detail, QLabel#detailMeta, QLabel#pv {
    color: #36475d;
}
QLabel#muted, QLabel#caps {
    color: #53667d;
}
QPushButton[nav="true"] {
    color: #40536c;
}
QPushButton[nav="true"][selected="true"] {
    background: #e6f0ff;
    color: #075fc5;
}
#settingsNav {
    color: #354960;
}
#settingsNav:checked {
    color: #ffffff;
    background: #163d73;
}
#footer QLabel {
    color: #41536b;
}
#windowControl {
    color: #394c65;
}
QPushButton {
    background: #ffffff;
    color: #172235;
    border-color: #d1ddeb;
}
#interactiveField, #searchField, #notesEditor, #gamesTable {
    background: #ffffff;
    color: #172235;
    border-color: #d1ddeb;
}
#movesScroll QScrollBar:vertical {
    background: #0b141d;
}
#movesScroll QScrollBar::handle:vertical {
    background: #394a5e;
}
#ledgerMove {
    color: #172235;
}
"""


def run(render_path: Path | None = None, screen: str = "analyze", theme: str = "dark", settings_category: str = "", show_suggestion: bool = False) -> int:
    app = QApplication(sys.argv)
    window = DeepTacticsReferenceWindow(screen=screen)
    if screen == "setting" and settings_category:
        window.pages["Setting"].show_category(settings_category)
    if screen == "play" and show_suggestion:
        window.play_board.board.show_suggestion()
    window.apply_theme(theme.title())
    window.show()
    if render_path:
        app.processEvents()
        render_path.parent.mkdir(parents=True, exist_ok=True)
        if not window.grab().save(str(render_path)):
            raise RuntimeError(f"Could not save preview: {render_path}")
        window.close()
        return 0
    return app.exec()


def run_qt_game() -> int:
    """Launch the Qt shell from the application's public UI switch."""
    settings = SETTINGS_STORE.load()
    screen = str(settings.get("start_with", "Analyze")).lower().replace(" ", "_")
    screen = {"openings": "opening", "settings": "setting"}.get(screen, screen)
    if screen not in {"analyze", "play", "ai_match", "games", "opening", "database", "setting"}:
        screen = "analyze"
    return run(screen=screen, theme=str(settings.get("theme", "Dark")).lower())


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or render the DeepTactics Qt interface.")
    parser.add_argument("--render", type=Path, help="Write a PNG screenshot and exit.")
    parser.add_argument("--screen", choices=("analyze", "play", "ai_match", "games", "opening", "database", "setting"), default="analyze", help="Preview screen.")
    parser.add_argument("--theme", choices=("dark", "light", "system"), default="dark", help="Preview theme.")
    parser.add_argument("--settings-category", choices=("General", "Engine Management", "Board & Pieces", "Gameplay", "Folders & Files", "Analysis", "Display", "Sound", "Advanced", "About"), default="", help="Settings page category to render.")
    parser.add_argument("--show-suggestion", action="store_true", help="Render the Play board with a suggested-move arrow.")
    args = parser.parse_args()
    return run(args.render, args.screen, args.theme, args.settings_category, args.show_suggestion)


if __name__ == "__main__":
    raise SystemExit(main())
