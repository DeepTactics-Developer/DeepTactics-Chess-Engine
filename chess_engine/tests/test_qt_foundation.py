"""Foundation tests for the optional PySide6 desktop interface."""

from __future__ import annotations

import os
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from chess_engine.ui.settings_store import DEFAULT_GUI_SETTINGS, SettingsStore
from chess_engine.ui.database_store import DatabaseStore
from chess_engine.ui.database_jobs import DatabaseJob, DatabaseJobRunner
from chess_engine.ui.engine_registry import EngineRegistry
from chess_engine.ui.game_library import GameLibraryStore
from chess_engine.ui.i18n import TranslationCatalog
from chess_engine.chess_core.enums import Color, PieceType
from chess_engine.chess_core.move import Move
from chess_engine.chess_core.piece import Piece
from chess_engine.chess_core.square import Square
from chess_engine.analysis import MoveClassification

try:
    from PySide6.QtCore import QPoint, QEventLoop, Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QLabel
    from chess_engine.ui.gui_state import GameDocument
    from chess_engine.ui.qt_ui import (
        CustomPositionDialog,
        DeepTacticsReferenceWindow,
        EvalMeter,
        Header,
        OpeningPage,
        SETTINGS_STORE,
        UciLogDialog,
        WoodenBoard,
        _terminal_display_info,
        _qt_move_classification,
        cp_to_expected_points,
        format_evaluation,
        format_play_clock,
        load_ai_match_opening_lines,
        pgn_with_metadata,
    )
    from chess_engine.ui.uci_client import UCIAnalysisInfo, UCIClient, parse_info_line, parse_result_white_score
except ModuleNotFoundError:
    GameDocument = None
    UCIClient = None


class SettingsStoreTest(unittest.TestCase):
    def test_settings_store_defaults_save_and_reset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gui_settings.json"
            store = SettingsStore(path)

            self.assertEqual(store.load()["theme"], "Dark")
            self.assertEqual(store.load()["piece_set_path"], "assets/pieces")
            self.assertEqual(store.load()["nnue_mode"], "Off")
            self.assertEqual(store.load()["default_ai_match_engine"], "DeepTactics")
            store.save({"theme": "Light", "language": "Korean"})
            loaded = store.load()
            self.assertEqual(loaded["theme"], "Light")
            self.assertEqual(loaded["language"], "Korean")
            self.assertEqual(loaded["engines"][0]["name"], "DeepTactics")

            reset = store.reset()
            self.assertEqual(reset["theme"], DEFAULT_GUI_SETTINGS["theme"])

    def test_extended_setting_defaults_remain_backward_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SettingsStore(Path(directory) / "gui_settings.json")
            store.save({"theme": "Light"})
            loaded = store.load()
            self.assertEqual(loaded["theme"], "Light")
            self.assertEqual(loaded["analysis_multipv"], 3)
            self.assertTrue(loaded["use_analysis_cache"])
            self.assertEqual(loaded["logs"], "artifacts/logs")
            self.assertFalse(loaded["use_tablebase"])
            self.assertEqual(loaded["syzygy_path"], "")
            self.assertEqual(loaded["tablebase_max_pieces"], 7)

    def test_engine_registry_preserves_default_off_tablebase_profile_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SettingsStore(Path(directory) / "gui_settings.json")
            store.save({"engines": [{"name": "DeepTactics", "type": "UCI", "path": "DeepTactics-UCI.cmd"}]})
            registry = EngineRegistry(store)

            entry = registry.entry("DeepTactics")
            self.assertIsNotNone(entry)
            self.assertFalse(entry["use_tablebase"])
            self.assertEqual(entry["syzygy_path"], "")
            self.assertEqual(entry["tablebase_max_pieces"], 7)

    def test_korean_translation_catalog_loads_real_resources(self) -> None:
        app = QApplication.instance() or QApplication([])
        root = Path(__file__).resolve().parents[2]
        catalog = TranslationCatalog(root / "assets", "Korean")
        self.assertEqual(catalog.text("tab.analyze"), "분석")
        self.assertEqual(catalog.text("settings.engines"), "엔진 관리")
        self.assertEqual(catalog.text("missing", "Fallback"), "Fallback")
        label = QLabel("Load PGN")
        catalog.apply_widgets(label)
        app.processEvents()
        self.assertEqual(label.text(), "PGN 불러오기")


class GameLibraryStoreTest(unittest.TestCase):
    def test_import_metadata_notes_favorites_and_non_destructive_removal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = GameLibraryStore(Path(directory) / "pgn")
            text = (
                '[Event "Training Game"]\n[Site "Local"]\n[Date "2026.05.27"]\n'
                '[White "Player"]\n[Black "DeepTactics"]\n[Result "*"]\n'
                '[ECO "C20"]\n[Opening "King Pawn Game"]\n\n1. e4 e5 2. Nf3 *\n'
            )
            records, warnings = store.import_text(text, source_type="Human vs AI")
            self.assertFalse(warnings)
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record.white, "Player")
            self.assertEqual(record.game_type, "Human vs AI")
            self.assertEqual(record.eco, "C20")
            self.assertEqual(store.uci_moves(record), ("e2e4", "e7e5", "g1f3"))
            managed = Path(record.managed_path)
            self.assertTrue(managed.exists())

            store.toggle_favorite(record)
            store.set_notes(record, "Review move two.")
            reloaded = GameLibraryStore(store.root)
            self.assertTrue(reloaded.records[0].favorite)
            self.assertEqual(reloaded.records[0].notes, "Review move two.")
            reloaded.remove_from_library(reloaded.records[0])
            self.assertFalse(reloaded.records)
            self.assertTrue(managed.exists(), "Removing from the index must preserve PGN files.")

    def test_collections_persist_without_copying_game_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = GameLibraryStore(Path(directory) / "pgn")
            records, warnings = store.import_text('[Result "*"]\n\n1. e4 e5 *\n')
            self.assertFalse(warnings)
            collection_id = store.create_collection("Opening Prep")
            store.add_to_collection(records[0], collection_id)

            reloaded = GameLibraryStore(store.root)
            self.assertEqual(reloaded.collections[collection_id], "Opening Prep")
            self.assertIn(collection_id, reloaded.records[0].collections)


@unittest.skipIf(GameDocument is None, "PySide6 is not installed")
class GameDocumentTest(unittest.TestCase):
    def test_document_owns_move_history_independently_from_an_engine(self) -> None:
        document = GameDocument()
        document.apply_uci_move("e2e4")

        snapshot = document.snapshot()
        self.assertEqual(snapshot.move_list, ("e4",))
        self.assertEqual(snapshot.current_ply, 1)
        self.assertIn(" b ", snapshot.fen)

        document.undo()
        self.assertEqual(document.snapshot().move_list, ())

    def test_timeline_seek_keeps_full_move_list_while_changing_fen(self) -> None:
        document = GameDocument()
        document.load_uci_line(("e2e4", "e7e5", "g1f3"))
        document.seek(1)

        snapshot = document.snapshot()
        self.assertEqual(snapshot.current_ply, 1)
        self.assertEqual(snapshot.move_list, ("e4", "e5", "Nf3"))
        self.assertIn(" b ", snapshot.fen)

    def test_live_game_moves_extend_mainline_instead_of_creating_analysis_variation(self) -> None:
        document = GameDocument()
        document.append_mainline_move("e2e4")
        document.append_mainline_move("e7e5")

        snapshot = document.snapshot()
        self.assertEqual(snapshot.move_list, ("e4", "e5"))
        self.assertEqual(snapshot.variation_moves, ())


@unittest.skipIf(UCIClient is None, "PySide6 is not installed")
class UCIClientTest(unittest.TestCase):
    def test_info_parser_normalizes_black_root_score_to_white(self) -> None:
        info = parse_info_line("info depth 4 score cp 35 nodes 120 nps 4000 time 30 tbhits 9 hashfull 81 pv e7e5", "b")
        self.assertIsNotNone(info)
        self.assertEqual(info.score_cp_white, -35)
        self.assertEqual(info.pv, ("e7e5",))
        self.assertEqual(info.tbhits, 9)
        self.assertEqual(info.hashfull, 81)
        self.assertEqual(parse_result_white_score("info string result searched_score_white=-81 bestmove=e7e5"), -81)
        alternate = parse_info_line("info depth 3 multipv 2 score cp 11 nodes 40 nps 80 time 2 pv d7d5", "b")
        self.assertEqual(alternate.multipv, 2)
        self.assertEqual(alternate.score_cp_white, -11)

    def test_evaluation_display_keeps_mate_distinct_and_clamps_large_cp(self) -> None:
        self.assertEqual(format_evaluation(67), "+0.67")
        self.assertEqual(format_evaluation(-230), "-2.30")
        self.assertEqual(format_evaluation(1240), "+10.00+")
        self.assertEqual(format_evaluation(-3000), "-10.00+")
        self.assertEqual(format_evaluation(mate_white=3), "+M3")
        self.assertEqual(format_evaluation(mate_white=-4), "-M4")

    def test_terminal_checkmate_display_keeps_white_winner_direction(self) -> None:
        fen = "k7/1Q6/2K5/8/8/8/8/8 b - - 0 1"
        info = _terminal_display_info(fen, UCIAnalysisInfo(score_cp_white=0))
        self.assertEqual(info.score_cp_white, 100_000)
        meter = EvalMeter()
        meter.set_score(info.score_cp_white, info.mate_white)
        self.assertEqual(float(meter._animation.endValue()), 100.0)

    def test_diagnostic_result_does_not_erase_parsed_mate_score(self) -> None:
        app = QApplication.instance() or QApplication([])
        client = UCIClient(Path.cwd())
        client._handle_line("info depth 4 score mate 3 nodes 1 nps 1 time 1 pv e2e4")
        client._handle_line("info string result searched_score_white=99999 bestmove=e2e4")
        app.processEvents()
        self.assertEqual(client.latest_info.mate_white, 3)
        self.assertIsNone(client.latest_info.score_cp_white)

    def test_profile_tablebase_options_are_sent_as_uci_compatible_values(self) -> None:
        client = UCIClient(Path.cwd())
        client.profile = {
            "name": "DeepTactics",
            "builtin": True,
            "threads": 2,
            "hash": 64,
            "use_tablebase": True,
            "syzygy_path": "C:/tb",
            "tablebase_max_pieces": 6,
        }
        commands: list[str] = []
        client._write = commands.append  # type: ignore[method-assign]
        client.supported_options.update({"Threads", "Hash", "Use Tablebase", "SyzygyPath", "Tablebase Max Pieces"})

        client._handle_line("uciok")

        self.assertIn("setoption name Use Tablebase value true", commands)
        self.assertIn("setoption name SyzygyPath value C:/tb", commands)
        self.assertIn("setoption name Tablebase Max Pieces value 6", commands)
        self.assertIn("setoption name Threads value 2", commands)
        self.assertIn("setoption name Hash value 64", commands)
        self.assertEqual(commands[-1], "isready")

    def test_expected_points_classification_softens_decided_positions_and_limits_miss(self) -> None:
        quiet = Move.from_uci("e2e4")
        other = Move.from_uci("d2d4")
        extreme, extreme_evidence = _qt_move_classification(
            player=Color.WHITE,
            before=UCIAnalysisInfo(score_cp_white=800),
            after=UCIAnalysisInfo(score_cp_white=500),
            previous=None,
            played_move=quiet,
            best_move=other,
            is_forced=False,
            is_book=False,
        )
        self.assertIsNone(extreme)
        self.assertLess(extreme_evidence.expected_loss, 0.10)

        meaningful, meaningful_evidence = _qt_move_classification(
            player=Color.WHITE,
            before=UCIAnalysisInfo(score_cp_white=50),
            after=UCIAnalysisInfo(score_cp_white=-100),
            previous=None,
            played_move=quiet,
            best_move=other,
            is_forced=False,
            is_book=False,
        )
        self.assertEqual(meaningful.key, "mistake")
        self.assertGreater(meaningful_evidence.expected_loss, 0.10)

        ordinary, ordinary_evidence = _qt_move_classification(
            player=Color.WHITE,
            before=UCIAnalysisInfo(score_cp_white=15),
            after=UCIAnalysisInfo(score_cp_white=5),
            previous=None,
            played_move=quiet,
            best_move=other,
            is_forced=False,
            is_book=False,
        )
        self.assertIsNone(ordinary)
        self.assertEqual(ordinary_evidence.label, "No icon")

        missed, missed_evidence = _qt_move_classification(
            player=Color.WHITE,
            before=UCIAnalysisInfo(score_cp_white=400),
            after=UCIAnalysisInfo(score_cp_white=50),
            previous=UCIAnalysisInfo(score_cp_white=-100),
            played_move=quiet,
            best_move=other,
            is_forced=False,
            is_book=False,
        )
        self.assertEqual(missed.key, "missed_win")
        self.assertGreater(missed_evidence.opportunity_gain, 0.24)

        black_mistake, black_evidence = _qt_move_classification(
            player=Color.BLACK,
            before=UCIAnalysisInfo(score_cp_white=-50),
            after=UCIAnalysisInfo(score_cp_white=100),
            previous=None,
            played_move=quiet,
            best_move=other,
            is_forced=False,
            is_book=False,
        )
        self.assertEqual(black_mistake.key, "mistake")
        self.assertGreater(black_evidence.expected_loss, 0.16)

    def test_expected_points_classification_preserves_mate_type(self) -> None:
        classification, evidence = _qt_move_classification(
            player=Color.WHITE,
            before=UCIAnalysisInfo(mate_white=3),
            after=UCIAnalysisInfo(score_cp_white=0),
            previous=None,
            played_move=Move.from_uci("e2e4"),
            best_move=Move.from_uci("d2d4"),
            is_forced=False,
            is_book=False,
        )
        self.assertEqual(classification.key, "blunder")
        self.assertIsNone(evidence.cp_loss)
        self.assertIn("+M3", evidence.tooltip())
        self.assertAlmostEqual(cp_to_expected_points(0), 0.5)

    def test_pgn_metadata_includes_gui_player_names_and_date_fields(self) -> None:
        text = pgn_with_metadata(
            '[Result "*"]\n\n1. e4 *\n',
            Event="DeepTactics Play",
            Date="2026.05.28",
            White="Human",
            Black="DeepTactics",
            Type="Human vs AI",
        )
        self.assertIn('[White "Human"]', text)
        self.assertIn('[Black "DeepTactics"]', text)
        self.assertIn('[Date "2026.05.28"]', text)

    def test_opening_page_reads_real_tsv_lines_and_emits_analyze_line(self) -> None:
        app = QApplication.instance() or QApplication([])
        page = OpeningPage()
        self.assertGreater(len(page.openings), 1000)
        self.assertIn("OPENING BOOK READY", page.notice.text())
        page.library.search.setText("Sicilian Defense")
        app.processEvents()
        self.assertTrue(page.library.visible)
        page.selected_opening_changed(page.library.visible[0])
        self.assertEqual(page.study.current.name, "Sicilian Defense")
        self.assertNotEqual(page.study.positions[-1], page.study.positions[0])
        emitted: list[tuple[str, ...]] = []
        page.open_analyze.connect(emitted.append)
        page.study.send_to_analyze()
        self.assertTrue(emitted and emitted[0])
        self.assertEqual(page.continuations.height(), 145)
        self.assertGreaterEqual(page.study.name.minimumHeight(), 55)


    def test_real_deeptactics_process_analyzes_and_stops_without_blocking_qt(self) -> None:
        app = QApplication.instance() or QApplication([])
        root = Path(__file__).resolve().parents[2]
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        client = UCIClient(root)
        bestmoves: list[str] = []
        infos = []
        errors: list[str] = []
        commands: list[str] = []
        loop = QEventLoop()
        client.info_received.connect(infos.append)
        client.error_received.connect(errors.append)
        client.command_sent.connect(commands.append)
        client.bestmove_received.connect(lambda move: (bestmoves.append(move), loop.quit()))

        client.analyze_fen(fen, depth=1)
        QTimer.singleShot(10000, loop.quit)
        loop.exec()
        self.assertFalse(errors, errors)
        self.assertTrue(bestmoves and bestmoves[-1] != "0000")
        self.assertTrue(infos and infos[-1].depth >= 1)
        self.assertIn("uci", commands)
        self.assertTrue(any(command.startswith("position fen ") for command in commands))

        bestmoves.clear()
        client.analyze_fen(fen, infinite=True)
        QTimer.singleShot(150, client.stop)
        QTimer.singleShot(10000, loop.quit)
        loop.exec()
        client.quit()
        app.processEvents()
        self.assertFalse(errors, errors)
        self.assertTrue(bestmoves, "stop did not return a bestmove")

    def test_uci_log_dialog_and_database_job_runner_record_real_activity(self) -> None:
        app = QApplication.instance() or QApplication([])
        dialog = UciLogDialog()
        dialog.append_line("Analyze", ">", "uci")
        dialog.append_line("Analyze", "<", "uciok")
        self.assertIn("[Analyze] > uci", dialog.text.toPlainText())
        self.assertIn("[Analyze] < uciok", dialog.text.toPlainText())

        root = Path(__file__).resolve().parents[2]
        runner = DatabaseJobRunner(root)
        completed: list[DatabaseJob] = []
        loop = QEventLoop()
        runner.finished.connect(lambda job: (completed.append(job), loop.quit()))
        runner.enqueue(DatabaseJob("Native status", "workspace", ["native-status"]))
        QTimer.singleShot(10000, loop.quit)
        loop.exec()
        app.processEvents()
        self.assertTrue(completed)
        self.assertEqual(completed[0].status, "Completed", completed[0].output)
        self.assertIn("native", completed[0].output.lower())

    def test_analyze_button_updates_live_qt_cards_from_uci_result(self) -> None:
        app = QApplication.instance() or QApplication([])
        window = DeepTacticsReferenceWindow(screen="analyze")
        blocked = window.analysis_console.engine.depth_selector.blockSignals(True)
        window.analysis_console.engine.depth_selector.setValue(1)
        window.analysis_console.engine.depth_selector.blockSignals(blocked)
        window.analysis_console.position_info.set_ply(0)
        loop = QEventLoop()
        bestmoves: list[str] = []
        window.uci_client.bestmove_received.connect(lambda move: (bestmoves.append(move), loop.quit()))

        window.analysis_console.engine.analyze_button.click()
        QTimer.singleShot(10000, loop.quit)
        loop.exec()
        QTimer.singleShot(10000, loop.quit)
        loop.exec()
        app.processEvents()

        self.assertNotEqual(window.analysis_console.engine.stat_values["Depth"].text(), "--")
        self.assertNotEqual(window.analysis_console.summary.score.text(), "--")
        self.assertNotIn("Nodes:  --", window.footer.metrics[2].text())
        self.assertNotIn("--", window.analysis_console.summary.candidate_moves[1].text())
        self.assertTrue(window.analysis_active)
        self.assertTrue(window.analysis_console.engine.analyze_button.isChecked())
        self.assertTrue(window.analysis_console.engine.stop_button.isEnabled())
        self.assertIn("active", window.analysis_console.engine.ready_state.text().lower())
        baseline_scores = dict(window.analysis_console.graph.canvas.scores)
        self.assertEqual(set(baseline_scores), {0})
        window.analysis_console.principal_variation.suggestion.setChecked(True)
        app.processEvents()
        self.assertIsNotNone(window.analysis_board.board.suggested_arrow)
        window.analysis_console.principal_variation.suggestion.setChecked(False)
        self.assertIsNone(window.analysis_board.board.suggested_arrow)
        window.apply_analysis_move("e2e4")
        QTimer.singleShot(10000, loop.quit)
        loop.exec()
        app.processEvents()
        self.assertEqual(window.analysis_document.snapshot().current_ply, 1)
        self.assertEqual(window.analysis_console.graph.canvas.scores, baseline_scores)
        window.analysis_console.engine.stop_button.click()
        self.assertFalse(window.analysis_active)
        self.assertFalse(window.analysis_console.engine.stop_button.isEnabled())
        window.close()

    def test_analyze_timeline_updates_board_highlight_and_builtin_nnue_is_off(self) -> None:
        app = QApplication.instance() or QApplication([])
        window = DeepTacticsReferenceWindow(screen="analyze")
        self.assertEqual(window.analysis_document.snapshot().current_ply, 0)
        self.assertFalse(window.analysis_console.ledger.move_buttons[0].isEnabled())
        window.analysis_document.load_uci_line(("e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"))
        window.analysis_console.ledger.move_buttons[0].click()
        app.processEvents()

        self.assertEqual(window.analysis_document.snapshot().current_ply, 1)
        self.assertEqual(window.analysis_board.board.last_move, ("e2", "e4"))
        self.assertTrue(window.analysis_console.ledger.move_buttons[0].property("selected"))
        self.assertFalse(window.analysis_console.engine.nnue.property("active"))
        window.analysis_console.graph.canvas.ply_selected.emit(6)
        app.processEvents()
        self.assertEqual(window.analysis_document.snapshot().current_ply, 6)
        self.assertEqual(window.analysis_board.board.last_move, ("a7", "a6"))
        window.close()

    def test_header_reserves_room_for_brand_version_and_cache_key_separates_profiles(self) -> None:
        app = QApplication.instance() or QApplication([])
        header = Header()
        self.assertGreaterEqual(header.identity.width(), 345)
        self.assertFalse(header.version_label.text().endswith("..."))
        window = DeepTacticsReferenceWindow(screen="analyze")
        with patch("chess_engine.ui.qt_ui.ENGINE_REGISTRY.entry", return_value={"path": "A.exe", "threads": 1, "hash": 256}):
            first = window._analysis_cache_key(window.analysis_document.fen)
        with patch("chess_engine.ui.qt_ui.ENGINE_REGISTRY.entry", return_value={"path": "B.exe", "threads": 1, "hash": 256}):
            second = window._analysis_cache_key(window.analysis_document.fen)
        self.assertNotEqual(first, second)
        window.close()

    def test_analyze_variation_preserves_loaded_mainline_and_annotations_are_available(self) -> None:
        app = QApplication.instance() or QApplication([])
        window = DeepTacticsReferenceWindow(screen="analyze")
        blocked = window.analysis_console.engine.depth_selector.blockSignals(True)
        window.analysis_console.engine.depth_selector.setValue(1)
        window.analysis_console.engine.depth_selector.blockSignals(blocked)
        window.analysis_document.load_uci_line(("e2e4", "e7e5", "g1f3"))
        window.analysis_document.seek_mainline(1)
        window.apply_analysis_move("c7c5")
        snapshot = window.analysis_document.snapshot()
        self.assertEqual(snapshot.move_list, ("e4", "e5", "Nf3"))
        self.assertEqual(snapshot.variation_moves, ("c5",))
        self.assertFalse(window.analysis_console.ledger.variation.isHidden())
        self.assertIn("Try: 1...", window.analysis_console.ledger.variation.text())
        variation_row = window.analysis_console.ledger.moves_grid.getItemPosition(
            window.analysis_console.ledger.moves_grid.indexOf(window.analysis_console.ledger.variation)
        )[0]
        next_mainline_row = window.analysis_console.ledger.moves_grid.getItemPosition(
            window.analysis_console.ledger.moves_grid.indexOf(window.analysis_console.ledger.move_buttons[2])
        )[0]
        self.assertLess(variation_row, next_mainline_row)
        self.assertEqual(window.analysis_board.board.last_move, ("c7", "c5"))
        self.assertIn("c5", window.analysis_console.position_info.values["OPENING LINE"].text())
        window.analysis_board.board.marked_squares.add("e4")
        window.analysis_board.board.annotation_arrows.append(("e4", "e5"))
        window.analysis_board.board.clear_annotations()
        self.assertFalse(window.analysis_board.board.marked_squares)
        self.assertFalse(window.analysis_board.board.annotation_arrows)
        window.close()

    def test_try_move_does_not_create_floating_top_level_widgets(self) -> None:
        app = QApplication.instance() or QApplication([])
        window = DeepTacticsReferenceWindow(screen="analyze")
        window.show()
        window.analysis_document.load_uci_line(("e2e4", "e7e5", "g1f3"))
        window.analysis_document.seek_mainline(1)
        app.processEvents()
        before = {id(widget) for widget in QApplication.topLevelWidgets() if widget.isVisible()}
        for _ in range(5):
            window.analysis_document.seek_mainline(1)
            window.apply_analysis_move("c7c5")
        app.processEvents()
        after = {id(widget) for widget in QApplication.topLevelWidgets() if widget.isVisible()}
        self.assertFalse(after - before)
        window.close()

    def test_cached_mainline_navigation_avoids_research_and_depth_change_invalidates_cache(self) -> None:
        app = QApplication.instance() or QApplication([])
        window = DeepTacticsReferenceWindow(screen="analyze")
        window.analysis_document.load_uci_line(("e2e4", "e7e5"))
        fen = window.analysis_document.mainline_positions()[1]
        window._baseline_positions = window.analysis_document.mainline_positions()
        window._baseline_results[1] = UCIAnalysisInfo(score_cp_white=35, depth=2, pv=("e7e5",))
        commands: list[str] = []
        window.uci_client.command_sent.connect(commands.append)
        window.analysis_active = True
        window.analysis_phase = "live"
        window.navigate_analysis_ply(1)
        app.processEvents()
        self.assertEqual(window.analysis_document.fen, fen)
        self.assertFalse(any(command.startswith("go ") for command in commands))
        window._analysis_cache[window._analysis_cache_key(fen)] = {1: UCIAnalysisInfo(score_cp_white=35)}
        window.change_analysis_depth(window._analysis_depth() + 1)
        self.assertFalse(window._analysis_cache)
        window.close()

    def test_eval_meter_keeps_animated_score_mapping(self) -> None:
        meter = EvalMeter()
        self.assertEqual(meter._animation.duration(), 180)
        meter.set_score(100)
        expected = 50 + 50 * (2 / (1 + math.exp(-0.004 * 100)) - 1)
        self.assertAlmostEqual(float(meter._animation.endValue()), expected)
        meter.set_score(mate_white=-2)
        self.assertEqual(float(meter._animation.endValue()), 0.0)

    def test_moves_card_scrolls_long_lines_and_knight_arrow_turns(self) -> None:
        app = QApplication.instance() or QApplication([])
        window = DeepTacticsReferenceWindow(screen="analyze")
        ledger = window.analysis_console.ledger
        ledger.set_timeline(tuple(f"M{ply}" for ply in range(1, 61)), 60)
        window.show()
        app.processEvents()
        self.assertGreater(ledger.scroll.verticalScrollBar().maximum(), 0)
        ledger.scroll.verticalScrollBar().setValue(ledger.scroll.verticalScrollBar().maximum() // 2)
        retained_scroll = ledger.scroll.verticalScrollBar().value()
        visible_top_levels_before = {id(widget) for widget in QApplication.topLevelWidgets() if widget.isVisible()}
        stable_buttons = tuple(ledger.move_buttons)
        ledger.set_timeline(tuple(f"M{ply}" for ply in range(1, 61)), 31, ("TryA",), 31)
        loop = QEventLoop()
        QTimer.singleShot(20, loop.quit)
        loop.exec()
        self.assertEqual(ledger.scroll.verticalScrollBar().value(), retained_scroll)
        visible_top_levels_after = {id(widget) for widget in QApplication.topLevelWidgets() if widget.isVisible()}
        self.assertFalse(visible_top_levels_after - visible_top_levels_before)
        self.assertEqual(tuple(ledger.move_buttons), stable_buttons)
        ledger.set_timeline(tuple(f"M{ply}" for ply in range(1, 61)), 60, ("TryA",), 31)
        app.processEvents()
        self.assertEqual(tuple(ledger.move_buttons), stable_buttons)
        self.assertFalse({id(widget) for widget in QApplication.topLevelWidgets() if widget.isVisible()} - visible_top_levels_after)
        board = WoodenBoard(starting_position=True)
        straight_route = board._arrow_route("e2", "e4", 0, 0, 64)
        knight_route = board._arrow_route("g1", "f3", 0, 0, 64)
        self.assertEqual(len(straight_route), 2)
        self.assertEqual(len(knight_route), 3)
        window.close()

    def test_analyze_board_click_does_not_raise_and_cached_ply_restores_evaluation(self) -> None:
        app = QApplication.instance() or QApplication([])
        window = DeepTacticsReferenceWindow(screen="analyze")
        board = window.analysis_board.board
        clicked: list[str] = []
        board.left_click_completed.connect(clicked.append)
        window.show()
        app.processEvents()
        QTest.mouseClick(board, Qt.MouseButton.LeftButton, pos=QPoint(20, 20))
        app.processEvents()
        self.assertTrue(clicked)

        window.analysis_document.load_uci_line(("e2e4", "e7e5"))
        window._baseline_positions = window.analysis_document.mainline_positions()
        window._baseline_results[1] = UCIAnalysisInfo(score_cp_white=124, depth=4, pv=("e7e5",))
        classification = MoveClassification("good", "Good", "OK", 20)
        window._analysis_classifications[1] = classification
        window.analysis_document.seek_mainline(1)
        self.assertEqual(window.analysis_console.summary.score.text(), "+1.24")
        self.assertIs(window.analysis_board.board.move_classification, classification)
        window.close()

    def test_moves_rebuild_repeatedly_keeps_scrollbar_embedded(self) -> None:
        app = QApplication.instance() or QApplication([])
        window = DeepTacticsReferenceWindow(screen="analyze")
        window.show()
        app.processEvents()
        before = {id(widget) for widget in QApplication.topLevelWidgets() if widget.isVisible()}
        for count in (36, 42, 36, 48, 36):
            window.analysis_console.ledger.set_timeline(tuple(f"M{ply}" for ply in range(1, count + 1)), count)
            window.analysis_console.ledger.set_classifications(
                {1: MoveClassification("good", "Good", "OK", 15)}
            )
            app.processEvents()
        after = {id(widget) for widget in QApplication.topLevelWidgets() if widget.isVisible()}
        self.assertFalse(after - before)
        self.assertIsNotNone(window.analysis_console.ledger.scroll.verticalScrollBar().parent())
        window.close()

    def test_original_line_preanalysis_builds_stable_graph_before_live_navigation(self) -> None:
        app = QApplication.instance() or QApplication([])
        window = DeepTacticsReferenceWindow(screen="analyze")
        blocked = window.analysis_console.engine.depth_selector.blockSignals(True)
        window.analysis_console.engine.depth_selector.setValue(1)
        window.analysis_console.engine.depth_selector.blockSignals(blocked)
        window.analysis_document.load_uci_line(("e2e4", "e7e5", "g1f3"))
        stable_buttons = tuple(window.analysis_console.ledger.move_buttons)
        bestmoves: list[str] = []
        loop = QEventLoop()
        window.uci_client.bestmove_received.connect(lambda move: (bestmoves.append(move), loop.quit()))
        window.analysis_console.engine.analyze_button.click()
        for _expected in range(5):  # four original positions, then the current live position
            QTimer.singleShot(10000, loop.quit)
            loop.exec()
        baseline_scores = dict(window.analysis_console.graph.canvas.scores)
        self.assertEqual(set(baseline_scores), {0, 1, 2, 3})
        self.assertTrue(any(not badge.isHidden() for badge in window.analysis_console.ledger.badges))
        self.assertEqual(tuple(window.analysis_console.ledger.move_buttons), stable_buttons)
        window.analysis_document.seek_mainline(1)
        app.processEvents()
        self.assertFalse(window.uci_client.is_searching)
        self.assertEqual(window.analysis_console.graph.canvas.scores, baseline_scores)
        window.stop_analysis()
        window.new_analysis_document()
        self.assertFalse(window.analysis_console.graph.canvas.scores)
        self.assertTrue(all(badge.isHidden() for badge in window.analysis_console.ledger.badges))
        window.close()

    def test_custom_single_position_analyzes_live_fen_instead_of_empty_review(self) -> None:
        app = QApplication.instance() or QApplication([])
        window = DeepTacticsReferenceWindow(screen="analyze")
        custom_fen = "4k3/8/8/8/3P4/8/4K3/8 w - - 0 1"
        window.analysis_document.load_fen(custom_fen)
        calls: list[tuple[str, dict[str, object]]] = []
        window.uci_client.analyze_fen = lambda fen, **kwargs: calls.append((fen, kwargs))  # type: ignore[method-assign]

        window.start_analysis()
        app.processEvents()

        self.assertEqual(window.analysis_phase, "live")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], custom_fen)
        self.assertEqual(calls[0][1].get("moves"), ())
        self.assertEqual(calls[0][1].get("initial_fen"), custom_fen)
        window.close()

    def test_custom_position_builds_fen_and_play_start_applies_setup(self) -> None:
        app = QApplication.instance() or QApplication([])
        dialog = CustomPositionDialog()
        dialog.board.set_piece(Square.from_algebraic("e1"), Piece(Color.WHITE, PieceType.KING))
        dialog.board.set_piece(Square.from_algebraic("e8"), Piece(Color.BLACK, PieceType.KING))
        dialog.side_to_move.setCurrentIndex(1)
        dialog.apply_position()
        self.assertIsNotNone(dialog.result_fen)
        self.assertIn(" b ", dialog.result_fen)

        window = DeepTacticsReferenceWindow(screen="play")
        window.play_console.player_setup.selections[1].setCurrentText("Human")
        window.play_board.new_button.click()
        window.apply_play_move("e2e4")
        window.apply_play_move("e7e5")
        self.assertEqual(window.play_document.snapshot().move_list, ("e4", "e5"))
        self.assertEqual(window.play_console.moves.table.item(0, 1).text(), "e4")
        self.assertEqual(window.play_console.moves.table.item(0, 2).text(), "e5")
        self.assertEqual(window.play_console.game_status.values["Rule Set"].text(), "Standard")
        self.assertEqual(window.play_console.game_status.values["Players"].text(), "Human vs Human")
        self.assertEqual(window.play_console.player_clock.move_number.text(), "Move: 2 ply")
        self.assertTrue(window.play_board.undo_button.isEnabled())
        window.play_board.undo_button.click()
        self.assertEqual(window.play_document.snapshot().move_list, ("e4",))
        self.assertEqual(window.play_document.snapshot().current_ply, 1)
        window.close()

    def test_play_time_formats_and_builtin_engine_can_answer_a_human_move(self) -> None:
        app = QApplication.instance() or QApplication([])
        self.assertEqual(format_play_clock(None), "\u221e")
        self.assertEqual(format_play_clock(3_661_000), "1:01:01")
        self.assertEqual(format_play_clock(61_000), "01:01")
        self.assertEqual(format_play_clock(9_950), "00:09.9")
        window = DeepTacticsReferenceWindow(screen="play")
        window.play_board.new_button.click()
        window.apply_play_move("e2e4")
        loop = QEventLoop()
        QTimer.singleShot(10000, loop.quit)
        window.play_uci_client.bestmove_received.connect(lambda _move: loop.quit())
        loop.exec()
        app.processEvents()
        self.assertGreaterEqual(len(window.play_document.snapshot().move_list), 2)
        self.assertEqual(window.play_document.game.board.side_to_move, Color.WHITE)
        window.close()

    def test_play_premove_chain_is_virtual_then_executes_one_move_after_opponent_turn(self) -> None:
        app = QApplication.instance() or QApplication([])
        window = DeepTacticsReferenceWindow(screen="play")
        window.play_console.player_setup.selections[1].setCurrentText("Human")
        window.play_board.new_button.click()
        window.apply_play_move("e2e4")
        window.play_board.board.last_input_was_drag = True
        window.apply_play_move("g1f3")
        window.play_board.board.last_input_was_drag = True
        window.apply_play_move("f3g5")
        self.assertEqual(window.play_premoves, ["g1f3", "f3g5"])
        self.assertIn("f3", window.play_board.board.marked_squares)
        window.play_board.board.last_input_was_drag = False
        window.apply_play_move("e7e5")
        app.processEvents()
        self.assertEqual(window.play_document.snapshot().move_list, ("e4", "e5", "Nf3"))
        self.assertEqual(window.play_premoves, ["f3g5"])
        window.clear_play_premoves()
        self.assertFalse(window.play_premoves)
        window.close()

    def test_play_pause_disables_board_and_ai_match_runs_separate_turn_sessions(self) -> None:
        app = QApplication.instance() or QApplication([])
        window = DeepTacticsReferenceWindow(screen="play")
        window.play_console.player_setup.selections[1].setCurrentText("Human")
        window.play_board.new_button.click()
        window.play_board.tail_action.click()
        self.assertTrue(window.play_paused)
        self.assertFalse(window.play_board.board.interactive)
        window.play_board.tail_action.click()
        self.assertFalse(window.play_paused)
        self.assertTrue(window.play_board.board.interactive)

        window.show_screen("AI Match")
        window.match_right.controls.games.setValue(1)
        changed: list[int] = []
        loop = QEventLoop()
        window.match_document.changed.connect(
            lambda snapshot: (changed.append(snapshot.current_ply), loop.quit()) if snapshot.current_ply >= 2 else None
        )
        window.match_right.controls.start_button.click()
        QTimer.singleShot(10000, loop.quit)
        loop.exec()
        window.stop_ai_match()
        self.assertGreaterEqual(window.match_live_ply, 2)
        self.assertIsNot(window.match_white_client, window.match_black_client)
        self.assertEqual(window.match_center.move_list.table.item(0, 1).text(), window.match_document.snapshot().move_list[0])
        window.close()

    def test_ai_match_opening_suite_and_undo_clock_restore_are_connected(self) -> None:
        app = QApplication.instance() or QApplication([])
        lines = load_ai_match_opening_lines()
        self.assertTrue(lines)
        window = DeepTacticsReferenceWindow(screen="play")
        window.match_game_number = 1
        window.match_total_games = 1
        window.match_running = True
        with patch.object(window, "_request_match_move"):
            window._start_match_game()
        self.assertEqual(window.match_document.mainline_uci_moves, lines[0])

        settings = SETTINGS_STORE.load()
        settings["restore_clock_on_undo"] = True
        with patch("chess_engine.ui.qt_ui.SETTINGS_STORE.load", return_value=settings):
            window.play_console.player_setup.selections[1].setCurrentText("Human")
            window.play_board.new_button.click()
            window.play_turn_started_at = None
            window.play_clock_ms = {Color.WHITE: 120_000, Color.BLACK: 120_000}
            window.apply_play_move("e2e4")
            window.play_clock_ms = {Color.WHITE: 110_000, Color.BLACK: 119_000}
            window.play_turn_started_at = None
            window.apply_play_move("e7e5")
            window.undo_play_move()
        self.assertEqual(window.play_clock_ms, {Color.WHITE: 110_000, Color.BLACK: 119_000})
        window.close()

    def test_ai_match_search_mode_controls_choose_the_expected_uci_path(self) -> None:
        window = DeepTacticsReferenceWindow(screen="ai_match")
        window.match_running = True
        window.match_paused = False
        window.match_time_control = window._match_control_config()
        window.match_clock_ms = {Color.WHITE: 300_000, Color.BLACK: 300_000}
        window.match_document.new_game()

        window.match_right.controls.search_mode.setCurrentText("Fixed Depth")
        window.match_right.controls.depth.setValue(7)
        with patch.object(window.match_white_client, "analyze_fen") as analyze, patch.object(
            window.match_white_client, "play_fen_with_clock"
        ) as clock:
            window._request_match_move()
        self.assertEqual(analyze.call_args.kwargs["depth"], 7)
        self.assertNotIn("movetime_ms", analyze.call_args.kwargs)
        clock.assert_not_called()

        window.match_right.controls.search_mode.setCurrentText("Move Time")
        window.match_right.controls.movetime_ms.setValue(650)
        with patch.object(window.match_white_client, "analyze_fen") as analyze, patch.object(
            window.match_white_client, "play_fen_with_clock"
        ) as clock:
            window._request_match_move()
        self.assertEqual(analyze.call_args.kwargs["movetime_ms"], 650)
        self.assertNotIn("depth", analyze.call_args.kwargs)
        clock.assert_not_called()

        window.match_right.controls.search_mode.setCurrentText("Clock")
        window.match_right.controls.time_control.setCurrentText("Blitz 5+3")
        window.match_time_control = window._match_control_config()
        with patch.object(window.match_white_client, "analyze_fen") as analyze, patch.object(
            window.match_white_client, "play_fen_with_clock"
        ) as clock:
            window._request_match_move()
        analyze.assert_not_called()
        self.assertEqual(clock.call_args.kwargs["white_ms"], 300_000)
        self.assertEqual(clock.call_args.kwargs["black_ms"], 300_000)
        self.assertEqual(clock.call_args.kwargs["white_increment_ms"], 3000)
        self.assertEqual(clock.call_args.kwargs["black_increment_ms"], 3000)

        window.match_right.controls.time_control.setCurrentText("Unlimited")
        window.match_time_control = window._match_control_config()
        window.match_right.controls.movetime_ms.setValue(900)
        with patch.object(window.match_white_client, "analyze_fen") as analyze, patch.object(
            window.match_white_client, "play_fen_with_clock"
        ) as clock:
            window._request_match_move()
        self.assertEqual(analyze.call_args.kwargs["movetime_ms"], 900)
        clock.assert_not_called()
        window.close()

    def test_analyze_modes_issue_depth_movetime_and_infinite_uci_commands(self) -> None:
        window = DeepTacticsReferenceWindow(screen="analyze")
        base = SETTINGS_STORE.load()
        for mode, expected in (
            ("Fixed Depth", {"depth": 4}),
            ("Move Time", {"movetime_ms": 7000}),
            ("Infinite until Stop", {"infinite": True}),
        ):
            settings = dict(base)
            settings["analysis_mode"] = mode
            settings["default_move_time"] = 7
            window.analysis_phase = "live"
            with patch("chess_engine.ui.qt_ui.SETTINGS_STORE.load", return_value=settings):
                with patch.object(window.uci_client, "analyze_fen") as analyze:
                    window._submit_analysis(window.analysis_document.fen, 4)
            kwargs = analyze.call_args.kwargs
            for key, value in expected.items():
                self.assertEqual(kwargs[key], value)
        window.close()

    def test_analyze_pgn_load_save_is_connected_to_document(self) -> None:
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "loaded.pgn"
            output = Path(directory) / "saved.pgn"
            source.write_text("[Result \"*\"]\n\n1. e4 e5 2. Nf3 *\n", encoding="utf-8")
            settings = dict(DEFAULT_GUI_SETTINGS)
            settings["pgn_auto_save"] = str(Path(directory) / "library")
            with patch("chess_engine.ui.qt_ui.SETTINGS_STORE.load", return_value=settings):
                window = DeepTacticsReferenceWindow(screen="analyze")
            with patch(
                "chess_engine.ui.qt_ui.QFileDialog.getOpenFileName",
                return_value=(str(source), "PGN Files (*.pgn)"),
            ):
                window.load_analysis_pgn()
            self.assertEqual(window.analysis_document.snapshot().move_list, ("e4", "e5", "Nf3"))
            with patch(
                "chess_engine.ui.qt_ui.QFileDialog.getSaveFileName",
                return_value=(str(output), "PGN Files (*.pgn)"),
            ):
                window.save_analysis_pgn()
            self.assertIn("1. e4 e5 2. Nf3", output.read_text(encoding="utf-8"))
            window.close()

    def test_games_library_record_routes_to_analyze_and_play_without_mock_data(self) -> None:
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as directory:
            settings = dict(DEFAULT_GUI_SETTINGS)
            settings["pgn_auto_save"] = str(Path(directory) / "library")
            with patch("chess_engine.ui.qt_ui.SETTINGS_STORE.load", return_value=settings):
                window = DeepTacticsReferenceWindow(screen="games")
            records, warnings = window.game_library.import_text(
                '[Event "Imported Test"]\n[White "A"]\n[Black "B"]\n[Result "*"]\n\n1. d4 d5 2. c4 *\n'
            )
            self.assertFalse(warnings)
            window.pages["Games"].refresh_status()
            self.assertEqual(window.pages["Games"].center.table.rowCount(), 1)
            self.assertNotIn("SAMPLE", window.pages["Games"].notice.text())

            window.open_library_game_in_analyze(records[0])
            self.assertEqual(window.analysis_document.snapshot().move_list, ("d4", "d5", "c4"))
            window.load_library_game_in_play(records[0])
            app.processEvents()
            self.assertEqual(window.play_document.snapshot().move_list, ("d4", "d5", "c4"))
            self.assertEqual(window.stack.currentWidget(), window.pages["Play"])
            window.close()


if __name__ == "__main__":
    unittest.main()
