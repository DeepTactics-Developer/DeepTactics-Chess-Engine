import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from time import perf_counter, sleep
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chess_engine.main import _resolve_weights_path, _save_compare_results
from chess_engine.ai.search import SearchResult
from chess_engine.ai.tablebase import TablebaseRootResult
from chess_engine.chess_core.fen import board_to_fen
from chess_engine.chess_core.move import Move
from chess_engine.native.native_eval import native_hash_status
from chess_engine.uci import UCIEngine


class MainEntrypointTest(unittest.TestCase):







    def test_self_play_command_prints_summary(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "chess_engine.main",
                "self-play",
                "--games",
                "1",
                "--depth",
                "1",
                "--max-plies",
                "1",
                "--seed",
                "3",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("Game 1:", result.stdout)
        self.assertIn("plies=1", result.stdout)
        self.assertIn("Summary:", result.stdout)

    def test_self_play_command_accepts_adaptive_time_arguments(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "chess_engine.main",
                "self-play",
                "--games",
                "1",
                "--depth",
                "1",
                "--max-plies",
                "1",
                "--time",
                "0.05",
                "--adaptive",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("Game 1:", result.stdout)

    def test_self_play_command_accepts_native_search_backend(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "chess_engine.main",
                "self-play",
                "--games",
                "1",
                "--depth",
                "1",
                "--max-plies",
                "1",
                "--search-backend",
                "native",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("Game 1:", result.stdout)

    def test_self_play_verbose_prints_move_progress(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "chess_engine.main",
                "self-play",
                "--games",
                "1",
                "--depth",
                "1",
                "--max-plies",
                "1",
                "--no-book",
                "--verbose",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("ply 1:", result.stdout)

    def test_learn_command_saves_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "rows.csv"
            output_path = Path(directory) / "weights.json"
            input_path.write_text(
                "game,ply,fen,move,san,score,result\n"
                "1,1,4k3/8/8/8/8/8/8/4KQ2 w - - 0 1,e1e2,Ke2,100,1-0\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "chess_engine.main",
                    "learn",
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0)
            self.assertTrue(output_path.exists())
            self.assertIn("Saved evaluation weights", result.stdout)

    def test_train_cycle_generates_data_and_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "cycle.csv"
            weights_path = Path(directory) / "weights.json"

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "chess_engine.main",
                    "train-cycle",
                    "--games",
                    "1",
                    "--depth",
                    "1",
                    "--max-plies",
                    "2",
                    "--no-book",
                    "--data-output",
                    str(data_path),
                    "--weights-output",
                    str(weights_path),
                    "--format",
                    "csv",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0)
            self.assertTrue(data_path.exists())
            self.assertTrue(weights_path.exists())
            self.assertIn("Saved evaluation weights", result.stdout)

    def test_train_cycle_appends_existing_data_and_updates_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "cycle.csv"
            weights_path = Path(directory) / "weights.json"
            data_path.write_text(
                "game,ply,side,fen,move,san,score,source,fen_after,result,result_value\n"
                "1,1,w,4k3/8/8/8/8/8/8/4KQ2 w - - 0 1,e1e2,Ke2,100,d1,"
                "4k3/8/8/8/8/8/4K3/5Q2 b - - 1 1,1-0,1.0\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "chess_engine.main",
                    "train-cycle",
                    "--games",
                    "1",
                    "--depth",
                    "1",
                    "--max-plies",
                    "1",
                    "--no-book",
                    "--data-output",
                    str(data_path),
                    "--weights-output",
                    str(weights_path),
                    "--format",
                    "csv",
                    "--compare-games",
                    "0",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0)
            rows = data_path.read_text(encoding="utf-8").splitlines()
            self.assertGreaterEqual(len(rows), 3)
            self.assertTrue(weights_path.exists())
            self.assertIn("Loaded existing training rows: 1", result.stdout)
            self.assertIn("Updated evaluation weights", result.stdout)
            self.assertIn("search_backend", rows[0])
            self.assertIn("weights", rows[0])

    def test_train_cycle_prints_monotonic_game_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "cycle.csv"
            weights_path = Path(directory) / "weights.json"

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "chess_engine.main",
                    "train-cycle",
                    "--games",
                    "2",
                    "--depth",
                    "1",
                    "--max-plies",
                    "1",
                    "--no-book",
                    "--data-output",
                    str(data_path),
                    "--weights-output",
                    str(weights_path),
                    "--format",
                    "csv",
                    "--compare-games",
                    "0",
                    "--max-position-duplicates",
                    "1",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0)
            self.assertIn("Game 1:", result.stdout)
            self.assertIn("Game 2:", result.stdout)

    def test_train_cycle_fast_generates_data_and_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "cycle.csv"
            weights_path = Path(directory) / "weights.json"

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "chess_engine.main",
                    "train-cycle-fast",
                    "--games",
                    "1",
                    "--time",
                    "0.05",
                    "--max-depth",
                    "2",
                    "--max-plies",
                    "1",
                    "--workers",
                    "1",
                    "--no-book",
                    "--data-output",
                    str(data_path),
                    "--weights-output",
                    str(weights_path),
                    "--format",
                    "csv",
                    "--compare-games",
                    "0",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(data_path.exists())
            self.assertTrue(weights_path.exists())
            self.assertIn("Fast train cycle:", result.stdout)
            self.assertIn("Saved evaluation weights", result.stdout)

    def test_compare_weights_command_prints_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate_path = Path(directory) / "candidate.json"
            output_path = Path(directory) / "compare.csv"
            candidate_path.write_text(
                '{"material": 1.05, "position": 1.0, "mobility": 1.0, "endgame": 1.0}',
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "chess_engine.main",
                    "compare-weights",
                    "--candidate",
                    str(candidate_path),
                    "--games",
                    "1",
                    "--depth",
                    "1",
                    "--max-plies",
                    "2",
                    "--verbose",
                    "--output",
                    str(output_path),
                    "--format",
                    "csv",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0)
            self.assertIn("Candidate results:", result.stdout)
            self.assertIn("Record:", result.stdout)
            self.assertIn("Score:", result.stdout)
            self.assertIn("Average plies:", result.stdout)
            self.assertIn("Verdict:", result.stdout)
            self.assertTrue(output_path.exists())
            self.assertIn("candidate_wins", output_path.read_text(encoding="utf-8"))
            self.assertIn("score_rate", output_path.read_text(encoding="utf-8"))


    def test_training_report_command_prints_quality_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "rows.csv"
            input_path.write_text(
                "game,ply,fen,move,san,score,result\n"
                "1,1,8/8/8/8/8/8/8/K6k w - - 0 1,a1a2,Ka2,120,1-0\n"
                "1,2,8/8/8/8/8/8/8/K6k w - - 2 2,a1a2,Ka2,-80,1/2-1/2\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "chess_engine.main",
                    "training-report",
                    "--input",
                    str(input_path),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0)
            self.assertIn("Rows: 2", result.stdout)
            self.assertIn("Unique positions: 1", result.stdout)
            self.assertIn("Duplicate rows: 1", result.stdout)
            self.assertIn("Average score: 20.0cp", result.stdout)

    def test_nnue_build_dataset_command_writes_sparse_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "rows.csv"
            output_path = Path(directory) / "nnue.csv"
            input_path.write_text(
                "game,ply,side,fen,move,san,score,result,result_value\n"
                "1,1,w,rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1,"
                "e2e4,e4,25,1-0,1.0\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "chess_engine.main",
                    "nnue-build-dataset",
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--label",
                    "blended",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0)
            self.assertTrue(output_path.exists())
            self.assertIn("NNUE dataset:", result.stdout)
            self.assertIn("Written rows: 1", result.stdout)
            self.assertIn("features", output_path.read_text(encoding="utf-8"))










    def test_uci_command_accepts_options_and_clock_go(self):
        result = subprocess.run(
            [sys.executable, "-m", "chess_engine.main", "uci", "--depth", "1"],
            cwd=PROJECT_ROOT,
            input=(
                "uci\n"
                "setoption name Backend value python\n"
                "setoption name Depth value 1\n"
                "setoption name Move Overhead value 5\n"
                "isready\n"
                "position startpos\n"
                "go wtime 1000 btime 1000 winc 0 binc 0 movestogo 20\n"
                "quit\n"
            ),
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("option name Backend", result.stdout)
        self.assertIn("option name NNUE Mode", result.stdout)
        self.assertIn("readyok", result.stdout)
        self.assertIn("bestmove ", result.stdout)




















    def test_native_uci_infinite_search_uses_deep_stop_responsive_depth(self):
        engine = UCIEngine(depth=2, backend="native", quiescence_depth=6)
        captured: dict[str, object] = {}

        class ImmediateExecutor:
            def submit(self, function, *args, **kwargs):
                captured["function"] = function
                captured["args"] = args
                captured.update(kwargs)

                class DoneFuture:
                    def done(self):
                        return True

                    def result(self, timeout=None):
                        return SearchResult(move=Move.from_uci("e2e4"), score=0, depth=64, completed_depth=1)

                    def cancel(self):
                        return False

                    def cancelled(self):
                        return False

                return DoneFuture()

        engine._executor = ImmediateExecutor()  # type: ignore[assignment]
        lines = engine.handle_line("go infinite")

        self.assertIn("requested_depth=64", "\n".join(lines))
        self.assertEqual(captured["depth"], 64)
        self.assertEqual(captured["backend"], "native")
        self.assertEqual(captured["time_limit"], 24 * 60 * 60)








    def test_native_uci_stop_interrupts_long_background_search(self):
        engine = UCIEngine(depth=64, backend="native")
        engine._start_pending_search(depth=64, time_limit=None, adaptive=False)
        sleep(0.02)
        started = perf_counter()
        responses = engine._finish_pending_search(request_stop=True)
        elapsed = perf_counter() - started

        self.assertLess(elapsed, 0.5)
        self.assertTrue(any(response.startswith("bestmove ") for response in responses))

    def test_native_uci_promoted_timed_search_defaults_are_reported(self):
        engine = UCIEngine(depth=64, backend="native")
        engine.handle_line("position startpos")
        responses = engine.handle_line("go movetime 10")

        config = next(response for response in responses if response.startswith("info string config "))
        self.assertIn("native_internal_iterative_deepening=true", config)
        self.assertIn("native_aspiration_windows=true", config)
        self.assertIn("native_clustered_tt=false", config)

    def test_native_uci_hash_option_changes_native_tt_capacity(self):
        engine = UCIEngine(depth=1, backend="native")
        initial = native_hash_status()
        if initial is None:
            self.skipTest("native DLL does not expose hash capacity controls")

        engine.handle_line("setoption name Hash value 32")
        updated = native_hash_status()
        self.assertIsNotNone(updated)
        assert updated is not None
        self.assertEqual(updated["hash_mb"], 32)
        self.assertGreater(updated["bucket_count"], initial["bucket_count"])

        engine.handle_line("position startpos")
        config = next(
            response
            for response in engine.handle_line("go depth 1")
            if response.startswith("info string config ")
        )
        self.assertIn("hash_mb=32", config)



    def test_latest_weights_alias_resolves_default_weights_path(self):
        self.assertEqual(
            _resolve_weights_path("latest"),
            Path("artifacts/evaluation_weights.json"),
        )
        self.assertEqual(
            _resolve_weights_path("custom.json"),
            Path("custom.json"),
        )

    def test_save_compare_results_supports_json_and_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            json_path = Path(directory) / "compare.json"
            csv_path = Path(directory) / "compare.csv"
            results = {
                "candidate_wins": 1,
                "candidate_losses": 0,
                "draws": 1,
                "average_plies": 12.5,
            }

            _save_compare_results(results, str(json_path), "json")
            _save_compare_results(results, str(csv_path), "csv")

            self.assertIn('"candidate_wins"', json_path.read_text(encoding="utf-8"))
            self.assertIn("candidate_wins", csv_path.read_text(encoding="utf-8"))

    def test_analyze_command_prints_pgn_report(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "game.pgn"
            input_path.write_text("1. e4 e5 *", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "chess_engine.main",
                    "analyze",
                    "--input",
                    str(input_path),
                    "--depth",
                    "1",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0)
            self.assertIn("White accuracy:", result.stdout)
            self.assertIn("e4:", result.stdout)

    def test_import_openings_command_converts_source_tsv(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "source.tsv"
            output_path = Path(directory) / "openings.tsv"
            input_path.write_text("C60\tRuy Lopez\t1. e4 e5 2. Nf3 Nc6 3. Bb5\n", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "chess_engine.main",
                    "import-openings",
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0)
            self.assertIn("Imported 1 opening", result.stdout)
            self.assertIn("C60\tRuy Lopez", output_path.read_text(encoding="utf-8"))

    def test_import_openings_command_accepts_multiple_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            input_a = Path(directory) / "a.tsv"
            input_b = Path(directory) / "b.tsv"
            output_path = Path(directory) / "openings.tsv"
            input_a.write_text("A00\tTest A\t1. a3\n", encoding="utf-8")
            input_b.write_text("B00\tTest B\t1. e4\n", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "chess_engine.main",
                    "import-openings",
                    "--input",
                    str(input_a),
                    str(input_b),
                    "--output",
                    str(output_path),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0)
            self.assertIn("Imported 2 opening", result.stdout)




if __name__ == "__main__":
    unittest.main()
