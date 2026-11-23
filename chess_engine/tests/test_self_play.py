import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chess_engine.self_play import (
    SelfPlayProgress,
    deduplicate_training_rows,
    results_to_training_rows,
    run_self_play_game,
    run_self_play_games,
    save_self_play_results,
)
from chess_engine.self_play.self_play_runner import _adjudicated_limit_result
from chess_engine.chess_core.game import Game
from chess_engine.chess_core.board import Board
from chess_engine.chess_core.enums import Color, PieceType
from chess_engine.chess_core.piece import Piece
from chess_engine.chess_core.square import Square


class SelfPlayTest(unittest.TestCase):
    def test_self_play_records_positions(self):
        result = run_self_play_game(depth=1, max_plies=4, opening_book_path=None)

        self.assertEqual(result.plies_played, 4)
        self.assertTrue(result.reached_move_limit)
        self.assertEqual(len(result.positions), 4)
        self.assertTrue(result.pgn)
        self.assertEqual(result.positions[0].ply, 1)
        self.assertIn(" w ", result.positions[0].fen_before)
        self.assertTrue(result.positions[0].move_uci)
        self.assertTrue(result.positions[0].move_san)

    def test_self_play_uses_opening_book_when_enabled(self):
        with tempfile.TemporaryDirectory() as directory:
            book_path = Path(directory) / "book.tsv"
            book_path.write_text("A00\tAnderssen Opening\ta3\n", encoding="utf-8")

            result = run_self_play_game(
                depth=1,
                max_plies=1,
                opening_plies=1,
                opening_book_path=str(book_path),
            )

        self.assertEqual(result.positions[0].move_san, "a3")

    def test_self_play_can_disable_opening_book(self):
        result = run_self_play_game(depth=1, max_plies=1, opening_book_path=None)

        self.assertNotEqual(result.positions[0].source, "book")

    def test_self_play_opening_seed_is_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            book_path = Path(directory) / "book.tsv"
            book_path.write_text(
                "A00\tTest A\t1. a3\n"
                "A04\tTest N\t1. Nf3\n"
                "C20\tTest E\t1. e4\n",
                encoding="utf-8",
            )

            first = run_self_play_game(
                depth=1,
                max_plies=1,
                opening_plies=1,
                opening_book_path=str(book_path),
                seed=7,
            )
            second = run_self_play_game(
                depth=1,
                max_plies=1,
                opening_plies=1,
                opening_book_path=str(book_path),
                seed=7,
            )

        self.assertEqual(first.positions[0].move_san, second.positions[0].move_san)

    def test_self_play_can_emit_move_progress(self):
        progress: list[SelfPlayProgress] = []

        result = run_self_play_game(
            depth=1,
            max_plies=2,
            opening_book_path=None,
            progress_callback=progress.append,
        )

        self.assertEqual(result.plies_played, 2)
        self.assertEqual(len(progress), 2)
        self.assertEqual(progress[0].game_index, 1)
        self.assertEqual(progress[0].ply, 1)
        self.assertTrue(progress[0].san)

    def test_self_play_training_rows_are_export_friendly(self):
        result = run_self_play_game(depth=1, max_plies=2)

        rows = result.to_training_rows()

        self.assertEqual(len(rows), 2)
        self.assertIn("fen", rows[0])
        self.assertIn("move", rows[0])
        self.assertIn("san", rows[0])
        self.assertIn("score", rows[0])
        self.assertIn("side", rows[0])
        self.assertIn("source", rows[0])
        self.assertIn("fen_after", rows[0])
        self.assertIn("search_backend", rows[0])
        self.assertIn("depth", rows[0])
        self.assertIn("time_limit", rows[0])
        self.assertIn("adaptive", rows[0])
        self.assertIn("weights", rows[0])
        self.assertIn("result_value", rows[0])
        self.assertIn("result", rows[0])

    def test_self_play_training_rows_include_search_metadata(self):
        result = run_self_play_game(
            depth=2,
            max_plies=1,
            time_limit=0.05,
            adaptive=True,
            opening_book_path=None,
            weights_label="evaluation_weights.json",
        )

        row = result.to_training_rows()[0]

        self.assertEqual(row["search_backend"], "native")
        self.assertEqual(row["depth"], 2)
        self.assertEqual(row["time_limit"], "0.05")
        self.assertEqual(row["adaptive"], 1)
        self.assertEqual(row["weights"], "evaluation_weights.json")

    def test_move_limit_adjudication_assigns_clear_eval_wins(self):
        board = Board.empty()
        board.set_piece(Square.from_algebraic("e1"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("e8"), Piece(Color.BLACK, PieceType.KING))
        board.set_piece(Square.from_algebraic("d1"), Piece(Color.WHITE, PieceType.QUEEN))
        game = Game(board=board)

        self.assertEqual(_adjudicated_limit_result(game, 250), "1-0")
        self.assertEqual(_adjudicated_limit_result(game, None), "*")

    def test_multiple_self_play_games_run(self):
        results = run_self_play_games(game_count=2, depth=1, max_plies=1)

        self.assertEqual(len(results), 2)
        self.assertTrue(all(result.plies_played == 1 for result in results))

    def test_self_play_can_use_time_limited_adaptive_search(self):
        result = run_self_play_game(depth=1, max_plies=1, time_limit=0.05, adaptive=True)

        self.assertEqual(result.plies_played, 1)
        self.assertEqual(len(result.positions), 1)

    def test_self_play_exploration_marks_training_source(self):
        result = run_self_play_game(
            depth=1,
            max_plies=6,
            opening_book_path=None,
            exploration_cp=1000,
            exploration_plies=6,
            seed=11,
        )

        self.assertTrue(any(position.source.startswith("explore:") for position in result.positions))

    def test_self_play_can_adjudicate_clear_leads_before_move_limit(self):
        result = run_self_play_game(
            depth=1,
            max_plies=8,
            opening_book_path=None,
            adjudication_threshold=0,
            adjudication_min_ply=1,
        )

        self.assertLess(result.plies_played, 8)
        self.assertFalse(result.reached_move_limit)
        self.assertIn(result.result, {"1-0", "0-1"})

    def test_results_to_training_rows_adds_game_index(self):
        results = run_self_play_games(game_count=2, depth=1, max_plies=1)

        rows = results_to_training_rows(results)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["game"], 1)
        self.assertEqual(rows[1]["game"], 2)

    def test_deduplicate_training_rows_caps_repeated_positions(self):
        rows = [
            {"game": 1, "fen": "8/8/8/8/8/8/8/K6k w - - 0 1", "move": "a1a2"},
            {"game": 2, "fen": "8/8/8/8/8/8/8/K6k w - - 4 9", "move": "a1a2"},
            {"game": 3, "fen": "8/8/8/8/8/8/8/K6k w - - 8 20", "move": "a1a2"},
        ]

        filtered = deduplicate_training_rows(rows, max_per_position=2)

        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[-1]["game"], 2)

    def test_save_self_play_results_as_json_and_csv(self):
        results = run_self_play_games(game_count=1, depth=1, max_plies=2)

        with tempfile.TemporaryDirectory() as directory:
            json_path = Path(directory) / "rows.json"
            csv_path = Path(directory) / "rows.csv"

            save_self_play_results(results, json_path)
            save_self_play_results(results, csv_path)

            self.assertIn('"fen"', json_path.read_text(encoding="utf-8"))
            self.assertIn("fen,move", csv_path.read_text(encoding="utf-8"))

    def test_save_self_play_results_rejects_unknown_format(self):
        results = run_self_play_games(game_count=1, depth=1, max_plies=1)

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                save_self_play_results(results, Path(directory) / "rows.txt")

    def test_self_play_rejects_invalid_arguments(self):
        with self.assertRaises(ValueError):
            run_self_play_game(depth=0)

        with self.assertRaises(ValueError):
            run_self_play_game(max_plies=0)

        with self.assertRaises(ValueError):
            run_self_play_game(time_limit=0)

        with self.assertRaises(ValueError):
            run_self_play_game(adjudication_threshold=-2)

        with self.assertRaises(ValueError):
            run_self_play_game(adjudication_min_ply=0)

        with self.assertRaises(ValueError):
            run_self_play_game(exploration_cp=-1)

        with self.assertRaises(ValueError):
            run_self_play_games(game_count=0)


if __name__ == "__main__":
    unittest.main()
