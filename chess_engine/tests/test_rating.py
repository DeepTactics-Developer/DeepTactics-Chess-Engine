import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chess_engine.rating.bayes_elo import estimate_bayes_elo
from chess_engine.rating.glicko import update_glicko_ratings
from chess_engine.rating.export import export_bayeselo_pgn, export_rating_csv
from chess_engine.rating.match_runner import EngineSpec, run_rating_match
from chess_engine.rating.models import EngineRating, MatchResult
from chess_engine.rating.store import (
    append_match_records,
    current_engine_rating_label,
    load_match_records,
    load_rating_store,
    record_rating_snapshot,
)
from chess_engine.ai.evaluator import get_evaluation_weights
from chess_engine.ai.search import SearchResult
from chess_engine.chess_core.move import Move


class RatingTest(unittest.TestCase):
    def test_bayes_elo_rewards_better_results(self):
        matches = [
            MatchResult("A", "B", 1.0),
            MatchResult("A", "B", 1.0),
            MatchResult("A", "B", 0.5),
        ]

        ratings = estimate_bayes_elo(matches)

        self.assertGreater(ratings["A"], ratings["B"])

    def test_glicko_updates_rating_and_rd(self):
        current = {"A": EngineRating(name="A"), "B": EngineRating(name="B")}

        updated = update_glicko_ratings(current, [MatchResult("A", "B", 1.0)])

        self.assertGreater(updated["A"][0], current["A"].glicko_rating)
        self.assertLess(updated["A"][1], current["A"].glicko_rd)

    def test_rating_store_label_reads_current_engine(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ratings.json"
            record_rating_snapshot(
                EngineRating(
                    name="DeepTactics Engine",
                    bayes_elo=1512.4,
                    glicko_rating=1508.2,
                    glicko_rd=120.0,
                    games=12,
                ),
                path=path,
            )

            label = current_engine_rating_label(path)

            self.assertEqual(label, "Elo: 1512 | Glicko-1: 1508")
            self.assertIn("history", load_rating_store(path))

    def test_match_records_are_persisted_and_exportable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ratings.json"
            pgn_path = Path(directory) / "rating_games.pgn"
            csv_path = Path(directory) / "rating_games.csv"
            append_match_records(
                [
                    MatchResult(
                        white="A",
                        black="B",
                        result=1.0,
                        result_text="1-0",
                        pgn='[Event "Test"]\n[White "A"]\n[Black "B"]\n[Result "1-0"]\n\n1-0',
                    )
                ],
                path,
            )

            self.assertEqual(len(load_match_records(path)), 1)
            export_bayeselo_pgn(pgn_path, path)
            export_rating_csv(csv_path, path)

            self.assertIn('[White "A"]', pgn_path.read_text(encoding="utf-8"))
            self.assertIn('[Result "1-0"]', pgn_path.read_text(encoding="utf-8"))
            self.assertIn("white,black,result", csv_path.read_text(encoding="utf-8"))

    def test_rating_match_keeps_pgn_and_configs_for_future_recalculation(self):
        result = run_rating_match(
            EngineSpec("A", use_weights=False, depth=1),
            EngineSpec("B", use_weights=False, depth=1),
            max_plies=1,
        )

        self.assertTrue(result.pgn)
        self.assertEqual(result.white_config["depth"], 1)
        self.assertIn(result.result_text, {"1-0", "0-1", "1/2-1/2"})

    def test_rating_match_activates_white_and_black_specs_on_their_turns(self):
        with tempfile.TemporaryDirectory() as directory:
            black_weights = Path(directory) / "black.json"
            black_weights.write_text('{"material": 1.2}', encoding="utf-8")
            observed: list[float] = []
            moves = iter((Move.from_uci("e2e4"), Move.from_uci("e7e5")))

            def fake_search(*args, **kwargs):
                observed.append(get_evaluation_weights().material)
                return SearchResult(move=next(moves), score=0, depth=1, completed_depth=1)

            with patch("chess_engine.rating.match_runner.find_backend_move", side_effect=fake_search):
                run_rating_match(
                    EngineSpec("White", use_weights=False, depth=1),
                    EngineSpec("Black", weights_path=str(black_weights), depth=1),
                    max_plies=2,
                )

        self.assertEqual(observed, [1.0, 1.2])


if __name__ == "__main__":
    unittest.main()
