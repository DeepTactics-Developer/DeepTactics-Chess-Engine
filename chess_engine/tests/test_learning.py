import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chess_engine.ai.evaluator import (
    EvaluationWeights,
    get_evaluation_weights,
    load_evaluation_weights,
    reset_evaluation_weights,
    set_evaluation_weights,
)
from chess_engine.chess_core.fen import STANDARD_STARTING_FEN
from chess_engine.learning import (
    PositionFeatures,
    features_from_fen,
    load_training_rows,
    train_weights_from_file,
    train_weights_from_rows,
)


class LearningTest(unittest.TestCase):
    def tearDown(self):
        reset_evaluation_weights()

    def test_features_from_fen_extracts_handcrafted_terms(self):
        features = features_from_fen(STANDARD_STARTING_FEN)

        self.assertIsInstance(features, PositionFeatures)
        self.assertEqual(features.material, 0)
        self.assertEqual(features.position, 0)
        self.assertEqual(features.mobility, 0)
        self.assertEqual(features.pawn_structure, 0)
        self.assertEqual(features.bishop_pair, 0)
        self.assertEqual(features.rook_activity, 0)
        self.assertEqual(features.king_safety, 0)
        self.assertEqual(features.passed_pawn, 0)
        self.assertEqual(features.endgame_pressure, 0)
        self.assertEqual(features.line_activity, 0)
        self.assertEqual(features.attack_momentum, 0)

    def test_fine_grained_features_capture_strategic_terms(self):
        # Rook shares the king's file so it registers native's distance-decayed
        # endgame_piece_pressure_score (a rook seven squares away, as in the
        # previous a7-vs-g8 FEN, scores zero pressure under that formula).
        features = features_from_fen("6k1/6R1/8/8/8/8/PPP4P/6K1 w - - 0 1")

        self.assertGreater(features.rook_activity, 0)
        self.assertGreater(features.king_safety, 0)
        self.assertGreater(features.endgame_pressure, 0)

    def test_training_rows_produce_bounded_weights(self):
        # Regularized updates barely move on a tiny sample by design (see
        # trainer.MIN_SAMPLES_FOR_FULL_TRUST) -- replicate past that
        # threshold so this actually exercises the correlation direction
        # instead of just re-testing the small-sample no-op case.
        rows = [
            {
                "fen": "4k3/8/8/8/8/8/8/4KQ2 w - - 0 1",
                "result": "1-0",
            },
            {
                "fen": "4kq2/8/8/8/8/8/8/4K3 b - - 0 1",
                "result": "0-1",
            },
        ] * 1500

        weights = train_weights_from_rows(rows)

        self.assertIsInstance(weights, EvaluationWeights)
        self.assertGreaterEqual(weights.material, 0.5)
        self.assertLessEqual(weights.material, 1.5)
        self.assertGreater(weights.material, 1.0)
        self.assertGreaterEqual(weights.endgame_pressure, 0.5)
        self.assertLessEqual(weights.endgame_pressure, 1.5)

    def test_training_outputs_fine_grained_weights(self):
        rows = [
            {
                "fen": "6k1/R7/8/8/8/8/PPP3PP/6K1 w - - 0 1",
                "result": "1-0",
            }
        ] * 2500

        weights = train_weights_from_rows(rows)

        self.assertGreaterEqual(weights.rook_activity, 1.0)
        self.assertGreaterEqual(weights.king_safety, 1.0)
        self.assertGreaterEqual(weights.line_activity, 1.0)


    def test_small_batches_barely_move_weights_from_previous(self):
        # A handful of rows shouldn't be enough to meaningfully move a weight
        # away from wherever it started -- see MIN_SAMPLES_FOR_FULL_TRUST.
        rows = [
            {"fen": "4k3/8/8/8/8/8/8/4KQ2 w - - 0 1", "result": "1-0"},
            {"fen": "4kq2/8/8/8/8/8/8/4K3 b - - 0 1", "result": "0-1"},
        ]

        weights = train_weights_from_rows(rows)

        self.assertAlmostEqual(weights.material, 1.0, delta=0.02)

    def test_large_batch_updates_from_a_nondefault_previous_weight(self):
        # With enough consistent-signal data, the update should move further
        # in the correlated direction starting from a higher prior than it
        # would starting from the default -- i.e. previous_weights is a real
        # starting point, not ignored.
        rows = [
            {"fen": "4k3/8/8/8/8/8/8/4KQ2 w - - 0 1", "result": "1-0"},
            {"fen": "4kq2/8/8/8/8/8/8/4K3 b - - 0 1", "result": "0-1"},
        ] * 1500

        from_default = train_weights_from_rows(rows)
        from_high_prior = train_weights_from_rows(rows, previous_weights=EvaluationWeights(material=1.4))

        self.assertGreater(from_high_prior.material, from_default.material)

    def test_blended_target_keeps_result_mode_backward_compatible(self):
        rows = [{"fen": "4k3/8/8/8/8/8/8/4KQ2 w - - 0 1", "result": "1-0"}]

        self.assertEqual(train_weights_from_rows(rows), train_weights_from_rows(rows, target_mode="result"))

    def test_training_ignores_draw_rows(self):
        weights = train_weights_from_rows(
            [{"fen": STANDARD_STARTING_FEN, "result": "1/2-1/2"}]
        )

        self.assertEqual(weights, EvaluationWeights())

    def test_save_load_and_activate_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.json"
            path.write_text(
                json.dumps({"material": 1.2, "position": 0.9, "mobility": 1.1}),
                encoding="utf-8",
            )

            loaded = load_evaluation_weights(path)

        self.assertEqual(
            loaded,
            EvaluationWeights(
                material=1.2,
                position=0.9,
                mobility=1.1,
                line_activity=0.81,
                attack_momentum=0.81,
            ),
        )
        self.assertEqual(get_evaluation_weights(), loaded)

    def test_load_weights_accepts_utf8_bom(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.json"
            path.write_text('\ufeff{"material": 1.2}', encoding="utf-8")

            loaded = load_evaluation_weights(path, apply=False)

        self.assertEqual(loaded.material, 1.2)

    def test_old_weight_file_preserves_position_scaled_activity_terms(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.json"
            path.write_text('{"position": 1.2}', encoding="utf-8")

            loaded = load_evaluation_weights(path, apply=False)

        self.assertAlmostEqual(loaded.line_activity, 1.44)
        self.assertAlmostEqual(loaded.attack_momentum, 1.44)

    def test_set_and_reset_weights(self):
        set_evaluation_weights(EvaluationWeights(material=1.1, position=1.2, mobility=1.3))
        self.assertEqual(get_evaluation_weights().position, 1.2)

        reset_evaluation_weights()

        self.assertEqual(get_evaluation_weights(), EvaluationWeights())

    def test_load_csv_and_train_file(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "rows.csv"
            output_path = Path(directory) / "weights.json"
            input_path.write_text(
                "game,ply,fen,move,san,score,result\n"
                "1,1,4k3/8/8/8/8/8/8/4KQ2 w - - 0 1,e1e2,Ke2,100,1-0\n",
                encoding="utf-8",
            )

            rows = load_training_rows(input_path)
            weights = train_weights_from_file(input_path, output_path)

            self.assertEqual(len(rows), 1)
            self.assertTrue(output_path.exists())
            self.assertGreaterEqual(weights.material, 1.0)


if __name__ == "__main__":
    unittest.main()
