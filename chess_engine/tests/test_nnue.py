import csv
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chess_engine.chess_core.fen import STANDARD_STARTING_FEN, board_from_fen
from chess_engine.chess_core.move import Move
from chess_engine.nnue import (
    NNUE_FEATURE_COUNT,
    AccumulatorNNUEModel,
    LinearNNUEModel,
    NNUEAccumulatorState,
    blend_evaluations,
    convert_self_play_to_nnue_dataset,
    evaluate_accumulator_nnue,
    evaluate_linear_nnue,
    compare_linear_nnue_to_dataset,
    label_from_training_row,
    normalize_fen,
    nnue_features_from_fen,
    compare_accumulator_nnue_to_dataset,
    save_nnue_compare_report,
    train_accumulator_nnue,
    train_deep_accumulator_nnue_from_linear,
    train_sparse_linear_nnue,
    train_sparse_linear_nnue_shards,
)


class NNUEPreparationTest(unittest.TestCase):
    def test_starting_position_features_are_sparse_and_stable(self):
        features = nnue_features_from_fen(STANDARD_STARTING_FEN)

        self.assertEqual(features.feature_count, NNUE_FEATURE_COUNT)
        self.assertGreater(len(features.active_indices), 32)
        self.assertEqual(features.active_indices, tuple(sorted(set(features.active_indices))))
        self.assertTrue(all(0 <= index < NNUE_FEATURE_COUNT for index in features.active_indices))

    def test_reference_four_field_fen_is_normalized(self):
        self.assertEqual(
            normalize_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"),
            STANDARD_STARTING_FEN,
        )

    def test_label_modes_are_white_perspective_centipawns(self):
        row = {"score": "120", "result": "1-0", "result_value": "1.0"}

        self.assertEqual(label_from_training_row(row, "score"), 120)
        self.assertEqual(label_from_training_row(row, "result"), 600)
        self.assertEqual(label_from_training_row(row, "blended"), 288)

    def test_convert_self_play_rows_to_nnue_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "self_play.csv"
            output_path = Path(directory) / "nnue.csv"
            input_path.write_text(
                "game,ply,side,fen,move,san,score,result,result_value\n"
                f"1,1,w,{STANDARD_STARTING_FEN},e2e4,e4,30,1-0,1.0\n",
                encoding="utf-8",
            )

            stats = convert_self_play_to_nnue_dataset(input_path, output_path, label_mode="blended")

            self.assertEqual(stats.input_rows, 1)
            self.assertEqual(stats.written_rows, 1)
            self.assertEqual(stats.skipped_rows, 0)
            with output_path.open("r", newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["label_mode"], "blended")
            self.assertEqual(int(rows[0]["feature_count"]), NNUE_FEATURE_COUNT)
            self.assertTrue(rows[0]["features"])







    def test_linear_nnue_probe_and_blend(self):
        board = board_from_fen(STANDARD_STARTING_FEN)
        first_feature = nnue_features_from_fen(STANDARD_STARTING_FEN).active_indices[0]
        model = LinearNNUEModel(
            feature_count=NNUE_FEATURE_COUNT,
            bias=10.0,
            weights={first_feature: 15.0},
        )

        self.assertEqual(evaluate_linear_nnue(board, model), 25)
        self.assertEqual(blend_evaluations(100, 0, 0.25), 75)
        self.assertEqual(blend_evaluations(100, None, 0.25), 100)

    def test_accumulator_nnue_binary_round_trip_matches_linear_probe(self):
        board = board_from_fen(STANDARD_STARTING_FEN)
        first_feature = nnue_features_from_fen(STANDARD_STARTING_FEN).active_indices[0]
        linear = LinearNNUEModel(
            feature_count=NNUE_FEATURE_COUNT,
            bias=10.0,
            weights={first_feature: 15.0},
        )
        accumulator = AccumulatorNNUEModel.from_linear(linear)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model_v2.dtnnue"
            accumulator.save_binary(path)
            loaded = AccumulatorNNUEModel.from_binary(path)

        self.assertEqual(evaluate_accumulator_nnue(board, loaded), evaluate_linear_nnue(board, linear))
        self.assertEqual(loaded.hidden_dim, 1)

    def test_accumulator_state_delta_matches_full_recompute_after_move(self):
        board = board_from_fen(STANDARD_STARTING_FEN)
        first_feature = nnue_features_from_fen(STANDARD_STARTING_FEN).active_indices[0]
        linear = LinearNNUEModel(
            feature_count=NNUE_FEATURE_COUNT,
            bias=10.0,
            weights={first_feature: 15.0},
        )
        accumulator = AccumulatorNNUEModel.from_linear(linear)
        state = NNUEAccumulatorState.from_board(board, accumulator)

        move = Move.from_uci("e2e4")
        board.apply_move_unchecked(move)
        state.update_after_move(board, move)

        self.assertEqual(state.evaluate(), evaluate_accumulator_nnue(board, accumulator))

    def test_deep_accumulator_export_creates_multi_hidden_model(self):
        first_feature = nnue_features_from_fen(STANDARD_STARTING_FEN).active_indices[0]
        linear = LinearNNUEModel(
            feature_count=NNUE_FEATURE_COUNT,
            bias=10.0,
            weights={first_feature: 16.0},
        )

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "linear.json"
            output_path = Path(directory) / "deep.dtnnue"
            linear.save_json(input_path)
            model = train_deep_accumulator_nnue_from_linear(input_path, output_path, hidden_dim=4)
            loaded = AccumulatorNNUEModel.from_binary(output_path)

        self.assertEqual(model.hidden_dim, 4)
        self.assertEqual(loaded.hidden_dim, 4)





if __name__ == "__main__":
    unittest.main()
