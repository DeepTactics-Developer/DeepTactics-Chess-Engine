"""Console entry point for the chess engine project."""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter, sleep

from chess_engine.ai.evaluator import (
    EvaluationWeights,
    evaluate_board,
    evaluate_breakdown,
    evaluate_weighted_breakdown,
    get_evaluation_weights,
    linear_nnue_status,
    load_accumulator_nnue_model,
    load_evaluation_weights,
    load_linear_nnue_model,
    native_evaluator_available,
    native_evaluator_enabled,
    native_evaluator_status,
    reset_evaluation_weights,
    set_evaluation_weights,
    score_for_color,
    use_native_evaluator,
)
from chess_engine.ai.backend import find_backend_move
from chess_engine.analysis import analyze_game_moves
from chess_engine.chess_core.enums import Color, GameStatus
from chess_engine.chess_core.fen import board_from_fen
from chess_engine.chess_core.game import Game
from chess_engine.chess_core.move import Move
from chess_engine.chess_core.pgn import game_from_pgn, result_from_status
from chess_engine.chess_core.rules import native_attacks_enabled, use_native_attacks
from chess_engine.learning import build_training_report, load_training_rows, train_weights_from_file
from chess_engine.nnue import (
    compare_linear_nnue_to_dataset,
    convert_self_play_to_nnue_dataset,
    save_nnue_compare_report,
    train_sparse_linear_nnue,
)
from chess_engine.openings import convert_opening_sources_to_tsv
from chess_engine.self_play import (
    SelfPlayProgress,
    deduplicate_training_rows,
    run_self_play_game,
    run_self_play_games,
    save_self_play_results,
)
from chess_engine.native.native_eval import (
    NATIVE_LIBRARY_PATH,
    evaluate_native,
    evaluate_native_breakdown,
    find_best_move_native,
    DEFAULT_FORCED_LINE_EXTENSION_BUDGET,
    DEFAULT_MAJOR_PIECE_SAFETY_ORDERING,
    native_available,
    native_hash_status,
    set_native_search_experiments,
)
from chess_engine.uci import UCIEngine, run_uci_loop

DEFAULT_WEIGHTS_PATH = Path("artifacts/evaluation_weights.json")


def _add_native_experiment_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--native-forced-extension-budget",
        type=int,
        choices=(0, 1, 2),
        default=DEFAULT_FORCED_LINE_EXTENSION_BUDGET,
        help=(
            "native forced-line horizon extension budget; default "
            f"{DEFAULT_FORCED_LINE_EXTENSION_BUDGET}, use 0 to disable"
        ),
    )
    parser.add_argument(
        "--native-mate-danger-ordering",
        action="store_true",
        help="experimental native immediate-mate/check move ordering; default off",
    )
    parser.add_argument(
        "--native-defensive-mate-threat-ordering",
        action="store_true",
        help="experimental native ordering for moves that reduce immediate opponent check/mate threats; default off",
    )
    parser.add_argument(
        "--native-major-piece-safety-ordering",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_MAJOR_PIECE_SAFETY_ORDERING,
        help=(
            "native ordering penalty for hanging queen/rook moves; default "
            f"{str(DEFAULT_MAJOR_PIECE_SAFETY_ORDERING).lower()}"
        ),
    )
    parser.add_argument(
        "--native-internal-iterative-deepening",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="timed native iterative deepening with shared TT/history; default on",
    )
    parser.add_argument(
        "--native-clustered-tt",
        action="store_true",
        help="experimental native two-way clustered TT replacement; default off",
    )
    parser.add_argument(
        "--native-aspiration-windows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="aspiration windows for native internal iterative deepening; default on",
    )
    parser.add_argument(
        "--native-pawn-endgame-extension-budget",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help="experimental pure-pawn endgame frontier extension budget; default off",
    )
    parser.add_argument(
        "--native-check-extension-budget",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help="experimental native checking-move extension budget; default off",
    )
    parser.add_argument(
        "--native-singular-extension",
        action="store_true",
        help="experimental native singular extension (extend a TT move no other move can match); default off",
    )
    parser.add_argument(
        "--native-futility-pruning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "native futility pruning (reverse + move-level, low depth only); default on "
            "(promoted 2026-08-30: 51.56%% score over 1600 paired self-play games, z~2.3)"
        ),
    )
    parser.add_argument(
        "--native-lmr-formula",
        action="store_true",
        help="experimental native LMR reduction scaling with log(depth)*log(index) instead of the flat 1-ply default; default off",
    )
    parser.add_argument(
        "--native-razoring",
        action="store_true",
        help="experimental native razoring (shallow-depth alpha cutoff verified by quiescence search); default off",
    )
    parser.add_argument(
        "--native-continuation-history",
        action="store_true",
        help=(
            "experimental native continuation history (decaying move-ordering score indexed by "
            "the previous move and the candidate reply, both by piece+destination); default off"
        ),
    )
    parser.add_argument(
        "--native-probcut",
        action="store_true",
        help=(
            "experimental native ProbCut (depth>=5 cutoff verified by a real reduced-depth search "
            "of tactical moves at a widened beta window); default off"
        ),
    )


def _configure_native_search_experiments(args: argparse.Namespace) -> dict[str, object]:
    config = {
        "forced_line_extension_budget": getattr(
            args,
            "native_forced_extension_budget",
            DEFAULT_FORCED_LINE_EXTENSION_BUDGET,
        ),
        "mate_danger_ordering": getattr(args, "native_mate_danger_ordering", False),
        "defensive_mate_threat_ordering": getattr(args, "native_defensive_mate_threat_ordering", False),
        "major_piece_safety_ordering": getattr(
            args,
            "native_major_piece_safety_ordering",
            DEFAULT_MAJOR_PIECE_SAFETY_ORDERING,
        ),
        "internal_iterative_deepening": getattr(args, "native_internal_iterative_deepening", True),
        "clustered_tt_replacement": getattr(args, "native_clustered_tt", False),
        "aspiration_windows": getattr(args, "native_aspiration_windows", True),
        "pawn_endgame_extension_budget": getattr(args, "native_pawn_endgame_extension_budget", 0),
        "check_extension_budget": getattr(args, "native_check_extension_budget", 0),
        "singular_extension": getattr(args, "native_singular_extension", False),
        "futility_pruning": getattr(args, "native_futility_pruning", True),
        "lmr_formula": getattr(args, "native_lmr_formula", False),
        "razoring": getattr(args, "native_razoring", False),
        "continuation_history": getattr(args, "native_continuation_history", False),
        "probcut": getattr(args, "native_probcut", False),
    }
    config["native_setter_available"] = set_native_search_experiments(
        forced_line_extension_budget=int(config["forced_line_extension_budget"]),
        mate_danger_ordering=bool(config["mate_danger_ordering"]),
        defensive_mate_threat_ordering=bool(config["defensive_mate_threat_ordering"]),
        major_piece_safety_ordering=bool(config["major_piece_safety_ordering"]),
        internal_iterative_deepening=bool(config["internal_iterative_deepening"]),
        clustered_tt_replacement=bool(config["clustered_tt_replacement"]),
        aspiration_windows=bool(config["aspiration_windows"]),
        pawn_endgame_extension_budget=int(config["pawn_endgame_extension_budget"]),
        check_extension_budget=int(config["check_extension_budget"]),
        singular_extension=bool(config["singular_extension"]),
        futility_pruning=bool(config["futility_pruning"]),
        lmr_formula=bool(config["lmr_formula"]),
        razoring=bool(config["razoring"]),
        continuation_history=bool(config["continuation_history"]),
        probcut=bool(config["probcut"]),
    )
    return config


def main(argv: list[str] | None = None) -> None:
    """Run the console application."""
    args = _parse_args(argv)
    if args.command == "self-play":
        _run_self_play_command(args)
        return
    if args.command == "learn":
        _run_learn_command(args)
        return
    if args.command == "train-cycle":
        _run_train_cycle_command(args)
        return
    if args.command == "train-cycle-fast":
        _run_train_cycle_fast_command(args)
        return
    if args.command == "compare-weights":
        _run_compare_weights_command(args)
        return
    if args.command == "training-report":
        _run_training_report_command(args)
        return
    if args.command == "nnue-build-dataset":
        _run_nnue_build_dataset_command(args)
        return
    if args.command == "nnue-train-linear":
        _run_nnue_train_linear_command(args)
        return
    if args.command == "nnue-compare-linear":
        _run_nnue_compare_linear_command(args)
        return
    if args.command == "analyze":
        _run_analyze_command(args)
        return
    if args.command == "import-openings":
        _run_import_openings_command(args)
        return
    if args.command == "native-status":
        _run_native_status_command()
        return
    if args.command == "uci":
        _run_uci_command(args)
        return

    _load_runtime_weights(args)
    _load_runtime_linear_nnue(args)

    from chess_engine.ui.qt_ui import run_qt_game

    run_qt_game()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Python chess engine.")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser(
        "native-status",
        help="report whether the native DLL is current with its C++ source",
    )

    self_play_parser = subparsers.add_parser(
        "self-play",
        help="run AI-vs-AI games and optionally save training rows",
    )
    self_play_parser.add_argument("--games", type=int, default=1, help="number of games")
    self_play_parser.add_argument("--depth", type=int, default=1, help="AI search depth")
    self_play_parser.add_argument("--max-plies", type=int, default=120, help="maximum plies per game")
    self_play_parser.add_argument("--time", type=float, default=None, help="AI thinking time per move")
    self_play_parser.add_argument("--adaptive", action="store_true", help="use adaptive bounded search")
    self_play_parser.add_argument(
        "--search-backend",
        choices=("native",),
        default="native",
        help="AI search backend",
    )
    self_play_parser.add_argument("--no-book", action="store_true", help="disable opening book moves")
    self_play_parser.add_argument("--opening-plies", type=int, default=8, help="max opening book plies")
    self_play_parser.add_argument("--exploration-cp", type=int, default=0, help="self-play only: randomly choose among moves within this centipawn window")
    self_play_parser.add_argument("--exploration-plies", type=int, default=24, help="self-play only: apply exploration up to this ply")
    self_play_parser.add_argument("--seed", type=int, default=None, help="optional seed for reproducible self-play")
    self_play_parser.add_argument("--verbose", action="store_true", help="print each self-play move as it finishes")
    self_play_parser.add_argument("--output", help="optional JSON or CSV output path")
    self_play_parser.add_argument("--weights", help="optional evaluation weights JSON to use")
    self_play_parser.add_argument("--no-weights", action="store_true", help="disable automatic latest weights loading")
    self_play_parser.add_argument(
        "--adjudication-threshold",
        type=int,
        default=250,
        help="centipawn lead used to assign a training result when max plies is reached; -1 disables",
    )
    self_play_parser.add_argument(
        "--adjudication-min-ply",
        type=int,
        default=40,
        help="minimum ply before self-play can stop early on a clear evaluation lead",
    )
    self_play_parser.add_argument(
        "--format",
        choices=("json", "csv"),
        help="output format; defaults to file extension",
    )

    learn_parser = subparsers.add_parser(
        "learn",
        help="train evaluation weights from saved self-play rows",
    )
    learn_parser.add_argument("--input", required=True, help="self-play JSON or CSV input")
    learn_parser.add_argument("--output", required=True, help="weights JSON output")
    learn_parser.add_argument(
        "--target-mode",
        choices=("result", "move-loss", "blended"),
        default="result",
        help="training target: final game result, move loss, or an experimental blend",
    )

    train_cycle_parser = subparsers.add_parser(
        "train-cycle",
        help="run self-play data generation and train evaluation weights",
    )
    train_cycle_parser.add_argument("--games", type=int, default=10, help="number of self-play games")
    train_cycle_parser.add_argument("--depth", type=int, default=1, help="AI search depth")
    train_cycle_parser.add_argument("--max-plies", type=int, default=120, help="maximum plies per game")
    train_cycle_parser.add_argument("--time", type=float, default=None, help="AI thinking time per move")
    train_cycle_parser.add_argument("--adaptive", action="store_true", help="use adaptive bounded search")
    train_cycle_parser.add_argument(
        "--search-backend",
        choices=("native",),
        default="native",
        help="AI search backend",
    )
    train_cycle_parser.add_argument("--no-book", action="store_true", help="disable opening book moves")
    train_cycle_parser.add_argument("--opening-plies", type=int, default=8, help="max opening book plies")
    train_cycle_parser.add_argument("--exploration-cp", type=int, default=25, help="self-play only: randomly choose among moves within this centipawn window")
    train_cycle_parser.add_argument("--exploration-plies", type=int, default=24, help="self-play only: apply exploration up to this ply")
    train_cycle_parser.add_argument("--seed", type=int, default=None, help="optional seed for reproducible self-play")
    train_cycle_parser.add_argument("--verbose", action="store_true", help="print each self-play move as it finishes")
    train_cycle_parser.add_argument("--input-weights", help="optional existing weights JSON to start from; use 'latest' for artifacts/evaluation_weights.json")
    train_cycle_parser.add_argument("--no-weights", action="store_true", help="disable automatic latest weights loading before self-play")
    train_cycle_parser.add_argument(
        "--adjudication-threshold",
        type=int,
        default=250,
        help="centipawn lead used to assign a training result when max plies is reached; -1 disables",
    )
    train_cycle_parser.add_argument(
        "--adjudication-min-ply",
        type=int,
        default=40,
        help="minimum ply before self-play can stop early on a clear evaluation lead",
    )
    train_cycle_parser.add_argument(
        "--data-output",
        default="artifacts/self_play_cycle.csv",
        help="cumulative CSV or JSON self-play training rows output",
    )
    train_cycle_parser.add_argument(
        "--weights-output",
        default="artifacts/evaluation_weights.json",
        help="trained evaluation weights JSON output",
    )
    train_cycle_parser.add_argument(
        "--format",
        choices=("json", "csv"),
        help="training data output format; defaults to file extension",
    )
    train_cycle_parser.add_argument(
        "--compare-games",
        type=int,
        default=2,
        help="comparison games after training; use 0 to skip",
    )
    train_cycle_parser.add_argument(
        "--reset-training-data",
        action="store_true",
        help="start a fresh cumulative data file instead of appending existing rows",
    )
    train_cycle_parser.add_argument(
        "--max-position-duplicates",
        type=int,
        default=2,
        help="maximum cumulative rows to keep for the same normalized position",
    )

    fast_cycle_parser = subparsers.add_parser(
        "train-cycle-fast",
        help="run parallel time-limited self-play and train evaluation weights as games finish",
    )
    fast_cycle_parser.add_argument("--games", type=int, default=100, help="number of self-play games")
    fast_cycle_parser.add_argument("--time", type=float, default=0.15, help="thinking time per move")
    fast_cycle_parser.add_argument("--max-depth", type=int, default=64, help="safety depth cap; time is the practical limit")
    fast_cycle_parser.add_argument("--max-plies", type=int, default=160, help="maximum plies per game")
    fast_cycle_parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1), help="parallel worker processes")
    fast_cycle_parser.add_argument("--search-backend", choices=("native",), default="native", help="AI search backend")
    fast_cycle_parser.add_argument("--no-book", action="store_true", help="disable opening book moves")
    fast_cycle_parser.add_argument("--opening-plies", type=int, default=8, help="max opening book plies")
    fast_cycle_parser.add_argument("--exploration-cp", type=int, default=25, help="near-equal move exploration window")
    fast_cycle_parser.add_argument("--exploration-plies", type=int, default=24, help="apply exploration up to this ply")
    fast_cycle_parser.add_argument("--seed", type=int, default=None, help="optional seed")
    fast_cycle_parser.add_argument("--verbose", action="store_true", help="print each finished game")
    fast_cycle_parser.add_argument("--input-weights", help="optional existing weights JSON; use 'latest' for artifacts/evaluation_weights.json")
    fast_cycle_parser.add_argument("--no-weights", action="store_true", help="disable latest weights loading before self-play")
    fast_cycle_parser.add_argument("--adjudication-threshold", type=int, default=250, help="cp lead for adjudication; -1 disables")
    fast_cycle_parser.add_argument("--adjudication-min-ply", type=int, default=40, help="minimum ply before adjudication")
    fast_cycle_parser.add_argument("--data-output", default="artifacts/self_play_cycle.csv", help="cumulative training rows")
    fast_cycle_parser.add_argument("--weights-output", default="artifacts/evaluation_weights.json", help="trained weights JSON")
    fast_cycle_parser.add_argument("--format", choices=("json", "csv"), help="training data output format")
    fast_cycle_parser.add_argument("--compare-games", type=int, default=0, help="comparison games after training")
    fast_cycle_parser.add_argument("--reset-training-data", action="store_true", help="start a fresh cumulative data file")
    fast_cycle_parser.add_argument("--max-position-duplicates", type=int, default=2, help="max rows for the same normalized position")
    fast_cycle_parser.add_argument(
        "--retrain-every-n-games",
        type=int,
        default=20,
        help="retrain weights from the full cumulative file every N completed games (always retrains on the last game); retraining cost grows with cumulative file size, so retraining after every single game gets expensive on large datasets",
    )

    compare_parser = subparsers.add_parser(
        "compare-weights",
        help="pit candidate evaluation weights against baseline weights",
    )
    compare_parser.add_argument("--candidate", required=True, help="candidate weights JSON; use 'latest' for artifacts/evaluation_weights.json")
    compare_parser.add_argument("--base", help="baseline weights JSON; defaults to handcrafted weights; use 'latest' for artifacts/evaluation_weights.json")
    compare_parser.add_argument("--games", type=int, default=10, help="number of comparison games")
    compare_parser.add_argument("--depth", type=int, default=1, help="AI search depth")
    compare_parser.add_argument("--max-plies", type=int, default=120, help="maximum plies per game")
    compare_parser.add_argument("--time", type=float, default=None, help="AI thinking time per move")
    compare_parser.add_argument("--adaptive", action="store_true", help="use adaptive bounded search")
    compare_parser.add_argument(
        "--search-backend",
        choices=("native",),
        default="native",
        help="search backend for comparison games",
    )
    compare_parser.add_argument("--seed", type=int, default=None, help="reserved for reproducible future options")
    compare_parser.add_argument("--verbose", action="store_true", help="print each comparison game")
    compare_parser.add_argument("--output", help="optional comparison summary JSON or CSV output path")
    compare_parser.add_argument(
        "--format",
        choices=("json", "csv"),
        help="comparison output format; defaults to file extension",
    )

    report_parser = subparsers.add_parser(
        "training-report",
        help="print a quality summary for self-play training rows",
    )
    report_parser.add_argument("--input", default="artifacts/self_play_cycle.csv", help="CSV or JSON training rows")

    nnue_dataset_parser = subparsers.add_parser(
        "nnue-build-dataset",
        help="convert self-play rows into a sparse NNUE training dataset",
    )
    nnue_dataset_parser.add_argument("--input", default="artifacts/self_play_cycle.csv", help="self-play CSV or JSON input")
    nnue_dataset_parser.add_argument("--output", default="artifacts/nnue_dataset.csv", help="NNUE CSV or JSONL output")
    nnue_dataset_parser.add_argument(
        "--label",
        choices=("blended", "score", "result"),
        default="blended",
        help="target label source: blended search/result, raw score, or final result",
    )
    nnue_dataset_parser.add_argument(
        "--format",
        choices=("csv", "jsonl"),
        help="output format; defaults to file extension",
    )
    nnue_dataset_parser.add_argument("--max-rows", type=int, default=None, help="optional row limit for quick experiments")
    nnue_dataset_parser.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="keep repeated normalized positions instead of capping duplicates",
    )
    nnue_dataset_parser.add_argument(
        "--max-position-duplicates",
        type=int,
        default=2,
        help="maximum rows to keep per normalized position before NNUE export",
    )

    nnue_train_parser = subparsers.add_parser(
        "nnue-train-linear",
        help="train or continue the first sparse linear NNUE probe",
    )
    nnue_train_parser.add_argument("--input", nargs="+", required=True, help="one or more NNUE CSV/JSONL dataset files")
    nnue_train_parser.add_argument("--output", default="artifacts/nnue_linear_model.json", help="model JSON output")
    nnue_train_parser.add_argument("--input-model", help="existing model JSON to continue training from")
    nnue_train_parser.add_argument("--epochs", type=int, default=1, help="training epochs")
    nnue_train_parser.add_argument("--learning-rate", type=float, default=0.001, help="SGD learning rate")
    nnue_train_parser.add_argument("--l2", type=float, default=0.000001, help="L2 decay")
    nnue_train_parser.add_argument("--max-rows", type=int, default=None, help="optional row limit for this run")
    nnue_train_parser.add_argument(
        "--reference-weight",
        type=float,
        default=1.0,
        help="sample weight for externally labelled reference rows",
    )
    nnue_train_parser.add_argument(
        "--self-play-weight",
        type=float,
        default=0.35,
        help="sample weight for self-play rows",
    )
    nnue_train_parser.add_argument("--progress-every", type=int, default=100000, help="print progress every N trained rows")

    nnue_compare_parser = subparsers.add_parser(
        "nnue-compare-linear",
        help="compare a sparse linear NNUE probe against labelled dataset rows",
    )
    nnue_compare_parser.add_argument("--input", nargs="+", required=True, help="one or more NNUE CSV/JSONL dataset files")
    nnue_compare_parser.add_argument("--model", required=True, help="sparse linear NNUE probe JSON")
    nnue_compare_parser.add_argument("--max-rows", type=int, default=None, help="optional row limit for quick reports")
    nnue_compare_parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="skip handcrafted/native baseline comparison and report only NNUE error",
    )
    nnue_compare_parser.add_argument("--output", help="optional JSON or CSV report output")
    nnue_compare_parser.add_argument("--format", choices=("json", "csv"), help="report output format")

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="analyze a PGN mainline and print move classifications",
    )
    analyze_parser.add_argument("--input", required=True, help="PGN input path")
    analyze_parser.add_argument("--depth", type=int, default=1, help="analysis depth")
    analyze_parser.add_argument("--time", type=float, default=None, help="optional time per position")
    analyze_parser.add_argument("--adaptive", action="store_true", help="use adaptive bounded search")
    analyze_parser.add_argument(
        "--search-backend",
        choices=("native",),
        default="native",
        help="analysis search backend",
    )
    analyze_parser.add_argument("--weights", help="optional evaluation weights JSON to use")
    analyze_parser.add_argument("--no-weights", action="store_true", help="disable automatic latest weights loading")

    import_openings_parser = subparsers.add_parser(
        "import-openings",
        help="convert ECO/opening TSV data into the local opening book format",
    )
    import_openings_parser.add_argument("--input", nargs="+", required=True, help="one or more source TSV paths")
    import_openings_parser.add_argument(
        "--output",
        default="assets/openings/openings.tsv",
        help="local opening TSV output path",
    )

    uci_parser = subparsers.add_parser(
        "uci",
        help="run DeepTactics Engine as a basic UCI-compatible engine",
    )
    uci_parser.add_argument("--depth", type=int, default=2, help="default UCI search depth")
    uci_parser.add_argument("--time", type=float, default=None, help="optional default thinking time per move")
    uci_parser.add_argument("--adaptive", action="store_true", help="use adaptive search for timed UCI go commands")
    uci_parser.add_argument("--quiescence-depth", type=int, default=4, help="native/Python UCI tactical leaf depth")
    uci_parser.add_argument("--hash-mb", type=int, default=16, help="native transposition-table capacity in MB")
    uci_parser.add_argument(
        "--search-backend",
        choices=("native",),
        default="native",
        help="UCI search backend",
    )
    uci_parser.add_argument("--weights", help="optional evaluation weights JSON to use")
    uci_parser.add_argument("--no-weights", action="store_true", help="disable automatic latest weights loading")
    uci_parser.add_argument("--nnue-linear-model", help="optional sparse linear NNUE probe JSON")
    uci_parser.add_argument(
        "--nnue-mode",
        choices=("off", "blend", "linear"),
        default="off",
        help="how to apply --nnue-linear-model in UCI mode",
    )
    uci_parser.add_argument("--nnue-blend", type=float, default=0.35, help="blend weight for UCI NNUE blend mode")
    uci_parser.add_argument(
        "--engine-mode",
        choices=("Pure Engine", "Practical Play", "Analysis", "Development Audit"),
        default="Pure Engine",
        help=(
            "declared runtime profile for diagnostics; default keeps pure engine comparison semantics"
        ),
    )
    uci_parser.add_argument(
        "--use-opening-book",
        action="store_true",
        help="allow explicit runtime opening-book moves outside Pure Engine mode; default off",
    )
    uci_parser.add_argument(
        "--opening-book-path",
        default="assets/openings/openings.tsv",
        help="TSV opening book used only when --use-opening-book is enabled",
    )
    uci_parser.add_argument(
        "--opening-book-max-ply",
        type=int,
        default=2,
        help="maximum ply for explicit runtime opening-book moves",
    )
    uci_parser.add_argument(
        "--use-experience-book",
        action="store_true",
        help="allow explicit JSON experience-book moves outside Pure Engine mode; default off",
    )
    uci_parser.add_argument(
        "--experience-book-path",
        default="artifacts/experience_book.json",
        help="JSON experience book used only when --use-experience-book is enabled",
    )
    uci_parser.add_argument(
        "--use-tablebase",
        action="store_true",
        help="allow optional Syzygy root tablebase moves outside Pure Engine mode; default off",
    )
    uci_parser.add_argument(
        "--syzygy-path",
        default="",
        help="directory containing Syzygy tablebase files used only when --use-tablebase is enabled",
    )
    uci_parser.add_argument(
        "--tablebase-max-pieces",
        type=int,
        default=7,
        help="maximum piece count for optional Syzygy root probing",
    )
    uci_parser.add_argument(
        "--time-manager",
        choices=("Legacy", "Conservative", "Balanced", "Aggressive"),
        default="Balanced",
        help=(
            "UCI clock budget policy; default Balanced (promoted 2026-09: 52.17%% score "
            "over 600 paired self-play games, z~2.26); Legacy preserves the original formula"
        ),
    )
    _add_native_experiment_arguments(uci_parser)

    parser.add_argument(
        "--ai",
        choices=("none", "white", "black"),
        default="none",
        help="choose a side for the AI",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        help="AI search depth, or max depth when --time is used",
    )
    parser.add_argument(
        "--time",
        type=float,
        default=None,
        help="AI thinking time per move in seconds; enables iterative deepening",
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="let the AI choose a bounded time budget per position",
    )
    parser.add_argument(
        "--search-backend",
        choices=("native",),
        default="native",
        help="AI search backend",
    )
    parser.add_argument(
        "--weights",
        help="optional evaluation weights JSON to use; use 'latest' for artifacts/evaluation_weights.json",
    )
    parser.add_argument(
        "--no-weights",
        action="store_true",
        help="disable automatic latest weights loading",
    )
    parser.add_argument(
        "--nnue-linear-model",
        help="optional sparse linear NNUE probe JSON to load for evaluation experiments",
    )
    parser.add_argument(
        "--nnue-mode",
        choices=("off", "blend", "linear"),
        default="off",
        help="how to apply --nnue-linear-model; blend is recommended for experiments",
    )
    parser.add_argument(
        "--nnue-blend",
        type=float,
        default=0.35,
        help="blend weight when --nnue-mode blend is used",
    )
    return parser.parse_args(argv)


def _run_self_play_command(args: argparse.Namespace) -> None:
    _load_runtime_weights(args)

    if args.output:
        _run_self_play_command_incremental(args)
        return

    results = run_self_play_games(
        game_count=args.games,
        depth=args.depth,
        max_plies=args.max_plies,
        time_limit=args.time,
        adaptive=args.adaptive,
        opening_book_path=None if args.no_book else "assets/openings/openings.tsv",
        opening_plies=args.opening_plies,
        seed=args.seed,
        search_backend=args.search_backend,
        progress_callback=_print_self_play_progress if args.verbose else None,
        adjudication_threshold=_parse_adjudication_threshold(args.adjudication_threshold),
        adjudication_min_ply=args.adjudication_min_ply,
        exploration_cp=args.exploration_cp,
        exploration_plies=args.exploration_plies,
    )
    total_plies = sum(result.plies_played for result in results)

    if args.output:
        save_self_play_results(results, args.output, args.format)
        print(f"Saved {total_plies} training rows to {args.output}")
    else:
        for index, result in enumerate(results, start=1):
            print(
                f"Game {index}: result={result.result}, "
                f"plies={result.plies_played}, limit={result.reached_move_limit}"
            )
    print(_format_self_play_summary(results))


def _run_self_play_command_incremental(args: argparse.Namespace) -> None:
    results = []
    try:
        for game_index in range(1, args.games + 1):
            result = run_self_play_game(
                depth=args.depth,
                max_plies=args.max_plies,
                time_limit=args.time,
                adaptive=args.adaptive,
                opening_book_path=None if args.no_book else "assets/openings/openings.tsv",
                opening_plies=args.opening_plies,
                seed=None if args.seed is None else args.seed + game_index - 1,
                game_index=game_index,
                search_backend=args.search_backend,
                progress_callback=_print_self_play_progress if args.verbose else None,
                adjudication_threshold=_parse_adjudication_threshold(args.adjudication_threshold),
                adjudication_min_ply=args.adjudication_min_ply,
                exploration_cp=args.exploration_cp,
                exploration_plies=args.exploration_plies,
            )
            results.append(result)
            _save_self_play_results_atomic(results, args.output, args.format)
            print(
                f"Game {game_index}: result={result.result}, "
                f"plies={result.plies_played}, saved={args.output}"
            )
        print(_format_self_play_summary(results))
    except KeyboardInterrupt:
        print(f"\nStopped. Saved {len(results)} completed game(s) to {args.output}.")
        if results:
            print(_format_self_play_summary(results))


def _save_self_play_results_atomic(results, output_path: str, output_format: str | None) -> None:
    path = Path(output_path)
    temp_path = path.with_name(f"{path.name}.tmp")
    selected_format = output_format or path.suffix.lower().lstrip(".")
    save_self_play_results(results, temp_path, selected_format)
    temp_path.replace(path)


def _format_self_play_summary(results) -> str:
    games = len(results)
    if games == 0:
        return "Summary: games=0"
    decisive = sum(1 for result in results if result.result in {"1-0", "0-1"})
    draws = sum(1 for result in results if result.result == "1/2-1/2")
    unfinished = sum(1 for result in results if result.result == "*")
    move_limits = sum(1 for result in results if result.reached_move_limit)
    total_plies = sum(result.plies_played for result in results)
    average_plies = total_plies / games
    decisive_rate = decisive / games * 100
    draw_rate = draws / games * 100
    limit_rate = move_limits / games * 100
    return (
        "Summary: "
        f"games={games}, decisive={decisive} ({decisive_rate:.1f}%), "
        f"draws={draws} ({draw_rate:.1f}%), unfinished={unfinished}, "
        f"move_limits={move_limits} ({limit_rate:.1f}%), "
        f"avg_plies={average_plies:.1f}"
    )


def _parse_adjudication_threshold(value: int) -> int | None:
    if value < 0:
        return None
    return value


def _run_learn_command(args: argparse.Namespace) -> None:
    weights = train_weights_from_file(args.input, args.output, target_mode=args.target_mode)
    print(f"Saved evaluation weights to {args.output}: {weights}")


























def _run_train_cycle_command(args: argparse.Namespace) -> None:
    if args.input_weights:
        weights = load_evaluation_weights(_resolve_weights_path(args.input_weights))
        print(f"Loaded evaluation weights: {weights}")
    else:
        _load_runtime_weights(args)
    baseline_weights = get_evaluation_weights()
    data_path = Path(args.data_output)
    selected_format = args.format or data_path.suffix.lower().lstrip(".")
    cumulative_rows = [] if args.reset_training_data else _load_existing_training_rows(data_path)
    if cumulative_rows:
        print(f"Loaded existing training rows: {len(cumulative_rows)} from {data_path}")
    elif args.reset_training_data:
        print(f"Reset cumulative training data: {data_path}")

    results = []
    next_game_index = _next_training_game_index(cumulative_rows) + 1
    try:
        for game_index in range(1, args.games + 1):
            absolute_game_index = next_game_index
            next_game_index += 1
            result = run_self_play_game(
                depth=args.depth,
                max_plies=args.max_plies,
                time_limit=args.time,
                adaptive=args.adaptive,
                opening_book_path=None if args.no_book else "assets/openings/openings.tsv",
                opening_plies=args.opening_plies,
                seed=None if args.seed is None else args.seed + game_index - 1,
                game_index=absolute_game_index,
                search_backend=args.search_backend,
                progress_callback=_print_self_play_progress if args.verbose else None,
                adjudication_threshold=_parse_adjudication_threshold(args.adjudication_threshold),
                adjudication_min_ply=args.adjudication_min_ply,
                weights_label=_weights_metadata_label(args.weights_output),
                exploration_cp=args.exploration_cp,
                exploration_plies=args.exploration_plies,
            )
            results.append(result)
            cumulative_rows.extend(_training_rows_for_result(result, absolute_game_index))
            before_filter = len(cumulative_rows)
            cumulative_rows = deduplicate_training_rows(cumulative_rows, args.max_position_duplicates)
            _save_training_rows_atomic(cumulative_rows, data_path, selected_format)
            weights = train_weights_from_file(data_path, args.weights_output)
            set_evaluation_weights(weights)
            print(
                f"Game {absolute_game_index}: result={result.result}, "
                f"plies={result.plies_played}, saved={data_path}"
            )
            removed = before_filter - len(cumulative_rows)
            suffix = f", filtered={removed}" if removed else ""
            print(f"Updated evaluation weights: {args.weights_output}{suffix}")
    except KeyboardInterrupt:
        print(f"\nStopped. Saved {len(results)} completed new game(s) to {data_path}.")

    if not results:
        print("No completed games. Skipped weight training.")
        return

    print(_format_self_play_summary(results))
    weights = train_weights_from_file(data_path, args.weights_output)
    set_evaluation_weights(weights)
    print(f"Saved evaluation weights to {args.weights_output}: {weights}")
    if args.compare_games > 0:
        comparison = _compare_weights(
            base_weights=baseline_weights,
            candidate_weights=weights,
            games=args.compare_games,
            depth=args.depth,
            max_plies=args.max_plies,
            time_limit=args.time,
            adaptive=args.adaptive,
            search_backend=args.search_backend,
        )
        print("Post-training comparison against previous weights:")
        _print_compare_summary(comparison)


def _run_train_cycle_fast_command(args: argparse.Namespace) -> None:
    if args.games < 1:
        raise ValueError("games must be at least 1")
    if args.time <= 0:
        raise ValueError("time must be positive")
    if args.max_depth < 1:
        raise ValueError("max-depth must be at least 1")
    if args.workers < 1:
        raise ValueError("workers must be at least 1")
    if args.retrain_every_n_games < 1:
        raise ValueError("retrain-every-n-games must be at least 1")

    if args.input_weights:
        weights_path = str(_resolve_weights_path(args.input_weights))
        weights = load_evaluation_weights(weights_path)
        print(f"Loaded evaluation weights: {weights}")
    else:
        _load_runtime_weights(args)
        weights_path = str(_resolve_weights_path("latest")) if DEFAULT_WEIGHTS_PATH.exists() and not args.no_weights else ""
    baseline_weights = get_evaluation_weights()
    data_path = Path(args.data_output)
    selected_format = args.format or data_path.suffix.lower().lstrip(".")
    cumulative_rows = [] if args.reset_training_data else _load_existing_training_rows(data_path)
    if cumulative_rows:
        print(f"Loaded existing training rows: {len(cumulative_rows)} from {data_path}")
    elif args.reset_training_data:
        print(f"Reset cumulative training data: {data_path}")

    next_game_index = _next_training_game_index(cumulative_rows) + 1
    completed_results = []
    worker_count = min(args.workers, args.games)
    print(
        "Fast train cycle:",
        f"games={args.games}",
        f"workers={worker_count}",
        f"time={args.time:.3f}s",
        f"max_depth={args.max_depth}",
        f"backend={args.search_backend}",
    )
    try:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {}
            for offset in range(args.games):
                absolute_game_index = next_game_index + offset
                seed = None if args.seed is None else args.seed + offset
                futures[
                    executor.submit(
                        _run_fast_cycle_worker,
                        absolute_game_index,
                        seed,
                        args.max_depth,
                        args.max_plies,
                        args.time,
                        args.search_backend,
                        None if args.no_book else "assets/openings/openings.tsv",
                        args.opening_plies,
                        _parse_adjudication_threshold(args.adjudication_threshold),
                        args.adjudication_min_ply,
                        _weights_metadata_label(args.weights_output),
                        args.exploration_cp,
                        args.exploration_plies,
                        weights_path,
                    )
                ] = absolute_game_index
            for future in as_completed(futures):
                absolute_game_index = futures[future]
                result = future.result()
                completed_results.append(result)
                cumulative_rows.extend(_training_rows_for_result(result, absolute_game_index))
                before_filter = len(cumulative_rows)
                cumulative_rows = deduplicate_training_rows(cumulative_rows, args.max_position_duplicates)
                _save_training_rows_atomic(cumulative_rows, data_path, selected_format)
                removed = before_filter - len(cumulative_rows)
                suffix = f", filtered={removed}" if removed else ""
                retrained = len(completed_results) % args.retrain_every_n_games == 0 or len(completed_results) == args.games
                if retrained:
                    weights = train_weights_from_file(data_path, args.weights_output)
                    set_evaluation_weights(weights)
                print(
                    f"Game {absolute_game_index}: result={result.result}, "
                    f"plies={result.plies_played}, saved={data_path}{suffix}"
                    + (" [retrained]" if retrained else ""),
                    flush=True,
                )
                if args.verbose:
                    print(f"  pgn={result.pgn.splitlines()[-1] if result.pgn else '-'}", flush=True)
    except KeyboardInterrupt:
        print(f"\nStopped. Saved {len(completed_results)} completed new game(s) to {data_path}.")

    if not completed_results:
        print("No completed games. Skipped weight training.")
        return

    print(_format_self_play_summary(completed_results))
    weights = train_weights_from_file(data_path, args.weights_output)
    set_evaluation_weights(weights)
    print(f"Saved evaluation weights to {args.weights_output}: {weights}")
    if args.compare_games > 0:
        comparison = _compare_weights(
            base_weights=baseline_weights,
            candidate_weights=weights,
            games=args.compare_games,
            depth=min(args.max_depth, 8),
            max_plies=args.max_plies,
            time_limit=args.time,
            adaptive=False,
            search_backend=args.search_backend,
        )
        print("Post-training comparison against previous weights:")
        _print_compare_summary(comparison)


def _run_fast_cycle_worker(
    game_index: int,
    seed: int | None,
    depth: int,
    max_plies: int,
    time_limit: float,
    search_backend: str,
    opening_book_path: str | None,
    opening_plies: int,
    adjudication_threshold: int | None,
    adjudication_min_ply: int,
    weights_label: str,
    exploration_cp: int,
    exploration_plies: int,
    weights_path: str,
):
    if weights_path:
        load_evaluation_weights(weights_path)
    return run_self_play_game(
        depth=depth,
        max_plies=max_plies,
        time_limit=time_limit,
        adaptive=False,
        opening_book_path=opening_book_path,
        opening_plies=opening_plies,
        seed=seed,
        game_index=game_index,
        search_backend=search_backend,
        progress_callback=None,
        adjudication_threshold=adjudication_threshold,
        adjudication_min_ply=adjudication_min_ply,
        weights_label=weights_label,
        exploration_cp=exploration_cp,
        exploration_plies=exploration_plies,
    )


def _load_existing_training_rows(path: Path) -> list[dict[str, str | int | float]]:
    if not path.exists():
        return []
    return [dict(row) for row in load_training_rows(path)]


def _next_training_game_index(rows: list[dict[str, str | int | float]]) -> int:
    max_game = 0
    for row in rows:
        try:
            max_game = max(max_game, int(row.get("game", 0)))
        except (TypeError, ValueError):
            continue
    return max_game


def _training_rows_for_result(result, game_index: int) -> list[dict[str, str | int]]:
    return [{"game": game_index, **row} for row in result.to_training_rows()]


def _save_training_rows_atomic(rows: list[dict[str, str | int | float]], path: Path, selected_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    if selected_format == "json":
        temp_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    elif selected_format == "csv":
        fieldnames = [
            "game",
            "ply",
            "side",
            "fen",
            "move",
            "san",
            "score",
            "source",
            "fen_after",
            "search_backend",
            "depth",
            "time_limit",
            "adaptive",
            "weights",
            "result",
            "result_value",
        ]
        with temp_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    else:
        raise ValueError("training data format must be 'json' or 'csv'")
    _replace_with_retry(temp_path, path)


def _replace_with_retry(temp_path: Path, path: Path, attempts: int = 5, delay: float = 0.2) -> None:
    """Retry the atomic rename a few times to ride out transient Windows file locks
    (antivirus/indexer/OneDrive briefly holding the destination open)."""
    for attempt in range(attempts):
        try:
            temp_path.replace(path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            sleep(delay)


def _weights_metadata_label(path: str | Path) -> str:
    return Path(path).name


def _print_self_play_progress(progress: SelfPlayProgress) -> None:
    print(
        f"  game {progress.game_index} ply {progress.ply}: "
        f"{progress.san} ({progress.move_uci}) score={progress.score} source={progress.source}",
        flush=True,
    )


def _run_compare_weights_command(args: argparse.Namespace) -> None:
    base_weights = (
        load_evaluation_weights(_resolve_weights_path(args.base), apply=False)
        if args.base
        else EvaluationWeights()
    )
    candidate_weights = load_evaluation_weights(_resolve_weights_path(args.candidate), apply=False)
    results = _compare_weights(
        base_weights=base_weights,
        candidate_weights=candidate_weights,
        games=args.games,
        depth=args.depth,
        max_plies=args.max_plies,
        time_limit=args.time,
        adaptive=args.adaptive,
        search_backend=args.search_backend,
        opening_lines=None,
        verbose=args.verbose,
    )
    _print_compare_summary(results)
    if args.output:
        _save_compare_results(results, args.output, args.format)
        print(f"Saved comparison results to {args.output}")


def _run_training_report_command(args: argparse.Namespace) -> None:
    report = build_training_report(args.input)
    for line in report.lines():
        print(line)






def _run_nnue_build_dataset_command(args: argparse.Namespace) -> None:
    stats = convert_self_play_to_nnue_dataset(
        args.input,
        args.output,
        label_mode=args.label,
        output_format=args.format,
        max_rows=args.max_rows,
        deduplicate=not args.no_deduplicate,
        max_position_duplicates=args.max_position_duplicates,
    )
    print("NNUE dataset:")
    for line in stats.lines():
        print(f"  {line}")






def _run_nnue_train_linear_command(args: argparse.Namespace) -> None:
    for input_path in args.input:
        if not Path(input_path).exists():
            raise FileNotFoundError(f"NNUE training input not found: {input_path}")
    if args.input_model and not Path(args.input_model).exists():
        raise FileNotFoundError(f"NNUE input model not found: {args.input_model}")
    stats = train_sparse_linear_nnue(
        args.input,
        args.output,
        input_model_path=args.input_model,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        max_rows=args.max_rows,
        reference_weight=args.reference_weight,
        self_play_weight=args.self_play_weight,
        progress_every=args.progress_every,
    )
    print("NNUE sparse linear training:")
    for line in stats.lines():
        print(f"  {line}")






def _run_nnue_compare_linear_command(args: argparse.Namespace) -> None:
    for input_path in args.input:
        if not Path(input_path).exists():
            raise FileNotFoundError(f"NNUE comparison input not found: {input_path}")
    if not Path(args.model).exists():
        raise FileNotFoundError(f"NNUE model not found: {args.model}")
    report = compare_linear_nnue_to_dataset(
        args.input,
        args.model,
        max_rows=args.max_rows,
        include_baseline=not args.no_baseline,
    )
    print("NNUE sparse linear comparison:")
    for line in report.lines():
        print(f"  {line}")
    if args.output:
        save_nnue_compare_report(report, args.output, args.format)
        print(f"Saved NNUE comparison report to {args.output}")










def _run_uci_command(args: argparse.Namespace) -> None:
    loaded_weights = _load_runtime_weights(args, quiet=True)
    _load_runtime_linear_nnue(args, quiet=True)
    run_uci_loop(
        depth=args.depth,
        backend=args.search_backend,
        time_limit=args.time,
        adaptive=args.adaptive,
        quiescence_depth=args.quiescence_depth,
        weights_path=(
            None
            if loaded_weights is None
            else (_resolve_weights_path(args.weights) if args.weights else DEFAULT_WEIGHTS_PATH)
        ),
        nnue_model_path=getattr(args, "nnue_linear_model", None),
        nnue_mode=getattr(args, "nnue_mode", "off"),
        nnue_blend=getattr(args, "nnue_blend", 0.35),
        native_forced_extension_budget=getattr(
            args,
            "native_forced_extension_budget",
            DEFAULT_FORCED_LINE_EXTENSION_BUDGET,
        ),
        native_mate_danger_ordering=getattr(args, "native_mate_danger_ordering", False),
        native_defensive_mate_threat_ordering=getattr(args, "native_defensive_mate_threat_ordering", False),
        native_major_piece_safety_ordering=getattr(
            args,
            "native_major_piece_safety_ordering",
            DEFAULT_MAJOR_PIECE_SAFETY_ORDERING,
        ),
        native_internal_iterative_deepening=getattr(args, "native_internal_iterative_deepening", True),
        native_clustered_tt=getattr(args, "native_clustered_tt", False),
        native_aspiration_windows=getattr(args, "native_aspiration_windows", True),
        native_pawn_endgame_extension_budget=getattr(args, "native_pawn_endgame_extension_budget", 0),
        native_check_extension_budget=getattr(args, "native_check_extension_budget", 0),
        native_singular_extension=getattr(args, "native_singular_extension", False),
        native_futility_pruning=getattr(args, "native_futility_pruning", True),
        native_lmr_formula=getattr(args, "native_lmr_formula", False),
        native_razoring=getattr(args, "native_razoring", False),
        native_continuation_history=getattr(args, "native_continuation_history", False),
        native_probcut=getattr(args, "native_probcut", False),
        hash_mb=max(1, min(4096, getattr(args, "hash_mb", 16))),
        engine_mode=getattr(args, "engine_mode", "Pure Engine"),
        use_opening_book=getattr(args, "use_opening_book", False),
        opening_book_path=getattr(args, "opening_book_path", "assets/openings/openings.tsv"),
        opening_book_max_ply=getattr(args, "opening_book_max_ply", 2),
        use_experience_book=getattr(args, "use_experience_book", False),
        experience_book_path=getattr(args, "experience_book_path", "artifacts/experience_book.json"),
        use_tablebase=getattr(args, "use_tablebase", False),
        syzygy_path=getattr(args, "syzygy_path", ""),
        tablebase_max_pieces=getattr(args, "tablebase_max_pieces", 7),
        time_manager=getattr(args, "time_manager", "Balanced"),
    )






def _run_native_status_command() -> None:
    source_path = Path("chess_engine/native/native_eval.cpp")
    dll_path = NATIVE_LIBRARY_PATH
    source_time = source_path.stat().st_mtime if source_path.exists() else None
    dll_time = dll_path.stat().st_mtime if dll_path.exists() else None
    older = source_time is not None and dll_time is not None and dll_time < source_time
    print("Native runtime status:")
    print(f"  source: {source_path} modified={source_time if source_time is not None else 'missing'}")
    print(f"  dll: {dll_path} modified={dll_time if dll_time is not None else 'missing'}")
    print(f"  native_available: {native_available()}")
    print(f"  native_enabled: {native_evaluator_enabled()}")
    print(f"  native_hash: {native_hash_status()}")
    if not dll_path.exists():
        print("  status: DLL missing, rebuild required")
    elif older:
        print("  status: DLL older than source, rebuild recommended")
    else:
        print("  status: DLL is current with source timestamp")






def _compare_weights(
    base_weights: EvaluationWeights,
    candidate_weights: EvaluationWeights,
    games: int,
    depth: int,
    max_plies: int,
    time_limit: float | None,
    adaptive: bool,
    search_backend: str = "native",
    opening_lines: list[tuple[str, ...]] | None = None,
    verbose: bool = False,
) -> dict[str, float | int]:
    if games < 1:
        raise ValueError("games must be at least 1")
    if depth < 1:
        raise ValueError("depth must be at least 1")
    if max_plies < 1:
        raise ValueError("max_plies must be at least 1")

    candidate_wins = 0
    candidate_losses = 0
    draws = 0
    total_plies = 0

    try:
        for game_index in range(1, games + 1):
            candidate_color = Color.WHITE if game_index % 2 == 1 else Color.BLACK
            opening_moves = (
                opening_lines[((game_index - 1) // 2) % len(opening_lines)]
                if opening_lines
                else ()
            )
            result, plies = _run_weight_comparison_game(
                base_weights=base_weights,
                candidate_weights=candidate_weights,
                candidate_color=candidate_color,
                depth=depth,
                max_plies=max_plies,
                time_limit=time_limit,
                adaptive=adaptive,
                search_backend=search_backend,
                opening_moves=opening_moves,
            )
            total_plies += plies
            if result == "1/2-1/2" or result == "*":
                draws += 1
                outcome = "draw"
            elif (result == "1-0" and candidate_color is Color.WHITE) or (
                result == "0-1" and candidate_color is Color.BLACK
            ):
                candidate_wins += 1
                outcome = "candidate win"
            else:
                candidate_losses += 1
                outcome = "candidate loss"
            if verbose:
                print(
                    f"Game {game_index}: candidate={candidate_color.value}, "
                    f"result={result}, plies={plies}, {outcome}",
                    flush=True,
                )
    finally:
        reset_evaluation_weights()

    candidate_score = candidate_wins + draws * 0.5
    score_rate = candidate_score / games
    return {
        "candidate_wins": candidate_wins,
        "candidate_losses": candidate_losses,
        "draws": draws,
        "games": games,
        "candidate_score": candidate_score,
        "score_rate": score_rate,
        "win_rate": candidate_wins / games,
        "draw_rate": draws / games,
        "loss_rate": candidate_losses / games,
        "average_plies": total_plies / games,
        "opening_lines": 0 if opening_lines is None else len(opening_lines),
    }


def _print_compare_summary(results: dict[str, float | int]) -> None:
    games = int(results["games"])
    candidate_wins = int(results["candidate_wins"])
    draws = int(results["draws"])
    candidate_losses = int(results["candidate_losses"])
    candidate_score = float(results["candidate_score"])
    score_rate = float(results["score_rate"])
    verdict = _compare_verdict(score_rate)
    print("Candidate results:")
    print(f"  Record: +{candidate_wins} ={draws} -{candidate_losses} ({games} games)")
    print(f"  Score: {candidate_score:.1f}/{games} ({score_rate * 100:.1f}%)")
    print(
        "  Rates: "
        f"win {float(results['win_rate']) * 100:.1f}% / "
        f"draw {float(results['draw_rate']) * 100:.1f}% / "
        f"loss {float(results['loss_rate']) * 100:.1f}%"
    )
    print(f"  Average plies: {float(results['average_plies']):.1f}")
    print(f"  Verdict: {verdict}")


def _compare_verdict(score_rate: float) -> str:
    if score_rate >= 0.6:
        return "candidate looks stronger"
    if score_rate <= 0.4:
        return "candidate looks weaker"
    return "candidate is roughly even"


def _run_weight_comparison_game(
    base_weights: EvaluationWeights,
    candidate_weights: EvaluationWeights,
    candidate_color: Color,
    depth: int,
    max_plies: int,
    time_limit: float | None,
    adaptive: bool,
    search_backend: str = "native",
    opening_moves: tuple[str, ...] = (),
) -> tuple[str, int]:
    game = Game.new()
    for move_uci in opening_moves:
        game.apply_move(Move.from_uci(move_uci))
    for _ in range(max_plies):
        if game.is_over():
            break
        weights = candidate_weights if game.board.side_to_move is candidate_color else base_weights
        set_evaluation_weights(weights)
        result = _search_comparison_move(game, depth, time_limit, adaptive, search_backend)
        if result.move is None:
            break
        game.apply_move(result.move)

    if game.status() is GameStatus.ONGOING and len(game.move_history) >= max_plies:
        return "*", len(game.move_history)
    return result_from_status(game.status(), game.board.side_to_move), len(game.move_history)


def _search_comparison_move(game: Game, depth: int, time_limit: float | None, adaptive: bool, search_backend: str):
    return find_backend_move(
        game.board,
        depth=depth,
        backend=search_backend,
        time_limit=time_limit,
        adaptive=adaptive,
    )


def _run_analyze_command(args: argparse.Namespace) -> None:
    _load_runtime_weights(args)
    _load_runtime_linear_nnue(args)
    game = game_from_pgn(Path(args.input).read_text(encoding="utf-8"))
    report = analyze_game_moves(
        game,
        depth=args.depth,
        time_limit=args.time,
        adaptive=args.adaptive,
        backend=args.search_backend,
    )
    print(f"White accuracy: {report.white_accuracy:.1f}")
    print(f"Black accuracy: {report.black_accuracy:.1f}")
    for move in report.moves:
        print(
            f"{move.ply}. {move.san}: {move.classification.label} "
            f"({move.classification.loss_cp}cp), best={move.best_move_san or '-'}"
        )


def _run_import_openings_command(args: argparse.Namespace) -> None:
    count = convert_opening_sources_to_tsv(args.input, args.output)
    print(f"Imported {count} opening row(s) to {args.output}")
















def _load_runtime_weights(args: argparse.Namespace, *, quiet: bool = False) -> EvaluationWeights | None:
    if getattr(args, "no_weights", False):
        reset_evaluation_weights()
        if not quiet:
            print("Evaluation weights disabled; using handcrafted defaults.")
        return None

    selected = getattr(args, "weights", None)
    path = _resolve_weights_path(selected) if selected else DEFAULT_WEIGHTS_PATH
    if not path.exists():
        if selected:
            raise FileNotFoundError(f"evaluation weights not found: {path}")
        return None

    weights = load_evaluation_weights(path)
    if not quiet:
        print(f"Loaded evaluation weights: {weights}")
    return weights


def _load_runtime_linear_nnue(args: argparse.Namespace, *, quiet: bool = False) -> None:
    model_path = getattr(args, "nnue_linear_model", None)
    if not model_path or getattr(args, "nnue_mode", "off") == "off":
        return
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"linear NNUE model not found: {path}")
    if path.suffix.lower() == ".dtnnue":
        load_accumulator_nnue_model(path, mode=args.nnue_mode, blend_weight=args.nnue_blend)
    else:
        load_linear_nnue_model(path, mode=args.nnue_mode, blend_weight=args.nnue_blend)
    if not quiet:
        print(f"Loaded NNUE probe: {linear_nnue_status()}")




def _resolve_weights_path(value: str, latest_path: Path = DEFAULT_WEIGHTS_PATH) -> Path:
    if value == "latest":
        return latest_path
    return Path(value)


def _save_compare_results(
    results: dict[str, float | int],
    output_path: str,
    output_format: str | None,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    selected_format = output_format or path.suffix.lower().lstrip(".")
    if selected_format == "json":
        path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        return
    if selected_format == "csv":
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(results.keys()))
            writer.writeheader()
            writer.writerow(results)
        return
    raise ValueError("comparison output format must be 'json' or 'csv'")


def _parse_ai_color(value: str) -> Color | None:
    if value == "white":
        return Color.WHITE
    if value == "black":
        return Color.BLACK
    return None


if __name__ == "__main__":
    main()
