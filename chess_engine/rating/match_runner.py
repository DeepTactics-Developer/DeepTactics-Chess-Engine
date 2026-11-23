"""Run lightweight engine-vs-engine rating matches."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from chess_engine.ai.backend import SearchBackend, find_backend_move
from chess_engine.ai.evaluator import (
    EvaluationWeights,
    get_evaluation_weights,
    load_evaluation_weights,
    reset_evaluation_weights,
    set_evaluation_weights,
)
from chess_engine.chess_core.enums import Color, GameStatus
from chess_engine.chess_core.game import Game
from chess_engine.chess_core.pgn import result_from_status
from chess_engine.rating.models import MatchResult


@dataclass(frozen=True, slots=True)
class EngineSpec:
    name: str
    weights_path: str | None = None
    use_weights: bool = True
    backend: SearchBackend = "native"
    depth: int = 2
    time_limit: float | None = None
    adaptive: bool = False

    def to_metadata(self) -> dict[str, str | int | float | bool]:
        return {
            "weights_path": self.weights_path or "",
            "use_weights": self.use_weights,
            "backend": self.backend,
            "depth": self.depth,
            "time_limit": "" if self.time_limit is None else self.time_limit,
            "adaptive": self.adaptive,
        }


def run_rating_match(
    white: EngineSpec,
    black: EngineSpec,
    max_plies: int = 160,
) -> MatchResult:
    """Run one rated game and return the result from White's perspective."""
    original_weights = get_evaluation_weights()
    game = Game.new()
    try:
        for _ in range(max_plies):
            if game.is_over():
                break
            spec = white if game.board.side_to_move is Color.WHITE else black
            _activate_spec_weights(spec)
            result = find_backend_move(
                game.board,
                depth=spec.depth,
                backend=spec.backend,
                time_limit=spec.time_limit,
                adaptive=spec.adaptive,
            )
            if result.move is None:
                break
            game.apply_move(result.move)
        result_text = result_from_status(game.status(), game.board.side_to_move)
        if result_text == "*" or (game.status() is GameStatus.ONGOING):
            result_text = "1/2-1/2"
        return MatchResult(
            white=white.name,
            black=black.name,
            result=_white_score(result_text),
            result_text=result_text,
            pgn=game.to_pgn(),
            played_at=datetime.now().isoformat(timespec="seconds"),
            white_config=white.to_metadata(),
            black_config=black.to_metadata(),
        )
    finally:
        set_evaluation_weights(original_weights)


def _activate_spec_weights(spec: EngineSpec) -> None:
    if not spec.use_weights:
        reset_evaluation_weights()
        return
    if spec.weights_path:
        load_evaluation_weights(spec.weights_path)


def _white_score(result_text: str) -> float:
    if result_text == "1-0":
        return 1.0
    if result_text == "0-1":
        return 0.0
    return 0.5
