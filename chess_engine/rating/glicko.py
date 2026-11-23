"""Glicko-1 rating period update."""

from __future__ import annotations

import math

from chess_engine.rating.models import EngineRating, MatchResult

Q = math.log(10) / 400


def update_glicko_ratings(
    current: dict[str, EngineRating],
    matches: list[MatchResult],
    default_rating: float = 1500.0,
    default_rd: float = 350.0,
) -> dict[str, tuple[float, float]]:
    """Return updated ``(rating, rd)`` values for every engine in ``matches``."""
    names = sorted({match.white for match in matches} | {match.black for match in matches})
    grouped = {name: [] for name in names}
    for match in matches:
        grouped[match.white].append((match.black, match.result))
        grouped[match.black].append((match.white, 1.0 - match.result))

    updated: dict[str, tuple[float, float]] = {}
    for name, games in grouped.items():
        rating = current.get(name, EngineRating(name=name, glicko_rating=default_rating)).glicko_rating
        rd = current.get(name, EngineRating(name=name, glicko_rd=default_rd)).glicko_rd
        if not games:
            updated[name] = (rating, min(default_rd, math.sqrt(rd * rd + 50 * 50)))
            continue

        d2_inv = 0.0
        delta_sum = 0.0
        for opponent, score in games:
            opponent_rating = current.get(opponent, EngineRating(name=opponent)).glicko_rating
            opponent_rd = current.get(opponent, EngineRating(name=opponent)).glicko_rd
            g_value = _g(opponent_rd)
            expected = _expected(rating, opponent_rating, opponent_rd)
            d2_inv += Q * Q * g_value * g_value * expected * (1 - expected)
            delta_sum += g_value * (score - expected)

        d2 = 1 / d2_inv if d2_inv else default_rd * default_rd
        new_rd = math.sqrt(1 / ((1 / (rd * rd)) + (1 / d2)))
        new_rating = rating + Q / ((1 / (rd * rd)) + (1 / d2)) * delta_sum
        updated[name] = (round(new_rating, 1), round(new_rd, 1))
    return updated


def _g(rd: float) -> float:
    return 1 / math.sqrt(1 + 3 * Q * Q * rd * rd / (math.pi * math.pi))


def _expected(rating: float, opponent_rating: float, opponent_rd: float) -> float:
    return 1 / (1 + math.pow(10, -_g(opponent_rd) * (rating - opponent_rating) / 400))
