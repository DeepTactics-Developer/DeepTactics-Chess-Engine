"""Small BayesElo-style batch estimator.

This is not a full clone of the BayesElo program. It uses the same practical
idea for this project: fit Elo-like Bradley-Terry ratings from game results
while applying a Bayesian prior so small samples do not swing wildly.
"""

from __future__ import annotations

import math

from chess_engine.rating.models import MatchResult


def estimate_bayes_elo(
    matches: list[MatchResult],
    initial: dict[str, float] | None = None,
    prior_rating: float = 1500.0,
    prior_sigma: float = 350.0,
    iterations: int = 200,
    learning_rate: float = 18.0,
) -> dict[str, float]:
    """Return fitted Elo values with a Gaussian prior around ``prior_rating``."""
    names = sorted({match.white for match in matches} | {match.black for match in matches})
    ratings = {name: float((initial or {}).get(name, prior_rating)) for name in names}
    if not matches:
        return ratings

    prior_variance = prior_sigma * prior_sigma
    for _ in range(iterations):
        gradients = {name: -(ratings[name] - prior_rating) / prior_variance for name in names}
        for match in matches:
            expected = _expected_score(ratings[match.white], ratings[match.black])
            error = match.result - expected
            gradients[match.white] += error / 400.0
            gradients[match.black] -= error / 400.0
        for name in names:
            ratings[name] += learning_rate * gradients[name]
    return {name: round(value, 1) for name, value in ratings.items()}


def _expected_score(first: float, second: float) -> float:
    return 1.0 / (1.0 + math.pow(10.0, (second - first) / 400.0))
