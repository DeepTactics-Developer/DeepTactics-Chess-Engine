"""Read and write engine rating snapshots."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from chess_engine.rating.models import EngineRating, MatchResult

DEFAULT_RATING_PATH = Path("artifacts/engine_ratings.json")
DEFAULT_ENGINE_NAME = "DeepTactics Engine"


def load_rating_store(path: str | Path = DEFAULT_RATING_PATH) -> dict:
    rating_path = Path(path)
    if not rating_path.exists():
        return _empty_store()
    store = json.loads(rating_path.read_text(encoding="utf-8-sig"))
    store.setdefault("current_engine", DEFAULT_ENGINE_NAME)
    store.setdefault("ratings", {})
    store.setdefault("history", [])
    store.setdefault("matches", [])
    return store


def _empty_store() -> dict:
    return {"current_engine": DEFAULT_ENGINE_NAME, "ratings": {}, "history": [], "matches": []}


def save_rating_store(store: dict, path: str | Path = DEFAULT_RATING_PATH) -> None:
    rating_path = Path(path)
    rating_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = rating_path.with_name(f"{rating_path.name}.tmp")
    temp_path.write_text(json.dumps(store, indent=2, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(rating_path)


def get_engine_rating(name: str = DEFAULT_ENGINE_NAME, path: str | Path = DEFAULT_RATING_PATH) -> EngineRating:
    store = load_rating_store(path)
    data = store.get("ratings", {}).get(name, {"name": name})
    return EngineRating.from_json(data)


def current_engine_rating_label(path: str | Path = DEFAULT_RATING_PATH) -> str:
    rating = get_engine_rating(DEFAULT_ENGINE_NAME, path)
    if rating.games <= 0:
        return "Elo: unrated | Glicko-1: unrated"
    return f"Elo: {rating.bayes_elo:.0f} | Glicko-1: {rating.glicko_rating:.0f}"


def record_rating_snapshot(
    rating: EngineRating,
    path: str | Path = DEFAULT_RATING_PATH,
    note: str = "",
) -> None:
    store = load_rating_store(path)
    store["current_engine"] = DEFAULT_ENGINE_NAME
    rating = EngineRating(
        name=rating.name,
        bayes_elo=round(rating.bayes_elo, 1),
        glicko_rating=round(rating.glicko_rating, 1),
        glicko_rd=round(rating.glicko_rd, 1),
        games=rating.games,
        version=rating.version,
        updated_at=rating.updated_at or datetime.now().isoformat(timespec="seconds"),
    )
    store.setdefault("ratings", {})[rating.name] = rating.to_json()
    snapshot = rating.to_json()
    snapshot["note"] = note
    store.setdefault("history", []).append(snapshot)
    save_rating_store(store, path)


def append_match_records(matches: list[MatchResult], path: str | Path = DEFAULT_RATING_PATH) -> None:
    store = load_rating_store(path)
    store.setdefault("matches", []).extend(match.to_json() for match in matches)
    save_rating_store(store, path)


def load_match_records(path: str | Path = DEFAULT_RATING_PATH) -> list[MatchResult]:
    store = load_rating_store(path)
    return [MatchResult.from_json(item) for item in store.get("matches", [])]
