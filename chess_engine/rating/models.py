"""Rating data structures."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True, slots=True)
class MatchResult:
    """One rated game from White's perspective."""

    white: str
    black: str
    result: float
    result_text: str = "1/2-1/2"
    pgn: str = ""
    played_at: str = ""
    event: str = "DeepTactics Rating"
    white_config: dict[str, str | int | float | bool] = field(default_factory=dict)
    black_config: dict[str, str | int | float | bool] = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "MatchResult":
        return cls(
            white=str(data.get("white", "")),
            black=str(data.get("black", "")),
            result=float(data.get("result", 0.5)),
            result_text=str(data.get("result_text", _result_text_from_score(float(data.get("result", 0.5))))),
            pgn=str(data.get("pgn", "")),
            played_at=str(data.get("played_at", "")),
            event=str(data.get("event", "DeepTactics Rating")),
            white_config=dict(data.get("white_config", {})),
            black_config=dict(data.get("black_config", {})),
        )


@dataclass(frozen=True, slots=True)
class EngineRating:
    """Stored rating values for one engine version."""

    name: str
    bayes_elo: float = 1500.0
    glicko_rating: float = 1500.0
    glicko_rd: float = 350.0
    games: int = 0
    version: str = ""
    updated_at: str = ""

    def to_json(self) -> dict[str, str | float | int]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "EngineRating":
        return cls(
            name=str(data.get("name", "DeepTactics Engine")),
            bayes_elo=float(data.get("bayes_elo", 1500.0)),
            glicko_rating=float(data.get("glicko_rating", 1500.0)),
            glicko_rd=float(data.get("glicko_rd", 350.0)),
            games=int(data.get("games", 0)),
            version=str(data.get("version", "")),
            updated_at=str(data.get("updated_at", "")),
        )


def _result_text_from_score(score: float) -> str:
    if score >= 1.0:
        return "1-0"
    if score <= 0.0:
        return "0-1"
    return "1/2-1/2"
