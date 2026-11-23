"""Export rating data for external rating tools."""

from __future__ import annotations

import csv
from pathlib import Path

from chess_engine.rating.models import MatchResult
from chess_engine.rating.store import DEFAULT_RATING_PATH, load_match_records


def export_bayeselo_pgn(
    output_path: str | Path,
    store_path: str | Path = DEFAULT_RATING_PATH,
) -> None:
    """Write stored rating games as PGN for external BayesElo workflows."""
    matches = load_match_records(store_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n\n".join(_match_to_pgn(match, index) for index, match in enumerate(matches, 1)), encoding="utf-8")


def export_rating_csv(
    output_path: str | Path,
    store_path: str | Path = DEFAULT_RATING_PATH,
) -> None:
    """Write compact match rows for spreadsheets or custom tools."""
    matches = load_match_records(store_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["white", "black", "result", "result_text", "played_at"])
        writer.writeheader()
        for match in matches:
            writer.writerow(
                {
                    "white": match.white,
                    "black": match.black,
                    "result": match.result,
                    "result_text": match.result_text,
                    "played_at": match.played_at,
                }
            )


def _match_to_pgn(match: MatchResult, index: int) -> str:
    body = _pgn_movetext(match.pgn) or match.result_text
    return "\n".join(
        [
            f'[Event "{match.event}"]',
            '[Site "Local"]',
            f'[Round "{index}"]',
            f'[White "{match.white}"]',
            f'[Black "{match.black}"]',
            f'[Result "{match.result_text}"]',
            "",
            body,
        ]
    )


def _pgn_movetext(pgn: str) -> str:
    lines = [line.strip() for line in pgn.splitlines() if line.strip() and not line.strip().startswith("[")]
    if not lines:
        return ""
    body = " ".join(lines)
    for result in ("1-0", "0-1", "1/2-1/2", "*"):
        if body.endswith(result):
            body = body[: -len(result)].strip()
            break
    return body
