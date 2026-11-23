# Contributing to DeepTactics Chess Engine

Thanks for helping improve the engine! Bug reports, ideas and pull requests are all welcome.

## Getting set up

1. Fork and clone the repository.
2. `pip install PySide6` and build the native core with `build_native.bat`.
3. Run the tests: `python -m unittest discover -s chess_engine/tests -t .`

## Search and evaluation changes

Strength changes are accepted on evidence, not intuition:

- Put new search ideas behind an option that defaults to **off**.
- Compare it against the current default with **paired self-play** (same openings, both colours) over a few hundred games or more and report the score and game count.
- Only changes that show a clear, repeatable gain become defaults.

## Pull requests

- Keep changes focused; add or update tests for behaviour changes.
- Do not add third-party code, data or images unless their license allows redistribution under GPL v3, and say where they come from.
- By contributing you agree that your work is released under the project's GPL v3 license.
