"""Core chess rules and game-state package.

Imports are resolved lazily so low-level modules can be used without pulling in
the full rules/PGN stack and creating circular imports with the native bridge.
"""

from __future__ import annotations

_EXPORTS = {
    "Board": ("chess_engine.chess_core.board", "Board"),
    "CastlingRights": ("chess_engine.chess_core.board", "CastlingRights"),
    "CastlingSide": ("chess_engine.chess_core.enums", "CastlingSide"),
    "Color": ("chess_engine.chess_core.enums", "Color"),
    "Game": ("chess_engine.chess_core.game", "Game"),
    "GameStatus": ("chess_engine.chess_core.enums", "GameStatus"),
    "Move": ("chess_engine.chess_core.move", "Move"),
    "PgnGame": ("chess_engine.chess_core.pgn", "PgnGame"),
    "Piece": ("chess_engine.chess_core.piece", "Piece"),
    "PieceType": ("chess_engine.chess_core.enums", "PieceType"),
    "STANDARD_STARTING_FEN": ("chess_engine.chess_core.fen", "STANDARD_STARTING_FEN"),
    "Square": ("chess_engine.chess_core.square", "Square"),
    "board_from_fen": ("chess_engine.chess_core.fen", "board_from_fen"),
    "board_to_fen": ("chess_engine.chess_core.fen", "board_to_fen"),
    "format_san_move_list": ("chess_engine.chess_core.pgn", "format_san_move_list"),
    "move_to_san": ("chess_engine.chess_core.pgn", "move_to_san"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attribute_name = _EXPORTS[name]
    from importlib import import_module

    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
