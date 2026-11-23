"""PGN recording and SAN export support."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from chess_engine.chess_core.board import Board
from chess_engine.chess_core.enums import Color, GameStatus, PieceType
from chess_engine.chess_core.move import Move
from chess_engine.chess_core.rules import generate_legal_moves, is_checkmate, is_in_check


PIECE_TO_SAN = {
    PieceType.KING: "K",
    PieceType.QUEEN: "Q",
    PieceType.ROOK: "R",
    PieceType.BISHOP: "B",
    PieceType.KNIGHT: "N",
}


@dataclass(slots=True)
class PgnGame:
    """A minimal PGN export object."""

    san_moves: list[str] = field(default_factory=list)
    tags: dict[str, str] = field(default_factory=dict)
    result: str = "*"

    def to_pgn(self) -> str:
        """Return a PGN string with tags and move text."""
        tag_lines = [
            f'[{name} "{value}"]'
            for name, value in self._ordered_tags().items()
        ]
        move_text = format_san_move_list(self.san_moves, self.result)
        if tag_lines:
            return "\n".join(tag_lines) + "\n\n" + move_text
        return move_text

    def _ordered_tags(self) -> dict[str, str]:
        defaults = {
            "Event": "?",
            "Site": "?",
            "Date": "????.??.??",
            "Round": "?",
            "White": "White",
            "Black": "Black",
            "Result": self.result,
        }
        return {**defaults, **self.tags, "Result": self.result}


def move_to_san(board: Board, move: Move) -> str:
    """Convert a legal move into Standard Algebraic Notation."""
    piece = board.get_piece(move.from_square)
    if piece is None:
        raise ValueError(f"no piece on {move.from_square}")

    if move.is_castling:
        san = "O-O" if move.to_square.file == 6 else "O-O-O"
    elif piece.piece_type is PieceType.PAWN:
        san = _pawn_move_to_san(board, move)
    else:
        san = _piece_move_to_san(board, move, piece.piece_type)

    next_board = board.copy()
    next_board.apply_move_unchecked(move)
    opponent = piece.color.opponent
    if is_checkmate(next_board, opponent):
        return san + "#"
    if is_in_check(next_board, opponent):
        return san + "+"
    return san


def format_san_move_list(san_moves: list[str], result: str = "*") -> str:
    """Format SAN moves as PGN move text."""
    parts: list[str] = []
    move_number = 1

    for index, san in enumerate(san_moves):
        if index % 2 == 0:
            parts.append(f"{move_number}. {san}")
        else:
            parts[-1] += f" {san}"
            move_number += 1

    if result:
        parts.append(result)
    return " ".join(parts)


def game_from_pgn(text: str) -> "Game":
    """Create a ``Game`` from a simple PGN mainline.

    Tags, comments, NAGs, and parenthesized variations are ignored. The parser
    resolves each SAN token by comparing it to legal moves from the current
    board, so special moves remain validated by the engine.
    """
    from chess_engine.chess_core.game import Game
    from chess_engine.chess_core.fen import board_from_fen

    fen = _tag_value(text, "FEN")
    game = Game(board=board_from_fen(fen)) if fen else Game.new()
    for san in pgn_to_san_tokens(text):
        move = move_from_san(game.board, san)
        game.apply_move(move)
    return game


def _tag_value(text: str, name: str) -> str | None:
    pattern = re.compile(rf'^\[{re.escape(name)}\s+"([^"]*)"\]\s*$', re.MULTILINE)
    match = pattern.search(text)
    return match.group(1) if match else None


def pgn_to_san_tokens(text: str) -> list[str]:
    """Return SAN tokens from PGN text, ignoring metadata and annotations."""
    body_lines = [
        line
        for line in text.splitlines()
        if not line.strip().startswith("[")
    ]
    body = "\n".join(body_lines)
    body = re.sub(r"\{[^}]*\}", " ", body)
    body = re.sub(r";[^\n]*", " ", body)
    body = _remove_parenthesized_variations(body)
    tokens: list[str] = []
    for token in body.replace("\n", " ").split():
        cleaned = _clean_pgn_token(token)
        if not cleaned or cleaned in {"1-0", "0-1", "1/2-1/2", "*"}:
            continue
        if cleaned.endswith("."):
            continue
        tokens.append(cleaned)
    return tokens


def move_from_san(board: Board, san: str) -> Move:
    """Resolve SAN text to a legal move on ``board``."""
    target = _normalize_san(san)
    for move in generate_legal_moves(board):
        if _normalize_san(move_to_san(board, move)) == target:
            return move
    raise ValueError(f"illegal or unsupported SAN move: {san!r}")


def result_from_status(status: GameStatus, side_to_move: Color) -> str:
    """Return a PGN result marker for a game status."""
    if status is GameStatus.CHECKMATE:
        return "0-1" if side_to_move is Color.WHITE else "1-0"
    if status in {
        GameStatus.STALEMATE,
        GameStatus.THREEFOLD_REPETITION,
        GameStatus.FIFTY_MOVE_RULE,
        GameStatus.INSUFFICIENT_MATERIAL,
    }:
        return "1/2-1/2"
    return "*"


def _clean_pgn_token(token: str) -> str:
    token = token.strip()
    token = re.sub(r"^\d+\.(\.\.)?", "", token)
    token = re.sub(r"\$\d+", "", token)
    token = token.rstrip("!?")
    return token


def _normalize_san(san: str) -> str:
    san = san.strip()
    san = san.replace("0-0-0", "O-O-O").replace("0-0", "O-O")
    return san.rstrip("+#!?")


def _remove_parenthesized_variations(text: str) -> str:
    depth = 0
    output: list[str] = []
    for char in text:
        if char == "(":
            depth += 1
            output.append(" ")
        elif char == ")":
            depth = max(0, depth - 1)
            output.append(" ")
        elif depth == 0:
            output.append(char)
    return "".join(output)


def _pawn_move_to_san(board: Board, move: Move) -> str:
    capture = _is_capture(board, move)
    san = ""
    if capture:
        san += str(move.from_square)[0] + "x"
    san += str(move.to_square)
    if move.promotion is not None:
        san += f"={PIECE_TO_SAN[move.promotion]}"
    return san


def _piece_move_to_san(board: Board, move: Move, piece_type: PieceType) -> str:
    san = PIECE_TO_SAN[piece_type]
    san += _disambiguation(board, move, piece_type)
    if _is_capture(board, move):
        san += "x"
    san += str(move.to_square)
    return san


def _disambiguation(board: Board, move: Move, piece_type: PieceType) -> str:
    same_piece_moves = [
        legal_move
        for legal_move in generate_legal_moves(board)
        if legal_move.to_square == move.to_square
        and legal_move.from_square != move.from_square
        and (piece := board.get_piece(legal_move.from_square)) is not None
        and piece.piece_type is piece_type
        and piece.color is board.side_to_move
    ]

    if not same_piece_moves:
        return ""

    same_file = any(other.from_square.file == move.from_square.file for other in same_piece_moves)
    same_rank = any(other.from_square.rank == move.from_square.rank for other in same_piece_moves)

    if not same_file:
        return str(move.from_square)[0]
    if not same_rank:
        return str(move.from_square)[1]
    return str(move.from_square)


def _is_capture(board: Board, move: Move) -> bool:
    return move.is_en_passant or board.get_piece(move.to_square) is not None
