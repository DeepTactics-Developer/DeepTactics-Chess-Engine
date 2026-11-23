"""FEN import and export support."""

from __future__ import annotations

from chess_engine.chess_core.board import Board, CastlingRights
from chess_engine.chess_core.enums import Color, PieceType
from chess_engine.chess_core.piece import Piece
from chess_engine.chess_core.square import BOARD_SIZE, Square


STANDARD_STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

FEN_TO_PIECE_TYPE = {
    "p": PieceType.PAWN,
    "n": PieceType.KNIGHT,
    "b": PieceType.BISHOP,
    "r": PieceType.ROOK,
    "q": PieceType.QUEEN,
    "k": PieceType.KING,
}


def board_to_fen(board: Board) -> str:
    """Serialize a board into a full FEN string."""
    return " ".join(
        [
            _piece_placement_to_fen(board),
            _side_to_move_to_fen(board.side_to_move),
            _castling_rights_to_fen(board.castling_rights),
            _en_passant_to_fen(board.en_passant_target),
            str(board.halfmove_clock),
            str(board.fullmove_number),
        ]
    )


def board_from_fen(fen: str) -> Board:
    """Create a board from a full FEN string."""
    fields = fen.split()
    if len(fields) != 6:
        raise ValueError("FEN must contain 6 fields")

    board = Board.empty()
    _load_piece_placement(board, fields[0])
    board.side_to_move = _side_to_move_from_fen(fields[1])
    board.castling_rights = _castling_rights_from_fen(fields[2])
    board.en_passant_target = _en_passant_from_fen(fields[3])
    board.halfmove_clock = _non_negative_int(fields[4], "halfmove clock")
    board.fullmove_number = _positive_int(fields[5], "fullmove number")
    return board


def _piece_placement_to_fen(board: Board) -> str:
    ranks: list[str] = []

    for rank in range(BOARD_SIZE - 1, -1, -1):
        empty_count = 0
        parts: list[str] = []

        for file in range(BOARD_SIZE):
            piece = board.get_piece(Square(file, rank))
            if piece is None:
                empty_count += 1
                continue

            if empty_count:
                parts.append(str(empty_count))
                empty_count = 0
            parts.append(piece.fen_symbol)

        if empty_count:
            parts.append(str(empty_count))

        ranks.append("".join(parts))

    return "/".join(ranks)


def _load_piece_placement(board: Board, placement: str) -> None:
    ranks = placement.split("/")
    if len(ranks) != BOARD_SIZE:
        raise ValueError("FEN piece placement must contain 8 ranks")

    for fen_rank_index, rank_text in enumerate(ranks):
        board_rank = BOARD_SIZE - 1 - fen_rank_index
        file = 0

        for char in rank_text:
            if char.isdigit():
                file += _empty_count(char)
                continue

            if file >= BOARD_SIZE:
                raise ValueError(f"too many squares in rank: {rank_text!r}")

            board.set_piece(Square(file, board_rank), _piece_from_fen_symbol(char))
            file += 1

        if file != BOARD_SIZE:
            raise ValueError(f"rank does not contain 8 squares: {rank_text!r}")


def _piece_from_fen_symbol(symbol: str) -> Piece:
    piece_type = FEN_TO_PIECE_TYPE.get(symbol.lower())
    if piece_type is None:
        raise ValueError(f"invalid FEN piece symbol: {symbol!r}")

    color = Color.WHITE if symbol.isupper() else Color.BLACK
    return Piece(color, piece_type)


def _side_to_move_to_fen(color: Color) -> str:
    return "w" if color is Color.WHITE else "b"


def _side_to_move_from_fen(value: str) -> Color:
    if value == "w":
        return Color.WHITE
    if value == "b":
        return Color.BLACK
    raise ValueError(f"invalid FEN side to move: {value!r}")


def _castling_rights_to_fen(castling_rights: CastlingRights) -> str:
    text = ""
    if castling_rights.white_king_side:
        text += "K"
    if castling_rights.white_queen_side:
        text += "Q"
    if castling_rights.black_king_side:
        text += "k"
    if castling_rights.black_queen_side:
        text += "q"
    return text or "-"


def _castling_rights_from_fen(value: str) -> CastlingRights:
    if value == "-":
        return CastlingRights(False, False, False, False)

    allowed = set("KQkq")
    if any(char not in allowed for char in value) or len(set(value)) != len(value):
        raise ValueError(f"invalid FEN castling rights: {value!r}")

    return CastlingRights(
        white_king_side="K" in value,
        white_queen_side="Q" in value,
        black_king_side="k" in value,
        black_queen_side="q" in value,
    )


def _en_passant_to_fen(square: Square | None) -> str:
    return "-" if square is None else square.to_algebraic()


def _en_passant_from_fen(value: str) -> Square | None:
    if value == "-":
        return None

    square = Square.from_algebraic(value)
    if square.rank not in (2, 5):
        raise ValueError(f"invalid FEN en passant target rank: {value!r}")
    return square


def _empty_count(value: str) -> int:
    count = int(value)
    if not 1 <= count <= BOARD_SIZE:
        raise ValueError(f"invalid empty square count: {value!r}")
    return count


def _non_negative_int(value: str, name: str) -> int:
    number = int(value)
    if number < 0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _positive_int(value: str, name: str) -> int:
    number = int(value)
    if number < 1:
        raise ValueError(f"{name} must be positive")
    return number
