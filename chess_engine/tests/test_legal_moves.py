import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chess_engine.chess_core.board import Board
from chess_engine.chess_core.board import CastlingRights
from chess_engine.chess_core.enums import Color, PieceType
from chess_engine.chess_core.move import Move
from chess_engine.chess_core.piece import Piece
from chess_engine.chess_core.rules import (
    _is_square_attacked_python,
    _generate_pseudo_legal_moves_python,
    _generate_legal_moves_python,
    find_king,
    generate_legal_moves,
    is_in_check,
    is_square_attacked,
    native_attacks_enabled,
    use_native_attacks,
)
from chess_engine.chess_core.square import Square
from chess_engine.native.native_eval import generate_legal_moves_native, generate_pseudo_legal_moves_native, is_square_attacked_native, native_available


def uci_moves(moves):
    return {move.to_uci() for move in moves}


def move_signatures(moves):
    return {
        (move.to_uci(), move.is_castling, move.is_en_passant)
        for move in moves
    }


class LegalMovesTest(unittest.TestCase):
    def test_find_king_returns_king_square(self):
        board = Board.empty()
        board.set_piece(Square.from_algebraic("e1"), Piece(Color.WHITE, PieceType.KING))

        self.assertEqual(find_king(board, Color.WHITE), Square.from_algebraic("e1"))

    def test_is_square_attacked_uses_pawn_attacks_not_pawn_forward_moves(self):
        board = Board.empty()
        board.set_piece(Square.from_algebraic("e5"), Piece(Color.BLACK, PieceType.PAWN))

        self.assertTrue(is_square_attacked(board, Square.from_algebraic("d4"), Color.BLACK))
        self.assertTrue(is_square_attacked(board, Square.from_algebraic("f4"), Color.BLACK))
        self.assertFalse(is_square_attacked(board, Square.from_algebraic("e4"), Color.BLACK))

    def test_rook_attack_detects_check(self):
        board = Board.empty()
        board.set_piece(Square.from_algebraic("e1"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("e8"), Piece(Color.BLACK, PieceType.ROOK))

        self.assertTrue(is_in_check(board, Color.WHITE))

    def test_blocked_rook_does_not_give_check(self):
        board = Board.empty()
        board.set_piece(Square.from_algebraic("e1"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("e4"), Piece(Color.WHITE, PieceType.BISHOP))
        board.set_piece(Square.from_algebraic("e8"), Piece(Color.BLACK, PieceType.ROOK))

        self.assertFalse(is_in_check(board, Color.WHITE))

    def test_legal_moves_do_not_leave_own_king_in_check(self):
        board = Board.empty()
        board.set_piece(Square.from_algebraic("e1"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("e2"), Piece(Color.WHITE, PieceType.ROOK))
        board.set_piece(Square.from_algebraic("e8"), Piece(Color.BLACK, PieceType.ROOK))
        board.set_piece(Square.from_algebraic("a8"), Piece(Color.BLACK, PieceType.KING))

        moves = uci_moves(generate_legal_moves(board, Color.WHITE))

        self.assertNotIn("e2d2", moves)
        self.assertNotIn("e2a2", moves)
        self.assertIn("e2e8", moves)

    def test_legal_moves_must_answer_current_check(self):
        board = Board.empty()
        board.set_piece(Square.from_algebraic("e1"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KNIGHT))
        board.set_piece(Square.from_algebraic("e8"), Piece(Color.BLACK, PieceType.ROOK))
        board.set_piece(Square.from_algebraic("a8"), Piece(Color.BLACK, PieceType.KING))

        moves = uci_moves(generate_legal_moves(board, Color.WHITE))

        self.assertNotIn("g1h3", moves)
        self.assertNotIn("g1f3", moves)

    def test_king_cannot_move_onto_attacked_square(self):
        board = Board.empty()
        board.set_piece(Square.from_algebraic("e1"), Piece(Color.WHITE, PieceType.KING))
        board.set_piece(Square.from_algebraic("e8"), Piece(Color.BLACK, PieceType.ROOK))
        board.set_piece(Square.from_algebraic("a8"), Piece(Color.BLACK, PieceType.KING))

        moves = uci_moves(generate_legal_moves(board, Color.WHITE))

        self.assertNotIn("e1e2", moves)
        self.assertIn("e1d1", moves)

    def test_starting_position_legal_moves_match_pseudo_legal_count(self):
        board = Board.starting_position()

        moves = generate_legal_moves(board, Color.WHITE)

        self.assertEqual(len(moves), 20)

    def test_native_attack_detection_matches_python_for_representative_positions(self):
        if not native_available():
            self.skipTest("native evaluator DLL is not available")
        positions = [Board.starting_position(), Board.empty()]
        custom = Board.empty()
        custom.set_piece(Square.from_algebraic("e1"), Piece(Color.WHITE, PieceType.KING))
        custom.set_piece(Square.from_algebraic("g2"), Piece(Color.WHITE, PieceType.BISHOP))
        custom.set_piece(Square.from_algebraic("b7"), Piece(Color.WHITE, PieceType.KNIGHT))
        custom.set_piece(Square.from_algebraic("e8"), Piece(Color.BLACK, PieceType.KING))
        custom.set_piece(Square.from_algebraic("e5"), Piece(Color.BLACK, PieceType.PAWN))
        custom.set_piece(Square.from_algebraic("a8"), Piece(Color.BLACK, PieceType.ROOK))
        custom.set_piece(Square.from_algebraic("h4"), Piece(Color.BLACK, PieceType.QUEEN))
        positions.append(custom)

        for board in positions:
            for file in range(8):
                for rank in range(8):
                    square = Square(file, rank)
                    for color in (Color.WHITE, Color.BLACK):
                        self.assertEqual(
                            is_square_attacked_native(board, file, rank, color),
                            _is_square_attacked_python(board, square, color),
                        )

    def test_native_attack_switch_preserves_legal_moves(self):
        if not native_available():
            self.skipTest("native evaluator DLL is not available")
        board = Board.starting_position()
        previous = native_attacks_enabled()
        try:
            use_native_attacks(False)
            python_moves = uci_moves(generate_legal_moves(board, Color.WHITE))
            use_native_attacks(True)
            native_moves = uci_moves(generate_legal_moves(board, Color.WHITE))
        finally:
            use_native_attacks(previous)

        self.assertEqual(native_moves, python_moves)

    def test_native_pseudo_legal_moves_match_python_for_representative_positions(self):
        if not native_available():
            self.skipTest("native evaluator DLL is not available")
        positions: list[tuple[Board, Color]] = [(Board.starting_position(), Color.WHITE), (Board.starting_position(), Color.BLACK)]

        castling = Board.empty()
        castling.castling_rights = CastlingRights(True, True, True, True)
        castling.set_piece(Square.from_algebraic("e1"), Piece(Color.WHITE, PieceType.KING))
        castling.set_piece(Square.from_algebraic("a1"), Piece(Color.WHITE, PieceType.ROOK))
        castling.set_piece(Square.from_algebraic("h1"), Piece(Color.WHITE, PieceType.ROOK))
        castling.set_piece(Square.from_algebraic("e8"), Piece(Color.BLACK, PieceType.KING))
        castling.set_piece(Square.from_algebraic("a8"), Piece(Color.BLACK, PieceType.ROOK))
        castling.set_piece(Square.from_algebraic("h8"), Piece(Color.BLACK, PieceType.ROOK))
        positions.extend([(castling, Color.WHITE), (castling, Color.BLACK)])

        special = Board.empty()
        special.set_piece(Square.from_algebraic("e1"), Piece(Color.WHITE, PieceType.KING))
        special.set_piece(Square.from_algebraic("a7"), Piece(Color.WHITE, PieceType.PAWN))
        special.set_piece(Square.from_algebraic("e5"), Piece(Color.WHITE, PieceType.PAWN))
        special.set_piece(Square.from_algebraic("e8"), Piece(Color.BLACK, PieceType.KING))
        special.set_piece(Square.from_algebraic("d7"), Piece(Color.BLACK, PieceType.PAWN))
        special.apply_move_unchecked(Move.from_uci("d7d5"))
        positions.extend([(special, Color.WHITE), (special, Color.BLACK)])

        for board, color in positions:
            self.assertEqual(
                move_signatures(generate_pseudo_legal_moves_native(board, color) or []),
                move_signatures(_generate_pseudo_legal_moves_python(board, color)),
            )

    def test_native_legal_moves_match_python_for_representative_positions(self):
        if not native_available():
            self.skipTest("native evaluator DLL is not available")
        positions: list[tuple[Board, Color]] = [(Board.starting_position(), Color.WHITE), (Board.starting_position(), Color.BLACK)]

        pinned = Board.empty()
        pinned.set_piece(Square.from_algebraic("e1"), Piece(Color.WHITE, PieceType.KING))
        pinned.set_piece(Square.from_algebraic("e2"), Piece(Color.WHITE, PieceType.ROOK))
        pinned.set_piece(Square.from_algebraic("e8"), Piece(Color.BLACK, PieceType.ROOK))
        pinned.set_piece(Square.from_algebraic("a8"), Piece(Color.BLACK, PieceType.KING))
        positions.append((pinned, Color.WHITE))

        checked = Board.empty()
        checked.set_piece(Square.from_algebraic("e1"), Piece(Color.WHITE, PieceType.KING))
        checked.set_piece(Square.from_algebraic("g1"), Piece(Color.WHITE, PieceType.KNIGHT))
        checked.set_piece(Square.from_algebraic("e8"), Piece(Color.BLACK, PieceType.ROOK))
        checked.set_piece(Square.from_algebraic("a8"), Piece(Color.BLACK, PieceType.KING))
        positions.append((checked, Color.WHITE))

        castling = Board.empty()
        castling.castling_rights = CastlingRights(True, True, False, False)
        castling.set_piece(Square.from_algebraic("e1"), Piece(Color.WHITE, PieceType.KING))
        castling.set_piece(Square.from_algebraic("a1"), Piece(Color.WHITE, PieceType.ROOK))
        castling.set_piece(Square.from_algebraic("h1"), Piece(Color.WHITE, PieceType.ROOK))
        castling.set_piece(Square.from_algebraic("a8"), Piece(Color.BLACK, PieceType.KING))
        positions.append((castling, Color.WHITE))

        en_passant_pin = Board.empty()
        en_passant_pin.set_piece(Square.from_algebraic("e1"), Piece(Color.WHITE, PieceType.KING))
        en_passant_pin.set_piece(Square.from_algebraic("e5"), Piece(Color.WHITE, PieceType.PAWN))
        en_passant_pin.set_piece(Square.from_algebraic("e8"), Piece(Color.BLACK, PieceType.ROOK))
        en_passant_pin.set_piece(Square.from_algebraic("a8"), Piece(Color.BLACK, PieceType.KING))
        en_passant_pin.set_piece(Square.from_algebraic("d7"), Piece(Color.BLACK, PieceType.PAWN))
        en_passant_pin.apply_move_unchecked(Move.from_uci("d7d5"))
        positions.append((en_passant_pin, Color.WHITE))

        for board, color in positions:
            self.assertEqual(
                move_signatures(generate_legal_moves_native(board, color) or []),
                move_signatures(_generate_legal_moves_python(board, color)),
            )


if __name__ == "__main__":
    unittest.main()
