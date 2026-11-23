#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C"
#endif

namespace {

constexpr int PAWN = 1;
constexpr int KNIGHT = 2;
constexpr int BISHOP = 3;
constexpr int ROOK = 4;
constexpr int QUEEN = 5;
constexpr int KING = 6;

constexpr int PIECE_VALUES[7] = {0, 100, 320, 330, 500, 900, 0};
constexpr int CHECKMATE_SCORE = 100000;
constexpr int ENDGAME_NON_PAWN_MATERIAL_THRESHOLD = 2200;
constexpr int MOBILITY_WEIGHT = 2;
constexpr uint64_t FNV_OFFSET = 1469598103934665603ULL;
constexpr uint64_t FNV_PRIME = 1099511628211ULL;
constexpr int TT_EXACT = 0;
constexpr int TT_LOWER = 1;
constexpr int TT_UPPER = 2;
constexpr int TRANSPOSITION_TABLE_CLUSTER_SIZE = 2;
constexpr int MAX_SEARCH_PLY = 128;
constexpr int MOVE_IDENTITY_COUNT = 1 << 16;
constexpr int TT_MATE_SCORE_THRESHOLD = CHECKMATE_SCORE - MAX_SEARCH_PLY;

thread_local uint64_t g_search_nodes = 0;
thread_local uint64_t g_null_move_probes = 0;
thread_local uint64_t g_null_move_cutoffs = 0;
thread_local uint64_t g_probcut_probes = 0;
thread_local uint64_t g_probcut_cutoffs = 0;
thread_local uint64_t g_nested_null_move_probes = 0;
thread_local int g_null_move_nesting = 0;
thread_local uint64_t g_aspiration_attempts = 0;
thread_local uint64_t g_aspiration_fail_lows = 0;
thread_local uint64_t g_aspiration_fail_highs = 0;
thread_local uint64_t g_aspiration_researches = 0;
thread_local uint64_t g_quiet_beta_cutoffs = 0;
thread_local uint64_t g_killer_order_hits = 0;
thread_local uint64_t g_history_order_hits = 0;
thread_local uint64_t g_continuation_history_order_hits = 0;
thread_local uint64_t g_history_saturated_updates = 0;
std::atomic<bool> g_stop_requested{false};
thread_local bool g_search_aborted = false;
thread_local bool g_has_deadline = false;
thread_local std::chrono::steady_clock::time_point g_search_deadline;
std::atomic<int> g_forced_line_extension_budget{1};
std::atomic<bool> g_mate_danger_ordering{false};
std::atomic<bool> g_defensive_mate_threat_ordering{false};
std::atomic<bool> g_major_piece_safety_ordering{true};
std::atomic<bool> g_clustered_tt_replacement{false};
std::atomic<bool> g_aspiration_windows{true};
std::atomic<int> g_pawn_endgame_extension_budget{0};
std::atomic<int> g_check_extension_budget{0};
std::atomic<bool> g_singular_extension{false};
std::atomic<bool> g_futility_pruning{true};
constexpr int REVERSE_FUTILITY_MARGIN_PER_PLY = 200;
constexpr int MOVE_FUTILITY_MARGIN_PER_PLY = 200;
constexpr int REVERSE_FUTILITY_MAX_DEPTH = 2;
constexpr int FUTILITY_MAX_DEPTH = 3;

// Phase 14: default-off razoring. Distinct from reverse futility (which
// prunes when far AHEAD of beta) -- razoring triggers when far BEHIND
// alpha at a shallow depth, and unlike reverse futility it doesn't trust
// the static eval enough to return it directly: it verifies with a
// quiescence search first, since a bad-looking quiet position can still
// have real tactical resources that only qsearch would see.
std::atomic<bool> g_razoring{false};
constexpr int RAZOR_MAX_DEPTH = 2;
constexpr int RAZOR_MARGIN_PER_PLY = 300;

// Default-off ProbCut. Distinct from futility (trusts a static eval
// directly) and razoring (verifies with quiescence only): this verifies a
// subset of tactical moves with a REAL, if shallow, recursive search at a
// widened beta window, so it should be less prone to the horizon-effect
// blind spots that hurt futility's first attempt. Operates at a much
// higher depth range (>=5) than futility/razoring (<=2-3), so the two
// families shouldn't compete for the same nodes.
std::atomic<bool> g_probcut{false};
constexpr int PROBCUT_MIN_DEPTH = 5;
constexpr int PROBCUT_REDUCTION = 4;
constexpr int PROBCUT_MARGIN = 150;

// Default-off continuation history. Replaces an earlier countermove
// heuristic (single remembered "best reply" per (piece,to) key, tested and
// found to be pure noise -- node count even rose slightly) with a
// continuous, decaying score built up over many nodes like history_scores,
// indexed by (piece,to) of both the previous move and the candidate reply.
std::atomic<bool> g_continuation_history{false};

// Phase 8b: default-off LMR reduction formula, scaling with remaining depth
// and move index (log(depth)*log(index)/divisor) instead of the always-on
// flat 1-ply reduction below. Table-driven so the hot path never calls
// std::log directly.
std::atomic<bool> g_lmr_formula{false};
constexpr int LMR_TABLE_MAX_DEPTH = 64;
constexpr int LMR_TABLE_MAX_INDEX = 256;
constexpr double LMR_BASE = 0.5;
constexpr double LMR_DIVISOR = 3.3;
int LMR_REDUCTION[LMR_TABLE_MAX_DEPTH][LMR_TABLE_MAX_INDEX];
bool g_lmr_table_initialized = false;

void init_lmr_table() {
    if (g_lmr_table_initialized) {
        return;
    }
    for (int depth = 0; depth < LMR_TABLE_MAX_DEPTH; ++depth) {
        for (int index = 0; index < LMR_TABLE_MAX_INDEX; ++index) {
            if (depth < 1 || index < 1) {
                LMR_REDUCTION[depth][index] = 0;
                continue;
            }
            double raw = LMR_BASE + std::log(static_cast<double>(depth)) * std::log(static_cast<double>(index)) / LMR_DIVISOR;
            int reduction = raw > 0.0 ? static_cast<int>(raw) : 0;
            LMR_REDUCTION[depth][index] = std::max(0, std::min(reduction, depth - 1));
        }
    }
    g_lmr_table_initialized = true;
}

int lmr_reduction(int depth, int index) {
    init_lmr_table();
    int clamped_depth = std::max(0, std::min(depth, LMR_TABLE_MAX_DEPTH - 1));
    int clamped_index = std::max(0, std::min(index, LMR_TABLE_MAX_INDEX - 1));
    return LMR_REDUCTION[clamped_depth][clamped_index];
}

int score_to_transposition_table(int score, int ply) {
    if (score >= TT_MATE_SCORE_THRESHOLD) {
        return score + ply;
    }
    if (score <= -TT_MATE_SCORE_THRESHOLD) {
        return score - ply;
    }
    return score;
}

int score_from_transposition_table(int score, int ply) {
    if (score >= TT_MATE_SCORE_THRESHOLD) {
        return score - ply;
    }
    if (score <= -TT_MATE_SCORE_THRESHOLD) {
        return score + ply;
    }
    return score;
}
std::atomic<int> g_hash_mb{16};

bool search_should_stop() {
    if (g_search_aborted) {
        return true;
    }
    if (g_stop_requested.load(std::memory_order_relaxed)) {
        g_search_aborted = true;
        return true;
    }
    if (g_has_deadline && (g_search_nodes & 0x3FU) == 0 &&
        std::chrono::steady_clock::now() >= g_search_deadline) {
        g_search_aborted = true;
        return true;
    }
    return false;
}

enum NativeMoveStage {
    STAGE_TT = 0,
    STAGE_GOOD_CAPTURE = 1,
    STAGE_GOOD_QUIET = 2,
    STAGE_BAD_CAPTURE = 3,
    STAGE_BAD_QUIET = 4
};

struct TranspositionEntry {
    uint64_t key = 0;
    int depth = -1;
    int score = 0;
    int bound = TT_EXACT;
    int best_move = 0;
    bool occupied = false;
};

struct TranspositionTable {
    std::vector<TranspositionEntry> entries;
    int bucket_count;
    int cluster_size;

    explicit TranspositionTable(int size)
        : entries(
            size * (
                g_clustered_tt_replacement.load(std::memory_order_relaxed)
                    ? TRANSPOSITION_TABLE_CLUSTER_SIZE
                    : 1
            )
        ),
          bucket_count(size),
          cluster_size(
              g_clustered_tt_replacement.load(std::memory_order_relaxed)
                  ? TRANSPOSITION_TABLE_CLUSTER_SIZE
                  : 1
          ) {}

    TranspositionEntry* find(uint64_t key) {
        int bucket = static_cast<int>(key % bucket_count) * cluster_size;
        for (int offset = 0; offset < cluster_size; ++offset) {
            TranspositionEntry& entry = entries[bucket + offset];
            if (entry.occupied && entry.key == key) {
                return &entry;
            }
        }
        return nullptr;
    }

    void store(uint64_t key, int depth, int score, int bound, int best_move) {
        int bucket = static_cast<int>(key % bucket_count) * cluster_size;
        TranspositionEntry* replacement = &entries[bucket];
        for (int offset = 0; offset < cluster_size; ++offset) {
            TranspositionEntry& entry = entries[bucket + offset];
            if (entry.occupied && entry.key == key) {
                if (depth >= entry.depth) {
                    replacement = &entry;
                } else {
                    return;
                }
                break;
            }
            if (!entry.occupied) {
                replacement = &entry;
                break;
            }
            if (entry.depth < replacement->depth) {
                replacement = &entry;
            }
        }
        replacement->key = key;
        replacement->depth = depth;
        replacement->score = score;
        replacement->bound = bound;
        replacement->best_move = best_move;
        replacement->occupied = true;
    }
};

int transposition_table_bucket_count() {
    const int cluster_size = (
        g_clustered_tt_replacement.load(std::memory_order_relaxed)
            ? TRANSPOSITION_TABLE_CLUSTER_SIZE
            : 1
    );
    const uint64_t bytes = static_cast<uint64_t>(
        std::max(1, g_hash_mb.load(std::memory_order_relaxed))
    ) * 1024ULL * 1024ULL;
    const uint64_t maximum_buckets = std::max<uint64_t>(
        1,
        bytes / (sizeof(TranspositionEntry) * cluster_size)
    );
    int bucket_count = 1;
    while (
        static_cast<uint64_t>(bucket_count) * 2ULL <= maximum_buckets &&
        bucket_count <= (1 << 29)
    ) {
        bucket_count *= 2;
    }
    return bucket_count;
}

// Indexed by (piece, to-square) of the move that led to the current node
// AND (piece, to-square) of the candidate reply -- a continuous, decaying
// score built up over many nodes (like history_scores), not a single
// remembered "best reply" (that was the countermove heuristic this
// replaces: one stored move per (piece,to) key, overwritten by whatever
// caused a cutoff most recently, tested and found to be pure noise).
// 13*64*13*64 ints (~2.7MB) is too large for a stack-local SearchHeuristics
// member, so it's heap-backed via std::vector.
constexpr int CONTINUATION_HISTORY_SIZE = 13 * 64 * 13 * 64;

int continuation_history_index(int previous_move_piece, int previous_move_to, int current_piece, int current_to) {
    int prev = (previous_move_piece + 6) * 64 + previous_move_to;
    int cur = (current_piece + 6) * 64 + current_to;
    return prev * (13 * 64) + cur;
}

struct SearchHeuristics {
    int killer_moves[MAX_SEARCH_PLY][2]{};
    int history_scores[MOVE_IDENTITY_COUNT]{};
    std::vector<int> continuation_history = std::vector<int>(CONTINUATION_HISTORY_SIZE, 0);

    int continuation_history_score(int encoded, int current_piece, int previous_move_piece, int previous_move_to) const {
        if (previous_move_piece < -6 || previous_move_piece > 6 || previous_move_to < 0) {
            return 0;
        }
        int current_to = (encoded >> 6) & 0x3F;
        int index = continuation_history_index(previous_move_piece, previous_move_to, current_piece, current_to);
        return std::min(5000, std::max(0, continuation_history[index]));
    }

    void record_continuation_history_cutoff(
        int encoded, int current_piece, int previous_move_piece, int previous_move_to, int depth
    ) {
        if (previous_move_piece < -6 || previous_move_piece > 6 || previous_move_to < 0) {
            return;
        }
        int current_to = (encoded >> 6) & 0x3F;
        int index = continuation_history_index(previous_move_piece, previous_move_to, current_piece, current_to);
        continuation_history[index] = std::min(5000, continuation_history[index] + depth * depth);
    }

    void record_continuation_history_failure(
        int encoded, int current_piece, int previous_move_piece, int previous_move_to, int depth
    ) {
        if (previous_move_piece < -6 || previous_move_piece > 6 || previous_move_to < 0) {
            return;
        }
        int current_to = (encoded >> 6) & 0x3F;
        int index = continuation_history_index(previous_move_piece, previous_move_to, current_piece, current_to);
        continuation_history[index] = std::max(0, continuation_history[index] - depth * depth);
    }

    int killer_score(int encoded, int ply) const {
        if (ply < 0 || ply >= MAX_SEARCH_PLY) {
            return 0;
        }
        int identity = encoded & 0xFFFF;
        if (killer_moves[ply][0] == identity) {
            return 7000;
        }
        if (killer_moves[ply][1] == identity) {
            return 6500;
        }
        return 0;
    }

    int history_score(int encoded) const {
        return std::min(5000, history_scores[encoded & 0xFFFF]);
    }

    void record_cutoff(int encoded, int depth, int ply, bool is_quiet) {
        if (!is_quiet || ply < 0 || ply >= MAX_SEARCH_PLY) {
            return;
        }
        int identity = encoded & 0xFFFF;
        if (killer_moves[ply][0] != identity) {
            killer_moves[ply][1] = killer_moves[ply][0];
            killer_moves[ply][0] = identity;
        }
        int& history = history_scores[identity];
        ++g_quiet_beta_cutoffs;
        if (history >= 5000) {
            ++g_history_saturated_updates;
        }
        history = std::min(5000, history + depth * depth);
    }

    void record_quiet_failure(int encoded, int depth) {
        int& history = history_scores[encoded & 0xFFFF];
        history = std::max(0, history - depth * depth);
    }
};

struct SparseNNUEFileModel {
    std::string path;
    uint32_t feature_count = 0;
    double bias = 0.0;
    std::unordered_map<int, double> weights;
    bool loaded = false;
};

struct AccumulatorNNUEFileModel {
    std::string path;
    uint32_t feature_count = 0;
    uint32_t hidden_dim = 0;
    uint32_t activation = 0;
    double output_bias = 0.0;
    std::vector<std::vector<std::pair<int, double>>> feature_weights;
    std::vector<double> output_weights;
    bool loaded = false;
};

bool line_clear_between(const int8_t* board, int from_file, int from_rank, int to_file, int to_rank);
bool square_has_pawn_attacker(const int8_t* board, int file, int rank, int by_color);

constexpr int PST[7][8][8] = {
    {},
    {
        {0, 0, 0, 0, 0, 0, 0, 0},
        {5, 10, 10, -20, -20, 10, 10, 5},
        {5, -5, -10, 0, 0, -10, -5, 5},
        {0, 0, 0, 20, 20, 0, 0, 0},
        {5, 5, 10, 25, 25, 10, 5, 5},
        {10, 10, 20, 30, 30, 20, 10, 10},
        {50, 50, 50, 50, 50, 50, 50, 50},
        {0, 0, 0, 0, 0, 0, 0, 0},
    },
    {
        {-50, -40, -30, -30, -30, -30, -40, -50},
        {-40, -20, 0, 5, 5, 0, -20, -40},
        {-30, 5, 10, 15, 15, 10, 5, -30},
        {-30, 0, 15, 20, 20, 15, 0, -30},
        {-30, 5, 15, 20, 20, 15, 5, -30},
        {-30, 0, 10, 15, 15, 10, 0, -30},
        {-40, -20, 0, 0, 0, 0, -20, -40},
        {-50, -40, -30, -30, -30, -30, -40, -50},
    },
    {
        {-20, -10, -10, -10, -10, -10, -10, -20},
        {-10, 5, 0, 0, 0, 0, 5, -10},
        {-10, 10, 10, 10, 10, 10, 10, -10},
        {-10, 0, 10, 10, 10, 10, 0, -10},
        {-10, 5, 5, 10, 10, 5, 5, -10},
        {-10, 0, 5, 10, 10, 5, 0, -10},
        {-10, 0, 0, 0, 0, 0, 0, -10},
        {-20, -10, -10, -10, -10, -10, -10, -20},
    },
    {
        {0, 0, 5, 10, 10, 5, 0, 0},
        {-5, 0, 0, 0, 0, 0, 0, -5},
        {-5, 0, 0, 0, 0, 0, 0, -5},
        {-5, 0, 0, 0, 0, 0, 0, -5},
        {-5, 0, 0, 0, 0, 0, 0, -5},
        {-5, 0, 0, 0, 0, 0, 0, -5},
        {5, 10, 10, 10, 10, 10, 10, 5},
        {0, 0, 0, 5, 5, 0, 0, 0},
    },
    {
        {-20, -10, -10, -5, -5, -10, -10, -20},
        {-10, 0, 5, 0, 0, 0, 0, -10},
        {-10, 5, 5, 5, 5, 5, 0, -10},
        {0, 0, 5, 5, 5, 5, 0, -5},
        {-5, 0, 5, 5, 5, 5, 0, -5},
        {-10, 0, 5, 5, 5, 5, 0, -10},
        {-10, 0, 0, 0, 0, 0, 0, -10},
        {-20, -10, -10, -5, -5, -10, -10, -20},
    },
    {
        {20, 30, 10, 0, 0, 10, 30, 20},
        {20, 20, 0, 0, 0, 0, 20, 20},
        {-10, -20, -20, -20, -20, -20, -20, -10},
        {-20, -30, -30, -40, -40, -30, -30, -20},
        {-30, -40, -40, -50, -50, -40, -40, -30},
        {-30, -40, -40, -50, -50, -40, -40, -30},
        {-30, -40, -40, -50, -50, -40, -40, -30},
        {-30, -40, -40, -50, -50, -40, -40, -30},
    },
};

int idx(int file, int rank) {
    return rank * 8 + file;
}

bool inside(int file, int rank) {
    return file >= 0 && file < 8 && rank >= 0 && rank < 8;
}

int color_of(int code) {
    return (code > 0) - (code < 0);
}

int type_of(int code) {
    return std::abs(code);
}

bool square_attacked_internal(const int8_t* board, int file, int rank, int by_color);

bool has_insufficient_material_draw(const int8_t* board) {
    int non_king_count = 0;
    int single_minor_piece = 0;
    int bishop_square_colors[2] = {-1, -1};
    int bishop_count = 0;

    for (int index = 0; index < 64; ++index) {
        int piece = type_of(board[index]);
        if (piece == 0 || piece == KING) {
            continue;
        }
        ++non_king_count;
        if (non_king_count > 2) {
            return false;
        }
        if (piece == BISHOP) {
            if (bishop_count < 2) {
                bishop_square_colors[bishop_count] = (index % 8 + index / 8) & 1;
            }
            ++bishop_count;
        } else if (piece == KNIGHT) {
            single_minor_piece = KNIGHT;
        } else {
            return false;
        }
    }

    if (non_king_count == 0) {
        return true;
    }
    if (non_king_count == 1) {
        return bishop_count == 1 || single_minor_piece == KNIGHT;
    }
    return bishop_count == 2 && bishop_square_colors[0] == bishop_square_colors[1];
}

uint64_t mix_hash(uint64_t hash, uint64_t value) {
    hash ^= value;
    hash *= FNV_PRIME;
    return hash;
}

int piece_square_value(int piece, int file, int rank, int color) {
    int table_rank = color > 0 ? rank : 7 - rank;
    return PST[piece][table_rank][file];
}

int central_score(int file, int rank) {
    double file_distance = std::abs(file - 3.5);
    double rank_distance = std::abs(rank - 3.5);
    return static_cast<int>(28 - (file_distance + rank_distance) * 6);
}

int central_control_score(const int8_t* board) {
    int score = 0;
    constexpr int center[4][2] = {{3, 3}, {4, 3}, {3, 4}, {4, 4}};
    for (const auto& square : center) {
        int file = square[0];
        int rank = square[1];
        int code = board[idx(file, rank)];
        if (type_of(code) == PAWN) {
            score += color_of(code) * 18;
        }
        if (square_attacked_internal(board, file, rank, 1)) {
            score += 7;
        }
        if (square_attacked_internal(board, file, rank, -1)) {
            score -= 7;
        }
    }
    return score;
}

int opening_development_score(const int8_t* board, double phase) {
    if (phase > 0.25) {
        return 0;
    }
    constexpr int white_home[4] = {1, 6, 2, 5};
    constexpr int black_home[4] = {57, 62, 58, 61};
    int score = 0;
    for (int color : {1, -1}) {
        int developed = 0;
        int blocked = 0;
        const int* squares = color > 0 ? white_home : black_home;
        for (int index = 0; index < 4; ++index) {
            int code = board[squares[index]];
            int piece = type_of(code);
            if (code == 0) {
                ++developed;
            } else if (color_of(code) == color && (piece == KNIGHT || piece == BISHOP)) {
                ++blocked;
            }
        }
        int value = developed * 10 - blocked * 2;
        score += color * value;
    }
    return score;
}

int king_opening_safety_score(
    int white_king_file,
    int white_king_rank,
    int black_king_file,
    int black_king_rank,
    double phase
) {
    if (phase > 0.35) {
        return 0;
    }
    int score = 0;
    if (white_king_rank == 0 && (white_king_file == 6 || white_king_file == 2)) {
        score += 24;
    } else if (white_king_rank == 0 && white_king_file == 4) {
        score -= 12;
    }
    if (black_king_rank == 7 && (black_king_file == 6 || black_king_file == 2)) {
        score -= 24;
    } else if (black_king_rank == 7 && black_king_file == 4) {
        score += 12;
    }
    return score;
}

int loose_piece_threat_score(const int8_t* board) {
    int score = 0;
    for (int square = 0; square < 64; ++square) {
        int code = board[square];
        int piece = type_of(code);
        int color = color_of(code);
        if (piece == 0 || piece == KING) {
            continue;
        }
        int file = square % 8;
        int rank = square / 8;
        bool attacked = square_attacked_internal(board, file, rank, -color);
        bool defended = square_attacked_internal(board, file, rank, color);
        if (!attacked || defended) {
            continue;
        }
        int value = std::min(90, PIECE_VALUES[piece] / 8);
        score -= color * value;
    }
    return score;
}

int king_distance(int first_file, int first_rank, int second_file, int second_rank) {
    return std::max(std::abs(first_file - second_file), std::abs(first_rank - second_rank));
}

int edge_score(int file, int rank) {
    int file_edge = std::min(file, 7 - file);
    int rank_edge = std::min(rank, 7 - rank);
    return 6 - (file_edge + rank_edge);
}

bool is_passed_pawn(const int8_t* board, int file, int rank, int color) {
    int direction = color > 0 ? 1 : -1;
    int enemy = -color;
    for (int r = rank + direction; r >= 0 && r < 8; r += direction) {
        for (int f = std::max(0, file - 1); f <= std::min(7, file + 1); ++f) {
            int code = board[idx(f, r)];
            if (color_of(code) == enemy && type_of(code) == PAWN) {
                return false;
            }
        }
    }
    return true;
}

int pawn_count_on_file(const int8_t* board, int file, int color) {
    int count = 0;
    for (int rank = 0; rank < 8; ++rank) {
        int code = board[idx(file, rank)];
        if (color_of(code) == color && type_of(code) == PAWN) {
            ++count;
        }
    }
    return count;
}

bool has_adjacent_friendly_pawn_file(const int8_t* board, int file, int color) {
    for (int adjacent_file : {file - 1, file + 1}) {
        if (inside(adjacent_file, 0) && pawn_count_on_file(board, adjacent_file, color) > 0) {
            return true;
        }
    }
    return false;
}

bool has_same_rank_pawn_partner(const int8_t* board, int file, int rank, int color) {
    for (int adjacent_file : {file - 1, file + 1}) {
        if (!inside(adjacent_file, rank)) {
            continue;
        }
        int code = board[idx(adjacent_file, rank)];
        if (color_of(code) == color && type_of(code) == PAWN) {
            return true;
        }
    }
    return false;
}

bool is_hanging_pawn(const int8_t* board, int file, int rank, int color) {
    if (!has_same_rank_pawn_partner(board, file, rank, color)) {
        return false;
    }
    for (int outer_file : {file - 2, file + 2}) {
        if (inside(outer_file, 0) && pawn_count_on_file(board, outer_file, color) > 0) {
            return false;
        }
    }
    return !square_has_pawn_attacker(board, file, rank, color);
}

bool can_create_pawn_break(const int8_t* board, int file, int rank, int color) {
    int direction = color > 0 ? 1 : -1;
    int target_rank = rank + direction;
    if (!inside(file, target_rank) || board[idx(file, target_rank)] != 0) {
        return false;
    }
    for (int adjacent_file : {file - 1, file + 1}) {
        if (!inside(adjacent_file, target_rank)) {
            continue;
        }
        int adjacent = board[idx(adjacent_file, target_rank)];
        if (color_of(adjacent) == -color && type_of(adjacent) == PAWN) {
            return true;
        }
        int next_rank = target_rank + direction;
        if (inside(adjacent_file, next_rank)) {
            int next = board[idx(adjacent_file, next_rank)];
            if (color_of(next) == -color && type_of(next) == PAWN) {
                return true;
            }
        }
    }
    return false;
}

int pawn_count_in_file_range(const int8_t* board, int first_file, int last_file, int color) {
    int count = 0;
    for (int file = first_file; file <= last_file; ++file) {
        count += pawn_count_on_file(board, file, color);
    }
    return count;
}

int most_advanced_pawn_in_file_range(const int8_t* board, int first_file, int last_file, int color) {
    int best = 0;
    for (int file = first_file; file <= last_file; ++file) {
        for (int rank = 0; rank < 8; ++rank) {
            int code = board[idx(file, rank)];
            if (color_of(code) == color && type_of(code) == PAWN) {
                best = std::max(best, color > 0 ? rank : 7 - rank);
            }
        }
    }
    return best;
}

int pawn_majority_score(const int8_t* board) {
    int score = 0;
    for (std::pair<int, int> span : {std::pair<int, int>(0, 3), std::pair<int, int>(4, 7)}) {
        int white = pawn_count_in_file_range(board, span.first, span.second, 1);
        int black = pawn_count_in_file_range(board, span.first, span.second, -1);
        if (white > black) {
            score += std::min(18, (white - black) * 8 + most_advanced_pawn_in_file_range(board, span.first, span.second, 1) * 2);
        } else if (black > white) {
            score -= std::min(18, (black - white) * 8 + most_advanced_pawn_in_file_range(board, span.first, span.second, -1) * 2);
        }
    }
    return score;
}

bool file_has_pawn(const int8_t* board, int file, int color) {
    return pawn_count_on_file(board, file, color) > 0;
}

bool square_has_pawn_attacker(const int8_t* board, int file, int rank, int by_color) {
    int direction = by_color > 0 ? 1 : -1;
    int pawn_rank = rank - direction;
    if (!inside(file, pawn_rank)) {
        return false;
    }
    for (int pawn_file : {file - 1, file + 1}) {
        if (!inside(pawn_file, pawn_rank)) {
            continue;
        }
        int code = board[idx(pawn_file, pawn_rank)];
        if (color_of(code) == by_color && type_of(code) == PAWN) {
            return true;
        }
    }
    return false;
}

bool light_square(int file, int rank) {
    return ((file + rank) % 2) == 1;
}

int pawn_structure_score(const int8_t* board) {
    int score = 0;
    for (int file = 0; file < 8; ++file) {
        for (int color : {1, -1}) {
            int count = pawn_count_on_file(board, file, color);
            if (count > 1) {
                score -= color * (count - 1) * 14;
            }
        }
    }
    for (int rank = 0; rank < 8; ++rank) {
        for (int file = 0; file < 8; ++file) {
            int code = board[idx(file, rank)];
            if (type_of(code) != PAWN) {
                continue;
            }
            int color = color_of(code);
            if (!has_adjacent_friendly_pawn_file(board, file, color)) {
                score -= color * 10;
            }
            if (is_hanging_pawn(board, file, rank, color)) {
                score -= color * 5;
            }
            if (can_create_pawn_break(board, file, rank, color)) {
                int value = 8 + (file >= 2 && file <= 5 ? 4 : 0);
                score += color * value;
            }
            // Passed pawns are NOT scored here -- they get a dedicated,
            // phase-aware accumulator (path bonus + support + connected
            // passer, gated by passed_pawn_weight) further down. Scoring
            // them here too double-counted every passed pawn through two
            // independent weights (pawn_structure_weight AND
            // passed_pawn_weight).
        }
    }
    return score + pawn_majority_score(board);
}

int bishop_pair_score(const int8_t* board) {
    int white_bishops = 0;
    int black_bishops = 0;
    for (int index = 0; index < 64; ++index) {
        int code = board[index];
        if (type_of(code) == BISHOP) {
            if (color_of(code) > 0) {
                ++white_bishops;
            } else {
                ++black_bishops;
            }
        }
    }
    return (white_bishops >= 2 ? 28 : 0) - (black_bishops >= 2 ? 28 : 0);
}

int rook_file_score(const int8_t* board) {
    int score = 0;
    for (int rank = 0; rank < 8; ++rank) {
        for (int file = 0; file < 8; ++file) {
            int code = board[idx(file, rank)];
            if (type_of(code) != ROOK) {
                continue;
            }
            int color = color_of(code);
            bool friendly_pawn = file_has_pawn(board, file, color);
            bool enemy_pawn = file_has_pawn(board, file, -color);
            if (!friendly_pawn && !enemy_pawn) {
                score += color * 24;
            } else if (!friendly_pawn && enemy_pawn) {
                score += color * 12;
            }
            int seventh_rank = color > 0 ? 6 : 1;
            if (rank == seventh_rank) {
                score += color * 18;
            }
        }
    }
    return score;
}

int knight_outpost_score(const int8_t* board) {
    int score = 0;
    for (int rank = 0; rank < 8; ++rank) {
        for (int file = 0; file < 8; ++file) {
            int code = board[idx(file, rank)];
            if (type_of(code) != KNIGHT) {
                continue;
            }
            int color = color_of(code);
            int advancement = color > 0 ? rank : 7 - rank;
            if (file < 2 || file > 5 || advancement < 3) {
                continue;
            }
            if (!square_has_pawn_attacker(board, file, rank, color)) {
                continue;
            }
            if (square_has_pawn_attacker(board, file, rank, -color)) {
                continue;
            }
            int value = 18 + std::max(0, advancement - 3) * 5 + std::max(0, central_score(file, rank)) / 4;
            score += color * value;
        }
    }
    return score;
}

int bad_bishop_score(const int8_t* board) {
    int score = 0;
    for (int rank = 0; rank < 8; ++rank) {
        for (int file = 0; file < 8; ++file) {
            int code = board[idx(file, rank)];
            if (type_of(code) != BISHOP) {
                continue;
            }
            int color = color_of(code);
            bool bishop_light = light_square(file, rank);
            int same_color_pawns = 0;
            int central_blockers = 0;
            for (int pawn_rank = 0; pawn_rank < 8; ++pawn_rank) {
                for (int pawn_file = 0; pawn_file < 8; ++pawn_file) {
                    int pawn = board[idx(pawn_file, pawn_rank)];
                    if (color_of(pawn) != color || type_of(pawn) != PAWN) {
                        continue;
                    }
                    if (light_square(pawn_file, pawn_rank) != bishop_light) {
                        continue;
                    }
                    ++same_color_pawns;
                    if (pawn_file >= 2 && pawn_file <= 5) {
                        ++central_blockers;
                    }
                }
            }
            int penalty = std::max(0, same_color_pawns - 2) * 3 + central_blockers * 4;
            score -= color * penalty;
        }
    }
    return score;
}

int space_advantage_score(const int8_t* board) {
    int score = 0;
    for (int rank = 0; rank < 8; ++rank) {
        for (int file = 0; file < 8; ++file) {
            int code = board[idx(file, rank)];
            if (type_of(code) != PAWN) {
                continue;
            }
            int color = color_of(code);
            int advancement = color > 0 ? rank : 7 - rank;
            if (advancement < 3) {
                continue;
            }
            int value = 3;
            if (file >= 2 && file <= 5) {
                value += 5;
            }
            if (advancement >= 4) {
                value += 4;
            }
            score += color * value;
        }
    }
    return score;
}

bool isolated_pawn(const int8_t* board, int file, int color) {
    for (int adjacent_file : {file - 1, file + 1}) {
        if (!inside(adjacent_file, 0)) {
            continue;
        }
        if (pawn_count_on_file(board, adjacent_file, color) > 0) {
            return false;
        }
    }
    return true;
}

bool backward_pawn(const int8_t* board, int file, int rank, int color) {
    if (!isolated_pawn(board, file, color)) {
        return false;
    }
    int direction = color > 0 ? 1 : -1;
    int front_rank = rank + direction;
    if (!inside(file, front_rank)) {
        return false;
    }
    for (int enemy_file : {file - 1, file + 1}) {
        if (!inside(enemy_file, front_rank)) {
            continue;
        }
        int code = board[idx(enemy_file, front_rank)];
        if (color_of(code) == -color && type_of(code) == PAWN) {
            return true;
        }
    }
    return false;
}

bool square_can_be_defended_by_pawn(const int8_t* board, int file, int rank, int color) {
    int direction = color > 0 ? 1 : -1;
    int pawn_rank = rank - direction;
    if (!inside(file, pawn_rank)) {
        return false;
    }
    for (int pawn_file : {file - 1, file + 1}) {
        if (!inside(pawn_file, pawn_rank)) {
            continue;
        }
        int code = board[idx(pawn_file, pawn_rank)];
        if (color_of(code) == color && type_of(code) == PAWN) {
            return true;
        }
    }
    return false;
}

bool minor_piece_controls(const int8_t* board, int file, int rank, int color) {
    for (int piece_rank = 0; piece_rank < 8; ++piece_rank) {
        for (int piece_file = 0; piece_file < 8; ++piece_file) {
            int code = board[idx(piece_file, piece_rank)];
            if (color_of(code) != color) {
                continue;
            }
            int piece = type_of(code);
            int file_delta = std::abs(piece_file - file);
            int rank_delta = std::abs(piece_rank - rank);
            if (piece == KNIGHT && ((file_delta == 1 && rank_delta == 2) || (file_delta == 2 && rank_delta == 1))) {
                return true;
            }
            if (piece == BISHOP && file_delta == rank_delta) {
                return true;
            }
        }
    }
    return false;
}

int advanced_pawn_structure_score(const int8_t* board) {
    int score = 0;
    for (int rank = 0; rank < 8; ++rank) {
        for (int file = 0; file < 8; ++file) {
            int code = board[idx(file, rank)];
            if (type_of(code) != PAWN) {
                continue;
            }
            int color = color_of(code);
            int penalty = 0;
            if (isolated_pawn(board, file, color)) {
                penalty += 10;
            }
            if (backward_pawn(board, file, rank, color)) {
                penalty += 8;
            }
            score -= color * penalty;
        }
    }
    return score;
}

int weak_square_score(const int8_t* board) {
    int score = 0;
    for (int file = 2; file <= 5; ++file) {
        for (int rank = 2; rank <= 5; ++rank) {
            bool weak_for_white = !square_can_be_defended_by_pawn(board, file, rank, 1);
            bool weak_for_black = !square_can_be_defended_by_pawn(board, file, rank, -1);
            if (weak_for_black && minor_piece_controls(board, file, rank, 1)) {
                score += 8;
            }
            if (weak_for_white && minor_piece_controls(board, file, rank, -1)) {
                score -= 8;
            }
        }
    }
    return score;
}

int bishop_scope_score(const int8_t* board, int file, int rank, int color) {
    constexpr int diagonal[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    int scope = 0;
    for (const auto& direction : diagonal) {
        int current_file = file + direction[0];
        int current_rank = rank + direction[1];
        while (inside(current_file, current_rank)) {
            int code = board[idx(current_file, current_rank)];
            if (code == 0) {
                scope += 2;
            } else {
                scope += color_of(code) == -color ? 1 : -2;
                break;
            }
            current_file += direction[0];
            current_rank += direction[1];
        }
    }
    return scope;
}

int piece_quality_score(const int8_t* board) {
    int score = 0;
    for (int rank = 0; rank < 8; ++rank) {
        for (int file = 0; file < 8; ++file) {
            int code = board[idx(file, rank)];
            int piece = type_of(code);
            int color = color_of(code);
            int value = 0;
            if (piece == KNIGHT) {
                value += std::max(0, central_score(file, rank)) / 3;
                if (square_has_pawn_attacker(board, file, rank, color)) {
                    value += 6;
                }
                if (square_has_pawn_attacker(board, file, rank, -color)) {
                    value -= 8;
                }
            } else if (piece == BISHOP) {
                value += bishop_scope_score(board, file, rank, color);
            } else {
                continue;
            }
            score += color * value;
        }
    }
    return score;
}

int locked_center_file_count(const int8_t* board) {
    int count = 0;
    for (int file = 2; file <= 5; ++file) {
        bool locked = false;
        for (int rank = 1; rank <= 6; ++rank) {
            int code = board[idx(file, rank)];
            if (type_of(code) != PAWN) {
                continue;
            }
            int color = color_of(code);
            int front_rank = rank + (color > 0 ? 1 : -1);
            if (!inside(file, front_rank)) {
                continue;
            }
            int front = board[idx(file, front_rank)];
            if (color_of(front) == -color && type_of(front) == PAWN) {
                locked = true;
                break;
            }
        }
        if (locked) {
            ++count;
        }
    }
    return count;
}

int bishop_own_pawn_blockers(const int8_t* board, int file, int rank, int color) {
    constexpr int diagonal[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    int blockers = 0;
    for (const auto& direction : diagonal) {
        int target_file = file + direction[0];
        int target_rank = rank + direction[1];
        if (!inside(target_file, target_rank)) {
            continue;
        }
        int code = board[idx(target_file, target_rank)];
        if (color_of(code) == color && type_of(code) == PAWN) {
            ++blockers;
        }
    }
    return blockers;
}

int locked_center_minor_score(const int8_t* board) {
    int locked_files = locked_center_file_count(board);
    if (locked_files <= 0) {
        return 0;
    }

    int score = 0;
    for (int rank = 0; rank < 8; ++rank) {
        for (int file = 0; file < 8; ++file) {
            int code = board[idx(file, rank)];
            int piece = type_of(code);
            int color = color_of(code);
            int value = 0;
            if (piece == KNIGHT) {
                int advancement = color > 0 ? rank : 7 - rank;
                if (file >= 2 && file <= 5 && square_has_pawn_attacker(board, file, rank, color)) {
                    value += 10 + std::max(0, advancement - 2) * 4 + locked_files * 3;
                }
            } else if (piece == BISHOP) {
                int blockers = bishop_own_pawn_blockers(board, file, rank, color);
                if (blockers > 0) {
                    value -= 8 + blockers * 5 + locked_files * 3;
                }
            }
            score += color * value;
        }
    }
    return score;
}

int central_color_complex_weakness(const int8_t* board, int color, bool light) {
    int count = 0;
    for (int file = 2; file <= 5; ++file) {
        for (int rank = 2; rank <= 5; ++rank) {
            if (light_square(file, rank) != light) {
                continue;
            }
            if (!square_can_be_defended_by_pawn(board, file, rank, color)) {
                ++count;
            }
        }
    }
    return count;
}

int enemy_bishop_color_pressure(const int8_t* board, int color, bool light) {
    int count = 0;
    for (int index = 0; index < 64; ++index) {
        int code = board[index];
        if (color_of(code) == -color && type_of(code) == BISHOP && light_square(index % 8, index / 8) == light) {
            ++count;
        }
    }
    return count;
}

int color_complex_score(const int8_t* board) {
    int score = 0;
    for (int color : {1, -1}) {
        for (bool light : {false, true}) {
            int undefended = central_color_complex_weakness(board, color, light);
            int pressure = enemy_bishop_color_pressure(board, color, light);
            int value = std::min(32, undefended * pressure * 3);
            score -= color * value;
        }
    }
    return score;
}

int line_activity_score(const int8_t* board) {
    // Rook open/semi-open-file activity lives solely in rook_file_score()
    // (rook_activity_weight), and "piece behind a passed pawn" lives solely
    // in passed_pawn_support_score() (passed_pawn_weight) -- both used to be
    // repeated here too, under line_activity_weight, double-counting the
    // same facts a third and fourth time. Bishop diagonal scope has no
    // other home, so it's the only thing left in this term.
    int score = 0;
    for (int rank = 0; rank < 8; ++rank) {
        for (int file = 0; file < 8; ++file) {
            int code = board[idx(file, rank)];
            int piece = type_of(code);
            int color = color_of(code);
            if (piece != BISHOP) {
                continue;
            }
            int value = std::max(0, bishop_scope_score(board, file, rank, color) - 8);
            score += color * value;
        }
    }
    return score;
}

int enemy_major_back_rank_pressure(const int8_t* board, int color, int king_file, int king_rank) {
    int pressure = 0;
    for (int rank = 0; rank < 8; ++rank) {
        for (int file = 0; file < 8; ++file) {
            int code = board[idx(file, rank)];
            int piece = type_of(code);
            if (color_of(code) != -color || (piece != ROOK && piece != QUEEN)) {
                continue;
            }
            if (rank == king_rank) {
                ++pressure;
            }
            if (std::abs(file - king_file) <= 1 && !file_has_pawn(board, file, color)) {
                ++pressure;
            }
        }
    }
    return std::min(3, pressure);
}

int back_rank_weakness_score(const int8_t* board) {
    int score = 0;
    for (int color : {1, -1}) {
        int king_file = -1;
        int king_rank = -1;
        for (int index = 0; index < 64; ++index) {
            if (color_of(board[index]) == color && type_of(board[index]) == KING) {
                king_file = index % 8;
                king_rank = index / 8;
                break;
            }
        }
        int home_rank = color > 0 ? 0 : 7;
        if (king_file < 0 || king_rank != home_rank) {
            continue;
        }
        int escape_rank = king_rank + (color > 0 ? 1 : -1);
        int blocked = 0;
        for (int file = std::max(0, king_file - 1); file <= std::min(7, king_file + 1); ++file) {
            int code = board[idx(file, escape_rank)];
            if (color_of(code) == color) {
                ++blocked;
            }
        }
        if (blocked < 2) {
            continue;
        }
        int pressure = enemy_major_back_rank_pressure(board, color, king_file, king_rank);
        int penalty = blocked * 7 + pressure * 8;
        score -= color * penalty;
    }
    return score;
}

int exchange_strategy_score(int material, double phase) {
    if (std::abs(material) < 250) {
        return 0;
    }
    int traded_non_pawn = static_cast<int>(ENDGAME_NON_PAWN_MATERIAL_THRESHOLD * phase);
    int value = std::min(36, std::abs(material) / 80 + traded_non_pawn / 120);
    return material > 0 ? value : -value;
}

bool piece_has_line_to_square(const int8_t* board, int from_file, int from_rank, int piece, int target_file, int target_rank) {
    int file_delta = target_file - from_file;
    int rank_delta = target_rank - from_rank;
    int abs_file = std::abs(file_delta);
    int abs_rank = std::abs(rank_delta);
    if (piece == BISHOP && abs_file != abs_rank) {
        return false;
    }
    if (piece == ROOK && file_delta != 0 && rank_delta != 0) {
        return false;
    }
    if (piece == QUEEN && !(file_delta == 0 || rank_delta == 0 || abs_file == abs_rank)) {
        return false;
    }
    if (piece != BISHOP && piece != ROOK && piece != QUEEN) {
        return false;
    }
    return line_clear_between(board, from_file, from_rank, target_file, target_rank);
}

int king_attack_pressure(const int8_t* board, int color, int enemy_king_file, int enemy_king_rank) {
    if (enemy_king_file < 0) {
        return 0;
    }
    int pressure = 0;
    for (int rank = 0; rank < 8; ++rank) {
        for (int file = 0; file < 8; ++file) {
            int code = board[idx(file, rank)];
            if (color_of(code) != color) {
                continue;
            }
            int piece = type_of(code);
            if (piece == KING || piece == PAWN) {
                continue;
            }
            int distance = king_distance(file, rank, enemy_king_file, enemy_king_rank);
            int value = std::max(0, 7 - distance);
            if (piece == QUEEN) {
                value *= 3;
            } else if (piece == ROOK) {
                value *= 2;
            }
            if (piece_has_line_to_square(board, file, rank, piece, enemy_king_file, enemy_king_rank)) {
                value += 8;
            }
            pressure += value;
        }
    }
    return std::min(45, pressure);
}

int attack_momentum_score(const int8_t* board, int white_king_file, int white_king_rank, int black_king_file, int black_king_rank) {
    return king_attack_pressure(board, 1, black_king_file, black_king_rank) -
        king_attack_pressure(board, -1, white_king_file, white_king_rank);
}

bool same_line(int first_file, int first_rank, int second_file, int second_rank, int third_file, int third_rank) {
    int first_file_delta = second_file - first_file;
    int first_rank_delta = second_rank - first_rank;
    int second_file_delta = third_file - first_file;
    int second_rank_delta = third_rank - first_rank;
    return first_file_delta * second_rank_delta == first_rank_delta * second_file_delta;
}

int queen_king_battery_score(const int8_t* board, int color, int enemy_king_file, int enemy_king_rank) {
    if (enemy_king_file < 0) {
        return 0;
    }
    int value = 0;
    for (int rank = 0; rank < 8; ++rank) {
        for (int file = 0; file < 8; ++file) {
            int code = board[idx(file, rank)];
            if (color_of(code) != color || type_of(code) != QUEEN) {
                continue;
            }
            if (
                piece_has_line_to_square(board, file, rank, QUEEN, enemy_king_file, enemy_king_rank) &&
                king_distance(file, rank, enemy_king_file, enemy_king_rank) <= 5
            ) {
                value += 18;
            }
            for (int support_rank = 0; support_rank < 8; ++support_rank) {
                for (int support_file = 0; support_file < 8; ++support_file) {
                    int support = board[idx(support_file, support_rank)];
                    int support_piece = type_of(support);
                    if (color_of(support) != color || (support_piece != BISHOP && support_piece != ROOK)) {
                        continue;
                    }
                    if (same_line(file, rank, support_file, support_rank, enemy_king_file, enemy_king_rank)) {
                        value += 10;
                    }
                }
            }
        }
    }
    return std::min(36, value);
}

bool square_in_king_ring(int file, int rank, int king_file, int king_rank) {
    return king_file >= 0 && std::max(std::abs(file - king_file), std::abs(rank - king_rank)) <= 1;
}

int minor_piece_fork_pressure_score(const int8_t* board, int color, int enemy_king_file, int enemy_king_rank) {
    constexpr int knight_offsets[8][2] = {{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}};
    int value = 0;
    for (int rank = 0; rank < 8; ++rank) {
        for (int file = 0; file < 8; ++file) {
            int code = board[idx(file, rank)];
            if (color_of(code) != color || type_of(code) != KNIGHT) {
                continue;
            }
            int attacked_high_value = 0;
            for (const auto& offset : knight_offsets) {
                int target_file = file + offset[0];
                int target_rank = rank + offset[1];
                if (!inside(target_file, target_rank)) {
                    continue;
                }
                int target = board[idx(target_file, target_rank)];
                int target_piece = type_of(target);
                if (color_of(target) == -color && (target_piece == ROOK || target_piece == QUEEN)) {
                    ++attacked_high_value;
                }
                if (square_in_king_ring(target_file, target_rank, enemy_king_file, enemy_king_rank)) {
                    ++attacked_high_value;
                }
            }
            if (attacked_high_value >= 2) {
                value += 22;
            }
        }
    }
    return std::min(44, value);
}

int rook_seventh_rank_tactical_score(const int8_t* board, int color, int enemy_king_rank) {
    int target_rank = color > 0 ? 6 : 1;
    int enemy_home_rank = color > 0 ? 7 : 0;
    int value = 0;
    for (int file = 0; file < 8; ++file) {
        int code = board[idx(file, target_rank)];
        if (color_of(code) != color || type_of(code) != ROOK) {
            continue;
        }
        value += 18;
        if (enemy_king_rank == enemy_home_rank) {
            value += 10;
        }
    }
    return std::min(42, value);
}

int knight_rim_penalty_score(const int8_t* board, int color) {
    int penalty = 0;
    for (int rank = 0; rank < 8; ++rank) {
        for (int file = 0; file < 8; ++file) {
            int code = board[idx(file, rank)];
            if (color_of(code) == color && type_of(code) == KNIGHT && (file == 0 || file == 7 || rank == 0 || rank == 7)) {
                penalty += 8;
            }
        }
    }
    return -penalty;
}

int tactical_motif_score(const int8_t* board, int white_king_file, int white_king_rank, int black_king_file, int black_king_rank) {
    int white_value = queen_king_battery_score(board, 1, black_king_file, black_king_rank) +
        minor_piece_fork_pressure_score(board, 1, black_king_file, black_king_rank) +
        rook_seventh_rank_tactical_score(board, 1, black_king_rank) +
        knight_rim_penalty_score(board, 1);
    int black_value = queen_king_battery_score(board, -1, white_king_file, white_king_rank) +
        minor_piece_fork_pressure_score(board, -1, white_king_file, white_king_rank) +
        rook_seventh_rank_tactical_score(board, -1, white_king_rank) +
        knight_rim_penalty_score(board, -1);
    return white_value - black_value;
}

int book_knowledge_score(const int8_t* board) {
    return knight_outpost_score(board) +
        bad_bishop_score(board) +
        space_advantage_score(board) +
        advanced_pawn_structure_score(board) +
        weak_square_score(board) +
        piece_quality_score(board) +
        locked_center_minor_score(board) +
        color_complex_score(board) +
        back_rank_weakness_score(board);
}

int king_shield_score(const int8_t* board, int king_file, int king_rank, int color) {
    if (king_file < 0) {
        return 0;
    }
    int score = 0;
    int direction = color > 0 ? 1 : -1;
    int shield_rank = king_rank + direction;
    if (inside(king_file, shield_rank)) {
        for (int file = std::max(0, king_file - 1); file <= std::min(7, king_file + 1); ++file) {
            int code = board[idx(file, shield_rank)];
            score += (color_of(code) == color && type_of(code) == PAWN) ? 12 : -8;
        }
    }
    for (int file = std::max(0, king_file - 1); file <= std::min(7, king_file + 1); ++file) {
        if (!file_has_pawn(board, file, color) && !file_has_pawn(board, file, -color)) {
            score -= 10;
        } else if (!file_has_pawn(board, file, color)) {
            score -= 5;
        }
    }
    return color * score;
}

int endgame_piece_pressure_score(const int8_t* board, int weaker_king_file, int weaker_king_rank, int stronger) {
    if (weaker_king_file < 0) {
        return 0;
    }
    int pressure = 0;
    for (int rank = 0; rank < 8; ++rank) {
        for (int file = 0; file < 8; ++file) {
            int code = board[idx(file, rank)];
            if (color_of(code) != stronger) {
                continue;
            }
            int piece = type_of(code);
            if (piece == KING || piece == PAWN) {
                continue;
            }
            int distance = std::abs(file - weaker_king_file) + std::abs(rank - weaker_king_rank);
            int closeness = std::max(0, 14 - distance * 2);
            int piece_factor = piece == QUEEN ? 4 : (piece == ROOK ? 3 : 2);
            pressure += closeness * piece_factor;
            if (file == weaker_king_file || rank == weaker_king_rank) {
                pressure += piece_factor * 10;
            }
        }
    }
    return stronger * pressure;
}

int passed_pawn_path_bonus(
    const int8_t* board,
    int file,
    int rank,
    int color,
    int enemy_king_file,
    int enemy_king_rank
) {
    int direction = color > 0 ? 1 : -1;
    int promotion_rank = color > 0 ? 7 : 0;
    int advancement = color > 0 ? rank : 7 - rank;
    int bonus = 0;

    bool clear_path = true;
    for (int r = rank + direction; r >= 0 && r < 8; r += direction) {
        if (board[idx(file, r)] != 0) {
            clear_path = false;
            break;
        }
    }
    if (clear_path) {
        bonus += 18 + advancement * 8;
    }

    if (enemy_king_file >= 0) {
        int pawn_steps = std::abs(promotion_rank - rank);
        int enemy_steps = king_distance(enemy_king_file, enemy_king_rank, file, promotion_rank);
        if (enemy_steps > pawn_steps) {
            bonus += 20 + (enemy_steps - pawn_steps) * 8;
        } else if (enemy_steps <= pawn_steps - 1) {
            bonus -= 12;
        }
    }

    return color * bonus;
}

int major_piece_behind_passed_pawn(const int8_t* board, int file, int rank, int color) {
    int count = 0;
    int start = color > 0 ? 0 : rank + 1;
    int stop = color > 0 ? rank : 8;
    for (int r = start; r < stop; ++r) {
        int code = board[idx(file, r)];
        int piece = type_of(code);
        if (color_of(code) == color && (piece == ROOK || piece == QUEEN)) {
            ++count;
        }
    }
    return count;
}

int passed_pawn_support_score(
    const int8_t* board,
    int file,
    int rank,
    int color,
    int friendly_king_file,
    int friendly_king_rank,
    int enemy_king_file,
    int enemy_king_rank
) {
    int promotion_rank = color > 0 ? 7 : 0;
    int value = 0;
    if (friendly_king_file >= 0) {
        value += std::max(0, 7 - king_distance(friendly_king_file, friendly_king_rank, file, rank)) * 2;
        value += std::max(0, 7 - king_distance(friendly_king_file, friendly_king_rank, file, promotion_rank));
    }
    if (enemy_king_file >= 0) {
        int pawn_steps = std::abs(promotion_rank - rank);
        int enemy_steps = king_distance(enemy_king_file, enemy_king_rank, file, promotion_rank);
        value += std::max(0, enemy_steps - pawn_steps) * 4;
    }
    value += major_piece_behind_passed_pawn(board, file, rank, color) * 10;
    return color * value;
}

int outside_passer_score(const int8_t* board) {
    int score = 0;
    for (int rank = 0; rank < 8; ++rank) {
        for (int file = 0; file < 8; ++file) {
            int code = board[idx(file, rank)];
            if (type_of(code) != PAWN) {
                continue;
            }
            int color = color_of(code);
            if (!is_passed_pawn(board, file, rank, color) || (file != 0 && file != 1 && file != 6 && file != 7)) {
                continue;
            }
            int advancement = color > 0 ? rank : 7 - rank;
            score += color * (6 + advancement * 2);
        }
    }
    return score;
}

int bishop_vs_knight_pawn_structure_score(const int8_t* board) {
    int locked_pawns = 0;
    int white_knights = 0;
    int black_knights = 0;
    int white_bishops = 0;
    int black_bishops = 0;
    for (int rank = 0; rank < 8; ++rank) {
        for (int file = 0; file < 8; ++file) {
            int code = board[idx(file, rank)];
            int piece = type_of(code);
            int color = color_of(code);
            if (piece == PAWN) {
                int front_rank = rank + (color > 0 ? 1 : -1);
                if (inside(file, front_rank)) {
                    int blocker = board[idx(file, front_rank)];
                    if (color_of(blocker) == -color && type_of(blocker) == PAWN) {
                        ++locked_pawns;
                    }
                }
            } else if (piece == KNIGHT) {
                if (color > 0) {
                    ++white_knights;
                } else if (color < 0) {
                    ++black_knights;
                }
            } else if (piece == BISHOP) {
                if (color > 0) {
                    ++white_bishops;
                } else if (color < 0) {
                    ++black_bishops;
                }
            }
        }
    }
    if (locked_pawns < 3) {
        return 0;
    }
    int white_value = white_knights * 8 - white_bishops * 4;
    int black_value = black_knights * 8 - black_bishops * 4;
    return white_value - black_value;
}

int endgame_theory_score(const int8_t* board) {
    return outside_passer_score(board) + bishop_vs_knight_pawn_structure_score(board);
}

int connected_passed_pawn_bonus(const int8_t* board, int file, int rank, int color) {
    int best = 0;
    for (int adjacent_file : {file - 1, file + 1}) {
        if (adjacent_file < 0 || adjacent_file >= 8) {
            continue;
        }
        for (int rank_delta : {-1, 0, 1}) {
            int partner_rank = rank + rank_delta;
            if (partner_rank < 0 || partner_rank >= 8) {
                continue;
            }
            int code = board[idx(adjacent_file, partner_rank)];
            if (color_of(code) != color || type_of(code) != PAWN) {
                continue;
            }
            if (!is_passed_pawn(board, adjacent_file, partner_rank, color)) {
                continue;
            }
            int advancement = color > 0 ? rank : 7 - rank;
            best = std::max(best, 10 + advancement * 4);
        }
    }
    return best;
}

int king_cutoff_score(
    const int8_t* board,
    int material,
    int white_king_file,
    int white_king_rank,
    int black_king_file,
    int black_king_rank
) {
    if (std::abs(material) < 300 || white_king_file < 0 || black_king_file < 0) {
        return 0;
    }
    int stronger = material > 0 ? 1 : -1;
    int weaker_king_file = stronger > 0 ? black_king_file : white_king_file;
    int weaker_king_rank = stronger > 0 ? black_king_rank : white_king_rank;
    int score = 0;
    for (int rank = 0; rank < 8; ++rank) {
        for (int file = 0; file < 8; ++file) {
            int code = board[idx(file, rank)];
            if (color_of(code) != stronger) {
                continue;
            }
            int piece = type_of(code);
            if (piece != ROOK && piece != QUEEN) {
                continue;
            }
            int controlled_escape_squares = 0;
            for (int df = -1; df <= 1; ++df) {
                for (int dr = -1; dr <= 1; ++dr) {
                    if (df == 0 && dr == 0) {
                        continue;
                    }
                    int target_file = weaker_king_file + df;
                    int target_rank = weaker_king_rank + dr;
                    if (inside(target_file, target_rank) && piece_has_line_to_square(board, file, rank, piece, target_file, target_rank)) {
                        ++controlled_escape_squares;
                    }
                }
            }
            int value = controlled_escape_squares * (piece == ROOK ? 8 : 10);
            if (piece_has_line_to_square(board, file, rank, piece, weaker_king_file, weaker_king_rank)) {
                value += 12;
            }
            score += std::min(42, value);
        }
    }
    return stronger * score;
}

int simplification_score(int material, int non_pawn_material) {
    if (std::abs(material) < 300) {
        return 0;
    }
    int stronger = material > 0 ? 1 : -1;
    int traded_material = std::max(0, ENDGAME_NON_PAWN_MATERIAL_THRESHOLD - non_pawn_material);
    int lead_bonus = std::min(60, std::abs(material) / 30);
    int trade_bonus = std::min(80, traded_material / 18);
    return stronger * (lead_bonus + trade_bonus);
}

void add_step_mobility(const int8_t* board, int file, int rank, int color, const int offsets[][2], int count, int& moves) {
    for (int i = 0; i < count; ++i) {
        int f = file + offsets[i][0];
        int r = rank + offsets[i][1];
        if (!inside(f, r)) {
            continue;
        }
        int target = board[idx(f, r)];
        if (target == 0 || color_of(target) == -color) {
            ++moves;
        }
    }
}

void add_slide_mobility(const int8_t* board, int file, int rank, int color, const int offsets[][2], int count, int& moves) {
    for (int i = 0; i < count; ++i) {
        int f = file + offsets[i][0];
        int r = rank + offsets[i][1];
        while (inside(f, r)) {
            int target = board[idx(f, r)];
            if (target == 0) {
                ++moves;
            } else {
                if (color_of(target) == -color) {
                    ++moves;
                }
                break;
            }
            f += offsets[i][0];
            r += offsets[i][1];
        }
    }
}

int pseudo_mobility(const int8_t* board, int color) {
    constexpr int knight_offsets[8][2] = {{1, 2}, {2, 1}, {-1, 2}, {-2, 1}, {1, -2}, {2, -1}, {-1, -2}, {-2, -1}};
    constexpr int bishop_offsets[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    constexpr int rook_offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    constexpr int queen_offsets[8][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}, {1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    constexpr int king_offsets[8][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}, {1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    int moves = 0;
    for (int rank = 0; rank < 8; ++rank) {
        for (int file = 0; file < 8; ++file) {
            int code = board[idx(file, rank)];
            if (color_of(code) != color) {
                continue;
            }
            int piece = type_of(code);
            if (piece == PAWN) {
                int dir = color > 0 ? 1 : -1;
                if (inside(file, rank + dir) && board[idx(file, rank + dir)] == 0) {
                    ++moves;
                }
                for (int df : {-1, 1}) {
                    int f = file + df;
                    int r = rank + dir;
                    if (inside(f, r) && color_of(board[idx(f, r)]) == -color) {
                        ++moves;
                    }
                }
            } else if (piece == KNIGHT) {
                add_step_mobility(board, file, rank, color, knight_offsets, 8, moves);
            } else if (piece == BISHOP) {
                add_slide_mobility(board, file, rank, color, bishop_offsets, 4, moves);
            } else if (piece == ROOK) {
                add_slide_mobility(board, file, rank, color, rook_offsets, 4, moves);
            } else if (piece == QUEEN) {
                add_slide_mobility(board, file, rank, color, queen_offsets, 8, moves);
            } else if (piece == KING) {
                add_step_mobility(board, file, rank, color, king_offsets, 8, moves);
            }
        }
    }
    return moves;
}

int evaluate_internal(
    const int8_t* board,
    double material_weight,
    double position_weight,
    double mobility_weight,
    double endgame_weight,
    double pawn_structure_weight,
    double bishop_pair_weight,
    double rook_activity_weight,
    double king_safety_weight,
    double passed_pawn_weight,
    double endgame_pressure_weight,
    double line_activity_weight,
    double attack_momentum_weight,
    int* breakdown = nullptr,
    int max_breakdown = 0
) {
    if (has_insufficient_material_draw(board)) {
        if (breakdown) {
            for (int index = 0; index < max_breakdown && index < 13; ++index) {
                breakdown[index] = 0;
            }
        }
        return 0;
    }

    int material = 0;
    int position = 0;
    int non_pawn_material = 0;
    int white_king_file = -1;
    int white_king_rank = -1;
    int black_king_file = -1;
    int black_king_rank = -1;

    for (int rank = 0; rank < 8; ++rank) {
        for (int file = 0; file < 8; ++file) {
            int code = board[idx(file, rank)];
            if (code == 0) {
                continue;
            }
            int color = color_of(code);
            int piece = type_of(code);
            material += color * PIECE_VALUES[piece];
            position += color * piece_square_value(piece, file, rank, color);
            if (piece != KING && piece != PAWN) {
                non_pawn_material += PIECE_VALUES[piece];
            }
            if (piece == KING && color > 0) {
                white_king_file = file;
                white_king_rank = rank;
            } else if (piece == KING && color < 0) {
                black_king_file = file;
                black_king_rank = rank;
            }
        }
    }

    int mobility = (pseudo_mobility(board, 1) - pseudo_mobility(board, -1)) * MOBILITY_WEIGHT;

    double phase = std::max(0.0, std::min(1.0, 1.0 - static_cast<double>(non_pawn_material) / ENDGAME_NON_PAWN_MATERIAL_THRESHOLD));
    int pawn_structure = pawn_structure_score(board);
    int bishop_pair = bishop_pair_score(board);
    int rook_activity = rook_file_score(board);
    int king_safety = king_shield_score(board, white_king_file, white_king_rank, 1) +
        king_shield_score(board, black_king_file, black_king_rank, -1);
    int line_activity = line_activity_score(board);
    int attack_momentum = attack_momentum_score(
        board, white_king_file, white_king_rank, black_king_file, black_king_rank
    );
    int opening_fundamentals = central_control_score(board) +
        opening_development_score(board, phase) +
        king_opening_safety_score(
            white_king_file,
            white_king_rank,
            black_king_file,
            black_king_rank,
            phase
        );
    int loose_threats = loose_piece_threat_score(board);
    int book_knowledge = book_knowledge_score(board);
    int exchange_strategy = exchange_strategy_score(material, phase);
    int tactical_motif = tactical_motif_score(
        board, white_king_file, white_king_rank, black_king_file, black_king_rank
    );
    // pawn_structure/bishop_pair/rook_activity/king_safety each already
    // carry their own dedicated weight -- keep them out of `strategic`,
    // which gets multiplied by position_weight again below. Folding them
    // in there silently coupled four "independent" weights to
    // position_weight (tuning position_weight rescaled all four too).
    int weighted_substructure = static_cast<int>(
        pawn_structure * pawn_structure_weight +
        bishop_pair * bishop_pair_weight +
        rook_activity * rook_activity_weight +
        king_safety * king_safety_weight * (1.0 - phase)
    );
    int strategic = static_cast<int>(
        (book_knowledge + exchange_strategy + tactical_motif) * position_weight +
        static_cast<int>((opening_fundamentals * (1.0 - phase) + loose_threats) * position_weight)
    );

    int passed_pawns = 0;
    for (int rank = 0; rank < 8; ++rank) {
        for (int file = 0; file < 8; ++file) {
            int code = board[idx(file, rank)];
            if (type_of(code) != PAWN) {
                continue;
            }
            int color = color_of(code);
            if (!is_passed_pawn(board, file, rank, color)) {
                continue;
            }
            int advancement = color > 0 ? rank : 7 - rank;
            int value = 18 + advancement * advancement * 5;
            if (advancement >= 5) {
                value += 35;
            }
            int enemy_king_file = color > 0 ? black_king_file : white_king_file;
            int enemy_king_rank = color > 0 ? black_king_rank : white_king_rank;
            int friendly_king_file = color > 0 ? white_king_file : black_king_file;
            int friendly_king_rank = color > 0 ? white_king_rank : black_king_rank;
            value += color * passed_pawn_path_bonus(board, file, rank, color, enemy_king_file, enemy_king_rank);
            value += color * passed_pawn_support_score(
                board,
                file,
                rank,
                color,
                friendly_king_file,
                friendly_king_rank,
                enemy_king_file,
                enemy_king_rank
            );
            value += connected_passed_pawn_bonus(board, file, rank, color);
            passed_pawns += color * value;
        }
    }

    int endgame = 0;
    int pressure = 0;
    if (phase > 0.0 && white_king_file >= 0 && black_king_file >= 0) {
        if (material != 0 || passed_pawns != 0) {
            endgame += central_score(white_king_file, white_king_rank) - central_score(black_king_file, black_king_rank);
        }
        if (std::abs(material) >= 300) {
            int stronger = material > 0 ? 1 : -1;
            int distance = king_distance(white_king_file, white_king_rank, black_king_file, black_king_rank);
            int proximity = std::max(0, 14 - distance * 2);
            int weaker_edge = stronger > 0 ? edge_score(black_king_file, black_king_rank) : edge_score(white_king_file, white_king_rank);
            endgame += stronger * (proximity * 2 + weaker_edge * 12);
            int weaker_file = stronger > 0 ? black_king_file : white_king_file;
            int weaker_rank = stronger > 0 ? black_king_rank : white_king_rank;
            pressure += endgame_piece_pressure_score(board, weaker_file, weaker_rank, stronger);
        }
        endgame += simplification_score(material, non_pawn_material);
        endgame += king_cutoff_score(board, material, white_king_file, white_king_rank, black_king_file, black_king_rank);
        endgame += endgame_theory_score(board);
    }
    // Same fix as weighted_substructure above: passed_pawn_weight and
    // endgame_pressure_weight are each already dedicated weights, so pull
    // their terms out before endgame gets multiplied by endgame_weight
    // below -- otherwise they'd ride endgame_weight a second time. Both
    // stay phase-scaled (that part was correct).
    int passed_pawn_component = static_cast<int>(passed_pawns * passed_pawn_weight * phase);
    int endgame_pressure_component = static_cast<int>(pressure * endgame_pressure_weight * phase);
    endgame = static_cast<int>(endgame * phase);

    int total = static_cast<int>(
        material * material_weight +
        (position + strategic) * position_weight +
        weighted_substructure +
        passed_pawn_component +
        endgame_pressure_component +
        line_activity * line_activity_weight +
        attack_momentum * attack_momentum_weight +
        mobility * mobility_weight +
        endgame * endgame_weight
    );
    if (breakdown && max_breakdown >= 15) {
        breakdown[0] = material;
        breakdown[1] = position;
        breakdown[2] = mobility;
        breakdown[3] = pawn_structure;
        breakdown[4] = bishop_pair;
        breakdown[5] = rook_activity;
        breakdown[6] = king_safety;
        breakdown[7] = book_knowledge;
        breakdown[8] = line_activity;
        breakdown[9] = attack_momentum;
        breakdown[10] = endgame;
        breakdown[11] = strategic;
        breakdown[12] = total;
        // Raw (unweighted, pre-phase-scale) passed-pawn and endgame-pressure
        // terms -- needed so training feature extraction can read the same
        // quantities the evaluator actually scores, instead of a separate
        // Python reimplementation that used different constants/logic.
        breakdown[13] = passed_pawns;
        breakdown[14] = pressure;
    }
    return total;
}

bool is_attacked_by_pawn(const int8_t* board, int file, int rank, int by_color) {
    int pawn_direction = by_color > 0 ? 1 : -1;
    int attacker_rank = rank - pawn_direction;
    if (!inside(file, attacker_rank)) {
        return false;
    }
    for (int df : {-1, 1}) {
        int attacker_file = file + df;
        if (!inside(attacker_file, attacker_rank)) {
            continue;
        }
        int code = board[idx(attacker_file, attacker_rank)];
        if (color_of(code) == by_color && type_of(code) == PAWN) {
            return true;
        }
    }
    return false;
}

bool is_attacked_by_step_piece(
    const int8_t* board,
    int file,
    int rank,
    int by_color,
    int piece,
    const int offsets[][2],
    int count
) {
    for (int i = 0; i < count; ++i) {
        int attacker_file = file + offsets[i][0];
        int attacker_rank = rank + offsets[i][1];
        if (!inside(attacker_file, attacker_rank)) {
            continue;
        }
        int code = board[idx(attacker_file, attacker_rank)];
        if (color_of(code) == by_color && type_of(code) == piece) {
            return true;
        }
    }
    return false;
}

bool is_attacked_by_slider(
    const int8_t* board,
    int file,
    int rank,
    int by_color,
    const int offsets[][2],
    int count,
    int first_piece,
    int second_piece
) {
    for (int i = 0; i < count; ++i) {
        int current_file = file + offsets[i][0];
        int current_rank = rank + offsets[i][1];
        while (inside(current_file, current_rank)) {
            int code = board[idx(current_file, current_rank)];
            if (code == 0) {
                current_file += offsets[i][0];
                current_rank += offsets[i][1];
                continue;
            }
            int piece = type_of(code);
            if (color_of(code) == by_color && (piece == first_piece || piece == second_piece)) {
                return true;
            }
            break;
        }
    }
    return false;
}

bool square_attacked_internal(const int8_t* board, int file, int rank, int by_color) {
    constexpr int knight_offsets[8][2] = {{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}};
    constexpr int king_offsets[8][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    constexpr int diagonal_offsets[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    constexpr int orthogonal_offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

    if (!inside(file, rank) || (by_color != 1 && by_color != -1)) {
        return false;
    }
    return (
        is_attacked_by_pawn(board, file, rank, by_color) ||
        is_attacked_by_step_piece(board, file, rank, by_color, KNIGHT, knight_offsets, 8) ||
        is_attacked_by_step_piece(board, file, rank, by_color, KING, king_offsets, 8) ||
        is_attacked_by_slider(board, file, rank, by_color, diagonal_offsets, 4, BISHOP, QUEEN) ||
        is_attacked_by_slider(board, file, rank, by_color, orthogonal_offsets, 4, ROOK, QUEEN)
    );
}

int promotion_code(int piece) {
    if (piece == QUEEN) {
        return 1;
    }
    if (piece == ROOK) {
        return 2;
    }
    if (piece == BISHOP) {
        return 3;
    }
    if (piece == KNIGHT) {
        return 4;
    }
    return 0;
}

int encode_move(int from_file, int from_rank, int to_file, int to_rank, int promotion, int flags) {
    int from = idx(from_file, from_rank);
    int to = idx(to_file, to_rank);
    return from | (to << 6) | (promotion << 12) | (flags << 16);
}

void push_move(int* output, int max_count, int& count, int encoded) {
    if (count < max_count) {
        output[count] = encoded;
    }
    ++count;
}

void push_pawn_move(int* output, int max_count, int& count, int from_file, int from_rank, int to_file, int to_rank, int promotion_rank, int flags = 0) {
    if (to_rank == promotion_rank) {
        push_move(output, max_count, count, encode_move(from_file, from_rank, to_file, to_rank, promotion_code(QUEEN), flags));
        push_move(output, max_count, count, encode_move(from_file, from_rank, to_file, to_rank, promotion_code(ROOK), flags));
        push_move(output, max_count, count, encode_move(from_file, from_rank, to_file, to_rank, promotion_code(BISHOP), flags));
        push_move(output, max_count, count, encode_move(from_file, from_rank, to_file, to_rank, promotion_code(KNIGHT), flags));
        return;
    }
    push_move(output, max_count, count, encode_move(from_file, from_rank, to_file, to_rank, 0, flags));
}

void generate_pawn_moves(
    const int8_t* board,
    int color,
    int file,
    int rank,
    int ep_file,
    int ep_rank,
    int* output,
    int max_count,
    int& count
) {
    int direction = color > 0 ? 1 : -1;
    int start_rank = color > 0 ? 1 : 6;
    int promotion_rank = color > 0 ? 7 : 0;
    int one_rank = rank + direction;
    if (inside(file, one_rank) && board[idx(file, one_rank)] == 0) {
        push_pawn_move(output, max_count, count, file, rank, file, one_rank, promotion_rank);
        int two_rank = rank + direction * 2;
        if (rank == start_rank && inside(file, two_rank) && board[idx(file, two_rank)] == 0) {
            push_move(output, max_count, count, encode_move(file, rank, file, two_rank, 0, 0));
        }
    }
    for (int df : {-1, 1}) {
        int to_file = file + df;
        int to_rank = rank + direction;
        if (!inside(to_file, to_rank)) {
            continue;
        }
        int target = board[idx(to_file, to_rank)];
        if (target != 0 && color_of(target) == -color) {
            push_pawn_move(output, max_count, count, file, rank, to_file, to_rank, promotion_rank);
        } else if (to_file == ep_file && to_rank == ep_rank) {
            push_move(output, max_count, count, encode_move(file, rank, to_file, to_rank, 0, 1));
        }
    }
}

void generate_step_moves(
    const int8_t* board,
    int color,
    int file,
    int rank,
    const int offsets[][2],
    int offset_count,
    int* output,
    int max_count,
    int& count
) {
    for (int i = 0; i < offset_count; ++i) {
        int to_file = file + offsets[i][0];
        int to_rank = rank + offsets[i][1];
        if (!inside(to_file, to_rank)) {
            continue;
        }
        int target = board[idx(to_file, to_rank)];
        if (target == 0 || color_of(target) == -color) {
            push_move(output, max_count, count, encode_move(file, rank, to_file, to_rank, 0, 0));
        }
    }
}

void generate_slider_moves(
    const int8_t* board,
    int color,
    int file,
    int rank,
    const int offsets[][2],
    int offset_count,
    int* output,
    int max_count,
    int& count
) {
    for (int i = 0; i < offset_count; ++i) {
        int to_file = file + offsets[i][0];
        int to_rank = rank + offsets[i][1];
        while (inside(to_file, to_rank)) {
            int target = board[idx(to_file, to_rank)];
            if (target == 0) {
                push_move(output, max_count, count, encode_move(file, rank, to_file, to_rank, 0, 0));
            } else {
                if (color_of(target) == -color) {
                    push_move(output, max_count, count, encode_move(file, rank, to_file, to_rank, 0, 0));
                }
                break;
            }
            to_file += offsets[i][0];
            to_rank += offsets[i][1];
        }
    }
}

void generate_castling_moves(
    const int8_t* board,
    int color,
    int file,
    int rank,
    int white_king_side,
    int white_queen_side,
    int black_king_side,
    int black_queen_side,
    int* output,
    int max_count,
    int& count
) {
    int home_rank = color > 0 ? 0 : 7;
    if (file != 4 || rank != home_rank || type_of(board[idx(file, rank)]) != KING) {
        return;
    }
    if (square_attacked_internal(board, file, rank, -color)) {
        return;
    }
    int king_side = color > 0 ? white_king_side : black_king_side;
    int queen_side = color > 0 ? white_queen_side : black_queen_side;
    if (king_side && board[idx(7, rank)] == color * ROOK && board[idx(5, rank)] == 0 && board[idx(6, rank)] == 0) {
        if (!square_attacked_internal(board, 5, rank, -color) && !square_attacked_internal(board, 6, rank, -color)) {
            push_move(output, max_count, count, encode_move(4, rank, 6, rank, 0, 2));
        }
    }
    if (queen_side && board[idx(0, rank)] == color * ROOK && board[idx(1, rank)] == 0 && board[idx(2, rank)] == 0 && board[idx(3, rank)] == 0) {
        if (!square_attacked_internal(board, 3, rank, -color) && !square_attacked_internal(board, 2, rank, -color)) {
            push_move(output, max_count, count, encode_move(4, rank, 2, rank, 0, 2));
        }
    }
}

int generate_pseudo_legal_moves_internal(
    const int8_t* board,
    int color,
    int white_king_side,
    int white_queen_side,
    int black_king_side,
    int black_queen_side,
    int ep_file,
    int ep_rank,
    int* output,
    int max_count
) {
    constexpr int knight_offsets[8][2] = {{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}};
    constexpr int bishop_offsets[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    constexpr int rook_offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    constexpr int queen_offsets[8][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}, {1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    constexpr int king_offsets[8][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    int count = 0;
    if (color != 1 && color != -1) {
        return 0;
    }
    for (int rank = 0; rank < 8; ++rank) {
        for (int file = 0; file < 8; ++file) {
            int code = board[idx(file, rank)];
            if (color_of(code) != color) {
                continue;
            }
            int piece = type_of(code);
            if (piece == PAWN) {
                generate_pawn_moves(board, color, file, rank, ep_file, ep_rank, output, max_count, count);
            } else if (piece == KNIGHT) {
                generate_step_moves(board, color, file, rank, knight_offsets, 8, output, max_count, count);
            } else if (piece == BISHOP) {
                generate_slider_moves(board, color, file, rank, bishop_offsets, 4, output, max_count, count);
            } else if (piece == ROOK) {
                generate_slider_moves(board, color, file, rank, rook_offsets, 4, output, max_count, count);
            } else if (piece == QUEEN) {
                generate_slider_moves(board, color, file, rank, queen_offsets, 8, output, max_count, count);
            } else if (piece == KING) {
                generate_step_moves(board, color, file, rank, king_offsets, 8, output, max_count, count);
                generate_castling_moves(
                    board,
                    color,
                    file,
                    rank,
                    white_king_side,
                    white_queen_side,
                    black_king_side,
                    black_queen_side,
                    output,
                    max_count,
                    count
                );
            }
        }
    }
    return count;
}

int piece_from_promotion_code(int promotion) {
    if (promotion == 1) {
        return QUEEN;
    }
    if (promotion == 2) {
        return ROOK;
    }
    if (promotion == 3) {
        return BISHOP;
    }
    if (promotion == 4) {
        return KNIGHT;
    }
    return 0;
}

void apply_encoded_move(const int8_t* board, int encoded, int8_t* next_board) {
    for (int i = 0; i < 64; ++i) {
        next_board[i] = board[i];
    }
    int from = encoded & 0x3F;
    int to = (encoded >> 6) & 0x3F;
    int promotion = (encoded >> 12) & 0xF;
    int flags = (encoded >> 16) & 0xF;
    int moving_piece = next_board[from];
    int color = color_of(moving_piece);
    next_board[from] = 0;

    if (flags & 1) {
        int from_rank = from / 8;
        int to_file = to % 8;
        next_board[idx(to_file, from_rank)] = 0;
    }

    if (flags & 2) {
        int rank = from / 8;
        if ((to % 8) == 6) {
            next_board[idx(5, rank)] = next_board[idx(7, rank)];
            next_board[idx(7, rank)] = 0;
        } else if ((to % 8) == 2) {
            next_board[idx(3, rank)] = next_board[idx(0, rank)];
            next_board[idx(0, rank)] = 0;
        }
    }

    int promoted_piece = piece_from_promotion_code(promotion);
    next_board[to] = promoted_piece ? color * promoted_piece : moving_piece;
}

bool find_king_internal(const int8_t* board, int color, int& file, int& rank) {
    for (int current_rank = 0; current_rank < 8; ++current_rank) {
        for (int current_file = 0; current_file < 8; ++current_file) {
            int code = board[idx(current_file, current_rank)];
            if (color_of(code) == color && type_of(code) == KING) {
                file = current_file;
                rank = current_rank;
                return true;
            }
        }
    }
    return false;
}

bool leaves_king_safe(const int8_t* board, int encoded, int color) {
    int8_t next_board[64];
    apply_encoded_move(board, encoded, next_board);
    int king_file = -1;
    int king_rank = -1;
    if (!find_king_internal(next_board, color, king_file, king_rank)) {
        return false;
    }
    return !square_attacked_internal(next_board, king_file, king_rank, -color);
}

struct NativeState {
    int8_t board[64];
    int side_to_move;
    int white_king_side;
    int white_queen_side;
    int black_king_side;
    int black_queen_side;
    int ep_file;
    int ep_rank;
};

int captured_piece_for_encoded_move(const NativeState& state, int encoded) {
    int from = encoded & 0x3F;
    int to = (encoded >> 6) & 0x3F;
    int flags = (encoded >> 16) & 0xF;
    if (flags & 1) {
        return state.board[idx(to % 8, from / 8)];
    }
    return state.board[to];
}

void revoke_castling_for_square(NativeState& state, int square, int piece) {
    if (type_of(piece) == KING) {
        if (color_of(piece) > 0) {
            state.white_king_side = 0;
            state.white_queen_side = 0;
        } else {
            state.black_king_side = 0;
            state.black_queen_side = 0;
        }
    }
    if (type_of(piece) == ROOK) {
        if (square == idx(0, 0)) {
            state.white_queen_side = 0;
        } else if (square == idx(7, 0)) {
            state.white_king_side = 0;
        } else if (square == idx(0, 7)) {
            state.black_queen_side = 0;
        } else if (square == idx(7, 7)) {
            state.black_king_side = 0;
        }
    }
}

NativeState make_state_move(const NativeState& state, int encoded) {
    NativeState next = state;
    int from = encoded & 0x3F;
    int to = (encoded >> 6) & 0x3F;
    int promotion = (encoded >> 12) & 0xF;
    int flags = (encoded >> 16) & 0xF;
    int moving_piece = state.board[from];
    int captured_piece = captured_piece_for_encoded_move(state, encoded);
    revoke_castling_for_square(next, from, moving_piece);
    if (captured_piece != 0) {
        revoke_castling_for_square(next, to, captured_piece);
        if (flags & 1) {
            revoke_castling_for_square(next, idx(to % 8, from / 8), captured_piece);
        }
    }

    apply_encoded_move(state.board, encoded, next.board);
    next.ep_file = -1;
    next.ep_rank = -1;
    if (type_of(moving_piece) == PAWN && std::abs((to / 8) - (from / 8)) == 2) {
        next.ep_file = from % 8;
        next.ep_rank = ((to / 8) + (from / 8)) / 2;
    }
    next.side_to_move = -state.side_to_move;
    (void)promotion;
    return next;
}

NativeState make_null_move(const NativeState& state) {
    NativeState next = state;
    next.side_to_move = -state.side_to_move;
    next.ep_file = -1;
    next.ep_rank = -1;
    return next;
}

int non_pawn_material_for_color(const int8_t* board, int color) {
    int material = 0;
    for (int index = 0; index < 64; ++index) {
        int code = board[index];
        if (color_of(code) != color) {
            continue;
        }
        int piece = type_of(code);
        if (piece != KING && piece != PAWN) {
            material += PIECE_VALUES[piece];
        }
    }
    return material;
}

int generate_state_legal_moves(const NativeState& state, int* output, int max_count) {
    int pseudo_moves[256];
    int pseudo_count = generate_pseudo_legal_moves_internal(
        state.board,
        state.side_to_move,
        state.white_king_side,
        state.white_queen_side,
        state.black_king_side,
        state.black_queen_side,
        state.ep_file,
        state.ep_rank,
        pseudo_moves,
        256
    );
    int count = 0;
    for (int index = 0; index < pseudo_count && index < 256; ++index) {
        if (leaves_king_safe(state.board, pseudo_moves[index], state.side_to_move)) {
            push_move(output, max_count, count, pseudo_moves[index]);
        }
    }
    return count;
}

int immediate_capture_gain(const NativeState& state, int encoded) {
    int captured = type_of(captured_piece_for_encoded_move(state, encoded));
    int promotion = (encoded >> 12) & 0xF;
    int gain = captured ? PIECE_VALUES[captured] : 0;
    if (promotion) {
        gain += PIECE_VALUES[piece_from_promotion_code(promotion)] - PIECE_VALUES[PAWN];
    }
    return gain;
}

int best_legal_capture_gain(const NativeState& state, int target_square, int depth) {
    if (depth >= 16 || state.board[target_square] == 0) {
        return 0;
    }

    int moves[256];
    int count = generate_state_legal_moves(state, moves, 256);
    int best_gain = 0;
    for (int index = 0; index < count && index < 256; ++index) {
        int encoded = moves[index];
        int to = (encoded >> 6) & 0x3F;
        if (to != target_square || captured_piece_for_encoded_move(state, encoded) == 0) {
            continue;
        }
        NativeState next = make_state_move(state, encoded);
        int gain = immediate_capture_gain(state, encoded) -
            best_legal_capture_gain(next, target_square, depth + 1);
        best_gain = std::max(best_gain, gain);
    }
    return best_gain;
}

int static_exchange_score_native(const NativeState& state, int encoded) {
    int to = (encoded >> 6) & 0x3F;
    NativeState next = make_state_move(state, encoded);
    return immediate_capture_gain(state, encoded) - best_legal_capture_gain(next, to, 0);
}

bool encoded_move_gives_check(const NativeState& state, int encoded) {
    NativeState next = make_state_move(state, encoded);
    int king_file = -1;
    int king_rank = -1;
    return find_king_internal(next.board, next.side_to_move, king_file, king_rank) &&
        square_attacked_internal(next.board, king_file, king_rank, -next.side_to_move);
}

bool encoded_move_is_checkmate(const NativeState& state, int encoded) {
    if (!encoded_move_gives_check(state, encoded)) {
        return false;
    }
    NativeState next = make_state_move(state, encoded);
    int replies[256];
    return generate_state_legal_moves(next, replies, 256) == 0;
}

struct ImmediateThreatSummary {
    int checking_moves = 0;
    int mating_moves = 0;
};

ImmediateThreatSummary summarize_immediate_check_threats(const NativeState& attacking_state) {
    ImmediateThreatSummary summary;
    int moves[256];
    int count = generate_state_legal_moves(attacking_state, moves, 256);
    for (int index = 0; index < count && index < 256; ++index) {
        int encoded = moves[index];
        if (!encoded_move_gives_check(attacking_state, encoded)) {
            continue;
        }
        ++summary.checking_moves;
        if (encoded_move_is_checkmate(attacking_state, encoded)) {
            ++summary.mating_moves;
        }
    }
    return summary;
}

int defensive_mate_threat_order_bonus(
    const NativeState& state,
    int encoded,
    const ImmediateThreatSummary& baseline_threat
) {
    if (baseline_threat.checking_moves == 0) {
        return 0;
    }
    NativeState next = make_state_move(state, encoded);
    ImmediateThreatSummary replies = summarize_immediate_check_threats(next);
    if (replies.mating_moves > 0) {
        return -500000;
    }
    int bonus = (baseline_threat.checking_moves - replies.checking_moves) * 900;
    if (replies.checking_moves == 0) {
        bonus += 3000;
    }
    return bonus;
}

int hanging_major_piece_order_risk(const NativeState& state, int encoded) {
    int from = encoded & 0x3F;
    int to = (encoded >> 6) & 0x3F;
    int moved_code = state.board[from];
    int moved_piece = type_of(moved_code);
    int moved_color = color_of(moved_code);
    if (moved_color == 0 || (moved_piece != ROOK && moved_piece != QUEEN)) {
        return 0;
    }

    NativeState next = make_state_move(state, encoded);
    int landed_code = next.board[to];
    if (color_of(landed_code) != moved_color || type_of(landed_code) != moved_piece) {
        return 0;
    }
    if (!square_attacked_internal(next.board, to % 8, to / 8, -moved_color)) {
        return 0;
    }

    int captured_code = captured_piece_for_encoded_move(state, encoded);
    int captured_value = PIECE_VALUES[type_of(captured_code)];
    int risk = PIECE_VALUES[moved_piece] - captured_value;
    if (risk <= 0) {
        return 0;
    }
    return std::min(2400, risk * 2);
}

bool line_clear_between(const int8_t* board, int from_file, int from_rank, int to_file, int to_rank) {
    int file_step = (to_file > from_file) - (to_file < from_file);
    int rank_step = (to_rank > from_rank) - (to_rank < from_rank);
    int file = from_file + file_step;
    int rank = from_rank + rank_step;
    while (file != to_file || rank != to_rank) {
        if (board[idx(file, rank)] != 0) {
            return false;
        }
        file += file_step;
        rank += rank_step;
    }
    return true;
}

bool piece_attacks_target(
    const int8_t* board,
    int from_file,
    int from_rank,
    int piece,
    int color,
    int target_file,
    int target_rank
) {
    int file_delta = target_file - from_file;
    int rank_delta = target_rank - from_rank;
    int abs_file = std::abs(file_delta);
    int abs_rank = std::abs(rank_delta);
    if (piece == PAWN) {
        int direction = color > 0 ? 1 : -1;
        return abs_file == 1 && rank_delta == direction;
    }
    if (piece == KNIGHT) {
        return (abs_file == 1 && abs_rank == 2) || (abs_file == 2 && abs_rank == 1);
    }
    if (piece == KING) {
        return std::max(abs_file, abs_rank) == 1;
    }
    if (piece == BISHOP) {
        return abs_file == abs_rank && line_clear_between(board, from_file, from_rank, target_file, target_rank);
    }
    if (piece == ROOK) {
        return (file_delta == 0 || rank_delta == 0) && line_clear_between(board, from_file, from_rank, target_file, target_rank);
    }
    if (piece == QUEEN) {
        bool straight = file_delta == 0 || rank_delta == 0;
        bool diagonal = abs_file == abs_rank;
        return (straight || diagonal) && line_clear_between(board, from_file, from_rank, target_file, target_rank);
    }
    return false;
}

int tactical_pressure_score(const NativeState& state, int encoded) {
    NativeState next = make_state_move(state, encoded);
    int to = (encoded >> 6) & 0x3F;
    int moved_code = next.board[to];
    int color = color_of(moved_code);
    int piece = type_of(moved_code);
    if (piece == 0) {
        return 0;
    }

    std::vector<int> attacked_values;
    bool attacks_king = false;
    int from_file = to % 8;
    int from_rank = to / 8;
    for (int target = 0; target < 64; ++target) {
        int target_code = next.board[target];
        if (color_of(target_code) != -color) {
            continue;
        }
        if (!piece_attacks_target(next.board, from_file, from_rank, piece, color, target % 8, target / 8)) {
            continue;
        }
        int target_piece = type_of(target_code);
        if (target_piece == KING) {
            attacks_king = true;
        } else {
            attacked_values.push_back(PIECE_VALUES[target_piece]);
        }
    }
    if (attacked_values.empty()) {
        return 0;
    }
    std::sort(attacked_values.begin(), attacked_values.end(), std::greater<int>());
    int score = 0;
    if (attacks_king) {
        score += 900 + attacked_values[0];
    }
    if (attacked_values.size() >= 2) {
        score += 700 + (attacked_values[0] + attacked_values[1]) / 2;
    }
    if (attacked_values[0] >= PIECE_VALUES[QUEEN]) {
        score += 250;
    }
    return score;
}

int ray_pin_value(
    const int8_t* board,
    int start_file,
    int start_rank,
    int attacker_color,
    int file_step,
    int rank_step
) {
    int file = start_file + file_step;
    int rank = start_rank + rank_step;
    int first_enemy_piece = 0;
    while (inside(file, rank)) {
        int code = board[idx(file, rank)];
        if (code == 0) {
            file += file_step;
            rank += rank_step;
            continue;
        }
        if (color_of(code) == attacker_color) {
            return 0;
        }
        if (first_enemy_piece == 0) {
            if (type_of(code) == KING) {
                return 0;
            }
            first_enemy_piece = type_of(code);
            file += file_step;
            rank += rank_step;
            continue;
        }
        if (color_of(code) == -attacker_color && type_of(code) == KING) {
            return PIECE_VALUES[first_enemy_piece];
        }
        return 0;
    }
    return 0;
}

int pin_creation_score(const NativeState& state, int encoded) {
    NativeState next = make_state_move(state, encoded);
    int to = (encoded >> 6) & 0x3F;
    int moved_code = next.board[to];
    int color = color_of(moved_code);
    int piece = type_of(moved_code);
    if (piece != BISHOP && piece != ROOK && piece != QUEEN) {
        return 0;
    }

    constexpr int diagonal[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    constexpr int straight[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    int best = 0;
    int file = to % 8;
    int rank = to / 8;
    if (piece == BISHOP || piece == QUEEN) {
        for (const auto& direction : diagonal) {
            best = std::max(best, ray_pin_value(next.board, file, rank, color, direction[0], direction[1]));
        }
    }
    if (piece == ROOK || piece == QUEEN) {
        for (const auto& direction : straight) {
            best = std::max(best, ray_pin_value(next.board, file, rank, color, direction[0], direction[1]));
        }
    }
    if (best <= 0) {
        return 0;
    }
    return 650 + best;
}

void defended_own_piece_values(
    const int8_t* board,
    int defender_square,
    int defender_code,
    std::vector<int>& values
) {
    int defender_color = color_of(defender_code);
    int defender_piece = type_of(defender_code);
    int defender_file = defender_square % 8;
    int defender_rank = defender_square / 8;
    for (int target = 0; target < 64; ++target) {
        if (target == defender_square) {
            continue;
        }
        int target_code = board[target];
        if (color_of(target_code) != defender_color || type_of(target_code) == KING) {
            continue;
        }
        if (piece_attacks_target(board, defender_file, defender_rank, defender_piece, defender_color, target % 8, target / 8)) {
            values.push_back(PIECE_VALUES[type_of(target_code)]);
        }
    }
}

int defender_tactic_score(const NativeState& state, int encoded) {
    int captured_code = captured_piece_for_encoded_move(state, encoded);
    if (captured_code == 0 || type_of(captured_code) == KING) {
        return 0;
    }
    int to = (encoded >> 6) & 0x3F;
    int flags = (encoded >> 16) & 0xF;
    int defender_square = (flags & 1) ? idx(to % 8, (encoded & 0x3F) / 8) : to;
    std::vector<int> defended;
    defended_own_piece_values(state.board, defender_square, captured_code, defended);
    if (defended.empty()) {
        return 0;
    }
    return 450 + *std::max_element(defended.begin(), defended.end());
}

int pawn_break_order_score(const NativeState& state, int encoded) {
    int from = encoded & 0x3F;
    int to = (encoded >> 6) & 0x3F;
    int from_file = from % 8;
    int to_file = to % 8;
    int to_rank = to / 8;
    int color = state.side_to_move;
    if (from_file != to_file || state.board[to] != 0) {
        return 0;
    }
    int direction = color > 0 ? 1 : -1;
    int score = 0;
    for (int adjacent_file : {to_file - 1, to_file + 1}) {
        if (!inside(adjacent_file, to_rank)) {
            continue;
        }
        int adjacent = state.board[idx(adjacent_file, to_rank)];
        if (color_of(adjacent) == -color && type_of(adjacent) == PAWN) {
            score += 220;
        }
        int next_rank = to_rank + direction;
        if (inside(adjacent_file, next_rank)) {
            int target = state.board[idx(adjacent_file, next_rank)];
            if (color_of(target) == -color && type_of(target) == PAWN) {
                score += 220;
            }
        }
    }
    if (score > 0 && to_file >= 2 && to_file <= 5) {
        score += 80;
    }
    return std::min(520, score);
}

struct MoveOrderFacts {
    int promotion;
    int captured;
    int see;
    bool gives_check;
};

MoveOrderFacts move_order_facts(const NativeState& state, int encoded) {
    const int promotion = (encoded >> 12) & 0xF;
    const int captured = type_of(captured_piece_for_encoded_move(state, encoded));
    return {
        promotion,
        captured,
        captured || promotion ? static_exchange_score_native(state, encoded) : 0,
        encoded_move_gives_check(state, encoded),
    };
}

int move_order_score(
    const NativeState& state,
    int encoded,
    const MoveOrderFacts& facts,
    const SearchHeuristics* heuristics = nullptr,
    int ply = 0,
    int previous_move_piece = 0,
    int previous_move_to = -1
) {
    int from = encoded & 0x3F;
    int to = (encoded >> 6) & 0x3F;
    int promotion = facts.promotion;
    int flags = (encoded >> 16) & 0xF;
    int attacker = type_of(state.board[from]);
    int captured = facts.captured;
    int score = 0;
    if (captured) {
        score += 10000 + PIECE_VALUES[captured] * 10 - PIECE_VALUES[attacker];
        int see = facts.see;
        if (see >= 0) {
            score += 1500 + see;
        } else {
            score -= 1500 + std::abs(see) * 2;
        }
    }
    if (promotion) {
        score += 8000 + PIECE_VALUES[piece_from_promotion_code(promotion)];
    }
    if (flags & 2) {
        score += 250;
    }
    bool gives_check = facts.gives_check;
    if (gives_check) {
        score += 1200;
    }
    if (g_mate_danger_ordering.load(std::memory_order_relaxed) && gives_check) {
        score += encoded_move_is_checkmate(state, encoded) ? 500000 : 2400;
    }
    if (ply <= 2) {
        score += tactical_pressure_score(state, encoded);
        score += pin_creation_score(state, encoded);
        score += defender_tactic_score(state, encoded);
    }
    if (attacker == PAWN) {
        int to_rank = to / 8;
        int relative_rank = state.side_to_move > 0 ? to_rank : 7 - to_rank;
        if (relative_rank >= 5) {
            score += 80 + relative_rank * 15;
        }
        score += pawn_break_order_score(state, encoded);
    }
    if (to == idx(3, 3) || to == idx(4, 3) || to == idx(3, 4) || to == idx(4, 4)) {
        score += 30;
    }
    if (heuristics && !captured && !promotion) {
        int killer = heuristics->killer_score(encoded, ply);
        int history = heuristics->history_score(encoded);
        if (killer > 0) {
            ++g_killer_order_hits;
        }
        if (history > 0) {
            ++g_history_order_hits;
        }
        score += killer;
        score += history;
        if (g_continuation_history.load(std::memory_order_relaxed)) {
            int continuation = heuristics->continuation_history_score(
                encoded, state.board[from], previous_move_piece, previous_move_to
            );
            if (continuation > 0) {
                ++g_continuation_history_order_hits;
            }
            score += continuation;
        }
    }
    return score;
}

int native_move_stage(
    const NativeState& state,
    int encoded,
    const MoveOrderFacts& facts,
    int preferred_move,
    const SearchHeuristics* heuristics = nullptr,
    int ply = 0,
    int previous_move_piece = 0,
    int previous_move_to = -1
) {
    int preferred_identity = preferred_move & 0xFFFF;
    if (preferred_identity != 0 && (encoded & 0xFFFF) == preferred_identity) {
        return STAGE_TT;
    }
    int promotion = facts.promotion;
    int captured = facts.captured;
    if (captured || promotion) {
        return facts.see >= -40 ? STAGE_GOOD_CAPTURE : STAGE_BAD_CAPTURE;
    }
    bool continuation_history_hit =
        heuristics &&
        g_continuation_history.load(std::memory_order_relaxed) &&
        heuristics->continuation_history_score(
            encoded, state.board[encoded & 0x3F], previous_move_piece, previous_move_to
        ) > 0;
    if (
        facts.gives_check ||
        ((encoded >> 16) & 0x2) ||
        continuation_history_hit ||
        (heuristics && (heuristics->killer_score(encoded, ply) > 0 || heuristics->history_score(encoded) > 0))
    ) {
        return STAGE_GOOD_QUIET;
    }
    int to = (encoded >> 6) & 0x3F;
    if (to == idx(3, 3) || to == idx(4, 3) || to == idx(3, 4) || to == idx(4, 4)) {
        return STAGE_GOOD_QUIET;
    }
    return STAGE_BAD_QUIET;
}

void sort_moves_for_search(
    const NativeState& state,
    int* moves,
    int count,
    int preferred_move = 0,
    const SearchHeuristics* heuristics = nullptr,
    int ply = 0,
    bool allow_defensive_mate_threat_ordering = false,
    int previous_move_piece = 0,
    int previous_move_to = -1
) {
    std::vector<std::pair<int, int>> scored_moves;
    scored_moves.reserve(count);
    ImmediateThreatSummary baseline_threat;
    bool defensive_ordering =
        allow_defensive_mate_threat_ordering &&
        g_defensive_mate_threat_ordering.load(std::memory_order_relaxed);
    if (defensive_ordering) {
        baseline_threat = summarize_immediate_check_threats(make_null_move(state));
    }
    for (int index = 0; index < count; ++index) {
        MoveOrderFacts facts = move_order_facts(state, moves[index]);
        int stage = native_move_stage(
            state, moves[index], facts, preferred_move, heuristics, ply,
            previous_move_piece, previous_move_to
        );
        int score = move_order_score(
            state, moves[index], facts, heuristics, ply,
            previous_move_piece, previous_move_to
        );
        if (g_major_piece_safety_ordering.load(std::memory_order_relaxed)) {
            int major_risk = hanging_major_piece_order_risk(state, moves[index]);
            if (major_risk > 0) {
                score -= major_risk;
                if (stage == STAGE_GOOD_QUIET && facts.captured == 0 && facts.promotion == 0) {
                    stage = STAGE_BAD_QUIET;
                }
            }
        }
        if (defensive_ordering && baseline_threat.checking_moves > 0) {
            int defensive_bonus = defensive_mate_threat_order_bonus(state, moves[index], baseline_threat);
            score += defensive_bonus;
            if (defensive_bonus > 0 && stage > STAGE_GOOD_QUIET) {
                stage = STAGE_GOOD_QUIET;
            }
        }
        scored_moves.push_back({-stage * 1000000 + score, moves[index]});
    }
    std::sort(scored_moves.begin(), scored_moves.end(), [](const auto& first, const auto& second) {
        return first.first > second.first;
    });
    for (int index = 0; index < count; ++index) {
        moves[index] = scored_moves[index].second;
    }
}

bool is_state_in_check(const NativeState& state) {
    int king_file = -1;
    int king_rank = -1;
    return find_king_internal(state.board, state.side_to_move, king_file, king_rank) &&
        square_attacked_internal(state.board, king_file, king_rank, -state.side_to_move);
}

uint64_t hash_state(const NativeState& state) {
    uint64_t hash = FNV_OFFSET;
    for (int index = 0; index < 64; ++index) {
        hash = mix_hash(hash, static_cast<uint64_t>(state.board[index] + 8));
    }
    hash = mix_hash(hash, state.side_to_move > 0 ? 65ULL : 66ULL);
    hash = mix_hash(hash, static_cast<uint64_t>(state.white_king_side ? 67 : 68));
    hash = mix_hash(hash, static_cast<uint64_t>(state.white_queen_side ? 69 : 70));
    hash = mix_hash(hash, static_cast<uint64_t>(state.black_king_side ? 71 : 72));
    hash = mix_hash(hash, static_cast<uint64_t>(state.black_queen_side ? 73 : 74));
    hash = mix_hash(hash, static_cast<uint64_t>(state.ep_file + 2));
    hash = mix_hash(hash, static_cast<uint64_t>(state.ep_rank + 2));
    return hash;
}

int side_to_move_score(
    const NativeState& state,
    double material_weight,
    double position_weight,
    double mobility_weight,
    double endgame_weight,
    double pawn_structure_weight,
    double bishop_pair_weight,
    double rook_activity_weight,
    double king_safety_weight,
    double passed_pawn_weight,
    double endgame_pressure_weight,
    double line_activity_weight,
    double attack_momentum_weight
) {
    int white_score = evaluate_internal(
        state.board,
        material_weight,
        position_weight,
        mobility_weight,
        endgame_weight,
        pawn_structure_weight,
        bishop_pair_weight,
        rook_activity_weight,
        king_safety_weight,
        passed_pawn_weight,
        endgame_pressure_weight,
        line_activity_weight,
        attack_momentum_weight
    );
    return state.side_to_move > 0 ? white_score : -white_score;
}

bool is_tactical_move(const NativeState& state, int encoded) {
    int promotion = (encoded >> 12) & 0xF;
    return promotion != 0 || captured_piece_for_encoded_move(state, encoded) != 0;
}

bool is_quiescence_candidate(const NativeState& state, int encoded) {
    int promotion = (encoded >> 12) & 0xF;
    int captured = captured_piece_for_encoded_move(state, encoded);
    if (promotion || encoded_move_gives_check(state, encoded)) {
        return true;
    }
    if (captured != 0) {
        return static_exchange_score_native(state, encoded) >= -120;
    }

    int from = encoded & 0x3F;
    int to = (encoded >> 6) & 0x3F;
    int moving_piece = type_of(state.board[from]);
    if (moving_piece != PAWN) {
        return false;
    }

    int to_rank = to / 8;
    int relative_rank = state.side_to_move > 0 ? to_rank : 7 - to_rank;
    return relative_rank >= 6;
}

int root_penalty_for_move(int encoded, const int* penalty_moves, const int* penalties, int penalty_count) {
    if (!penalty_moves || !penalties || penalty_count <= 0) {
        return 0;
    }
    int identity = encoded & 0xFFFF;
    for (int index = 0; index < penalty_count; ++index) {
        if ((penalty_moves[index] & 0xFFFF) == identity) {
            return penalties[index];
        }
    }
    return 0;
}

int quiescence(
    const NativeState& state,
    int alpha,
    int beta,
    int remaining_depth,
    int ply,
    double material_weight,
    double position_weight,
    double mobility_weight,
    double endgame_weight,
    double pawn_structure_weight,
    double bishop_pair_weight,
    double rook_activity_weight,
    double king_safety_weight,
    double passed_pawn_weight,
    double endgame_pressure_weight,
    double line_activity_weight,
    double attack_momentum_weight
) {
    ++g_search_nodes;
    if (search_should_stop()) {
        return 0;
    }
    if (has_insufficient_material_draw(state.board)) {
        return 0;
    }
    int moves[256];
    int count = generate_state_legal_moves(state, moves, 256);
    bool in_check = is_state_in_check(state);

    if (count == 0) {
        return in_check ? -CHECKMATE_SCORE + ply : 0;
    }
    if (ply >= MAX_SEARCH_PLY - 1) {
        return side_to_move_score(
            state,
            material_weight,
            position_weight,
            mobility_weight,
            endgame_weight,
            pawn_structure_weight,
            bishop_pair_weight,
            rook_activity_weight,
            king_safety_weight,
            passed_pawn_weight,
            endgame_pressure_weight,
            line_activity_weight,
            attack_momentum_weight
        );
    }

    int stand_pat = 0;
    if (!in_check) {
        stand_pat = side_to_move_score(
            state,
            material_weight,
            position_weight,
            mobility_weight,
            endgame_weight,
            pawn_structure_weight,
            bishop_pair_weight,
            rook_activity_weight,
            king_safety_weight,
            passed_pawn_weight,
            endgame_pressure_weight,
            line_activity_weight,
            attack_momentum_weight
        );
    }
    // A checked side may not "stand pat"; it must search legal evasions even
    // at the nominal qsearch horizon to match legal chess and Python search.
    if (remaining_depth <= 0 && !in_check) {
        return stand_pat;
    }

    if (!in_check) {
        if (stand_pat >= beta) {
            return beta;
        }
        alpha = std::max(alpha, stand_pat);
    }

    sort_moves_for_search(state, moves, std::min(count, 256), 0, nullptr, ply);
    bool searched_move = false;
    for (int index = 0; index < count && index < 256; ++index) {
        if (!in_check && !is_quiescence_candidate(state, moves[index])) {
            continue;
        }
        searched_move = true;
        NativeState next = make_state_move(state, moves[index]);
        int score = -quiescence(
            next,
            -beta,
            -alpha,
            remaining_depth - 1,
            ply + 1,
            material_weight,
            position_weight,
            mobility_weight,
            endgame_weight,
            pawn_structure_weight,
            bishop_pair_weight,
            rook_activity_weight,
            king_safety_weight,
            passed_pawn_weight,
            endgame_pressure_weight,
            line_activity_weight,
            attack_momentum_weight
        );
        if (g_search_aborted) {
            return 0;
        }
        if (score >= beta) {
            return beta;
        }
        alpha = std::max(alpha, score);
    }

    if (!searched_move && !in_check) {
        return stand_pat;
    }
    return alpha;
}

int negamax(
    const NativeState& state,
    int depth,
    int alpha,
    int beta,
    int ply,
    int quiescence_depth,
    int forced_line_extension_budget,
    int pawn_endgame_extension_budget,
    int check_extension_budget,
    bool allow_null_move,
    TranspositionTable& transposition_table,
    SearchHeuristics& heuristics,
    double material_weight,
    double position_weight,
    double mobility_weight,
    double endgame_weight,
    double pawn_structure_weight,
    double bishop_pair_weight,
    double rook_activity_weight,
    double king_safety_weight,
    double passed_pawn_weight,
    double endgame_pressure_weight,
    double line_activity_weight,
    double attack_momentum_weight,
    int previous_move_piece = 0,
    int previous_move_to = -1
) {
    ++g_search_nodes;
    if (search_should_stop()) {
        return 0;
    }
    if (has_insufficient_material_draw(state.board)) {
        return 0;
    }
    int original_alpha = alpha;
    int original_beta = beta;
    uint64_t key = hash_state(state);
    int preferred_move = 0;
    int tt_depth = -1;
    int tt_score = 0;
    int tt_bound = TT_EXACT;
    if (TranspositionEntry* entry = transposition_table.find(key)) {
        preferred_move = entry->best_move;
        tt_depth = entry->depth;
        tt_score = score_from_transposition_table(entry->score, ply);
        tt_bound = entry->bound;
        if (entry->depth >= depth) {
            if (entry->bound == TT_EXACT) {
                return tt_score;
            }
            if (entry->bound == TT_LOWER) {
                alpha = std::max(alpha, tt_score);
            } else if (entry->bound == TT_UPPER) {
                beta = std::min(beta, tt_score);
            }
            if (alpha >= beta) {
                return tt_score;
            }
        }
    }

    int moves[256];
    int count = generate_state_legal_moves(state, moves, 256);
    bool in_check = is_state_in_check(state);

    if (count == 0) {
        return in_check ? -CHECKMATE_SCORE + ply : 0;
    }
    if (depth == 0 && forced_line_extension_budget > 0 && (in_check || count == 1)) {
        depth = 1;
        --forced_line_extension_budget;
    }
    if (
        depth == 0 &&
        pawn_endgame_extension_budget > 0 &&
        non_pawn_material_for_color(state.board, 1) == 0 &&
        non_pawn_material_for_color(state.board, -1) == 0
    ) {
        depth = 1;
        --pawn_endgame_extension_budget;
    }
    if (depth == 0) {
        return quiescence(
            state,
            alpha,
            beta,
            quiescence_depth,
            ply,
            material_weight,
            position_weight,
            mobility_weight,
            endgame_weight,
            pawn_structure_weight,
            bishop_pair_weight,
            rook_activity_weight,
            king_safety_weight,
            passed_pawn_weight,
            endgame_pressure_weight,
            line_activity_weight,
            attack_momentum_weight
        );
    }

    bool futility_eligible = (
        g_futility_pruning.load(std::memory_order_relaxed) &&
        !in_check &&
        depth <= FUTILITY_MAX_DEPTH
    );
    bool razoring_eligible = (
        g_razoring.load(std::memory_order_relaxed) &&
        !in_check &&
        depth <= RAZOR_MAX_DEPTH
    );
    int static_eval = 0;
    if (futility_eligible || razoring_eligible) {
        static_eval = side_to_move_score(
            state,
            material_weight,
            position_weight,
            mobility_weight,
            endgame_weight,
            pawn_structure_weight,
            bishop_pair_weight,
            rook_activity_weight,
            king_safety_weight,
            passed_pawn_weight,
            endgame_pressure_weight,
            line_activity_weight,
            attack_momentum_weight
        );
        // Reverse futility (static null move) pruning: the position already
        // looks so good, even conceding a per-ply margin, that further search
        // is unlikely to overturn a beta cutoff. Skip near mate scores, where
        // the margin arithmetic doesn't mean anything.
        if (
            futility_eligible &&
            depth <= REVERSE_FUTILITY_MAX_DEPTH &&
            std::abs(beta) < TT_MATE_SCORE_THRESHOLD &&
            static_eval - depth * REVERSE_FUTILITY_MARGIN_PER_PLY >= beta
        ) {
            return static_eval;
        }
        // Razoring: the position looks far worse than alpha even with a
        // generous margin. Rather than trusting the static eval outright
        // (unsafe -- a quiet-looking bad position can still have tactics),
        // verify with a quiescence search and only prune if it confirms.
        if (
            razoring_eligible &&
            std::abs(alpha) < TT_MATE_SCORE_THRESHOLD &&
            static_eval + depth * RAZOR_MARGIN_PER_PLY <= alpha
        ) {
            int razor_score = quiescence(
                state,
                alpha,
                beta,
                quiescence_depth,
                ply,
                material_weight,
                position_weight,
                mobility_weight,
                endgame_weight,
                pawn_structure_weight,
                bishop_pair_weight,
                rook_activity_weight,
                king_safety_weight,
                passed_pawn_weight,
                endgame_pressure_weight,
                line_activity_weight,
                attack_momentum_weight
            );
            if (g_search_aborted) {
                return 0;
            }
            if (razor_score <= alpha) {
                return razor_score;
            }
        }
    }

    if (
        allow_null_move &&
        depth >= 3 &&
        !in_check &&
        non_pawn_material_for_color(state.board, state.side_to_move) >= 700
    ) {
        ++g_null_move_probes;
        if (g_null_move_nesting > 0) {
            ++g_nested_null_move_probes;
        }
        NativeState null_state = make_null_move(state);
        int reduction = depth >= 5 ? 3 : 2;
        int null_depth = std::max(0, depth - 1 - reduction);
        ++g_null_move_nesting;
        int null_score = -negamax(
            null_state,
            null_depth,
            -beta,
            -beta + 1,
            ply + 1,
            quiescence_depth,
            forced_line_extension_budget,
            pawn_endgame_extension_budget,
            check_extension_budget,
            false,
            transposition_table,
            heuristics,
            material_weight,
            position_weight,
            mobility_weight,
            endgame_weight,
            pawn_structure_weight,
            bishop_pair_weight,
            rook_activity_weight,
            king_safety_weight,
            passed_pawn_weight,
            endgame_pressure_weight,
            line_activity_weight,
            attack_momentum_weight
        );
        --g_null_move_nesting;
        if (g_search_aborted) {
            return 0;
        }
        if (null_score >= beta) {
            ++g_null_move_cutoffs;
            return beta;
        }
    }

    bool extend_singular_move = false;
    if (
        g_singular_extension.load(std::memory_order_relaxed) &&
        ply > 0 &&
        depth >= 6 &&
        preferred_move != 0 &&
        tt_depth >= depth - 3 &&
        (tt_bound == TT_EXACT || tt_bound == TT_LOWER) &&
        tt_score > -TT_MATE_SCORE_THRESHOLD &&
        tt_score < TT_MATE_SCORE_THRESHOLD
    ) {
        int singular_beta = tt_score - 2 * depth;
        int singular_depth = depth / 2;
        int best_other_score = -1000000000;
        for (int index = 0; index < count && index < 256; ++index) {
            if (moves[index] == preferred_move) {
                continue;
            }
            NativeState next = make_state_move(state, moves[index]);
            int score = -negamax(
                next,
                singular_depth,
                -singular_beta - 1,
                -singular_beta,
                ply + 1,
                quiescence_depth,
                forced_line_extension_budget,
                pawn_endgame_extension_budget,
                check_extension_budget,
                true,
                transposition_table,
                heuristics,
                material_weight,
                position_weight,
                mobility_weight,
                endgame_weight,
                pawn_structure_weight,
                bishop_pair_weight,
                rook_activity_weight,
                king_safety_weight,
                passed_pawn_weight,
                endgame_pressure_weight,
                line_activity_weight,
                attack_momentum_weight
            );
            if (g_search_aborted) {
                return 0;
            }
            best_other_score = std::max(best_other_score, score);
            if (best_other_score >= singular_beta) {
                break;
            }
        }
        extend_singular_move = best_other_score < singular_beta;
    }

    sort_moves_for_search(
        state, moves, std::min(count, 256), preferred_move, &heuristics, ply,
        false, previous_move_piece, previous_move_to
    );

    // ProbCut: at moderately high depth, verify a handful of tactical moves
    // (already front-loaded by the sort above) with a REAL, reduced-depth
    // search at a widened beta window. If any of them meets that widened
    // bar, assume with high confidence the full-depth search would also
    // fail high at the real beta, and cut immediately without searching the
    // rest of the move list.
    if (
        g_probcut.load(std::memory_order_relaxed) &&
        !in_check &&
        depth >= PROBCUT_MIN_DEPTH &&
        std::abs(beta) < TT_MATE_SCORE_THRESHOLD
    ) {
        int probcut_beta = beta + PROBCUT_MARGIN;
        int probcut_depth = std::max(0, depth - PROBCUT_REDUCTION);
        for (int index = 0; index < count && index < 256; ++index) {
            if (!is_tactical_move(state, moves[index])) {
                continue;
            }
            ++g_probcut_probes;
            NativeState next = make_state_move(state, moves[index]);
            int probcut_previous_piece = state.board[moves[index] & 0x3F];
            int probcut_previous_to = (moves[index] >> 6) & 0x3F;
            int score = -negamax(
                next,
                probcut_depth,
                -probcut_beta,
                -probcut_beta + 1,
                ply + 1,
                quiescence_depth,
                forced_line_extension_budget,
                pawn_endgame_extension_budget,
                check_extension_budget,
                true,
                transposition_table,
                heuristics,
                material_weight,
                position_weight,
                mobility_weight,
                endgame_weight,
                pawn_structure_weight,
                bishop_pair_weight,
                rook_activity_weight,
                king_safety_weight,
                passed_pawn_weight,
                endgame_pressure_weight,
                line_activity_weight,
                attack_momentum_weight,
                probcut_previous_piece,
                probcut_previous_to
            );
            if (g_search_aborted) {
                return 0;
            }
            if (score >= probcut_beta) {
                ++g_probcut_cutoffs;
                return beta;
            }
        }
    }

    int best_score = -1000000000;
    int best_move = moves[0];
    for (int index = 0; index < count && index < 256; ++index) {
        bool quiet_for_history = !is_tactical_move(state, moves[index]);
        bool gives_check = encoded_move_gives_check(state, moves[index]);
        bool tactical = !quiet_for_history || gives_check;
        int child_depth = depth - 1;
        int child_check_extension_budget = check_extension_budget;
        if (gives_check && check_extension_budget > 0) {
            child_depth = depth;
            --child_check_extension_budget;
        }
        if (extend_singular_move && moves[index] == preferred_move) {
            ++child_depth;
        }
        // Move-level futility pruning: a quiet move this deep into the list,
        // this close to the horizon, is unlikely to raise alpha even with a
        // generous per-ply margin -- skip searching it at all (unlike LMR,
        // which still searches, just at reduced depth). Never skip index 0
        // (the TT/best candidate), so best_move always names a searched move.
        if (
            futility_eligible &&
            index > 0 &&
            !tactical &&
            std::abs(alpha) < TT_MATE_SCORE_THRESHOLD &&
            static_eval + depth * MOVE_FUTILITY_MARGIN_PER_PLY <= alpha
        ) {
            continue;
        }
        NativeState next = make_state_move(state, moves[index]);
        int child_previous_piece = state.board[moves[index] & 0x3F];
        int child_previous_to = (moves[index] >> 6) & 0x3F;
        int score = -1000000000;
        if (depth >= 3 && index >= 4 && !tactical) {
            int lmr_depth = depth - 2;
            if (g_lmr_formula.load(std::memory_order_relaxed)) {
                lmr_depth = std::max(0, depth - 1 - lmr_reduction(depth, index));
            }
            score = -negamax(
                next,
                lmr_depth,
                -alpha - 1,
                -alpha,
                ply + 1,
                quiescence_depth,
                forced_line_extension_budget,
                pawn_endgame_extension_budget,
                check_extension_budget,
                true,
                transposition_table,
                heuristics,
                material_weight,
                position_weight,
                mobility_weight,
                endgame_weight,
                pawn_structure_weight,
                bishop_pair_weight,
                rook_activity_weight,
                king_safety_weight,
                passed_pawn_weight,
                endgame_pressure_weight,
                line_activity_weight,
                attack_momentum_weight,
                child_previous_piece,
                child_previous_to
            );
            if (g_search_aborted) {
                return 0;
            }
        }
        if (score > alpha || score == -1000000000) {
            score = -negamax(
                next,
                child_depth,
                -beta,
                -alpha,
                ply + 1,
                quiescence_depth,
                forced_line_extension_budget,
                pawn_endgame_extension_budget,
                child_check_extension_budget,
                true,
                transposition_table,
                heuristics,
                material_weight,
                position_weight,
                mobility_weight,
                endgame_weight,
                pawn_structure_weight,
                bishop_pair_weight,
                rook_activity_weight,
                king_safety_weight,
                passed_pawn_weight,
                endgame_pressure_weight,
                line_activity_weight,
                attack_momentum_weight,
                child_previous_piece,
                child_previous_to
            );
            if (g_search_aborted) {
                return 0;
            }
        }
        if (score > best_score) {
            best_score = score;
            best_move = moves[index];
        }
        alpha = std::max(alpha, score);
        if (alpha >= beta) {
            bool track_continuation_history =
                g_continuation_history.load(std::memory_order_relaxed) &&
                previous_move_piece >= -6 && previous_move_piece <= 6 && previous_move_to >= 0;
            for (int failed_index = 0; failed_index < index; ++failed_index) {
                if (!is_tactical_move(state, moves[failed_index])) {
                    heuristics.record_quiet_failure(moves[failed_index], depth);
                    if (track_continuation_history) {
                        heuristics.record_continuation_history_failure(
                            moves[failed_index], state.board[moves[failed_index] & 0x3F],
                            previous_move_piece, previous_move_to, depth
                        );
                    }
                }
            }
            heuristics.record_cutoff(
                moves[index],
                depth,
                ply,
                quiet_for_history
            );
            if (track_continuation_history && quiet_for_history) {
                heuristics.record_continuation_history_cutoff(
                    moves[index], state.board[moves[index] & 0x3F],
                    previous_move_piece, previous_move_to, depth
                );
            }
            break;
        }
    }
    int bound = TT_EXACT;
    if (best_score <= original_alpha) {
        bound = TT_UPPER;
    } else if (best_score >= original_beta) {
        bound = TT_LOWER;
    }
    transposition_table.store(key, depth, score_to_transposition_table(best_score, ply), bound, best_move);
    return best_score;
}

int collect_principal_variation(
    NativeState state,
    TranspositionTable& transposition_table,
    int max_depth,
    int* output,
    int max_count
) {
    if (!output || max_depth <= 0 || max_count <= 0) {
        return 0;
    }

    int count = 0;
    for (int ply = 0; ply < max_depth && count < max_count; ++ply) {
        TranspositionEntry* entry = transposition_table.find(hash_state(state));
        if (!entry || entry->best_move == 0) {
            break;
        }

        int legal_moves[256];
        int legal_count = generate_state_legal_moves(state, legal_moves, 256);
        int selected_move = 0;
        int best_identity = entry->best_move & 0xFFFF;
        for (int index = 0; index < legal_count && index < 256; ++index) {
            if ((legal_moves[index] & 0xFFFF) == best_identity) {
                selected_move = legal_moves[index];
                break;
            }
        }
        if (selected_move == 0) {
            break;
        }

        output[count++] = selected_move;
        state = make_state_move(state, selected_move);
    }
    return count;
}

}  // namespace

EXPORT int deeptactics_evaluate(
    const int8_t* board,
    double material_weight,
    double position_weight,
    double mobility_weight,
    double endgame_weight,
    double pawn_structure_weight,
    double bishop_pair_weight,
    double rook_activity_weight,
    double king_safety_weight,
    double passed_pawn_weight,
    double endgame_pressure_weight,
    double line_activity_weight,
    double attack_momentum_weight
) {
    return evaluate_internal(
        board,
        material_weight,
        position_weight,
        mobility_weight,
        endgame_weight,
        pawn_structure_weight,
        bishop_pair_weight,
        rook_activity_weight,
        king_safety_weight,
        passed_pawn_weight,
        endgame_pressure_weight,
        line_activity_weight,
        attack_momentum_weight
    );
}

EXPORT int deeptactics_evaluate_breakdown(
    const int8_t* board,
    double material_weight,
    double position_weight,
    double mobility_weight,
    double endgame_weight,
    double pawn_structure_weight,
    double bishop_pair_weight,
    double rook_activity_weight,
    double king_safety_weight,
    double passed_pawn_weight,
    double endgame_pressure_weight,
    double line_activity_weight,
    double attack_momentum_weight,
    int* output,
    int max_output
) {
    if (!output || max_output < 15) {
        return 0;
    }
    evaluate_internal(
        board,
        material_weight,
        position_weight,
        mobility_weight,
        endgame_weight,
        pawn_structure_weight,
        bishop_pair_weight,
        rook_activity_weight,
        king_safety_weight,
        passed_pawn_weight,
        endgame_pressure_weight,
        line_activity_weight,
        attack_momentum_weight,
        output,
        max_output
    );
    return 15;
}

EXPORT double deeptactics_cp_to_win_probability(int cp) {
    // Convert centipawns from the mover's perspective into an estimated win
    // probability. The logistic curve mirrors the CAPS-style idea that small
    // centipawn losses near equality matter more than extra points in a won game.
    return 50.0 + 50.0 * ((2.0 / (1.0 + std::exp(-0.004 * static_cast<double>(cp)))) - 1.0);
}

EXPORT double deeptactics_move_accuracy(int cp_before, int cp_after, int is_book_move) {
    // Book moves are considered perfect because opening theory should not be
    // penalized by shallow engine noise.
    if (is_book_move) {
        return 100.0;
    }

    const double wp_before = deeptactics_cp_to_win_probability(cp_before);
    const double wp_after = deeptactics_cp_to_win_probability(cp_after);
    const double wp_loss = std::max(0.0, wp_before - wp_after);
    return std::max(0.0, std::min(100.0, 100.0 - wp_loss * 1.5));
}

EXPORT double deeptactics_game_accuracy(const double* accuracies, int count) {
    if (accuracies == nullptr || count <= 0) {
        return 100.0;
    }

    // Harmonic mean weights weak moves more heavily than a simple average,
    // which keeps a single major tactical miss from disappearing into the game
    // score. A small epsilon avoids division by zero for 0.0% moves.
    double reciprocal_sum = 0.0;
    constexpr double EPSILON = 1e-6;
    for (int index = 0; index < count; ++index) {
        const double clamped = std::max(0.0, std::min(100.0, accuracies[index]));
        reciprocal_sum += 1.0 / std::max(EPSILON, clamped);
    }
    return std::max(0.0, std::min(100.0, static_cast<double>(count) / reciprocal_sum));
}

EXPORT int deeptactics_nnue_sparse_linear_infer(
    const int* active_indices,
    int active_count,
    const double* feature_weights,
    int feature_count,
    double bias
) {
    if (!active_indices || !feature_weights || active_count < 0 || feature_count < 0) {
        return 0;
    }
    double score = bias;
    for (int index = 0; index < active_count; ++index) {
        const int feature_index = active_indices[index];
        if (feature_index >= 0 && feature_index < feature_count) {
            score += feature_weights[feature_index];
        }
    }
    return static_cast<int>(std::round(score));
}

bool load_sparse_nnue_file(const char* model_path, SparseNNUEFileModel& model) {
    std::ifstream input(model_path, std::ios::binary);
    if (!input) {
        return false;
    }

    char magic[8]{};
    input.read(magic, 8);
    const char expected[8] = {'D', 'T', 'N', 'N', 'U', 'E', '1', '\0'};
    if (!input || !std::equal(std::begin(magic), std::end(magic), std::begin(expected))) {
        return false;
    }

    uint32_t feature_count = 0;
    double bias = 0.0;
    uint32_t weight_count = 0;
    input.read(reinterpret_cast<char*>(&feature_count), sizeof(feature_count));
    input.read(reinterpret_cast<char*>(&bias), sizeof(bias));
    input.read(reinterpret_cast<char*>(&weight_count), sizeof(weight_count));
    if (!input || feature_count == 0) {
        return false;
    }

    SparseNNUEFileModel next_model;
    next_model.path = model_path;
    next_model.feature_count = feature_count;
    next_model.bias = bias;
    next_model.weights.reserve(weight_count);
    for (uint32_t i = 0; i < weight_count; ++i) {
        uint32_t index = 0;
        double value = 0.0;
        input.read(reinterpret_cast<char*>(&index), sizeof(index));
        input.read(reinterpret_cast<char*>(&value), sizeof(value));
        if (!input) {
            return false;
        }
        if (index < feature_count) {
            next_model.weights[static_cast<int>(index)] = value;
        }
    }
    next_model.loaded = true;
    model = std::move(next_model);
    return true;
}

bool extract_sparse_nnue_features(
    const int8_t* board,
    int side_to_move,
    int white_king_side,
    int white_queen_side,
    int black_king_side,
    int black_queen_side,
    int en_passant_file,
    std::vector<int>& output
) {
    constexpr int PIECE_SQUARE_COUNT = 12 * 64;
    constexpr int KING_CONDITIONED_COUNT = 2 * 64 * PIECE_SQUARE_COUNT;
    constexpr int SIDE_TO_MOVE_FEATURE = KING_CONDITIONED_COUNT;
    constexpr int CASTLING_FEATURE_OFFSET = SIDE_TO_MOVE_FEATURE + 1;
    constexpr int EN_PASSANT_FILE_OFFSET = CASTLING_FEATURE_OFFSET + 4;

    int white_king_square = -1;
    int black_king_square = -1;
    for (int square = 0; square < 64; ++square) {
        const int code = board[square];
        if (type_of(code) == KING) {
            if (color_of(code) > 0) {
                white_king_square = square;
            } else {
                black_king_square = square;
            }
        }
    }
    if (white_king_square < 0 || black_king_square < 0) {
        return false;
    }

    for (int perspective = 0; perspective < 2; ++perspective) {
        const int king_square = perspective == 0 ? white_king_square : black_king_square;
        const int perspective_base = perspective * 64 * PIECE_SQUARE_COUNT;
        const int king_base = perspective_base + king_square * PIECE_SQUARE_COUNT;
        for (int square = 0; square < 64; ++square) {
            const int code = board[square];
            const int piece = type_of(code);
            if (piece == 0) {
                continue;
            }
            const int piece_index = (color_of(code) > 0 ? 0 : 6) + piece - 1;
            output.push_back(king_base + piece_index * 64 + square);
        }
    }

    if (side_to_move < 0) {
        output.push_back(SIDE_TO_MOVE_FEATURE);
    }
    if (white_king_side) {
        output.push_back(CASTLING_FEATURE_OFFSET);
    }
    if (white_queen_side) {
        output.push_back(CASTLING_FEATURE_OFFSET + 1);
    }
    if (black_king_side) {
        output.push_back(CASTLING_FEATURE_OFFSET + 2);
    }
    if (black_queen_side) {
        output.push_back(CASTLING_FEATURE_OFFSET + 3);
    }
    if (en_passant_file >= 0 && en_passant_file < 8) {
        output.push_back(EN_PASSANT_FILE_OFFSET + en_passant_file);
    }
    std::sort(output.begin(), output.end());
    output.erase(std::unique(output.begin(), output.end()), output.end());
    return true;
}

EXPORT int deeptactics_nnue_sparse_linear_infer_file(
    const char* model_path,
    const int* active_indices,
    int active_count,
    int* ok
) {
    if (ok != nullptr) {
        *ok = 0;
    }
    if (model_path == nullptr || active_indices == nullptr || active_count < 0) {
        return 0;
    }

    static SparseNNUEFileModel cached_model;
    if (!cached_model.loaded || cached_model.path != model_path) {
        if (!load_sparse_nnue_file(model_path, cached_model)) {
            return 0;
        }
    }

    double score = cached_model.bias;
    for (int index = 0; index < active_count; ++index) {
        const int feature_index = active_indices[index];
        if (feature_index < 0 || static_cast<uint32_t>(feature_index) >= cached_model.feature_count) {
            continue;
        }
        auto found = cached_model.weights.find(feature_index);
        if (found != cached_model.weights.end()) {
            score += found->second;
        }
    }
    if (ok != nullptr) {
        *ok = 1;
    }
    return static_cast<int>(std::round(score));
}

EXPORT int deeptactics_nnue_extract_features(
    const int8_t* board,
    int side_to_move,
    int white_king_side,
    int white_queen_side,
    int black_king_side,
    int black_queen_side,
    int en_passant_file,
    int* output,
    int max_output
) {
    if (board == nullptr || output == nullptr || max_output <= 0) {
        return 0;
    }
    std::vector<int> features;
    features.reserve(80);
    if (!extract_sparse_nnue_features(
            board,
            side_to_move,
            white_king_side,
            white_queen_side,
            black_king_side,
            black_queen_side,
            en_passant_file,
            features
        )) {
        return 0;
    }
    const int count = std::min(max_output, static_cast<int>(features.size()));
    for (int index = 0; index < count; ++index) {
        output[index] = features[index];
    }
    return static_cast<int>(features.size());
}

EXPORT int deeptactics_nnue_evaluate_board_file(
    const char* model_path,
    const int8_t* board,
    int side_to_move,
    int white_king_side,
    int white_queen_side,
    int black_king_side,
    int black_queen_side,
    int en_passant_file,
    int* ok
) {
    if (ok != nullptr) {
        *ok = 0;
    }
    if (model_path == nullptr || board == nullptr) {
        return 0;
    }
    std::vector<int> features;
    features.reserve(80);
    if (!extract_sparse_nnue_features(
            board,
            side_to_move,
            white_king_side,
            white_queen_side,
            black_king_side,
            black_queen_side,
            en_passant_file,
            features
        )) {
        return 0;
    }
    return deeptactics_nnue_sparse_linear_infer_file(
        model_path,
        features.data(),
        static_cast<int>(features.size()),
        ok
    );
}

double activate_accumulator_value(double value, uint32_t activation) {
    if (activation == 0) {
        return value;
    }
    if (activation == 1) {
        return std::max(0.0, std::min(127.0, value));
    }
    return value;
}

bool load_accumulator_nnue_file(const char* model_path, AccumulatorNNUEFileModel& model) {
    std::ifstream input(model_path, std::ios::binary);
    if (!input) {
        return false;
    }
    char magic[8]{};
    input.read(magic, 8);
    const char expected[8] = {'D', 'T', 'N', 'N', 'U', '2', '\0', '\0'};
    if (!input || !std::equal(std::begin(magic), std::end(magic), std::begin(expected))) {
        return false;
    }

    uint32_t feature_count = 0;
    uint32_t hidden_dim = 0;
    uint32_t activation = 0;
    double output_bias = 0.0;
    uint32_t transform_count = 0;
    input.read(reinterpret_cast<char*>(&feature_count), sizeof(feature_count));
    input.read(reinterpret_cast<char*>(&hidden_dim), sizeof(hidden_dim));
    input.read(reinterpret_cast<char*>(&activation), sizeof(activation));
    input.read(reinterpret_cast<char*>(&output_bias), sizeof(output_bias));
    input.read(reinterpret_cast<char*>(&transform_count), sizeof(transform_count));
    if (!input || feature_count == 0 || hidden_dim == 0) {
        return false;
    }

    AccumulatorNNUEFileModel next_model;
    next_model.path = model_path;
    next_model.feature_count = feature_count;
    next_model.hidden_dim = hidden_dim;
    next_model.activation = activation;
    next_model.output_bias = output_bias;
    next_model.feature_weights.resize(feature_count);
    for (uint32_t i = 0; i < transform_count; ++i) {
        uint32_t feature_index = 0;
        uint32_t hidden_index = 0;
        double value = 0.0;
        input.read(reinterpret_cast<char*>(&feature_index), sizeof(feature_index));
        input.read(reinterpret_cast<char*>(&hidden_index), sizeof(hidden_index));
        input.read(reinterpret_cast<char*>(&value), sizeof(value));
        if (!input) {
            return false;
        }
        if (feature_index < feature_count && hidden_index < hidden_dim) {
            next_model.feature_weights[feature_index].push_back({static_cast<int>(hidden_index), value});
        }
    }
    next_model.output_weights.resize(hidden_dim);
    for (uint32_t index = 0; index < hidden_dim; ++index) {
        input.read(reinterpret_cast<char*>(&next_model.output_weights[index]), sizeof(double));
        if (!input) {
            return false;
        }
    }
    next_model.loaded = true;
    model = std::move(next_model);
    return true;
}

EXPORT int deeptactics_nnue_accumulator_evaluate_board_file(
    const char* model_path,
    const int8_t* board,
    int side_to_move,
    int white_king_side,
    int white_queen_side,
    int black_king_side,
    int black_queen_side,
    int en_passant_file,
    int* ok
) {
    if (ok != nullptr) {
        *ok = 0;
    }
    if (model_path == nullptr || board == nullptr) {
        return 0;
    }
    static AccumulatorNNUEFileModel cached_model;
    if (!cached_model.loaded || cached_model.path != model_path) {
        if (!load_accumulator_nnue_file(model_path, cached_model)) {
            return 0;
        }
    }

    std::vector<int> features;
    features.reserve(80);
    if (!extract_sparse_nnue_features(
            board,
            side_to_move,
            white_king_side,
            white_queen_side,
            black_king_side,
            black_queen_side,
            en_passant_file,
            features
        )) {
        return 0;
    }

    std::vector<double> accumulator(cached_model.hidden_dim, 0.0);
    for (int feature_index : features) {
        if (feature_index < 0 || static_cast<uint32_t>(feature_index) >= cached_model.feature_count) {
            continue;
        }
        for (const auto& entry : cached_model.feature_weights[feature_index]) {
            accumulator[entry.first] += entry.second;
        }
    }

    double score = cached_model.output_bias;
    for (uint32_t hidden_index = 0; hidden_index < cached_model.hidden_dim; ++hidden_index) {
        score += activate_accumulator_value(accumulator[hidden_index], cached_model.activation) *
            cached_model.output_weights[hidden_index];
    }
    if (ok != nullptr) {
        *ok = 1;
    }
    return static_cast<int>(std::round(score));
}

EXPORT int deeptactics_is_square_attacked(
    const int8_t* board,
    int file,
    int rank,
    int by_color
) {
    return square_attacked_internal(board, file, rank, by_color) ? 1 : 0;
}

EXPORT int deeptactics_generate_pseudo_legal_moves(
    const int8_t* board,
    int color,
    int white_king_side,
    int white_queen_side,
    int black_king_side,
    int black_queen_side,
    int ep_file,
    int ep_rank,
    int* output,
    int max_count
) {
    return generate_pseudo_legal_moves_internal(
        board,
        color,
        white_king_side,
        white_queen_side,
        black_king_side,
        black_queen_side,
        ep_file,
        ep_rank,
        output,
        max_count
    );
}

EXPORT int deeptactics_generate_legal_moves(
    const int8_t* board,
    int color,
    int white_king_side,
    int white_queen_side,
    int black_king_side,
    int black_queen_side,
    int ep_file,
    int ep_rank,
    int* output,
    int max_count
) {
    int pseudo_moves[256];
    int pseudo_count = generate_pseudo_legal_moves_internal(
        board,
        color,
        white_king_side,
        white_queen_side,
        black_king_side,
        black_queen_side,
        ep_file,
        ep_rank,
        pseudo_moves,
        256
    );
    int count = 0;
    for (int index = 0; index < pseudo_count && index < 256; ++index) {
        int encoded = pseudo_moves[index];
        if (leaves_king_safe(board, encoded, color)) {
            push_move(output, max_count, count, encoded);
        }
    }
    return count;
}

EXPORT int deeptactics_static_exchange_score(
    const int8_t* board,
    int side_to_move,
    int white_king_side,
    int white_queen_side,
    int black_king_side,
    int black_queen_side,
    int ep_file,
    int ep_rank,
    int encoded_move
) {
    NativeState state{};
    for (int index = 0; index < 64; ++index) {
        state.board[index] = board[index];
    }
    state.side_to_move = side_to_move;
    state.white_king_side = white_king_side;
    state.white_queen_side = white_queen_side;
    state.black_king_side = black_king_side;
    state.black_queen_side = black_queen_side;
    state.ep_file = ep_file;
    state.ep_rank = ep_rank;
    return static_exchange_score_native(state, encoded_move);
}

int find_best_move_internal(
    const int8_t* board,
    int side_to_move,
    int white_king_side,
    int white_queen_side,
    int black_king_side,
    int black_queen_side,
    int ep_file,
    int ep_rank,
    int depth,
    int quiescence_depth,
    double material_weight,
    double position_weight,
    double mobility_weight,
    double endgame_weight,
    double pawn_structure_weight,
    double bishop_pair_weight,
    double rook_activity_weight,
    double king_safety_weight,
    double passed_pawn_weight,
    double endgame_pressure_weight,
    double line_activity_weight,
    double attack_momentum_weight,
    const int* penalty_moves,
    const int* penalties,
    int penalty_count,
    int* out_move,
    int* out_score,
    int* out_pv,
    int max_pv,
    int* out_pv_count,
    uint64_t* out_nodes,
    int time_limit_ms,
    int* out_aborted
) {
    if (depth < 1 || quiescence_depth < 0 || (side_to_move != 1 && side_to_move != -1)) {
        return 0;
    }
    NativeState state{};
    g_search_nodes = 0;
    g_null_move_probes = 0;
    g_null_move_cutoffs = 0;
    g_probcut_probes = 0;
    g_probcut_cutoffs = 0;
    g_nested_null_move_probes = 0;
    g_null_move_nesting = 0;
    g_aspiration_attempts = 0;
    g_aspiration_fail_lows = 0;
    g_aspiration_fail_highs = 0;
    g_aspiration_researches = 0;
    g_quiet_beta_cutoffs = 0;
    g_killer_order_hits = 0;
    g_history_order_hits = 0;
    g_continuation_history_order_hits = 0;
    g_history_saturated_updates = 0;
    g_search_aborted = false;
    g_stop_requested.store(false, std::memory_order_relaxed);
    g_has_deadline = time_limit_ms >= 0;
    if (g_has_deadline) {
        g_search_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(time_limit_ms);
    }
    for (int i = 0; i < 64; ++i) {
        state.board[i] = board[i];
    }
    state.side_to_move = side_to_move;
    state.white_king_side = white_king_side;
    state.white_queen_side = white_queen_side;
    state.black_king_side = black_king_side;
    state.black_queen_side = black_queen_side;
    state.ep_file = ep_file;
    state.ep_rank = ep_rank;

    TranspositionTable transposition_table(transposition_table_bucket_count());
    SearchHeuristics heuristics{};
    uint64_t root_key = hash_state(state);
    int moves[256];
    int count = generate_state_legal_moves(state, moves, 256);
    if (count == 0) {
        int terminal_score = is_state_in_check(state) ? -CHECKMATE_SCORE : 0;
        if (out_move) {
            *out_move = 0;
        }
        if (out_score) {
            *out_score = terminal_score;
        }
        if (out_pv_count) {
            *out_pv_count = 0;
        }
        if (out_nodes) {
            *out_nodes = g_search_nodes;
        }
        if (out_aborted) {
            *out_aborted = 0;
        }
        g_has_deadline = false;
        return 1;
    }
    int preferred_move = 0;
    if (TranspositionEntry* entry = transposition_table.find(root_key)) {
        preferred_move = entry->best_move;
    }
    sort_moves_for_search(state, moves, std::min(count, 256), preferred_move, &heuristics, 0, true);
    int best_move = moves[0];
    int best_score = -1000000000;
    int alpha = -1000000000;
    int beta = 1000000000;
    int original_alpha = alpha;
    for (int index = 0; index < count && index < 256; ++index) {
        NativeState next = make_state_move(state, moves[index]);
        int root_previous_piece = state.board[moves[index] & 0x3F];
        int root_previous_to = (moves[index] >> 6) & 0x3F;
        int score = -negamax(
            next,
            depth - 1,
            -beta,
            -alpha,
            1,
            quiescence_depth,
        g_forced_line_extension_budget.load(std::memory_order_relaxed),
        g_pawn_endgame_extension_budget.load(std::memory_order_relaxed),
        g_check_extension_budget.load(std::memory_order_relaxed),
            true,
            transposition_table,
            heuristics,
            material_weight,
            position_weight,
            mobility_weight,
            endgame_weight,
            pawn_structure_weight,
            bishop_pair_weight,
            rook_activity_weight,
            king_safety_weight,
            passed_pawn_weight,
            endgame_pressure_weight,
            line_activity_weight,
            attack_momentum_weight,
            root_previous_piece,
            root_previous_to
        );
        if (g_search_aborted) {
            break;
        }
        score -= root_penalty_for_move(moves[index], penalty_moves, penalties, penalty_count);
        if (score > best_score) {
            best_score = score;
            best_move = moves[index];
        }
        alpha = std::max(alpha, score);
    }
    int bound = TT_EXACT;
    if (best_score <= original_alpha) {
        bound = TT_UPPER;
    } else if (best_score >= beta) {
        bound = TT_LOWER;
    }
    if (!g_search_aborted) {
        transposition_table.store(root_key, depth, best_score, bound, best_move);
    }
    if (out_move) {
        *out_move = best_move;
    }
    if (out_score) {
        *out_score = best_score;
    }
    if (out_pv_count) {
        *out_pv_count = collect_principal_variation(
            state,
            transposition_table,
            depth,
            out_pv,
            max_pv
        );
    }
    if (out_nodes) {
        *out_nodes = g_search_nodes;
    }
    if (out_aborted) {
        *out_aborted = g_search_aborted ? 1 : 0;
    }
    g_has_deadline = false;
    return 1;
}

int find_best_move_iterative_internal(
    const int8_t* board,
    int side_to_move,
    int white_king_side,
    int white_queen_side,
    int black_king_side,
    int black_queen_side,
    int ep_file,
    int ep_rank,
    int max_depth,
    int quiescence_depth,
    double material_weight,
    double position_weight,
    double mobility_weight,
    double endgame_weight,
    double pawn_structure_weight,
    double bishop_pair_weight,
    double rook_activity_weight,
    double king_safety_weight,
    double passed_pawn_weight,
    double endgame_pressure_weight,
    double line_activity_weight,
    double attack_momentum_weight,
    int time_limit_ms,
    const int* penalty_moves,
    const int* penalties,
    int penalty_count,
    int* out_move,
    int* out_score,
    int* out_pv,
    int max_pv,
    int* out_pv_count,
    uint64_t* out_nodes,
    int* out_aborted,
    int* out_completed_depth
) {
    if (max_depth < 1 || quiescence_depth < 0 || time_limit_ms < 1 ||
        (side_to_move != 1 && side_to_move != -1)) {
        return 0;
    }
    NativeState state{};
    for (int i = 0; i < 64; ++i) {
        state.board[i] = board[i];
    }
    state.side_to_move = side_to_move;
    state.white_king_side = white_king_side;
    state.white_queen_side = white_queen_side;
    state.black_king_side = black_king_side;
    state.black_queen_side = black_queen_side;
    state.ep_file = ep_file;
    state.ep_rank = ep_rank;

    g_search_nodes = 0;
    g_null_move_probes = 0;
    g_null_move_cutoffs = 0;
    g_probcut_probes = 0;
    g_probcut_cutoffs = 0;
    g_nested_null_move_probes = 0;
    g_null_move_nesting = 0;
    g_aspiration_attempts = 0;
    g_aspiration_fail_lows = 0;
    g_aspiration_fail_highs = 0;
    g_aspiration_researches = 0;
    g_quiet_beta_cutoffs = 0;
    g_killer_order_hits = 0;
    g_history_order_hits = 0;
    g_continuation_history_order_hits = 0;
    g_history_saturated_updates = 0;
    g_search_aborted = false;
    g_stop_requested.store(false, std::memory_order_relaxed);
    g_has_deadline = true;
    g_search_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(time_limit_ms);

    TranspositionTable transposition_table(transposition_table_bucket_count());
    SearchHeuristics heuristics{};
    uint64_t root_key = hash_state(state);
    int best_move = 0;
    int best_score = 0;
    int completed_depth = 0;
    int root_moves[256];
    int root_count = generate_state_legal_moves(state, root_moves, 256);
    if (root_count == 0) {
        int terminal_score = is_state_in_check(state) ? -CHECKMATE_SCORE : 0;
        if (out_move) {
            *out_move = 0;
        }
        if (out_score) {
            *out_score = terminal_score;
        }
        if (out_pv_count) {
            *out_pv_count = 0;
        }
        if (out_nodes) {
            *out_nodes = g_search_nodes;
        }
        if (out_aborted) {
            *out_aborted = 0;
        }
        if (out_completed_depth) {
            *out_completed_depth = 0;
        }
        g_has_deadline = false;
        return 1;
    }

    for (int depth = 1; depth <= max_depth; ++depth) {
        int moves[256];
        int count = generate_state_legal_moves(state, moves, 256);
        if (count == 0) {
            break;
        }
        int preferred_move = best_move;
        if (TranspositionEntry* entry = transposition_table.find(root_key)) {
            preferred_move = entry->best_move;
        }
        sort_moves_for_search(state, moves, std::min(count, 256), preferred_move, &heuristics, 0, true);
        int iteration_best_move = moves[0];
        int iteration_best_score = -1000000000;
        int original_alpha = -1000000000;
        int original_beta = 1000000000;
        bool use_aspiration = (
            g_aspiration_windows.load(std::memory_order_relaxed) &&
            completed_depth > 0 &&
            depth >= 3
        );
        if (use_aspiration) {
            ++g_aspiration_attempts;
            constexpr int ASPIRATION_MARGIN = 75;
            original_alpha = best_score - ASPIRATION_MARGIN;
            original_beta = best_score + ASPIRATION_MARGIN;
        }
        auto search_root = [&](int search_alpha, int search_beta) {
            iteration_best_move = moves[0];
            iteration_best_score = -1000000000;
            int alpha = search_alpha;
            for (int index = 0; index < count && index < 256; ++index) {
                NativeState next = make_state_move(state, moves[index]);
                int root_previous_piece = state.board[moves[index] & 0x3F];
                int root_previous_to = (moves[index] >> 6) & 0x3F;
                int score = -negamax(
                    next,
                    depth - 1,
                    -search_beta,
                    -alpha,
                    1,
                    quiescence_depth,
                g_forced_line_extension_budget.load(std::memory_order_relaxed),
                g_pawn_endgame_extension_budget.load(std::memory_order_relaxed),
                g_check_extension_budget.load(std::memory_order_relaxed),
                    true,
                    transposition_table,
                    heuristics,
                    material_weight,
                    position_weight,
                    mobility_weight,
                    endgame_weight,
                    pawn_structure_weight,
                    bishop_pair_weight,
                    rook_activity_weight,
                    king_safety_weight,
                    passed_pawn_weight,
                    endgame_pressure_weight,
                    line_activity_weight,
                    attack_momentum_weight,
                    root_previous_piece,
                    root_previous_to
                );
                if (g_search_aborted) {
                    break;
                }
                score -= root_penalty_for_move(moves[index], penalty_moves, penalties, penalty_count);
                if (score > iteration_best_score) {
                    iteration_best_score = score;
                    iteration_best_move = moves[index];
                }
                alpha = std::max(alpha, score);
            }
        };
        search_root(original_alpha, original_beta);
        if (
            !g_search_aborted &&
            use_aspiration &&
            (iteration_best_score <= original_alpha || iteration_best_score >= original_beta)
        ) {
            if (iteration_best_score <= original_alpha) {
                ++g_aspiration_fail_lows;
            } else {
                ++g_aspiration_fail_highs;
            }
            ++g_aspiration_researches;
            original_alpha = -1000000000;
            original_beta = 1000000000;
            search_root(original_alpha, original_beta);
        }
        if (g_search_aborted) {
            break;
        }
        int bound = iteration_best_score <= original_alpha
            ? TT_UPPER
            : (iteration_best_score >= original_beta ? TT_LOWER : TT_EXACT);
        transposition_table.store(root_key, depth, iteration_best_score, bound, iteration_best_move);
        best_move = iteration_best_move;
        best_score = iteration_best_score;
        completed_depth = depth;
    }

    if (best_move == 0) {
        int moves[256];
        int count = generate_state_legal_moves(state, moves, 256);
        if (count == 0) {
            g_has_deadline = false;
            return 0;
        }
        best_move = moves[0];
    }
    if (out_move) {
        *out_move = best_move;
    }
    if (out_score) {
        *out_score = best_score;
    }
    if (out_pv_count) {
        *out_pv_count = collect_principal_variation(
            state,
            transposition_table,
            completed_depth,
            out_pv,
            max_pv
        );
    }
    if (out_nodes) {
        *out_nodes = g_search_nodes;
    }
    if (out_aborted) {
        *out_aborted = g_search_aborted ? 1 : 0;
    }
    if (out_completed_depth) {
        *out_completed_depth = completed_depth;
    }
    g_has_deadline = false;
    return 1;
}

EXPORT int deeptactics_find_best_move(
    const int8_t* board, int side_to_move, int white_king_side, int white_queen_side,
    int black_king_side, int black_queen_side, int ep_file, int ep_rank,
    int depth, int quiescence_depth, double material_weight, double position_weight,
    double mobility_weight, double endgame_weight, double pawn_structure_weight,
    double bishop_pair_weight, double rook_activity_weight, double king_safety_weight,
    double passed_pawn_weight, double endgame_pressure_weight, double line_activity_weight,
    double attack_momentum_weight, const int* penalty_moves, const int* penalties,
    int penalty_count, int* out_move, int* out_score, int* out_pv, int max_pv,
    int* out_pv_count, uint64_t* out_nodes
) {
    return find_best_move_internal(
        board, side_to_move, white_king_side, white_queen_side, black_king_side,
        black_queen_side, ep_file, ep_rank, depth, quiescence_depth, material_weight,
        position_weight, mobility_weight, endgame_weight, pawn_structure_weight,
        bishop_pair_weight, rook_activity_weight, king_safety_weight, passed_pawn_weight,
        endgame_pressure_weight, line_activity_weight, attack_momentum_weight, penalty_moves,
        penalties, penalty_count, out_move, out_score, out_pv, max_pv, out_pv_count,
        out_nodes, -1, nullptr
    );
}

EXPORT int deeptactics_find_best_move_timed(
    const int8_t* board, int side_to_move, int white_king_side, int white_queen_side,
    int black_king_side, int black_queen_side, int ep_file, int ep_rank,
    int depth, int quiescence_depth, double material_weight, double position_weight,
    double mobility_weight, double endgame_weight, double pawn_structure_weight,
    double bishop_pair_weight, double rook_activity_weight, double king_safety_weight,
    double passed_pawn_weight, double endgame_pressure_weight, double line_activity_weight,
    double attack_momentum_weight, int time_limit_ms, const int* penalty_moves,
    const int* penalties, int penalty_count, int* out_move, int* out_score, int* out_pv,
    int max_pv, int* out_pv_count, uint64_t* out_nodes, int* out_aborted
) {
    return find_best_move_internal(
        board, side_to_move, white_king_side, white_queen_side, black_king_side,
        black_queen_side, ep_file, ep_rank, depth, quiescence_depth, material_weight,
        position_weight, mobility_weight, endgame_weight, pawn_structure_weight,
        bishop_pair_weight, rook_activity_weight, king_safety_weight, passed_pawn_weight,
        endgame_pressure_weight, line_activity_weight, attack_momentum_weight, penalty_moves,
        penalties, penalty_count, out_move, out_score, out_pv, max_pv, out_pv_count,
        out_nodes, time_limit_ms, out_aborted
    );
}

EXPORT int deeptactics_find_best_move_iterative_timed(
    const int8_t* board, int side_to_move, int white_king_side, int white_queen_side,
    int black_king_side, int black_queen_side, int ep_file, int ep_rank,
    int depth, int quiescence_depth, double material_weight, double position_weight,
    double mobility_weight, double endgame_weight, double pawn_structure_weight,
    double bishop_pair_weight, double rook_activity_weight, double king_safety_weight,
    double passed_pawn_weight, double endgame_pressure_weight, double line_activity_weight,
    double attack_momentum_weight, int time_limit_ms, const int* penalty_moves,
    const int* penalties, int penalty_count, int* out_move, int* out_score, int* out_pv,
    int max_pv, int* out_pv_count, uint64_t* out_nodes, int* out_aborted,
    int* out_completed_depth
) {
    return find_best_move_iterative_internal(
        board, side_to_move, white_king_side, white_queen_side, black_king_side,
        black_queen_side, ep_file, ep_rank, depth, quiescence_depth, material_weight,
        position_weight, mobility_weight, endgame_weight, pawn_structure_weight,
        bishop_pair_weight, rook_activity_weight, king_safety_weight, passed_pawn_weight,
        endgame_pressure_weight, line_activity_weight, attack_momentum_weight, time_limit_ms,
        penalty_moves, penalties, penalty_count, out_move, out_score, out_pv, max_pv,
        out_pv_count, out_nodes, out_aborted, out_completed_depth
    );
}

EXPORT void deeptactics_request_stop() {
    g_stop_requested.store(true, std::memory_order_relaxed);
}

EXPORT void deeptactics_set_search_experiments(int forced_line_extension_budget, int mate_danger_ordering) {
    g_forced_line_extension_budget.store(
        std::max(0, std::min(2, forced_line_extension_budget)),
        std::memory_order_relaxed
    );
    g_mate_danger_ordering.store(mate_danger_ordering != 0, std::memory_order_relaxed);
}

EXPORT void deeptactics_set_defensive_mate_threat_ordering(int enabled) {
    g_defensive_mate_threat_ordering.store(enabled != 0, std::memory_order_relaxed);
}

EXPORT void deeptactics_set_major_piece_safety_ordering(int enabled) {
    g_major_piece_safety_ordering.store(enabled != 0, std::memory_order_relaxed);
}

EXPORT void deeptactics_set_clustered_tt_replacement(int enabled) {
    g_clustered_tt_replacement.store(enabled != 0, std::memory_order_relaxed);
}

EXPORT void deeptactics_set_aspiration_windows(int enabled) {
    g_aspiration_windows.store(enabled != 0, std::memory_order_relaxed);
}

EXPORT void deeptactics_set_pawn_endgame_extension_budget(int budget) {
    g_pawn_endgame_extension_budget.store(std::max(0, std::min(2, budget)), std::memory_order_relaxed);
}

EXPORT void deeptactics_set_check_extension_budget(int budget) {
    g_check_extension_budget.store(std::max(0, std::min(2, budget)), std::memory_order_relaxed);
}

EXPORT void deeptactics_set_singular_extension(int enabled) {
    g_singular_extension.store(enabled != 0, std::memory_order_relaxed);
}

EXPORT void deeptactics_set_futility_pruning(int enabled) {
    g_futility_pruning.store(enabled != 0, std::memory_order_relaxed);
}

EXPORT void deeptactics_set_lmr_formula(int enabled) {
    g_lmr_formula.store(enabled != 0, std::memory_order_relaxed);
}

EXPORT void deeptactics_set_razoring(int enabled) {
    g_razoring.store(enabled != 0, std::memory_order_relaxed);
}

EXPORT void deeptactics_set_continuation_history(int enabled) {
    g_continuation_history.store(enabled != 0, std::memory_order_relaxed);
}

EXPORT void deeptactics_set_probcut(int enabled) {
    g_probcut.store(enabled != 0, std::memory_order_relaxed);
}

EXPORT void deeptactics_set_hash_mb(int hash_mb) {
    g_hash_mb.store(std::max(1, std::min(4096, hash_mb)), std::memory_order_relaxed);
}

EXPORT int deeptactics_get_hash_mb() {
    return g_hash_mb.load(std::memory_order_relaxed);
}

EXPORT int deeptactics_get_hash_bucket_count() {
    return transposition_table_bucket_count();
}

EXPORT uint64_t deeptactics_get_null_move_probes() {
    return g_null_move_probes;
}

EXPORT uint64_t deeptactics_get_null_move_cutoffs() {
    return g_null_move_cutoffs;
}

EXPORT uint64_t deeptactics_get_probcut_probes() {
    return g_probcut_probes;
}

EXPORT uint64_t deeptactics_get_probcut_cutoffs() {
    return g_probcut_cutoffs;
}

EXPORT uint64_t deeptactics_get_nested_null_move_probes() {
    return g_nested_null_move_probes;
}

EXPORT uint64_t deeptactics_get_aspiration_attempts() {
    return g_aspiration_attempts;
}

EXPORT uint64_t deeptactics_get_aspiration_fail_lows() {
    return g_aspiration_fail_lows;
}

EXPORT uint64_t deeptactics_get_aspiration_fail_highs() {
    return g_aspiration_fail_highs;
}

EXPORT uint64_t deeptactics_get_aspiration_researches() {
    return g_aspiration_researches;
}

EXPORT uint64_t deeptactics_get_quiet_beta_cutoffs() {
    return g_quiet_beta_cutoffs;
}

EXPORT uint64_t deeptactics_get_killer_order_hits() {
    return g_killer_order_hits;
}

EXPORT uint64_t deeptactics_get_history_order_hits() {
    return g_history_order_hits;
}

EXPORT uint64_t deeptactics_get_continuation_history_order_hits() {
    return g_continuation_history_order_hits;
}

EXPORT uint64_t deeptactics_get_history_saturated_updates() {
    return g_history_saturated_updates;
}
