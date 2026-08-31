#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr int N = 5;
constexpr uint32_t FULL = (uint32_t{1} << 25) - 1;
std::vector<uint32_t> lines;
std::array<std::array<int, 25>, 8> maps{};

struct Entry {
    int8_t value;
    int8_t bound;  // 0 exact, 1 lower, 2 upper
};

std::unordered_map<uint64_t, Entry> table;
uint64_t nodes = 0;
uint64_t hits = 0;
size_t max_table_entries = 32'000'000;

int cell(int row, int col) { return row * N + col; }

void initialize() {
    if (const char* configured = std::getenv("LAYER0_MAX_CACHE_ENTRIES")) {
        const auto parsed = std::strtoull(configured, nullptr, 10);
        if (parsed >= 26'000'000) max_table_entries = static_cast<size_t>(parsed);
    }
    const int directions[4][2] = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            for (const auto& direction : directions) {
                uint32_t mask = 0;
                bool valid = true;
                for (int step = 0; step < 4; ++step) {
                    int r = row + step * direction[0];
                    int c = col + step * direction[1];
                    if (r < 0 || r >= N || c < 0 || c >= N) {
                        valid = false;
                        break;
                    }
                    mask |= uint32_t{1} << cell(r, c);
                }
                if (valid && std::find(lines.begin(), lines.end(), mask) == lines.end()) {
                    lines.push_back(mask);
                }
            }
        }
    }
    for (int symmetry = 0; symmetry < 8; ++symmetry) {
        for (int source = 0; source < 25; ++source) {
            int row = source / N;
            int col = source % N;
            int rotations = symmetry;
            if (symmetry >= 4) {
                col = N - 1 - col;
                rotations -= 4;
            }
            for (int rotation = 0; rotation < rotations; ++rotation) {
                int old_row = row;
                row = col;
                col = N - 1 - old_row;
            }
            maps[symmetry][source] = cell(row, col);
        }
    }
}

bool has_four(uint32_t bits) {
    for (uint32_t line : lines) {
        if ((bits & line) == line) return true;
    }
    return false;
}

bool can_win(uint32_t blocker) {
    for (uint32_t line : lines) {
        if ((blocker & line) == 0) return true;
    }
    return false;
}

uint32_t winning_cells(uint32_t bits, uint32_t occupied) {
    uint32_t result = 0;
    uint32_t empty = FULL ^ occupied;
    while (empty) {
        uint32_t move = empty & (~empty + 1);
        if (has_four(bits | move)) result |= move;
        empty ^= move;
    }
    return result;
}

uint32_t transform(uint32_t bits, int symmetry) {
    uint32_t result = 0;
    while (bits) {
        uint32_t low = bits & (~bits + 1);
        int source = __builtin_ctz(bits);
        result |= uint32_t{1} << maps[symmetry][source];
        bits ^= low;
    }
    return result;
}

uint64_t canonical_key(uint32_t current, uint32_t opponent) {
    uint64_t best = ~uint64_t{0};
    for (int symmetry = 0; symmetry < 8; ++symmetry) {
        uint64_t key = uint64_t{transform(current, symmetry)} |
                       (uint64_t{transform(opponent, symmetry)} << 25);
        best = std::min(best, key);
    }
    return best;
}

int move_score(uint32_t current, uint32_t opponent, uint32_t move) {
    uint32_t occupied = current | opponent | move;
    int forks = __builtin_popcount(winning_cells(current | move, occupied));
    int cell_index = __builtin_ctz(move);
    int row = cell_index / N;
    int col = cell_index % N;
    int centrality = 4 - std::abs(row - 2) - std::abs(col - 2);
    int live = 0;
    int blocks = 0;
    for (uint32_t line : lines) {
        if ((line & opponent) == 0 && (line & current) && (line & move)) ++live;
        if ((line & current) == 0 && (line & opponent) && (line & move)) ++blocks;
    }
    return forks * 1000 + blocks * 50 + live * 10 + centrality;
}

std::vector<uint32_t> ordered_moves(uint32_t current, uint32_t opponent) {
    uint32_t occupied = current | opponent;
    uint32_t wins = winning_cells(current, occupied);
    uint32_t threats = winning_cells(opponent, occupied);
    uint32_t candidates = wins ? wins : (__builtin_popcount(threats) == 1 ? threats : FULL ^ occupied);
    std::vector<std::pair<int, uint32_t>> scored;
    while (candidates) {
        uint32_t move = candidates & (~candidates + 1);
        candidates ^= move;
        scored.emplace_back(move_score(current, opponent, move), move);
    }
    std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) {
        if (a.first != b.first) return a.first > b.first;
        return a.second < b.second;
    });
    std::vector<uint32_t> result;
    result.reserve(scored.size());
    for (const auto& item : scored) result.push_back(item.second);
    return result;
}

int solve(uint32_t current, uint32_t opponent, int alpha, int beta) {
    ++nodes;
    uint64_t key = canonical_key(current, opponent);
    auto found = table.find(key);
    if (found != table.end()) {
        ++hits;
        if (found->second.bound == 0) return found->second.value;
        if (found->second.bound == 1) alpha = std::max(alpha, int(found->second.value));
        if (found->second.bound == 2) beta = std::min(beta, int(found->second.value));
        if (alpha >= beta) return found->second.value;
    }
    const int original_alpha = alpha;
    const int original_beta = beta;
    uint32_t occupied = current | opponent;
    int value = 0;
    if (occupied == FULL || (!can_win(opponent) && !can_win(current))) {
        value = 0;
    } else if (winning_cells(current, occupied)) {
        value = 1;
    } else {
        uint32_t threats = winning_cells(opponent, occupied);
        if (__builtin_popcount(threats) >= 2) {
            value = -1;
        } else {
            value = -2;
            for (uint32_t move : ordered_moves(current, opponent)) {
                int child = has_four(current | move)
                                ? 1
                                : -solve(opponent, current | move, -beta, -alpha);
                value = std::max(value, child);
                alpha = std::max(alpha, value);
                if (alpha >= beta) break;
            }
        }
    }
    int bound = 0;
    if (value <= original_alpha) bound = 2;
    else if (value >= original_beta) bound = 1;
    table[key] = Entry{static_cast<int8_t>(value), static_cast<int8_t>(bound)};
    return value;
}

}  // namespace

std::string analyze_position(uint32_t red, uint32_t blue, int player) {
    if (has_four(red) || has_four(blue)) {
        return "{\"terminal\":true}";
    }
    bool cache_reset = false;
    if (table.size() > max_table_entries) {
        std::unordered_map<uint64_t, Entry> empty;
        table.swap(empty);
        cache_reset = true;
    }
    uint32_t current = player == 1 ? red : blue;
    uint32_t opponent = player == 1 ? blue : red;
    const uint64_t start_nodes = nodes;
    const uint64_t start_hits = hits;
    auto start = std::chrono::steady_clock::now();
    int root_value = solve(current, opponent, -1, 1);
    std::vector<int> optimal;
    for (uint32_t move : ordered_moves(current, opponent)) {
        int value;
        if (has_four(current | move)) {
            value = 1;
        } else if (root_value == 1) {
            value = -solve(opponent, current | move, -1, 0);
        } else if (root_value == 0) {
            value = -solve(opponent, current | move, 0, 1);
        } else {
            value = -1;
        }
        if (value == root_value) optimal.push_back(__builtin_ctz(move) + 1);
    }
    auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::sort(optimal.begin(), optimal.end());
    std::ostringstream output;
    output << "{\"value\":" << root_value << ",\"optimal_moves\":[";
    for (size_t index = 0; index < optimal.size(); ++index) {
        if (index) output << ',';
        output << optimal[index];
    }
    output << "],\"nodes\":" << (nodes - start_nodes)
           << ",\"cache_hits\":" << (hits - start_hits)
           << ",\"cache_size\":" << table.size()
           << ",\"cache_reset\":" << (cache_reset ? "true" : "false")
           << ",\"seconds\":" << elapsed << "}";
    return output.str();
}

int main(int argc, char** argv) {
    initialize();
    if (argc >= 2 && std::string(argv[1]) == "--server") {
        std::string first;
        while (std::cin >> first) {
            if (first == "quit") break;
            uint32_t red = static_cast<uint32_t>(std::stoul(first));
            uint32_t blue = 0;
            int player = 1;
            if (!(std::cin >> blue >> player)) return 5;
            std::cout << analyze_position(red, blue, player) << std::endl;
        }
        return 0;
    }
    uint32_t red = 0;
    uint32_t blue = 0;
    int player = 1;
    int first_move_argument = 1;
    if (argc >= 5 && std::string(argv[1]) == "--state") {
        red = static_cast<uint32_t>(std::stoul(argv[2]));
        blue = static_cast<uint32_t>(std::stoul(argv[3]));
        player = std::stoi(argv[4]);
        first_move_argument = argc;
    }
    for (int index = first_move_argument; index < argc; ++index) {
        std::string token(argv[index]);
        if (token != "pass" && token != "p") {
            int position = std::stoi(token);
            if (position < 1 || position > 25) return 3;
            uint32_t move = uint32_t{1} << (position - 1);
            if ((red | blue) & move) return 4;
            (player == 1 ? red : blue) |= move;
        }
        player = -player;
    }
    std::cout << analyze_position(red, blue, player) << '\n';
    return 0;
}
