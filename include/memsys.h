#pragma once
#include "config.h"
#include "cache.h"
#include <vector>
#include <unordered_map>
#include <cstdint>

struct MemAccessResult {
    int total_latency = 0;
    bool l1_hit = false;
    bool l2_hit = false;
};

class MemorySystem {
public:
    MemorySystem() = default;
    MemorySystem(const SystemConfig& cfg);

    MemAccessResult access(int core_id,
                           std::uint64_t addr,
                           bool is_write,
                           std::uint64_t cycle);

    AccessStats get_access_stats() const { return stats; }

private:
    SystemConfig cfg;
    std::vector<Cache> l1s;
    Cache l2;

    // Simple coherence directory: block_addr -> bitmask of sharers
    std::unordered_map<std::uint64_t, std::uint32_t> directory;

    AccessStats stats;

    void add_sharer(std::uint64_t block_addr, int core_id);
    void remove_sharer(std::uint64_t block_addr, int core_id);
    std::uint32_t get_sharers(std::uint64_t block_addr) const;

    void invalidate_others(std::uint64_t block_addr, int writer_core_id);

    std::uint64_t block_addr(std::uint64_t addr) const {
        return addr / cfg.l1_config.line_size; // assume same line size for L1/L2
    }
};

