#pragma once
#include "config.h"
#include <vector>
#include <cstdint>

struct CacheAccessResult {
    bool hit = false;
    bool evicted = false;
    std::uint64_t evicted_block_addr = 0; // block-aligned
};

struct CacheLine {
    bool valid = false;
    bool dirty = false;
    std::uint64_t tag = 0;
    std::uint64_t last_access_time = 0;  // LRU
    std::uint64_t insert_time = 0;       // FIFO
};

class Cache {
public:
    Cache() = default;
    Cache(const CacheConfig& cfg);

    // Returns hit/miss + eviction info
    CacheAccessResult access(std::uint64_t addr,
                             bool is_write,
                             std::uint64_t cycle);

    // Invalidate a line for coherence
    bool invalidate(std::uint64_t addr);

    std::size_t get_line_size() const { return line_size; }

private:
    CacheConfig cfg;
    std::size_t num_sets = 0;
    std::size_t line_size = 0;

    std::vector<std::vector<CacheLine>> sets;

    std::uint64_t addr_to_tag(std::uint64_t addr) const;
    std::size_t addr_to_index(std::uint64_t addr) const;
    std::uint64_t addr_to_block(std::uint64_t addr) const;

    CacheLine* find_line(std::size_t index, std::uint64_t tag);

    CacheLine* choose_victim(std::size_t index,
                             std::uint64_t cycle,
                             bool& was_valid_before,
                             std::uint64_t& evicted_block_addr);
};

