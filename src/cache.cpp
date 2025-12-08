#include "cache.h"
#include <stdexcept>

//
// Constructor
//
Cache::Cache(const CacheConfig& c) : cfg(c)
{
    line_size = cfg.line_size;

    if (line_size == 0 || cfg.size_bytes == 0 || cfg.associativity == 0) {
        throw std::runtime_error("Invalid cache configuration");
    }

    std::size_t num_lines = cfg.size_bytes / line_size;
    num_sets = num_lines / cfg.associativity;

    if (num_sets == 0) {
        throw std::runtime_error("Cache size too small for given associativity");
    }

    // Allocate cache sets
    sets.resize(num_sets);
    for (auto& set : sets) {
        set.resize(cfg.associativity);
    }
}

//
// Address helpers
//
std::uint64_t Cache::addr_to_block(std::uint64_t addr) const {
    return addr / line_size;
}

std::size_t Cache::addr_to_index(std::uint64_t addr) const {
    return addr_to_block(addr) % num_sets;
}

std::uint64_t Cache::addr_to_tag(std::uint64_t addr) const {
    return addr_to_block(addr) / num_sets;
}

//
// Find a matching tag in the set (if valid)
//
CacheLine* Cache::find_line(std::size_t index, std::uint64_t tag)
{
    auto& set = sets[index];
    for (auto& line : set) {
        if (line.valid && line.tag == tag) {
            return &line;
        }
    }
    return nullptr;
}

//
// Choose a victim based on replacement policy
//
CacheLine* Cache::choose_victim(std::size_t index,
                                std::uint64_t cycle,
                                bool& was_valid_before,
                                std::uint64_t& evicted_block_addr)
{
    auto& set = sets[index];

    // Prefer invalid lines
    for (auto& line : set) {
        if (!line.valid) {
            was_valid_before = false;
            evicted_block_addr = 0;
            return &line;
        }
    }

    // Otherwise pick LRU or FIFO
    std::size_t victim = 0;

    if (cfg.replacement_policy == "FIFO") {
        // Pick line with the oldest insert_time
        std::uint64_t oldest = set[0].insert_time;
        for (std::size_t i = 1; i < set.size(); ++i) {
            if (set[i].insert_time < oldest) {
                oldest = set[i].insert_time;
                victim = i;
            }
        }
    }
    else {
        // Default = LRU
        std::uint64_t oldest = set[0].last_access_time;
        for (std::size_t i = 1; i < set.size(); ++i) {
            if (set[i].last_access_time < oldest) {
                oldest = set[i].last_access_time;
                victim = i;
            }
        }
    }

    // If victim was valid, compute block address for inclusive/exclusive logic
    was_valid_before = set[victim].valid;

    if (was_valid_before) {
        std::uint64_t block = (set[victim].tag * num_sets + index);
        evicted_block_addr = block * line_size;
    } else {
        evicted_block_addr = 0;
    }

    return &set[victim];
}

//
// Main access function (hit/miss + eviction handling)
//
CacheAccessResult Cache::access(std::uint64_t addr,
                                bool is_write,
                                std::uint64_t cycle)
{
    CacheAccessResult result;

    std::size_t index = addr_to_index(addr);
    std::uint64_t tag = addr_to_tag(addr);

    // Check for hit
    CacheLine* line = find_line(index, tag);
    if (line) {
        result.hit = true;

        // Update timestamps
        line->last_access_time = cycle;
        if (is_write) line->dirty = true;

        return result;
    }

    // Miss: choose victim to replace
    bool was_valid = false;
    std::uint64_t evicted_addr = 0;

    CacheLine* victim = choose_victim(index, cycle, was_valid, evicted_addr);

    if (was_valid) {
        result.evicted = true;
        result.evicted_block_addr = evicted_addr;
    }

    // Fill in the new line
    victim->valid = true;
    victim->dirty = is_write;
    victim->tag = tag;
    victim->last_access_time = cycle;
    victim->insert_time = cycle;

    return result;
}

//
// Invalidate a block (used by coherence protocol)
//
bool Cache::invalidate(std::uint64_t addr)
{
    std::size_t index = addr_to_index(addr);
    std::uint64_t tag = addr_to_tag(addr);

    bool invalidated = false;

    for (auto& line : sets[index]) {
        if (line.valid && line.tag == tag) {
            line.valid = false;
            invalidated = true;
        }
    }

    return invalidated;
}

