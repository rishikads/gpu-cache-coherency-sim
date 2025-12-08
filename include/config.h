#pragma once
#include <string>
#include <cstdint>

struct CacheConfig {
    std::size_t size_bytes = 32 * 1024;
    std::size_t line_size  = 64;
    std::size_t associativity = 4;
    std::string replacement_policy = "LRU"; // or "FIFO"
    bool inclusive = true;                  // true = inclusive, false = exclusive
    int hit_latency = 4;                    // L1/L2 specific when used
};

struct SystemConfig {
    int num_cores = 2;
    int warps_per_core = 4;
    int warp_size = 32;

    CacheConfig l1_config;
    CacheConfig l2_config;

    int l1_hit_latency = 4;
    int l2_hit_latency = 20;
    int dram_latency   = 200;
};

struct AccessStats {
    std::uint64_t l1_hits = 0;
    std::uint64_t l1_misses = 0;
    std::uint64_t l2_hits = 0;
    std::uint64_t l2_misses = 0;
    std::uint64_t dram_accesses = 0;

    std::uint64_t coherence_invalidations = 0;
};

struct SimStats {
    std::uint64_t total_cycles = 0;
    std::uint64_t total_instructions = 0;
    double ipc = 0.0;
    AccessStats access;
};

