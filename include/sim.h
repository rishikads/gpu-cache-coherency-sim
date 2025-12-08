#pragma once
#include "config.h"
#include "memsys.h"
#include "workload.h"
#include <vector>
#include <cstdint>

struct WarpState {
    int core_id = 0;
    int warp_id = 0;
    std::uint64_t next_addr = 0;
    std::uint64_t remaining = 0;
    std::uint64_t stride = 4;

    bool stalled = false;
    std::uint64_t unblock_cycle = 0;
};

struct CoreState {
    std::vector<WarpState> warps;
};

class Simulator {
public:
    Simulator() = default;

    void set_config(const SystemConfig& c);
    bool load_workload(const std::string& path);
    void run();
    void print_stats() const;

    SimStats get_stats() const { return stats; }

private:
    SystemConfig cfg;
    MemorySystem mem;
    Workload workload;

    std::vector<CoreState> cores;
    SimStats stats;

    bool all_done() const;
};
