#include "sim.h"
#include <iostream>

void Simulator::set_config(const SystemConfig& c) {
    cfg = c;
    mem = MemorySystem(cfg);

    cores.clear();
    cores.resize(cfg.num_cores);
}

bool Simulator::load_workload(const std::string& path) {
    if (!workload.load_from_file(path)) return false;

    // Initialize warps from workload
    for (const auto& wp : workload.patterns()) {
        if (wp.core_id < 0 || wp.core_id >= cfg.num_cores) {
            std::cerr << "Ignoring pattern with invalid core: " << wp.core_id << "\n";
            continue;
        }
        if ((int)cores[wp.core_id].warps.size() <= wp.warp_id) {
            cores[wp.core_id].warps.resize(wp.warp_id + 1);
        }
        WarpState ws;
        ws.core_id = wp.core_id;
        ws.warp_id = wp.warp_id;
        ws.next_addr = wp.start_addr;
        ws.remaining = wp.count;
        ws.stride = wp.stride;
        cores[wp.core_id].warps[wp.warp_id] = ws;
    }
    return true;
}

bool Simulator::all_done() const {
    for (const auto& core : cores) {
        for (const auto& warp : core.warps) {
            if (warp.remaining > 0) return false;
        }
    }
    return true;
}

void Simulator::run() {
    std::uint64_t cycle = 0;
    std::uint64_t total_instr = 0;

    while (!all_done()) {
        for (int c = 0; c < cfg.num_cores; ++c) {
            auto& core = cores[c];
            for (auto& warp : core.warps) {
                if (warp.remaining == 0) continue;

                if (warp.stalled) {
                    if (cycle >= warp.unblock_cycle) {
                        warp.stalled = false;
                    } else {
                        continue;
                    }
                }

                // Issue 1 memory instruction per active warp per cycle
                auto res = mem.access(c, warp.next_addr, false, cycle);
                warp.stalled = true;
                warp.unblock_cycle = cycle + res.total_latency;

                warp.next_addr += warp.stride;
                warp.remaining--;
                total_instr++;
            }
        }
        cycle++;
    }

    stats.total_cycles = cycle;
    stats.total_instructions = total_instr;
    stats.ipc = (cycle > 0) ? (double)total_instr / (double)cycle : 0.0;
    stats.access = mem.get_access_stats();
}

void Simulator::print_stats() const {
    std::cout << "Total cycles: " << stats.total_cycles << "\n";
    std::cout << "Total instructions: " << stats.total_instructions << "\n";
    std::cout << "IPC: " << stats.ipc << "\n";

    std::cout << "L1 hits: " << stats.access.l1_hits
              << ", L1 misses: " << stats.access.l1_misses << "\n";
    std::cout << "L2 hits: " << stats.access.l2_hits
              << ", L2 misses: " << stats.access.l2_misses << "\n";
    std::cout << "DRAM accesses: " << stats.access.dram_accesses << "\n";
    std::cout << "Coherence invalidations: " << stats.access.coherence_invalidations << "\n";
}
