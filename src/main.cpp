#include "sim.h"
#include <iostream>

int main(int argc, char** argv) {
    std::string workload_file = "workloads/seq_workload.txt";
    if (argc > 1) {
        workload_file = argv[1];
    }

    SystemConfig cfg;
    cfg.num_cores = 2;
    cfg.warps_per_core = 4;
    cfg.l1_config.size_bytes = 32 * 1024;
    cfg.l1_config.associativity = 4;
    cfg.l1_config.line_size = 64;
    cfg.l1_config.replacement_policy = "LRU";

    cfg.l2_config.size_bytes = 512 * 1024;
    cfg.l2_config.associativity = 8;
    cfg.l2_config.line_size = 64;
    cfg.l2_config.replacement_policy = "LRU";
    cfg.l2_config.inclusive = true; // toggle for inclusive vs exclusive

    cfg.l1_hit_latency = 4;
    cfg.l2_hit_latency = 20;
    cfg.dram_latency = 200;

    Simulator sim;
    sim.set_config(cfg);
    if (!sim.load_workload(workload_file)) {
        std::cerr << "Failed to load workload\n";
        return 1;
    }

    sim.run();
    sim.print_stats();

    return 0;
}
