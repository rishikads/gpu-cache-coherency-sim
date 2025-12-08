GPU Cache Coherency Modeling & Performance Estimation

This project implements a cycle-approximate GPU cache simulator in C++.
It models L1/L2 cache behavior, replacement policies, memory latency, coherence, and warp-level execution patterns for a simplified multi-core GPU-like architecture.

The goal of this project is to study how architectural choices affect performance metrics such as IPC, cache hit rates, memory bottlenecks, and warp divergence behavior.
CUDA microbenchmarks are included for real-hardware comparison.

1. Baseline Simulation

The default configuration is:

LRU replacement

Inclusive hierarchy

Warp size = 32

Two cores

Sequential synthetic workload

Run:

make
./sim workloads/seq_workload.txt


Save output:

./sim workloads/seq_workload.txt > results/run_lru_inclusive.txt


View results in ASCII chart format:

python3 plot_results_cli.py

2. CUDA Microbenchmarks (Hardware Validation)

These benchmarks measure real GPU latency trends for comparison with the simulator.

cd cuda
nvcc -O3 stride_bench.cu -o stride_bench
./stride_bench

nvcc -O3 divergence_bench.cu -o divergence_bench
./divergence_bench
cd ..

3. Modifying Architecture Settings

All architectural knobs are in config.h.

Change Replacement Policy
l1_config.replacement_policy = "FIFO";
l2_config.replacement_policy = "FIFO";

Toggle Inclusive / Exclusive L2
l2_config.inclusive = false;   // false means exclusive mode

Change Warp Size
warp_size = 64;  // default is 32


Rebuild and run:

make
./sim workloads/seq_workload.txt > results/run_<config>.txt
