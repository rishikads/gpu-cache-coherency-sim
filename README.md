# GPU Cache Coherency Modeling & Performance Estimation

This project implements a cycle-approximate GPU cache simulator in C++.  
It models L1/L2 cache behavior, replacement policies, memory latency, coherence, and warp-level execution patterns for a simplified multi-core GPU-like architecture.

The goal of this project is to study how architectural choices affect performance metrics such as IPC, cache hit rates, memory bottlenecks, and warp divergence behavior.  
CUDA microbenchmarks are included for real-hardware comparison.

---

## Baseline Simulation (W1)

The default configuration corresponds to:

| Component   | Value                 |
| ----------- | --------------------- |
| L1 size     | 32KB                  |
| L1 assoc    | 4-way                 |
| L2 size     | 512KB                 |
| L2 assoc    | 8-way                 |
| Replacement | LRU                   |
| L2 policy   | Inclusive             |
| Latencies   | L1=4, L2=20, DRAM=200 | 

Run:

```bash
make clean
make
```

Save output:
Using reuse workload and sequential workload.
```bash
./sim workloads/reuse_workload.txt > results/run_W1_reuse.txt
./sim workloads/seq_workload.txt > results/run_W1_seq.txt
```

View results in ASCII chart format:
(Requires your working Python environment to be activated)
```bash
source ~/miniconda/bin/activate
conda activate gpu

python plot_results_cli.py
mv results/plot.png results/plot_W1.png
```

---

## CUDA Microbenchmarks (Hardware Validation)(W0)

These benchmarks measure real GPU latency trends for comparison with the simulator.

```bash
cd cuda
nvcc -O3 stride_bench.cu -o stride_bench
./stride_bench

nvcc -O3 divergence_bench.cu -o divergence_bench
./divergence_bench
cd ..
```

---

## Modifying Architecture Settings

All architectural knobs are in `main.cpp`.

### Change Replacement Policy

```cpp
cfg.l1_config.replacement_policy = "LRU";
cfg.l2_config.replacement_policy = "LRU";

cfg.l1_config.replacement_policy = "FIFO";
cfg.l2_config.replacement_policy = "FIFO";

```

### Toggle Inclusive

```cpp
cfg.l2_config.inclusive = true;
cfg.l2_config.inclusive = false;
```

### Change Warp Size

```cpp
cfg.warp_size = 64;    // default is 32
```

Rebuild and run

The analysis script automatically detects all `.txt` files inside the `results/` directory and compares IPC, cycles, L1/L2 hit rates, and memory behavior across configurations.

---
