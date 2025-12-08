# GPU Cache Coherency Modeling & Performance Estimation

This project implements a cycle-approximate GPU cache simulator in C++.  
It models L1/L2 cache behavior, replacement policies, memory latency, coherence, and warp-level execution patterns for a simplified multi-core GPU-like architecture.

The goal of this project is to study how architectural choices affect performance metrics such as IPC, cache hit rates, memory bottlenecks, and warp divergence behavior.  
CUDA microbenchmarks are included for real-hardware comparison.

---

## Baseline Simulation (W1)

The default configuration corresponds to:

- LRU replacement  
- Inclusive hierarchy  
- Warp size = 32  
- Two cores  
- Sequential synthetic workload  

Run:

```bash
make clean
make
./sim workloads/seq_workload.txt
```

Save output:

```bash
./sim workloads/seq_workload.txt > results/run_W1_lru_inclusive_ws32.txt
```

View results in ASCII chart format:

```bash
python3 plot_results_cli.py
```

---

## CUDA Microbenchmarks (Hardware Validation)

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

All architectural knobs are in `config.h`.

### Change Replacement Policy

```cpp
l1_config.replacement_policy = "FIFO";
l2_config.replacement_policy = "FIFO";
```

### Toggle Inclusive / Exclusive L2

```cpp
l2_config.inclusive = false;    // false means exclusive mode
```

### Change Warp Size

```cpp
warp_size = 64;    // default is 32
```

Rebuild and run:

```bash
make clean
make
./sim workloads/seq_workload.txt > results/run_<config>.txt
```

---

## Saving Results for Each Configuration (W1–W8)

Use the following naming convention when saving each experiment:

```
run_W<id>_<replacement>_<hierarchy>_ws<warp_size>.txt
```

Examples:

```bash
# W1: LRU, Inclusive, Warp Size 32 (baseline)
./sim workloads/seq_workload.txt > results/run_W1_lru_inclusive_ws32.txt

# W2: LRU, Inclusive, Warp Size 64
./sim workloads/seq_workload.txt > results/run_W2_lru_inclusive_ws64.txt

# W3: FIFO, Inclusive, Warp Size 32
./sim workloads/seq_workload.txt > results/run_W3_fifo_inclusive_ws32.txt

# W4: FIFO, Inclusive, Warp Size 64
./sim workloads/seq_workload.txt > results/run_W4_fifo_inclusive_ws64.txt

# W5: LRU, Exclusive, Warp Size 32
./sim workloads/seq_workload.txt > results/run_W5_lru_exclusive_ws32.txt

# W6: LRU, Exclusive, Warp Size 64
./sim workloads/seq_workload.txt > results/run_W6_lru_exclusive_ws64.txt

# W7: FIFO, Exclusive, Warp Size 32
./sim workloads/seq_workload.txt > results/run_W7_fifo_exclusive_ws32.txt

# W8: FIFO, Exclusive, Warp Size 64
./sim workloads/seq_workload.txt > results/run_W8_fifo_exclusive_ws64.txt
```

The analysis script automatically detects all `.txt` files inside the `results/` directory and compares IPC, cycles, L1/L2 hit rates, and memory behavior across configurations.

---
