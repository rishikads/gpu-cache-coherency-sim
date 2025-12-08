#include "memsys.h"
#include <stdexcept>

MemorySystem::MemorySystem(const SystemConfig& c)
    : cfg(c),
      l1s(c.num_cores, Cache(c.l1_config)),
      l2(c.l2_config) {
}

void MemorySystem::add_sharer(std::uint64_t block_addr, int core_id) {
    std::uint32_t mask = directory[block_addr];
    mask |= (1u << core_id);
    directory[block_addr] = mask;
}

void MemorySystem::remove_sharer(std::uint64_t block_addr, int core_id) {
    auto it = directory.find(block_addr);
    if (it == directory.end()) return;
    it->second &= ~(1u << core_id);
    if (it->second == 0) {
        directory.erase(it);
    }
}

std::uint32_t MemorySystem::get_sharers(std::uint64_t block_addr) const {
    auto it = directory.find(block_addr);
    if (it == directory.end()) return 0;
    return it->second;
}

void MemorySystem::invalidate_others(std::uint64_t block, int writer_core_id) {
    std::uint32_t mask = get_sharers(block);
    if (!mask) return;

    for (int c = 0; c < cfg.num_cores; ++c) {
        if (c == writer_core_id) continue;
        if (mask & (1u << c)) {
            bool inv = l1s[c].invalidate(block * cfg.l1_config.line_size);
            if (inv) {
                stats.coherence_invalidations++;
            }
        }
    }

    // Only writer core remains as sharer
    directory[block] = (1u << writer_core_id);
}

MemAccessResult MemorySystem::access(int core_id,
                                     std::uint64_t addr,
                                     bool is_write,
                                     std::uint64_t cycle) {
    MemAccessResult res;
    auto& l1 = l1s.at(core_id);
    std::uint64_t blk = block_addr(addr);

    // Coherence: if write, invalidate others first (write-invalidate protocol)
    if (is_write) {
        invalidate_others(blk, core_id);
    }

    // L1 access
    auto l1_res = l1.access(addr, is_write, cycle);
    if (l1_res.hit) {
        res.l1_hit = true;
        res.total_latency = cfg.l1_hit_latency;
        stats.l1_hits++;
        add_sharer(blk, core_id);
        return res;
    }

    stats.l1_misses++;

    // L2 access (inclusive / exclusive behavior simplified)
    auto l2_res = l2.access(addr, is_write, cycle);
    if (l2_res.hit) {
        res.l2_hit = true;
        stats.l2_hits++;
        // Fill L1
        auto fill_res = l1.access(addr, is_write, cycle);
        (void)fill_res; // we don't care about eviction from L1 here

        // Exclusive: remove from L2 after moving into L1
        if (!cfg.l2_config.inclusive) {
            l2.invalidate(addr);
        }

        res.total_latency = cfg.l1_hit_latency + cfg.l2_hit_latency;
        add_sharer(blk, core_id);
        return res;
    }

    stats.l2_misses++;

    // DRAM access
    stats.dram_accesses++;

    // On DRAM fill:
    // Inclusive: DRAM -> L2 -> L1
    // Exclusive: DRAM -> L1 (optionally push evicted L1 line to L2 – omitted here for simplicity)
    if (cfg.l2_config.inclusive) {
        auto fill_l2 = l2.access(addr, is_write, cycle);
        (void)fill_l2;
    }
    auto fill_l1 = l1.access(addr, is_write, cycle);
    (void)fill_l1;

    res.total_latency = cfg.l1_hit_latency + cfg.l2_hit_latency + cfg.dram_latency;
    add_sharer(blk, core_id);
    return res;
}
