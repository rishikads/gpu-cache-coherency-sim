#pragma once
#include <vector>
#include <string>
#include <cstdint>

struct WarpPattern {
    int core_id = 0;
    int warp_id = 0;
    std::uint64_t start_addr = 0;
    std::uint64_t count = 0;
    std::uint64_t stride = 4;
};

class Workload {
public:
    bool load_from_file(const std::string& path);
    const std::vector<WarpPattern>& patterns() const { return pats; }

private:
    std::vector<WarpPattern> pats;
};
