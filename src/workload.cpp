#include "workload.h"
#include <fstream>
#include <sstream>
#include <iostream>

bool Workload::load_from_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "Failed to open workload file: " << path << "\n";
        return false;
    }
    pats.clear();

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::string core_kw, warp_kw, seq_kw;
        WarpPattern wp;

        iss >> core_kw >> wp.core_id >> warp_kw >> wp.warp_id >> seq_kw;
        if (!iss || core_kw != "CORE" || warp_kw != "WARP" || seq_kw != "SEQ") {
            std::cerr << "Skipping malformed line: " << line << "\n";
            continue;
        }
        iss >> wp.start_addr >> wp.count >> wp.stride;
        if (!iss) {
            std::cerr << "Skipping malformed line: " << line << "\n";
            continue;
        }
        pats.push_back(wp);
    }
    return true;
}
