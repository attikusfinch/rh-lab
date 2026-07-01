#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

uint64_t parse_u64(const char* value, const char* name) {
    try {
        size_t pos = 0;
        const auto parsed = std::stoull(value, &pos);
        if (pos != std::string(value).size()) {
            throw std::invalid_argument("trailing characters");
        }
        return parsed;
    } catch (const std::exception& error) {
        throw std::runtime_error(std::string("Invalid ") + name + ": " + error.what());
    }
}

std::vector<uint32_t> sieve_primes(uint32_t limit) {
    std::vector<uint8_t> composite(limit + 1, 0);
    std::vector<uint32_t> primes;
    if (limit >= 16) {
        primes.reserve(static_cast<size_t>(limit / std::log(limit)));
    }

    for (uint32_t n = 2; n <= limit; ++n) {
        if (!composite[n]) {
            primes.push_back(n);
            if (static_cast<uint64_t>(n) * n <= limit) {
                for (uint64_t m = static_cast<uint64_t>(n) * n; m <= limit; m += n) {
                    composite[static_cast<size_t>(m)] = 1;
                }
            }
        }
    }
    return primes;
}

void write_binary(const std::string& path, const std::vector<uint32_t>& primes) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Cannot open output binary: " + path);
    }
    out.write(reinterpret_cast<const char*>(primes.data()), static_cast<std::streamsize>(primes.size() * sizeof(uint32_t)));
    if (!out) {
        throw std::runtime_error("Failed while writing output binary: " + path);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: primes_dump <limit> <output_u32_bin>\n";
        std::cerr << "Example: primes_dump 10000000 outputs/primes_10m.u32\n";
        return 2;
    }

    try {
        const auto started = std::chrono::steady_clock::now();
        const uint64_t raw_limit = parse_u64(argv[1], "limit");
        if (raw_limit < 2 || raw_limit > UINT32_MAX) {
            throw std::runtime_error("limit must be in [2, 4294967295]");
        }

        const uint32_t limit = static_cast<uint32_t>(raw_limit);
        const std::string output_bin = argv[2];

        std::cerr << "sieving primes up to " << limit << "...\n";
        const auto primes = sieve_primes(limit);
        std::cerr << "primes: " << primes.size() << "\n";

        write_binary(output_bin, primes);

        const auto finished = std::chrono::steady_clock::now();
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(finished - started).count();
        std::cerr << "wrote " << output_bin << "\n";
        std::cerr << "elapsed: " << ms << " ms\n";
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
