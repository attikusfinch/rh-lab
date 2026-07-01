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

std::vector<int8_t> compute_liouville(uint32_t limit) {
    std::vector<int8_t> lambda(limit + 1, 0);
    std::vector<uint8_t> composite(limit + 1, 0);
    std::vector<uint32_t> primes;
    primes.reserve(limit >= 16 ? static_cast<size_t>(limit / std::log(limit)) : 8);

    lambda[1] = 1;
    for (uint32_t n = 2; n <= limit; ++n) {
        if (!composite[n]) {
            primes.push_back(n);
            lambda[n] = -1;
        }

        for (const uint32_t p : primes) {
            const uint64_t m = static_cast<uint64_t>(n) * p;
            if (m > limit) {
                break;
            }
            composite[static_cast<size_t>(m)] = 1;
            lambda[static_cast<size_t>(m)] = static_cast<int8_t>(-lambda[n]);
            if (n % p == 0) {
                break;
            }
        }
    }

    return lambda;
}

void write_binary(const std::string& path, const std::vector<int8_t>& lambda) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Cannot open output binary: " + path);
    }
    out.write(reinterpret_cast<const char*>(lambda.data()), static_cast<std::streamsize>(lambda.size()));
    if (!out) {
        throw std::runtime_error("Failed while writing output binary: " + path);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: liouville_dump <limit> <output_bin>\n";
        std::cerr << "Example: liouville_dump 67108864 outputs/lambda_67m.i8\n";
        return 2;
    }

    try {
        const auto started = std::chrono::steady_clock::now();
        const uint64_t raw_limit = parse_u64(argv[1], "limit");
        if (raw_limit < 1 || raw_limit > UINT32_MAX) {
            throw std::runtime_error("limit must be in [1, 4294967295]");
        }

        const uint32_t limit = static_cast<uint32_t>(raw_limit);
        const std::string output_bin = argv[2];

        std::cerr << "computing liouville up to " << limit << "...\n";
        const auto lambda = compute_liouville(limit);

        std::cerr << "writing " << lambda.size() << " int8 values...\n";
        write_binary(output_bin, lambda);

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
