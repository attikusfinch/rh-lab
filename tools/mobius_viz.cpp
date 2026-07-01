#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
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

std::vector<uint32_t> make_targets(uint32_t limit, uint32_t linear_count) {
    std::vector<uint32_t> targets;
    targets.reserve(linear_count + 5000);

    for (uint32_t i = 1; i <= linear_count; ++i) {
        uint64_t x = (static_cast<uint64_t>(i) * limit) / linear_count;
        if (x < 1) {
            x = 1;
        }
        targets.push_back(static_cast<uint32_t>(x));
    }

    const uint32_t log_count = std::min<uint32_t>(5000, std::max<uint32_t>(1000, linear_count / 20));
    const double log_min = 0.0;
    const double log_max = std::log(static_cast<double>(limit));
    for (uint32_t i = 0; i < log_count; ++i) {
        const double t = log_count == 1 ? 1.0 : static_cast<double>(i) / (log_count - 1);
        uint64_t x = static_cast<uint64_t>(std::llround(std::exp(log_min + t * (log_max - log_min))));
        if (x < 1) {
            x = 1;
        }
        if (x > limit) {
            x = limit;
        }
        targets.push_back(static_cast<uint32_t>(x));
    }

    targets.push_back(1);
    targets.push_back(limit);
    std::sort(targets.begin(), targets.end());
    targets.erase(std::unique(targets.begin(), targets.end()), targets.end());
    return targets;
}

void write_mobius_csv(uint32_t limit, const std::vector<uint32_t>& targets, const std::string& path) {
    std::vector<int8_t> mu(limit + 1, 0);
    std::vector<uint8_t> composite(limit + 1, 0);
    std::vector<uint32_t> primes;
    primes.reserve(limit >= 16 ? static_cast<size_t>(limit / std::log(limit)) : 8);

    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Cannot open output CSV: " + path);
    }
    out << "x,M_x,mu_x,max_abs_M,x_at_max,M_over_sqrt,max_over_sqrt,max_over_sqrt_log,empirical_exponent\n";
    out << std::setprecision(17);

    mu[1] = 1;
    int64_t m_sum = 0;
    uint64_t max_abs = 0;
    uint32_t x_at_max = 1;
    size_t target_index = 0;

    auto emit_until = [&](uint32_t x) {
        while (target_index < targets.size() && targets[target_index] == x) {
            const double xd = static_cast<double>(x);
            const double sqrt_x = std::sqrt(xd);
            const double log_x = x > 1 ? std::log(xd) : 1.0;
            const double exponent = max_abs > 0 && x > 1
                ? std::log(static_cast<double>(max_abs)) / std::log(xd)
                : 0.0;

            out << x << ','
                << m_sum << ','
                << static_cast<int>(mu[x]) << ','
                << max_abs << ','
                << x_at_max << ','
                << (static_cast<double>(m_sum) / sqrt_x) << ','
                << (static_cast<double>(max_abs) / sqrt_x) << ','
                << (static_cast<double>(max_abs) / (sqrt_x * log_x)) << ','
                << exponent << '\n';
            ++target_index;
        }
    };

    for (uint32_t n = 1; n <= limit; ++n) {
        if (n >= 2 && !composite[n]) {
            primes.push_back(n);
            mu[n] = -1;
        }

        if (n >= 2) {
            for (const uint32_t p : primes) {
                const uint64_t m = static_cast<uint64_t>(n) * p;
                if (m > limit) {
                    break;
                }
                composite[static_cast<size_t>(m)] = 1;
                if (n % p == 0) {
                    mu[static_cast<size_t>(m)] = 0;
                    break;
                }
                mu[static_cast<size_t>(m)] = static_cast<int8_t>(-mu[n]);
            }
        }

        m_sum += mu[n];
        const uint64_t abs_sum = static_cast<uint64_t>(m_sum < 0 ? -m_sum : m_sum);
        if (abs_sum > max_abs) {
            max_abs = abs_sum;
            x_at_max = n;
        }
        emit_until(n);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: mobius_viz <limit> <sample_count> <output_csv>\n";
        std::cerr << "Example: mobius_viz 100000000 100000 outputs/mobius_100m.csv\n";
        return 2;
    }

    try {
        const auto started = std::chrono::steady_clock::now();
        const uint64_t raw_limit = parse_u64(argv[1], "limit");
        const uint64_t raw_samples = parse_u64(argv[2], "sample_count");
        if (raw_limit < 1 || raw_limit > UINT32_MAX) {
            throw std::runtime_error("limit must be in [1, 4294967295]");
        }
        if (raw_samples < 2 || raw_samples > UINT32_MAX) {
            throw std::runtime_error("sample_count must be in [2, 4294967295]");
        }

        const uint32_t limit = static_cast<uint32_t>(raw_limit);
        const uint32_t sample_count = static_cast<uint32_t>(raw_samples);
        const std::string output_csv = argv[3];

        std::cerr << "building targets...\n";
        const auto targets = make_targets(limit, sample_count);
        std::cerr << "targets: " << targets.size() << "\n";

        std::cerr << "sieving mobius up to " << limit << "...\n";
        write_mobius_csv(limit, targets, output_csv);

        const auto finished = std::chrono::steady_clock::now();
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(finished - started).count();
        std::cerr << "wrote rows to " << output_csv << "\n";
        std::cerr << "elapsed: " << ms << " ms\n";
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
