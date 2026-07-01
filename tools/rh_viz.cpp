#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Sample {
    uint32_t x;
    uint32_t pi_x;
    double psi_x;
};

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
    std::vector<bool> composite(limit + 1, false);
    std::vector<uint32_t> primes;
    if (limit >= 2) {
        primes.reserve(static_cast<size_t>(limit / std::max(1.0, std::log(limit))));
    }

    for (uint32_t n = 2; n <= limit; ++n) {
        if (!composite[n]) {
            primes.push_back(n);
            if (static_cast<uint64_t>(n) * n <= limit) {
                for (uint64_t m = static_cast<uint64_t>(n) * n; m <= limit; m += n) {
                    composite[static_cast<size_t>(m)] = true;
                }
            }
        }
    }
    return primes;
}

std::vector<uint32_t> make_linear_samples(uint32_t limit, uint32_t sample_count) {
    std::vector<uint32_t> samples;
    samples.reserve(sample_count);
    uint32_t previous = 0;
    for (uint32_t i = 1; i <= sample_count; ++i) {
        uint64_t x = (static_cast<uint64_t>(i) * limit) / sample_count;
        if (x < 2) {
            x = 2;
        }
        if (x != previous) {
            samples.push_back(static_cast<uint32_t>(x));
            previous = static_cast<uint32_t>(x);
        }
    }
    return samples;
}

std::vector<std::pair<uint32_t, double>> make_prime_power_events(
    const std::vector<uint32_t>& primes,
    uint32_t limit
) {
    std::vector<std::pair<uint32_t, double>> events;
    events.reserve(primes.size() + static_cast<size_t>(std::sqrt(limit)) + 16);

    for (const uint32_t p : primes) {
        const double log_p = std::log(static_cast<double>(p));
        uint64_t power = p;
        while (power <= limit) {
            events.emplace_back(static_cast<uint32_t>(power), log_p);
            if (power > limit / p) {
                break;
            }
            power *= p;
        }
    }

    std::sort(events.begin(), events.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });
    return events;
}

std::vector<Sample> compute_samples(
    const std::vector<uint32_t>& primes,
    const std::vector<std::pair<uint32_t, double>>& events,
    const std::vector<uint32_t>& sample_points
) {
    std::vector<Sample> samples;
    samples.reserve(sample_points.size());

    size_t prime_index = 0;
    size_t event_index = 0;
    double psi = 0.0;

    for (const uint32_t x : sample_points) {
        while (prime_index < primes.size() && primes[prime_index] <= x) {
            ++prime_index;
        }
        while (event_index < events.size() && events[event_index].first <= x) {
            psi += events[event_index].second;
            ++event_index;
        }
        samples.push_back(Sample{x, static_cast<uint32_t>(prime_index), psi});
    }

    return samples;
}

void write_csv(const std::string& path, const std::vector<Sample>& samples) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Cannot open output CSV: " + path);
    }

    out << "x,pi_x,psi_x,psi_minus_x,norm_sqrt_log2,norm_sqrt_log\n";
    out << std::setprecision(17);
    for (const auto& sample : samples) {
        const double x = static_cast<double>(sample.x);
        const double log_x = std::log(x);
        const double error = sample.psi_x - x;
        const double sqrt_x = std::sqrt(x);
        const double norm_log2 = error / (sqrt_x * log_x * log_x);
        const double norm_log = error / (sqrt_x * log_x);
        out << sample.x << ','
            << sample.pi_x << ','
            << sample.psi_x << ','
            << error << ','
            << norm_log2 << ','
            << norm_log << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: rh_viz <limit> <sample_count> <output_csv>\n";
        std::cerr << "Example: rh_viz 10000000 50000 outputs/rh_viz_10m.csv\n";
        return 2;
    }

    try {
        const auto started = std::chrono::steady_clock::now();
        const uint64_t raw_limit = parse_u64(argv[1], "limit");
        const uint64_t raw_samples = parse_u64(argv[2], "sample_count");
        if (raw_limit < 2 || raw_limit > UINT32_MAX) {
            throw std::runtime_error("limit must be in [2, 4294967295]");
        }
        if (raw_samples < 2 || raw_samples > UINT32_MAX) {
            throw std::runtime_error("sample_count must be in [2, 4294967295]");
        }

        const uint32_t limit = static_cast<uint32_t>(raw_limit);
        const uint32_t sample_count = static_cast<uint32_t>(raw_samples);
        const std::string output_csv = argv[3];

        std::cerr << "sieving up to " << limit << "...\n";
        const auto primes = sieve_primes(limit);
        std::cerr << "primes: " << primes.size() << "\n";

        std::cerr << "building prime-power events...\n";
        const auto events = make_prime_power_events(primes, limit);
        std::cerr << "events: " << events.size() << "\n";

        std::cerr << "sampling...\n";
        const auto sample_points = make_linear_samples(limit, sample_count);
        const auto samples = compute_samples(primes, events, sample_points);

        write_csv(output_csv, samples);

        const auto finished = std::chrono::steady_clock::now();
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(finished - started).count();
        std::cerr << "wrote " << samples.size() << " rows to " << output_csv << "\n";
        std::cerr << "elapsed: " << ms << " ms\n";
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
