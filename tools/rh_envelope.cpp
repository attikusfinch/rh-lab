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

std::vector<uint32_t> make_log_targets(uint32_t limit, uint32_t target_count) {
    std::vector<uint32_t> targets;
    targets.reserve(target_count + 8);

    const double log_min = std::log(2.0);
    const double log_max = std::log(static_cast<double>(limit));
    uint32_t previous = 0;

    for (uint32_t i = 0; i < target_count; ++i) {
        const double t = target_count == 1 ? 1.0 : static_cast<double>(i) / (target_count - 1);
        uint64_t x = static_cast<uint64_t>(std::llround(std::exp(log_min + t * (log_max - log_min))));
        if (x < 2) {
            x = 2;
        }
        if (x > limit) {
            x = limit;
        }
        if (x != previous) {
            targets.push_back(static_cast<uint32_t>(x));
            previous = static_cast<uint32_t>(x);
        }
    }

    if (targets.empty() || targets.back() != limit) {
        targets.push_back(limit);
    }
    return targets;
}

void write_envelope(
    const std::string& path,
    const std::vector<std::pair<uint32_t, double>>& events,
    const std::vector<uint32_t>& targets
) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Cannot open output CSV: " + path);
    }

    out << "X,max_abs_error,x_at_max,error_at_max,max_over_sqrt_log,max_over_sqrt_log2,empirical_exponent\n";
    out << std::setprecision(17);

    size_t event_index = 0;
    double psi = 0.0;
    double max_abs_error = 0.0;
    int64_t x_at_max = 0;
    double error_at_max = 0.0;

    auto observe = [&](uint32_t x, double error) {
        const double abs_error = std::abs(error);
        if (abs_error > max_abs_error) {
            max_abs_error = abs_error;
            x_at_max = x;
            error_at_max = error;
        }
    };

    for (const uint32_t target : targets) {
        while (event_index < events.size() && events[event_index].first <= target) {
            const uint32_t event_x = events[event_index].first;

            if (event_x > 2) {
                observe(event_x - 1, psi - static_cast<double>(event_x - 1));
            }

            while (event_index < events.size() && events[event_index].first == event_x) {
                psi += events[event_index].second;
                ++event_index;
            }
            observe(event_x, psi - static_cast<double>(event_x));
        }

        observe(target, psi - static_cast<double>(target));

        const double x = static_cast<double>(target);
        const double log_x = std::log(x);
        const double sqrt_x = std::sqrt(x);
        const double norm_log = max_abs_error / (sqrt_x * log_x);
        const double norm_log2 = max_abs_error / (sqrt_x * log_x * log_x);
        const double exponent = max_abs_error > 0.0 ? std::log(max_abs_error) / std::log(x) : 0.0;

        out << target << ','
            << max_abs_error << ','
            << x_at_max << ','
            << error_at_max << ','
            << norm_log << ','
            << norm_log2 << ','
            << exponent << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: rh_envelope <limit> <target_count> <output_csv>\n";
        std::cerr << "Example: rh_envelope 100000000 2000 outputs/rh_envelope_100m.csv\n";
        return 2;
    }

    try {
        const auto started = std::chrono::steady_clock::now();
        const uint64_t raw_limit = parse_u64(argv[1], "limit");
        const uint64_t raw_targets = parse_u64(argv[2], "target_count");
        if (raw_limit < 2 || raw_limit > UINT32_MAX) {
            throw std::runtime_error("limit must be in [2, 4294967295]");
        }
        if (raw_targets < 2 || raw_targets > UINT32_MAX) {
            throw std::runtime_error("target_count must be in [2, 4294967295]");
        }

        const uint32_t limit = static_cast<uint32_t>(raw_limit);
        const uint32_t target_count = static_cast<uint32_t>(raw_targets);
        const std::string output_csv = argv[3];

        std::cerr << "sieving up to " << limit << "...\n";
        const auto primes = sieve_primes(limit);
        std::cerr << "primes: " << primes.size() << "\n";

        std::cerr << "building prime-power events...\n";
        const auto events = make_prime_power_events(primes, limit);
        std::cerr << "events: " << events.size() << "\n";

        std::cerr << "scanning envelope...\n";
        const auto targets = make_log_targets(limit, target_count);
        write_envelope(output_csv, events, targets);

        const auto finished = std::chrono::steady_clock::now();
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(finished - started).count();
        std::cerr << "wrote " << targets.size() << " rows to " << output_csv << "\n";
        std::cerr << "elapsed: " << ms << " ms\n";
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
