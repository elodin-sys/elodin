// Real-time pacing and per-phase timings, the parts of Elodin's tick loop and run receipt this
// program keeps: a fixed schedule, counts of late ticks and schedule resets, and mean, p95 and
// max of each phase.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;

class Phase {
public:
    void add(Clock::duration elapsed)
    {
        samples_.push_back(std::chrono::duration<double, std::micro>(elapsed).count());
    }

    // `{"mean_us": ..., "p95_us": ..., "max_us": ...}`
    std::string json() const
    {
        if (samples_.empty())
            return "{\"mean_us\":0,\"p95_us\":0,\"max_us\":0}";
        std::vector<double> sorted = samples_;
        std::sort(sorted.begin(), sorted.end());
        double sum = 0;
        for (double sample : sorted)
            sum += sample;
        const double p95 = sorted[std::min(sorted.size() - 1, sorted.size() * 95 / 100)];
        return "{\"mean_us\":" + std::to_string(sum / sorted.size()) + ",\"p95_us\":"
            + std::to_string(p95) + ",\"max_us\":" + std::to_string(sorted.back()) + "}";
    }

private:
    std::vector<double> samples_;
};

class Pacer {
public:
    Pacer(Clock::duration period, bool realtime)
        : period_(period)
        , realtime_(realtime)
        , deadline_(Clock::now() + period)
    {
    }

    // Waits for the next tick's deadline. A tick that ends after its deadline is late; one more
    // than two periods late restarts the schedule instead of sprinting to catch up, as Elodin's
    // pacing does.
    void wait()
    {
        if (!realtime_)
            return;
        const auto now = Clock::now();
        if (now > deadline_) {
            ++late_;
            if (now > deadline_ + 2 * period_) {
                ++resets_;
                deadline_ = now;
            }
        } else {
            std::this_thread::sleep_until(deadline_);
        }
        deadline_ += period_;
    }

    uint64_t late() const { return late_; }
    uint64_t resets() const { return resets_; }

private:
    Clock::duration period_;
    bool realtime_;
    Clock::time_point deadline_;
    uint64_t late_ = 0;
    uint64_t resets_ = 0;
};
