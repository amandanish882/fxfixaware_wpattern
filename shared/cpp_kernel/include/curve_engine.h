// Discount curve engine - same as rates project but works for any OIS curve
// Used for both USD (SOFR) and foreign currency OIS curves

#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace fx_kernel {

enum class InstrumentType { Deposit = 0, Swap = 1 };
enum class DayCount { ACT_360 = 0, ACT_365 = 1, THIRTY_360 = 2, ACT_ACT = 3 };

struct CurveInstrument {
    InstrumentType type;
    double maturity_years;
    double rate;
    DayCount day_count;
    int payment_frequency;  // per year

    CurveInstrument(int t, double m, double r, int dc, int freq)
        : type(static_cast<InstrumentType>(t)), maturity_years(m), rate(r),
          day_count(static_cast<DayCount>(dc)), payment_frequency(freq) {}
};

class DiscountCurve {
    std::vector<double> times_;
    std::vector<double> log_dfs_;
    std::string valuation_date_;

    double day_count_fraction(double years, DayCount dc) const {
        // Simplified: use years directly (already in year fractions)
        switch (dc) {
            case DayCount::ACT_360: return years * 365.0 / 360.0;
            case DayCount::ACT_365: return years;
            case DayCount::THIRTY_360: return years;
            case DayCount::ACT_ACT: return years;
            default: return years;
        }
    }

public:
    DiscountCurve() = default;

    DiscountCurve(const std::vector<double>& times,
                  const std::vector<double>& dfs,
                  const std::string& valuation_date)
        : times_(times), valuation_date_(valuation_date) {
        log_dfs_.resize(dfs.size());
        for (size_t i = 0; i < dfs.size(); ++i) {
            log_dfs_[i] = std::log(dfs[i]);
        }
    }

    double df(double t) const {
        if (t <= 0.0) return 1.0;
        if (t <= times_.front()) {
            return std::exp(log_dfs_.front() * t / times_.front());
        }
        if (t >= times_.back()) {
            return std::exp(log_dfs_.back() * t / times_.back());
        }
        // Log-linear interpolation
        auto it = std::lower_bound(times_.begin(), times_.end(), t);
        size_t i = std::distance(times_.begin(), it);
        if (i == 0) i = 1;
        double t0 = times_[i-1], t1 = times_[i];
        double w = (t - t0) / (t1 - t0);
        return std::exp(log_dfs_[i-1] * (1.0 - w) + log_dfs_[i] * w);
    }

    double zero_rate(double t) const {
        if (t <= 1e-10) return -log_dfs_.front() / times_.front();
        return -std::log(df(t)) / t;
    }

    double forward_rate(double t1, double t2) const {
        if (t2 <= t1) return zero_rate(t1);
        return -(std::log(df(t2)) - std::log(df(t1))) / (t2 - t1);
    }

    double instantaneous_forward(double t) const {
        double dt = 0.001;
        return forward_rate(std::max(0.0, t - dt/2), t + dt/2);
    }

    double par_rate(double maturity, int freq = 2) const {
        double dt = 1.0 / freq;
        double annuity = 0.0;
        for (double t = dt; t <= maturity + 1e-10; t += dt) {
            annuity += dt * df(std::min(t, maturity));
        }
        if (annuity < 1e-15) return 0.0;
        return (1.0 - df(maturity)) / annuity;
    }

    const std::vector<double>& times() const { return times_; }
    const std::string& valuation_date() const { return valuation_date_; }
    size_t size() const { return times_.size(); }
};

// Sequential bootstrap: deposits then swaps in maturity order
inline DiscountCurve bootstrap_curve(std::vector<CurveInstrument> instruments,
                                      const std::string& valuation_date = "2026-03-05") {
    std::sort(instruments.begin(), instruments.end(),
              [](const CurveInstrument& a, const CurveInstrument& b) {
                  return a.maturity_years < b.maturity_years;
              });

    // Anchor at T=0, D=1.0 (required for interpolation)
    std::vector<double> times = {0.0};
    std::vector<double> dfs = {1.0};

    for (const auto& inst : instruments) {
        double t = inst.maturity_years;
        double alpha = t;  // simplified day count fraction

        if (inst.type == InstrumentType::Deposit) {
            double d = 1.0 / (1.0 + inst.rate * alpha);
            times.push_back(t);
            dfs.push_back(d);
        } else {
            // Swap: D(t_n) = (1 - S * sum(alpha_j * D(t_j))) / (1 + S * alpha_n)
            int freq = std::max(inst.payment_frequency, 1);
            double dt = 1.0 / freq;
            double pv_fixed = 0.0;

            // Build temporary curve from nodes bootstrapped so far
            DiscountCurve temp_curve(times, dfs, valuation_date);

            for (double tj = dt; tj < t - 1e-10; tj += dt) {
                pv_fixed += dt * temp_curve.df(tj);
            }
            double d_t = (1.0 - inst.rate * pv_fixed) / (1.0 + inst.rate * dt);
            if (d_t <= 0.0 || std::isnan(d_t)) {
                // Fallback: use zero-coupon formula
                d_t = std::exp(-inst.rate * t);
            }
            times.push_back(t);
            dfs.push_back(d_t);
        }
    }

    return DiscountCurve(times, dfs, valuation_date);
}

} // namespace fx_kernel
