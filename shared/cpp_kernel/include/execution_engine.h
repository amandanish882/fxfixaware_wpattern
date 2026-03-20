// Execution Engine for FX market-making
// Almgren-Chriss optimal execution with adaptive scheduling
// Costs in pips (e.g., 0.5 pips = 0.00005 for EURUSD)

#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <tuple>
#include <string>

namespace fx_kernel {

// Single execution slice: (time_fraction, quantity, cumulative_quantity)
// Example: (0.25, 300000, 300000) means at 25% of horizon, trade 300k, total so far 300k
struct ExecutionSlice {
    double time_frac;     // fraction of horizon elapsed, e.g. 0.25
    int quantity;          // lots to trade in this slice, e.g. 300000
    int cumulative;        // running total traded so far, e.g. 300000

    ExecutionSlice(double t, int q, int c) : time_frac(t), quantity(q), cumulative(c) {}
};

// Almgren-Chriss execution cost breakdown
// All costs in pips (1 pip = 0.0001 for most pairs, 0.01 for JPY pairs)
struct ExecutionCost {
    double permanent_cost;   // permanent market impact in pips, e.g. 0.35
    double temporary_cost;   // temporary market impact in pips, e.g. 0.82
    double total_cost;       // permanent + temporary, e.g. 1.17
    double risk_cost;        // variance penalty in pips, e.g. 0.23
    double total_with_risk;  // total + risk, e.g. 1.40
    int n_slices;            // number of execution slices used, e.g. 10
};

// Adaptive execution schedule using sinh kernel
// Concentrates trading at start (aggressive) or spreads evenly (passive)
// kappa > 1: front-loaded (e.g., kappa=2.0 -> 60% done by 30% of horizon)
// kappa < 1: back-loaded
// kappa = 1: roughly linear
inline std::vector<ExecutionSlice> adaptive_schedule(
        int total_qty,
        double kappa,
        int n_slices) {

    if (n_slices <= 0) {
        throw std::invalid_argument("n_slices must be positive, got: " + std::to_string(n_slices));
    }
    if (total_qty <= 0) {
        throw std::invalid_argument("total_qty must be positive, got: " + std::to_string(total_qty));
    }
    if (kappa <= 0.0) {
        throw std::invalid_argument("kappa must be positive, got: " + std::to_string(kappa));
    }

    std::vector<ExecutionSlice> schedule;
    schedule.reserve(n_slices);

    // Compute raw weights using sinh kernel
    // w_j = sinh(kappa * (1 - t_j)) / sinh(kappa) where t_j = j / n_slices
    // Example: kappa=2, n=5 -> weights ~ [0.434, 0.272, 0.166, 0.091, 0.037]
    std::vector<double> weights(n_slices);
    double sinh_kappa = std::sinh(kappa);
    double total_weight = 0.0;

    for (int j = 0; j < n_slices; ++j) {
        double t_j = static_cast<double>(j + 1) / n_slices;
        weights[j] = std::sinh(kappa * (1.0 - t_j + 1.0 / n_slices)) / sinh_kappa;
        total_weight += weights[j];
    }

    // Normalize and allocate quantities
    int allocated = 0;
    for (int j = 0; j < n_slices; ++j) {
        double t_j = static_cast<double>(j + 1) / n_slices;
        int qty;
        if (j == n_slices - 1) {
            qty = total_qty - allocated;  // ensure exact total
        } else {
            qty = static_cast<int>(std::round(total_qty * weights[j] / total_weight));
        }
        qty = std::max(qty, 0);
        allocated += qty;
        schedule.emplace_back(t_j, qty, allocated);
    }

    return schedule;
}

class AlmgrenChrissEngine {
    // Market microstructure parameters for FX
    double eta_;     // temporary impact coefficient, e.g. 0.1 for liquid G10 pairs
    double gamma_;   // permanent impact coefficient, e.g. 0.05
    double lambda_;  // risk aversion parameter, e.g. 1e-6

public:
    AlmgrenChrissEngine() : eta_(0.1), gamma_(0.05), lambda_(1e-6) {}

    AlmgrenChrissEngine(double eta, double gamma, double lambda)
        : eta_(eta), gamma_(gamma), lambda_(lambda) {
        if (eta < 0.0) throw std::invalid_argument("eta must be non-negative");
        if (gamma < 0.0) throw std::invalid_argument("gamma must be non-negative");
        if (lambda < 0.0) throw std::invalid_argument("lambda must be non-negative");
    }

    // Compute optimal execution cost for an FX order
    // n_contracts: notional in base currency units, e.g. 5000000 (5M EUR)
    // daily_volume: average daily volume, e.g. 500000000 (500M EUR)
    // daily_vol_pips: daily volatility in pips, e.g. 80.0 (80 pips/day for EURUSD)
    // horizon_minutes: execution horizon, e.g. 30.0 (half hour)
    //
    // Example: 5M EUR in 30min, ADV=500M, vol=80 pips/day
    //   participation = 5M / (500M * 30/480) = 0.16 (16%)
    //   permanent = 0.05 * 0.16 * 80 = 0.64 pips
    //   temporary = 0.1 * 0.16 * 80 / sqrt(30/480) = 5.12 pips (scaled by urgency)
    //   total ~ 5.76 pips
    ExecutionCost total_cost(int n_contracts,
                             int daily_volume,
                             double daily_vol_pips,
                             double horizon_minutes) const {

        if (daily_volume <= 0) {
            throw std::invalid_argument("daily_volume must be positive");
        }
        if (daily_vol_pips <= 0.0) {
            throw std::invalid_argument("daily_vol_pips must be positive");
        }
        if (horizon_minutes <= 0.0) {
            throw std::invalid_argument("horizon_minutes must be positive");
        }

        // Trading day = 480 minutes (8 hours) for major FX pairs
        const double trading_day_minutes = 480.0;
        double horizon_frac = horizon_minutes / trading_day_minutes;

        // Volume available in horizon window
        double volume_in_horizon = static_cast<double>(daily_volume) * horizon_frac;
        double participation = static_cast<double>(n_contracts) / volume_in_horizon;

        // Permanent impact: proportional to participation rate
        // Scales linearly: trading 10% of volume -> 10% * gamma * vol
        double permanent = gamma_ * participation * daily_vol_pips;

        // Temporary impact: inversely proportional to sqrt(horizon)
        // Faster execution = more impact (urgency premium)
        double urgency = 1.0 / std::sqrt(horizon_frac);
        double temporary = eta_ * participation * daily_vol_pips * urgency;

        // Variance / risk cost: holding risk during execution
        // Proportional to vol * sqrt(horizon) * notional fraction
        double notional_frac = static_cast<double>(n_contracts) / static_cast<double>(daily_volume);
        double risk = lambda_ * daily_vol_pips * daily_vol_pips * horizon_frac * notional_frac * 1e6;

        // Optimal number of slices: more slices for longer horizons
        int n_slices = std::max(2, static_cast<int>(std::ceil(horizon_minutes / 3.0)));
        n_slices = std::min(n_slices, 50);  // cap at 50 slices

        ExecutionCost result;
        result.permanent_cost = permanent;
        result.temporary_cost = temporary;
        result.total_cost = permanent + temporary;
        result.risk_cost = risk;
        result.total_with_risk = permanent + temporary + risk;
        result.n_slices = n_slices;

        return result;
    }

    // Compute full optimal trajectory
    // Returns schedule + cost in one call
    std::pair<std::vector<ExecutionSlice>, ExecutionCost> optimal_trajectory(
            int n_contracts,
            int daily_volume,
            double daily_vol_pips,
            double horizon_minutes) const {

        ExecutionCost cost = total_cost(n_contracts, daily_volume, daily_vol_pips, horizon_minutes);

        // Optimal kappa from Almgren-Chriss:
        // kappa = sqrt(lambda * sigma^2 / eta)
        // Higher risk aversion or vol -> more aggressive (front-loaded)
        double sigma_per_min = daily_vol_pips / std::sqrt(480.0);
        double kappa_raw = std::sqrt(lambda_ * sigma_per_min * sigma_per_min / (eta_ + 1e-15));
        double kappa = std::max(0.1, std::min(kappa_raw * horizon_minutes, 5.0));

        auto schedule = adaptive_schedule(n_contracts, kappa, cost.n_slices);
        return {schedule, cost};
    }

    // Accessors
    double eta() const { return eta_; }
    double gamma() const { return gamma_; }
    double lambda() const { return lambda_; }
};

} // namespace fx_kernel
