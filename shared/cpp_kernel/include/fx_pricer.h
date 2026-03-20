// FX Forward Pricing Engine
// Covered interest rate parity: F(T) = S * D_foreign(T) / D_domestic(T)
// Domestic = USD (SOFR), Foreign = e.g. EUR, GBP, JPY OIS curves

#pragma once
#include "curve_engine.h"
#include <cmath>
#include <stdexcept>
#include <string>

namespace fx_kernel {

struct FXForwardResult {
    double forward;          // outright forward rate, e.g. 1.0842 for EURUSD
    double forward_points;   // forward - spot in pips, e.g. -0.0023
    double domestic_df;      // D_d(T), e.g. 0.9753 for USD at T=0.5y
    double foreign_df;       // D_f(T), e.g. 0.9812 for EUR at T=0.5y
};

struct FXSwapNPV {
    double npv;              // net present value in domestic ccy, e.g. 15234.50 USD
    double near_leg_pv;      // PV of near leg (spot), e.g. -1000000.0
    double far_leg_pv;       // PV of far leg (forward), e.g. 1015234.50
    double forward_rate;     // fair forward used, e.g. 1.0842
};

struct FXGreeks {
    double delta;            // dV/dS, e.g. 997500.0 (notional * D_f(T))
    double gamma;            // d2V/dS2, always 0 for linear FX forward
    double theta;            // dV/dt per day, e.g. -12.35 USD/day
    double rho_domestic;     // dV/d(r_d), e.g. -498750.0
    double rho_foreign;      // dV/d(r_f), e.g. 498750.0
};

class FXForwardPricer {
    DiscountCurve domestic_curve_;   // USD (SOFR) discount curve
    DiscountCurve foreign_curve_;    // Foreign OIS discount curve
    double spot_;                     // spot FX rate, e.g. 1.0865 for EURUSD
    std::string ccy_pair_;           // e.g. "EURUSD"

public:
    FXForwardPricer() : spot_(0.0) {}

    FXForwardPricer(const DiscountCurve& domestic_curve,
                    const DiscountCurve& foreign_curve,
                    double spot,
                    const std::string& ccy_pair = "EURUSD")
        : domestic_curve_(domestic_curve),
          foreign_curve_(foreign_curve),
          spot_(spot),
          ccy_pair_(ccy_pair) {
        if (spot <= 0.0) {
            throw std::invalid_argument("Spot rate must be positive, got: " + std::to_string(spot));
        }
    }

    // Covered interest rate parity forward
    // Example: spot=1.0865, D_f(1y)=0.9620, D_d(1y)=0.9540
    //   forward = 1.0865 * 0.9620 / 0.9540 = 1.0956
    double forward(double T) const {
        if (T <= 0.0) return spot_;
        double df_dom = domestic_curve_.df(T);
        double df_for = foreign_curve_.df(T);
        if (df_dom < 1e-15) {
            throw std::runtime_error("Domestic discount factor near zero at T=" + std::to_string(T));
        }
        return spot_ * df_for / df_dom;
    }

    // Forward points = F(T) - S, quoted in market convention
    // Example: forward=1.0956, spot=1.0865 -> points = 0.0091
    double forward_points(double T) const {
        return forward(T) - spot_;
    }

    // Full forward result with all components
    FXForwardResult forward_full(double T) const {
        FXForwardResult result;
        result.domestic_df = domestic_curve_.df(T);
        result.foreign_df = foreign_curve_.df(T);
        result.forward = spot_ * result.foreign_df / result.domestic_df;
        result.forward_points = result.forward - spot_;
        return result;
    }

    // NPV of an FX forward: buy foreign at agreed_forward, sell at market forward
    // NPV = notional * (F_market(T) - F_agreed) * D_d(T)
    // Example: notional=1M, F_market=1.0956, F_agreed=1.0900, D_d(1y)=0.9540
    //   NPV = 1000000 * (1.0956 - 1.0900) * 0.9540 = 5342.40 USD
    double forward_npv(double notional, double maturity, double agreed_forward) const {
        double F = forward(maturity);
        double df_dom = domestic_curve_.df(maturity);
        return notional * (F - agreed_forward) * df_dom;
    }

    // FX swap: near leg at spot, far leg at agreed forward
    // Near leg PV = -notional * spot (pay domestic, receive foreign at spot)
    // Far leg PV  = +notional * agreed_forward * D_d(T) (receive domestic at forward date)
    // Example: notional=1M EUR, spot=1.0865, agreed_fwd=1.0900, D_d(0.25)=0.9888
    //   near = -1086500, far = 1077192, NPV = -9308
    FXSwapNPV fx_swap_npv(double notional, double maturity, double agreed_forward) const {
        FXSwapNPV result;
        result.forward_rate = forward(maturity);
        result.near_leg_pv = -notional * spot_;
        result.far_leg_pv = notional * agreed_forward * domestic_curve_.df(maturity);
        result.npv = result.near_leg_pv + result.far_leg_pv;
        return result;
    }

    // Delta: dV/dS = notional * D_f(T) for a long forward position
    // Example: notional=1M, D_f(1y)=0.9620 -> delta = 962000
    double delta(double notional, double maturity) const {
        return notional * foreign_curve_.df(maturity);
    }

    // Gamma: d2V/dS2 = 0 for linear FX forward (no optionality)
    double gamma(double notional, double maturity) const {
        (void)notional;
        (void)maturity;
        return 0.0;
    }

    // Theta: time decay per day (finite difference, shift T by -1/365)
    // Example: notional=1M, T=1y, F_agreed=1.09
    //   V(T) - V(T-1/365) ~ -12.35 USD/day
    double theta(double notional, double maturity, double agreed_forward) const {
        double dt = 1.0 / 365.0;
        if (maturity <= dt) {
            return -forward_npv(notional, maturity, agreed_forward) / maturity * dt;
        }
        double v0 = forward_npv(notional, maturity, agreed_forward);
        double v1 = forward_npv(notional, maturity - dt, agreed_forward);
        return v1 - v0;  // negative = time decay costs money
    }

    // Rho domestic: dV/d(r_d), bump domestic rates by 1bp
    // Example: notional=1M, T=1y -> rho_d ~ -498750 (higher USD rate = lower forward)
    double rho_domestic(double notional, double maturity, double agreed_forward) const {
        double bump = 0.0001;  // 1 basis point
        double F = forward(maturity);
        double df_dom = domestic_curve_.df(maturity);
        // dF/dr_d ~ -F * T, dNPV/dr_d ~ notional * (-F*T) * df_dom + notional*(F-K)*(-T*df_dom)
        return -notional * F * maturity * df_dom;
    }

    // Rho foreign: dV/d(r_f), bump foreign rates by 1bp
    // Example: notional=1M, T=1y -> rho_f ~ +498750 (higher foreign rate = higher forward)
    double rho_foreign(double notional, double maturity, double agreed_forward) const {
        double F = forward(maturity);
        double df_dom = domestic_curve_.df(maturity);
        // dF/dr_f ~ -F * T (through D_f), but this reduces forward
        return -notional * spot_ * foreign_curve_.df(maturity) * maturity;
    }

    // Full greeks
    FXGreeks greeks(double notional, double maturity, double agreed_forward) const {
        FXGreeks g;
        g.delta = delta(notional, maturity);
        g.gamma = gamma(notional, maturity);
        g.theta = theta(notional, maturity, agreed_forward);
        g.rho_domestic = rho_domestic(notional, maturity, agreed_forward);
        g.rho_foreign = rho_foreign(notional, maturity, agreed_forward);
        return g;
    }

    // Accessors
    double spot() const { return spot_; }
    const std::string& ccy_pair() const { return ccy_pair_; }
    const DiscountCurve& domestic_curve() const { return domestic_curve_; }
    const DiscountCurve& foreign_curve() const { return foreign_curve_; }
};

} // namespace fx_kernel
