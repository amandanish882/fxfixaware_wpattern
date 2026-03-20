// pybind11 bindings for fx_pricing_kernel
// Exposes: CurveInstrument, DiscountCurve, bootstrap_curve,
//          FXForwardPricer, adaptive_schedule, AlmgrenChrissEngine

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include "../include/curve_engine.h"
#include "../include/fx_pricer.h"
#include "../include/execution_engine.h"

namespace py = pybind11;
using namespace fx_kernel;

// SwapSpec: lightweight struct for convexity benchmarking
// E.g., SwapSpec{1e6, 5.0, 0.04, 2} -> 5Y swap, 1M notional, 4% fixed, semi-annual
struct SwapSpec {
    double notional;
    double maturity;
    double fixed_rate;
    int frequency;
};

PYBIND11_MODULE(fx_pricing_kernel, m) {
    m.doc() = "FX pricing kernel - discount curves, forward pricing, and execution engine";

    // ──────────────────────────────────────────────
    // Enums
    // ──────────────────────────────────────────────
    py::enum_<InstrumentType>(m, "InstrumentType")
        .value("Deposit", InstrumentType::Deposit)
        .value("Swap", InstrumentType::Swap)
        .export_values();

    py::enum_<DayCount>(m, "DayCount")
        .value("ACT_360", DayCount::ACT_360)
        .value("ACT_365", DayCount::ACT_365)
        .value("THIRTY_360", DayCount::THIRTY_360)
        .value("ACT_ACT", DayCount::ACT_ACT)
        .export_values();

    // ──────────────────────────────────────────────
    // CurveInstrument
    // ──────────────────────────────────────────────
    py::class_<CurveInstrument>(m, "CurveInstrument")
        .def(py::init<int, double, double, int, int>(),
             py::arg("type"), py::arg("maturity_years"), py::arg("rate"),
             py::arg("day_count"), py::arg("payment_frequency"),
             "Create instrument: type(0=Deposit,1=Swap), maturity(years), rate, "
             "day_count(0=ACT360,1=ACT365,2=30/360,3=ACTACT), freq(payments/year)")
        .def_readwrite("type", &CurveInstrument::type)
        .def_readwrite("maturity_years", &CurveInstrument::maturity_years)
        .def_readwrite("rate", &CurveInstrument::rate)
        .def_readwrite("day_count", &CurveInstrument::day_count)
        .def_readwrite("payment_frequency", &CurveInstrument::payment_frequency)
        .def("__repr__", [](const CurveInstrument& ci) {
            return "<CurveInstrument "
                   + std::string(ci.type == InstrumentType::Deposit ? "Deposit" : "Swap")
                   + " mat=" + std::to_string(ci.maturity_years)
                   + "y rate=" + std::to_string(ci.rate * 100.0) + "%>";
        });

    // ──────────────────────────────────────────────
    // DiscountCurve
    // ──────────────────────────────────────────────
    py::class_<DiscountCurve>(m, "DiscountCurve")
        .def(py::init<>())
        .def(py::init<const std::vector<double>&, const std::vector<double>&, const std::string&>(),
             py::arg("times"), py::arg("discount_factors"), py::arg("valuation_date"))
        .def("df", &DiscountCurve::df, py::arg("t"),
             "Discount factor at time t (years). E.g., df(1.0) -> 0.9540")
        .def("zero_rate", &DiscountCurve::zero_rate, py::arg("t"),
             "Continuously compounded zero rate. E.g., zero_rate(1.0) -> 0.0471")
        .def("forward_rate", &DiscountCurve::forward_rate, py::arg("t1"), py::arg("t2"),
             "Forward rate between t1 and t2. E.g., forward_rate(1.0, 2.0) -> 0.0485")
        .def("instantaneous_forward", &DiscountCurve::instantaneous_forward, py::arg("t"),
             "Instantaneous forward rate at t")
        .def("par_rate", &DiscountCurve::par_rate, py::arg("maturity"), py::arg("freq") = 2,
             "Par swap rate. E.g., par_rate(5.0) -> 0.0462")
        .def("times", &DiscountCurve::times)
        .def("valuation_date", &DiscountCurve::valuation_date)
        .def("size", &DiscountCurve::size)
        .def("__repr__", [](const DiscountCurve& c) {
            return "<DiscountCurve " + std::to_string(c.size()) + " pillars, val="
                   + c.valuation_date() + ">";
        });

    // ──────────────────────────────────────────────
    // bootstrap_curve
    // ──────────────────────────────────────────────
    m.def("bootstrap_curve", &bootstrap_curve,
          py::arg("instruments"), py::arg("valuation_date") = "2026-03-05",
          "Bootstrap a discount curve from deposit/swap instruments");

    // ──────────────────────────────────────────────
    // FXForwardResult, FXSwapNPV, FXGreeks
    // ──────────────────────────────────────────────
    py::class_<FXForwardResult>(m, "FXForwardResult")
        .def_readonly("forward", &FXForwardResult::forward)
        .def_readonly("forward_points", &FXForwardResult::forward_points)
        .def_readonly("domestic_df", &FXForwardResult::domestic_df)
        .def_readonly("foreign_df", &FXForwardResult::foreign_df)
        .def("__repr__", [](const FXForwardResult& r) {
            return "<FXForwardResult fwd=" + std::to_string(r.forward)
                   + " pts=" + std::to_string(r.forward_points) + ">";
        });

    py::class_<FXSwapNPV>(m, "FXSwapNPV")
        .def_readonly("npv", &FXSwapNPV::npv)
        .def_readonly("near_leg_pv", &FXSwapNPV::near_leg_pv)
        .def_readonly("far_leg_pv", &FXSwapNPV::far_leg_pv)
        .def_readonly("forward_rate", &FXSwapNPV::forward_rate)
        .def("__repr__", [](const FXSwapNPV& s) {
            return "<FXSwapNPV npv=" + std::to_string(s.npv)
                   + " fwd=" + std::to_string(s.forward_rate) + ">";
        });

    py::class_<FXGreeks>(m, "FXGreeks")
        .def_readonly("delta", &FXGreeks::delta)
        .def_readonly("gamma", &FXGreeks::gamma)
        .def_readonly("theta", &FXGreeks::theta)
        .def_readonly("rho_domestic", &FXGreeks::rho_domestic)
        .def_readonly("rho_foreign", &FXGreeks::rho_foreign)
        .def("__repr__", [](const FXGreeks& g) {
            return "<FXGreeks delta=" + std::to_string(g.delta)
                   + " gamma=" + std::to_string(g.gamma)
                   + " theta=" + std::to_string(g.theta) + ">";
        });

    // ──────────────────────────────────────────────
    // FXForwardPricer
    // ──────────────────────────────────────────────
    py::class_<FXForwardPricer>(m, "FXForwardPricer")
        .def(py::init<>())
        .def(py::init<const DiscountCurve&, const DiscountCurve&, double, const std::string&>(),
             py::arg("domestic_curve"), py::arg("foreign_curve"),
             py::arg("spot"), py::arg("ccy_pair") = "EURUSD")
        .def("forward", &FXForwardPricer::forward, py::arg("T"),
             "CIP forward: S * D_f(T) / D_d(T). E.g., forward(1.0) -> 1.0956")
        .def("forward_points", &FXForwardPricer::forward_points, py::arg("T"),
             "Forward points = F(T) - S. E.g., forward_points(1.0) -> 0.0091")
        .def("forward_full", &FXForwardPricer::forward_full, py::arg("T"),
             "Full forward result with DFs and points")
        .def("forward_npv", &FXForwardPricer::forward_npv,
             py::arg("notional"), py::arg("maturity"), py::arg("agreed_forward"),
             "NPV of FX forward position in domestic ccy")
        .def("fx_swap_npv", &FXForwardPricer::fx_swap_npv,
             py::arg("notional"), py::arg("maturity"), py::arg("agreed_forward"),
             "NPV of FX swap (spot + forward)")
        .def("delta", &FXForwardPricer::delta,
             py::arg("notional"), py::arg("maturity"),
             "dV/dS = notional * D_f(T). E.g., delta(1e6, 1.0) -> 962000")
        .def("gamma", &FXForwardPricer::gamma,
             py::arg("notional"), py::arg("maturity"),
             "d2V/dS2 = 0 for linear forward")
        .def("theta", &FXForwardPricer::theta,
             py::arg("notional"), py::arg("maturity"), py::arg("agreed_forward"),
             "Time decay per day in domestic ccy")
        .def("rho_domestic", &FXForwardPricer::rho_domestic,
             py::arg("notional"), py::arg("maturity"), py::arg("agreed_forward"),
             "Sensitivity to domestic rate")
        .def("rho_foreign", &FXForwardPricer::rho_foreign,
             py::arg("notional"), py::arg("maturity"), py::arg("agreed_forward"),
             "Sensitivity to foreign rate")
        .def("greeks", &FXForwardPricer::greeks,
             py::arg("notional"), py::arg("maturity"), py::arg("agreed_forward"),
             "All greeks in one call")
        .def("spot", &FXForwardPricer::spot)
        .def("ccy_pair", &FXForwardPricer::ccy_pair)
        .def("__repr__", [](const FXForwardPricer& p) {
            return "<FXForwardPricer " + p.ccy_pair() + " spot=" + std::to_string(p.spot()) + ">";
        });

    // ──────────────────────────────────────────────
    // ExecutionSlice, ExecutionCost
    // ──────────────────────────────────────────────
    py::class_<ExecutionSlice>(m, "ExecutionSlice")
        .def(py::init<double, int, int>(),
             py::arg("time_frac"), py::arg("quantity"), py::arg("cumulative"))
        .def_readwrite("time_frac", &ExecutionSlice::time_frac)
        .def_readwrite("quantity", &ExecutionSlice::quantity)
        .def_readwrite("cumulative", &ExecutionSlice::cumulative)
        .def("__repr__", [](const ExecutionSlice& s) {
            return "<Slice t=" + std::to_string(s.time_frac)
                   + " qty=" + std::to_string(s.quantity)
                   + " cum=" + std::to_string(s.cumulative) + ">";
        });

    py::class_<ExecutionCost>(m, "ExecutionCost")
        .def_readonly("permanent_cost", &ExecutionCost::permanent_cost)
        .def_readonly("temporary_cost", &ExecutionCost::temporary_cost)
        .def_readonly("total_cost", &ExecutionCost::total_cost)
        .def_readonly("risk_cost", &ExecutionCost::risk_cost)
        .def_readonly("total_with_risk", &ExecutionCost::total_with_risk)
        .def_readonly("n_slices", &ExecutionCost::n_slices)
        .def("__repr__", [](const ExecutionCost& c) {
            return "<ExecutionCost perm=" + std::to_string(c.permanent_cost)
                   + " temp=" + std::to_string(c.temporary_cost)
                   + " total=" + std::to_string(c.total_cost) + " pips>";
        });

    // ──────────────────────────────────────────────
    // adaptive_schedule
    // ──────────────────────────────────────────────
    m.def("adaptive_schedule", &adaptive_schedule,
          py::arg("total_qty"), py::arg("kappa"), py::arg("n_slices"),
          "Adaptive execution schedule using sinh kernel.\n"
          "kappa>1: front-loaded, kappa<1: back-loaded.\n"
          "Returns list of ExecutionSlice(time_frac, quantity, cumulative).");

    // ──────────────────────────────────────────────
    // AlmgrenChrissEngine
    // ──────────────────────────────────────────────
    py::class_<AlmgrenChrissEngine>(m, "AlmgrenChrissEngine")
        .def(py::init<>(), "Default: eta=0.1, gamma=0.05, lambda=1e-6")
        .def(py::init<double, double, double>(),
             py::arg("eta"), py::arg("gamma"), py::arg("lambda"),
             "Custom parameters: eta(temp impact), gamma(perm impact), lambda(risk aversion)")
        .def("total_cost", &AlmgrenChrissEngine::total_cost,
             py::arg("n_contracts"), py::arg("daily_volume"),
             py::arg("daily_vol_pips"), py::arg("horizon_minutes"),
             "Compute execution cost in pips.\n"
             "E.g., total_cost(5000000, 500000000, 80.0, 30.0) -> ~5.76 pips total")
        .def("optimal_trajectory", &AlmgrenChrissEngine::optimal_trajectory,
             py::arg("n_contracts"), py::arg("daily_volume"),
             py::arg("daily_vol_pips"), py::arg("horizon_minutes"),
             "Returns (schedule, cost) tuple with optimal execution plan")
        .def("eta", &AlmgrenChrissEngine::eta)
        .def("gamma", &AlmgrenChrissEngine::gamma)
        .def("lambda_", &AlmgrenChrissEngine::lambda,
             "Risk aversion parameter (named lambda_ to avoid Python keyword)")
        .def("__repr__", [](const AlmgrenChrissEngine& e) {
            return "<AlmgrenChrissEngine eta=" + std::to_string(e.eta())
                   + " gamma=" + std::to_string(e.gamma())
                   + " lambda=" + std::to_string(e.lambda()) + ">";
        });

    // ──────────────────────────────────────────────
    // SwapSpec for convexity benchmark
    // ──────────────────────────────────────────────
    py::class_<SwapSpec>(m, "SwapSpec")
        .def(py::init([](double notional, double maturity, double fixed_rate, int freq) {
            return SwapSpec{notional, maturity, fixed_rate, freq};
        }),
        py::arg("notional") = 1000000.0,
        py::arg("maturity") = 5.0,
        py::arg("fixed_rate") = 0.04,
        py::arg("frequency") = 2,
        "Swap specification for convexity benchmarking.\n"
        "E.g., SwapSpec(1e6, 5.0, 0.04, 2) -> 5Y swap, 1M notional, 4% fixed, semi-annual")
        .def_readwrite("notional", &SwapSpec::notional)
        .def_readwrite("maturity", &SwapSpec::maturity)
        .def_readwrite("fixed_rate", &SwapSpec::fixed_rate)
        .def_readwrite("frequency", &SwapSpec::frequency)
        .def("__repr__", [](const SwapSpec& s) {
            return "<SwapSpec notional=" + std::to_string(s.notional)
                   + " mat=" + std::to_string(s.maturity)
                   + "y rate=" + std::to_string(s.fixed_rate * 100.0) + "%>";
        });

    // Convexity adjustment for FX forward from interest rate differential
    m.def("convexity_adjustment", [](const DiscountCurve& dom_curve,
                                      const DiscountCurve& for_curve,
                                      double maturity,
                                      double vol) {
        // Convexity adjustment ~ 0.5 * sigma^2 * T^2 * (r_f - r_d)
        // Example: vol=0.08, T=5, r_f=0.035, r_d=0.045
        //   adj ~ 0.5 * 0.0064 * 25 * (-0.01) = -0.0008
        double r_d = dom_curve.zero_rate(maturity);
        double r_f = for_curve.zero_rate(maturity);
        double rate_diff = r_f - r_d;
        return 0.5 * vol * vol * maturity * maturity * rate_diff;
    }, py::arg("domestic_curve"), py::arg("foreign_curve"),
       py::arg("maturity"), py::arg("vol"),
       "Convexity adjustment for long-dated FX forwards");
}
