"""
Module A: FX Curves -- Data loading, OIS bootstrapping, and FX forward pricing.
"""

from module_a_curves.data_loader import FXDataLoader
from module_a_curves.curve_bootstrapper import (
    CurveBootstrapper,
    CurveInstrument,
    DiscountCurve,
)
from module_a_curves.interpolation import LogLinearInterpolator, MonotoneConvexInterpolator
from module_a_curves.fx_forward_curve import FXForwardCurve, CrossCurrencyBasis

__all__ = [
    "FXDataLoader",
    "CurveBootstrapper",
    "CurveInstrument",
    "DiscountCurve",
    "LogLinearInterpolator",
    "MonotoneConvexInterpolator",
    "FXForwardCurve",
    "CrossCurrencyBasis",
]
