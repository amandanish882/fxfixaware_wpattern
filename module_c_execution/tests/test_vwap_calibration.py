"""Tests for VWAP volume profile calibration."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pytest

from module_c_execution.vwap_calibration import calibrate_vwap_profile


def test_calibrate_vwap_profile_sums_to_one():
    p = calibrate_vwap_profile()
    assert isinstance(p, np.ndarray)
    assert p.shape == (96,)
    assert abs(p.sum() - 1.0) < 1e-6


def test_calibrate_vwap_profile_peaks_in_overlap():
    """The London/NY overlap window (buckets 52-68) should hold the bulk of volume."""
    p = calibrate_vwap_profile()
    overlap_share = p[52:68].sum()
    asia_share = p[0:24].sum()  # 00:00-06:00 UTC
    # Overlap should be at least 2x the Asia tail
    assert overlap_share > asia_share * 2
