"""Calibrated Completion Verifiers (CCV): calibrated, prefix-conditional verification
and budgeted sequential search for coding and terminal agents."""
from .calibration import PlattCalibrator, expected_calibration_error
from .crc import crc_threshold, false_finish_rate
from .controller import SearchConfig, SearchResult, adaptive_search, best_of_k

__all__ = [
    "PlattCalibrator", "expected_calibration_error", "crc_threshold", "false_finish_rate",
    "SearchConfig", "SearchResult", "adaptive_search", "best_of_k",
]
__version__ = "0.1.0"
