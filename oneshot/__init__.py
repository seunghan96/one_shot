"""OneShot: weighting a pool of anomaly detectors from a single confirmed event."""

from .core import (
    combine,
    observed_window,
    pick_threshold,
    soft_weights,
)
from .data import events, load_series
from .metrics import vus_pr

__all__ = [
    'combine',
    'events',
    'load_series',
    'observed_window',
    'pick_threshold',
    'soft_weights',
    'vus_pr',
]
__version__ = '0.1.0'
