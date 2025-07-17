#!/usr/bin/env python
"""Utility functions for aggregating per-step performance statistics.

The performance profiling utilities in *job_main* scripts output a sequence of
summary dictionaries, e.g.::

    {
        "steps/s": 1912.2,
        "env_ms": 0.16,
        "infer_ms": 0.118,
        "over_ms": 0.245,
    }

`compute_field_statistics` consolidates such a list into means and standard
deviations for each numeric field.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple


def _mean(values: Iterable[float]) -> float:
    """Return the arithmetic mean of *values* (assumes non-empty iterable)."""
    total = 0.0
    count = 0
    for val in values:
        total += val
        count += 1
    return total / count


def _std(values: Iterable[float], mean: float | None = None) -> float:
    """Return the population standard deviation of *values*.

    If *mean* is provided it will be reused, otherwise it is computed.
    Uses the population formula (divide by *N*, not *N − 1*).
    """
    if mean is None:
        values = list(values)
        mean = _mean(values)
    variance_acc = 0.0
    count = 0
    for val in values:
        variance_acc += (val - mean) ** 2
        count += 1
    return math.sqrt(variance_acc / count) if count else float("nan")


def compute_field_statistics(
    summaries: List[Dict[str, float]] | Iterable[Dict[str, float]]
) -> Dict[str, Tuple[float, float]]:
    """Compute mean and standard deviation for each numeric key.

    Parameters
    ----------
    summaries:
        An iterable of dictionaries where each dictionary maps metric names to
        numeric values (e.g. output from the job-scheduling performance
        summaries).

    Returns
    -------
    Dict[str, Tuple[float, float]]
        A mapping from metric name to a ``(mean, std)`` tuple.
    """
    # Convert to list to allow multiple passes over the data.
    summaries = list(summaries)
    if not summaries:
        return {}

    # Collect values per key.
    buckets: Dict[str, List[float]] = {}
    for summary in summaries:
        for key, value in summary.items():
            buckets.setdefault(key, []).append(float(value))

    # Compute statistics.
    stats: Dict[str, Tuple[float, float]] = {}
    for key, vals in buckets.items():
        mu = _mean(vals)
        sigma = _std(vals, mu)
        stats[key] = (mu, sigma)

    return stats


__all__ = ["compute_field_statistics"]
