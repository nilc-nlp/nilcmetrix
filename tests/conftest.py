# -*- coding: utf-8 -*-
"""Shared helpers and constants for the regression test harness."""

import json
import math
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent
INPUTS_DIR = TESTS_DIR / "inputs"
GOLDENS_DIR = TESTS_DIR / "goldens"

RTOL = 1e-6
ATOL = 1e-9


def read_text(path):
    raw = Path(path).read_text(encoding="utf-8")
    return raw.encode("utf-8", "surrogateescape").decode("utf-8")


def compute_metrics(text_str):
    import text_metrics

    t = text_metrics.Text(text_str)
    flat = text_metrics.no_palavras_metrics.values_for_text(t).as_flat_dict()
    return dict(flat)


def load_golden(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_golden(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, sort_keys=True, indent=2, ensure_ascii=False)
        f.write("\n")


def _values_equal(a, b):
    if a is None and b is None:
        return True
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return math.isclose(a, b, rel_tol=RTOL, abs_tol=ATOL)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=RTOL, abs_tol=ATOL)
    return a == b


def compare(actual, expected):
    """Return a list of human-readable diff lines. Empty list means match."""
    diffs = []

    actual_keys = set(actual.keys())
    expected_keys = set(expected.keys())

    missing = expected_keys - actual_keys
    extra = actual_keys - expected_keys
    for k in sorted(missing):
        diffs.append("missing key (in golden, not in actual): {!r}".format(k))
    for k in sorted(extra):
        diffs.append("extra key (in actual, not in golden): {!r}".format(k))

    for k in sorted(actual_keys & expected_keys):
        a, e = actual[k], expected[k]
        if not _values_equal(a, e):
            delta = None
            if isinstance(a, (int, float)) and isinstance(e, (int, float)):
                try:
                    delta = float(a) - float(e)
                except (TypeError, ValueError):
                    delta = None
            if delta is None:
                diffs.append("{}: golden={!r} actual={!r}".format(k, e, a))
            else:
                diffs.append(
                    "{}: golden={!r} actual={!r} delta={:+.3e}".format(k, e, a, delta)
                )

    return diffs
