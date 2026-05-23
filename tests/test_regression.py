# -*- coding: utf-8 -*-
"""Regression tests: each input text is compared against a stored golden."""

import pytest

from tests.conftest import (
    GOLDENS_DIR,
    INPUTS_DIR,
    compare,
    compute_metrics,
    load_golden,
    read_text,
)


_INPUTS = sorted(INPUTS_DIR.glob("*.txt"))


@pytest.mark.parametrize("input_path", _INPUTS, ids=lambda p: p.stem)
def test_metrics_match_golden(input_path):
    golden_path = GOLDENS_DIR / (input_path.stem + ".json")
    if not golden_path.exists():
        pytest.fail(
            "No golden for {!r}. Generate it with:\n"
            "    tests/run_tests.sh update {}".format(input_path.name, input_path.stem)
        )

    text_str = read_text(input_path)
    actual = compute_metrics(text_str)
    expected = load_golden(golden_path)

    diffs = compare(actual, expected)
    if diffs:
        pytest.fail(
            "Metric drift vs golden ({} differences):\n  ".format(len(diffs))
            + "\n  ".join(diffs)
        )
