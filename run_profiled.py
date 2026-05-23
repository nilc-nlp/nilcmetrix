# -*- coding: utf-8 -*-
"""Run the full NILC-Metrix metric set on a single text with profiling on.

Same input as run_all.py — the text is the first positional argument. The
NILC_PROFILE env var is set before text_metrics is imported, which turns
on the @timed / timed_block instrumentation declared in
text_metrics/profiling.py. At process exit a per-bucket report is printed
to stderr covering DB methods, JVM calls, ResourcePool cache misses, and
per-metric wall time.
"""

import os
os.environ.setdefault("NILC_PROFILE", "1")

import sys
import text_metrics


def normalize_text(text):
    raw = text.replace("{{quotes}}", '"')
    raw = raw.replace("{{exclamation}}", "!")
    raw = raw.replace("{{enter}}", "\n")
    raw = raw.replace("{{sharp}}", "#")
    raw = raw.replace("{{ampersand}}", "&")
    raw = raw.replace("{{percent}}", "%")
    raw = raw.replace("{{dollar}}", "$")
    return raw.encode("utf-8", "surrogateescape").decode("utf-8")


if __name__ == "__main__":
    raw = normalize_text(sys.argv[1])
    t = text_metrics.Text(raw)
    text_metrics.nilc_metrics.values_for_text(t)
