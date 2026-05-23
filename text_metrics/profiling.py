"""Lightweight, toggleable timing instrumentation.

Enabled when the env var NILC_PROFILE is set to a truthy value at import
time. When disabled, every public name in this module becomes a near-no-op
so the production HTTP path is unaffected.

Usage:

    from text_metrics.profiling import profiler, timed, timed_block

    @timed("db.get_delaf_verb")
    def get_delaf_verb(self, verb): ...

    with timed_block("jvm.stanford"):
        subprocess.Popen(...).communicate()

    profiler.start_text("book.txt")
    ...
    profiler.end_text("book.txt")

    profiler.report()  # at process exit; prints to stderr
"""

from __future__ import unicode_literals, print_function, division

import os
import sys
import time
import atexit
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps


ENABLED = os.environ.get("NILC_PROFILE", "").lower() not in ("", "0", "false", "no")


class _Profiler(object):
    def __init__(self):
        # bucket -> [count, total_seconds]
        self._buckets = defaultdict(lambda: [0, 0.0])
        # (label, elapsed_s)
        self._texts = []
        self._cur_text_label = None
        self._cur_text_start = None
        self._run_start = time.perf_counter()

    def record(self, bucket, elapsed):
        b = self._buckets[bucket]
        b[0] += 1
        b[1] += elapsed

    def start_text(self, label):
        self._cur_text_label = label
        self._cur_text_start = time.perf_counter()

    def end_text(self, label):
        elapsed = time.perf_counter() - self._cur_text_start
        self._texts.append((label, elapsed))
        # live per-book line to stderr
        print("[profile] %s  %.2fs" % (label, elapsed), file=sys.stderr)

    def report(self):
        out = sys.stderr
        total = time.perf_counter() - self._run_start
        print("\n=== NILC-Metrix profile ===", file=out)
        print("books: %d  run wall: %.2fs" % (len(self._texts), total), file=out)
        if self._texts:
            print("", file=out)
            for label, elapsed in self._texts:
                print("  %-40s  %7.2fs" % (label, elapsed), file=out)

        def dump_section(title, prefix, with_per_call=True, top=None):
            rows = [(k, v[0], v[1]) for k, v in self._buckets.items()
                    if k.startswith(prefix)]
            if not rows:
                return
            rows.sort(key=lambda r: -r[2])
            if top is not None:
                rows = rows[:top]
            print("\n[%s]" % title, file=out)
            if with_per_call:
                print("  %-44s %10s %10s %12s" %
                      ("bucket", "calls", "total_s", "per_call_ms"),
                      file=out)
                for name, count, total_s in rows:
                    per_call = (total_s / count) * 1000.0 if count else 0.0
                    print("  %-44s %10d %10.3f %12.3f" %
                          (name, count, total_s, per_call), file=out)
            else:
                print("  %-44s %10s %10s" %
                      ("bucket", "calls", "total_s"), file=out)
                for name, count, total_s in rows:
                    print("  %-44s %10d %10.3f" %
                          (name, count, total_s), file=out)

        dump_section("db", "db.")
        dump_section("jvm", "jvm.", with_per_call=False)
        dump_section("rp", "rp.")
        dump_section("metric (top 20)", "metric.", top=20)


class _NullProfiler(object):
    def record(self, bucket, elapsed): pass
    def start_text(self, label): pass
    def end_text(self, label): pass
    def report(self): pass


profiler = _Profiler() if ENABLED else _NullProfiler()


if ENABLED:
    @contextmanager
    def timed_block(bucket):
        start = time.perf_counter()
        try:
            yield
        finally:
            profiler.record(bucket, time.perf_counter() - start)

    def timed(bucket):
        def deco(fn):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    return fn(*args, **kwargs)
                finally:
                    profiler.record(bucket, time.perf_counter() - start)
            return wrapper
        return deco

    atexit.register(profiler.report)
else:
    @contextmanager
    def timed_block(bucket):
        yield

    def timed(bucket):
        def deco(fn):
            return fn
        return deco
