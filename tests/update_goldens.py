# -*- coding: utf-8 -*-
"""Regenerate JSON goldens for one or more input texts.

Usage:
    python tests/update_goldens.py             # regenerate all
    python tests/update_goldens.py stem1 stem2 # regenerate specific stems
"""

import sys
from pathlib import Path

# Allow running as a plain script from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.conftest import (  # noqa: E402  (sys.path mutation above)
    GOLDENS_DIR,
    INPUTS_DIR,
    compute_metrics,
    read_text,
    write_golden,
)


def _resolve_inputs(stems):
    if not stems:
        paths = sorted(INPUTS_DIR.glob("*.txt"))
        if not paths:
            print("No inputs found in {}".format(INPUTS_DIR))
        return paths

    paths = []
    missing = []
    for stem in stems:
        candidate = INPUTS_DIR / (stem + ".txt")
        if candidate.exists():
            paths.append(candidate)
        else:
            missing.append(stem)
    if missing:
        sys.stderr.write("Missing input file(s): {}\n".format(", ".join(missing)))
        sys.exit(1)
    return paths


def main(argv):
    paths = _resolve_inputs(argv[1:])
    for input_path in paths:
        text_str = read_text(input_path)
        metrics = compute_metrics(text_str)
        golden_path = GOLDENS_DIR / (input_path.stem + ".json")
        write_golden(golden_path, metrics)
        print("wrote {} ({} metrics)".format(golden_path, len(metrics)))


if __name__ == "__main__":
    main(sys.argv)
