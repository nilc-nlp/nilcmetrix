# Regression tests

Goldens-based regression tests for `text_metrics.no_palavras_metrics`. Each
input text under `inputs/` is paired (by filename stem) with a JSON golden
under `goldens/`. The test fails when computed metrics drift from the golden
beyond a small float tolerance.

## Requirements

- Docker, with the `cohmetrix:focal` image available locally.
- A running Postgres container exposing the cohmetrix lexical data, linked
  into the test container as `pgs_cohmetrix`. By default the host container
  is also named `pgs_cohmetrix`; override with `--pgs-container NAME`.

## Run the tests

```sh
tests/run_tests.sh                       # uses pgs_cohmetrix
tests/run_tests.sh test --pgs-container my_pgs
tests/run_tests.sh test -- -k cartomante # extra pytest args after `--`
```

## Add a new input

1. Drop a UTF-8 `.txt` file in `tests/inputs/`, e.g. `tests/inputs/foo.txt`.
2. Generate its golden:
   ```sh
   tests/run_tests.sh update foo
   ```
   (No stem regenerates all goldens.)
3. Commit `tests/inputs/foo.txt` and `tests/goldens/foo.json` together.

## Interpreting failures

A failing test prints one line per drifted metric:

```
mean_noun_phrase: golden=2.500000 actual=2.428571 delta=-7.143e-02
```

Tolerance lives in `conftest.py` (`RTOL = 1e-6`, `ATOL = 1e-9`). If a change
is intentional (e.g. an algorithm fix), regenerate the affected goldens with
`tests/run_tests.sh update <stem>` and commit the updated JSON.

## Layout

```
tests/
  conftest.py           # paths, tolerance, compute/compare helpers
  test_regression.py    # parametrized over inputs/*.txt
  update_goldens.py     # CLI to (re)generate goldens
  run_tests.sh          # docker entry point
  inputs/               # input texts (one .txt per case)
  goldens/              # JSON goldens, one per input by stem
```
