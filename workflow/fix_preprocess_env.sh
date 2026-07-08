#!/usr/bin/env bash
# Post-install patches for the `preprocess` pixi env, applied to the installed sources
# in place. Idempotent + python-version-agnostic. Re-run (via the `fix-preprocess-env`
# pixi task) after every `pixi install`/clean that re-fetches these packages. See
# SETUP_NOTES.md. Patches 1-2 are upstream-compat no-ops; patch 3 is a project-specific
# feature (our UMI extraction).
#
#  1. hammerpede (unmaintained since 2020): opens files with the "rU" mode, which
#     Python REMOVED in 3.11 -> hp_bootstrap.py dies with `ValueError: invalid
#     mode: 'rU'`. Fix: "rU" -> "r" (universal newlines are the py3 text default).
#  2. pychopper (_plot_stats, report-PDF step): does `float(<single-row Series>)`,
#     which pandas >= 2 rejects (`TypeError: float() argument must be ... not
#     'Series'`). All processing/outputs finish first; only the final report
#     crashes, but that returns a non-zero exit. Fix: `.Value)` -> `.Value.iloc[0])`
#     on the three calls at pychopper.py:184 ONLY (the next line already uses the
#     correct `.iloc[0]` form). NOT a global replace -- `sum(df.Value)` elsewhere
#     in the file is valid and must be left alone.

set -euo pipefail
root="${PIXI_PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
lib="$root/.pixi/envs/preprocess/lib"

# 1. hammerpede "rU" -> "r"
hp=$(find "$lib" -path '*hammerpede/seq_utils.py' -type f -print -quit)
sed -i 's/"rU"/"r"/g' "$hp"
echo "fix: hammerpede seq_utils.py  ('rU' -> 'r')   residual rU: $(grep -c '"rU"' "$hp" || true)"

# 2. pychopper float(Series) -> float(Series.iloc[0]) on the 3 calls at line 184
pc=$(find "$lib" -path '*pychopper/scripts/pychopper.py' -type f -print -quit)
sed -i \
  -e 's/"Primers_found", ].Value)/"Primers_found", ].Value.iloc[0])/' \
  -e 's/"Rescue", ].Value)/"Rescue", ].Value.iloc[0])/' \
  -e 's/"Unusable", ].Value)/"Unusable", ].Value.iloc[0])/' \
  "$pc"
echo "fix: pychopper.py _plot_stats (float(Series) -> .iloc[0])  remaining bad: $(grep -cE 'float\([^)]*(Primers_found|Rescue|Unusable)[^)]*\.Value\)' "$pc" || true)"
