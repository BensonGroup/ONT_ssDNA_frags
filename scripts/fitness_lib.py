#!/usr/bin/env python3
"""The single definition of a group's "fitness-top" clonotypes, shared by the clone-dominance
test and the FASTQ extractor so the two cannot disagree (same reason `fragvec_lib` and
`complexity_lib` are shared).

A clonotype is "fitness-top" when its summed empty-corrected log2 fold-change across the round
steps is at least a threshold. That summed score -- `log2_sum` -- is the same enrichment score
`scripts/top_vector_reads.py` ranks vectors by:

    log2_sum[v] = sum over the log2_* step columns of <group>_vs_<control>.log2_step_ratios.tsv

which `fitness_step_ratios.py` already wrote (empty-corrected, one row per signal vector).

`log2_sum` is an ENDPOINT score, not a trajectory one
-----------------------------------------------------
`fitness_step_ratios.py` adds EPS on the round-level rpm scale, so each intermediate
`(rpm_r + EPS)` appears once as a numerator and once as a denominator WITHIN its own arm and
cancels; the shared R0 baseline carries identical depth in every group's long table, so it
cancels across arms too. The sum therefore telescopes exactly to

    log2_sum[v] == log2( (subject_last_rpm + EPS) / (control_last_rpm + EPS) )

Intermediate rounds contribute NOTHING to `log2_sum` or to the ranking, and no unweighted
linear combination of the step columns can change that -- an average is `log2_sum` divided by
a constant, so it yields the identical ordering and the identical survivors.

This is why `min_step` / `n_pos_steps` / `is_monotone` are also returned. They are non-linear
in the steps, so they DO see the trajectory, and they are close to orthogonal to `log2_sum`:
measured on disk, only 31 of the 306 A549 clonotypes passing `log2_sum >= 4.1` are monotone,
while 287 vectors rising in >=4 of 5 steps fall below that endpoint bar. They are reported and
never filtered on -- treat them as a shape annotation beside the magnitude gate, and read them
with the depth caveat below.

CAVEAT on the shape columns: a step is empty-corrected, so it can be positive because the
CONTROL fell rather than because the subject rose, and at 1-5 reads a single step is mostly
EPS-floor noise. Do not read `is_monotone` as proof of sustained selection on a shallow vector;
check `subject_max_reads` alongside it.

Deriving the threshold instead of typing it
-------------------------------------------
The telescoping identity above also makes the threshold COMPUTABLE, so `auto_min_log2_sum`
replaces the hand-calibrated constant that used to be passed in. Since the control's endpoint
rpm cannot be negative,

    log2_sum[v] == log2( (S_last + EPS) / (C_last + EPS) )  <=  log2( S_last + EPS )

so a vector carrying a single read at the endpoint round scores AT MOST
`log2(1e6 / T_last / EPS + 1)`, where `T_last` is that sample's rpm denominator. Putting the bar
just above that number is therefore not a heuristic: it is an EXACT sufficient condition for
">= 2 reads at the endpoint round". Verified on disk -- of the vectors clearing their group's
auto threshold, the minimum endpoint read count is exactly 2 in both the A549 and MDA arms.

The converse does NOT hold, and is not meant to: plenty of >=2-read vectors fall below the bar
because the control grew too. The threshold is a magnitude gate whose FLOOR is pinned to the
depth, not a read-count filter.

Because `T_last` is per group, so is the threshold. Two arms sequenced to different depths get
different numbers (A549 4.0951 at T=62,611; MDA 3.4101 at T=104,641) -- uniform in MEANING, not
in magnitude. Their clonotype counts are consequently not comparable across arms.

Run the self-test:  pixi run -e default python scripts/fitness_lib.py
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import polars as pl

RPM = 1_000_000.0

# Separates the (k-1)-read score from the k-read score. It must clear the rounding slop -- the
# step columns are written at 4dp, so a sum of n_steps of them sits within n_steps * 5e-5 (2.5e-4
# for 5 steps) of its analytic value -- while staying far below the gap it is protecting, which is
# log2(2x+1) - log2(x+1) ~= 0.95 at these depths. 0.01 clears the slop by ~40x and the gap by ~95x.
AUTO_MARGIN = 0.01

_STEP_RE = re.compile(r"^log2_R(\d+)_vs_R(\d+)$")


def fitness_table_path(fitness_dir, group: str, control: str) -> Path:
    """Where fitness_step_ratios.py wrote the empty-corrected step table for a subject group."""
    return Path(fitness_dir) / f"{group}_vs_{control}.log2_step_ratios.tsv"


def group_long_path(fragvec_dir, group: str) -> Path:
    """The group's vector x sample long table -- the file the rpm denominators come from."""
    return Path(fragvec_dir) / f"{group}.fragvec_x_sample.long.parquet"


def sample_depths(long_df: pl.DataFrame) -> dict[str, int]:
    """sample -> T, the depth `rpm = 1e6 * n_reads / T` divides by.

    PURE (takes the already-loaded long table) so `fitness_step_ratios.py` can pass the frame it
    has already read and this stays the single definition of T. It must be summed over the FULL
    table: the signal-filtered `<group>.log2_step_ratios.tsv` (max_reads >= min_reads) cannot
    reproduce it, and using the filtered total would silently inflate every rpm.
    """
    return dict(long_df.group_by("sample").agg(pl.col("n_reads").sum().alias("T")).iter_rows())


def endpoint_round(fitness_dir, group: str, control: str) -> int | None:
    """The round `log2_sum` telescopes TO, read off the last `log2_R{c}_vs_R{p}` column.

    Not "R5" and not "the group's max round": `fitness_step_ratios.py` subtracts only the steps
    shared with the control, so a group with an R5 sample but a control stopping at R4 has a sum
    that ends at R4, and R4's depth is then the right denominator. The column name is the only
    place that truth is recorded.

    Returns None when the group has no `<group>_vs_<control>` table (same contract as
    `load_fitness_top`); raises when the table exists but carries no parsable step column.
    """
    p = fitness_table_path(fitness_dir, group, control)
    if not p.exists():
        return None
    header = p.read_text().split("\n", 1)[0].rstrip("\r").split("\t")
    steps = [m for m in (_STEP_RE.match(c) for c in header) if m]
    if not steps:
        raise ValueError(f"{p}: no log2_R<c>_vs_R<p> step columns to read an endpoint from")
    return int(steps[-1].group(1))


def one_read_log2(T: int, eps: float) -> float:
    """Score of a single endpoint read against an absent control: log2(1e6 / T / eps + 1).

    The maximum `log2_sum` any 1-read-at-the-endpoint vector can reach -- see the module docstring.
    """
    return math.log2(RPM / float(T) / eps + 1.0)


def auto_min_log2_sum(fitness_dir, fragvec_dir, group: str, control: str,
                      round_to_sample: dict[int, str], eps: float = 1.0,
                      min_last_reads: int = 2, margin: float = AUTO_MARGIN):
    """(threshold, provenance) derived from the group's own endpoint depth, or None.

        threshold = log2((min_last_reads - 1) * 1e6 / T_endpoint / eps + 1) + margin

    Any vector scoring >= threshold carries >= `min_last_reads` reads at the endpoint round,
    because `log2_sum` telescopes to `log2((S_last + eps) / (C_last + eps)) <= log2(S_last + eps)`.
    `min_last_reads = 2` reduces to "the 1-read floor + margin".

    `round_to_sample` maps this group's round index -> sample name. It is supplied by the caller
    because the round labelling already exists in both call sites (`fitness_step_ratios.sample_round`,
    `clonotype_clone_ranks.sample_round`) and a third copy here would be one more thing to disagree.

    Returns None when the group has no fitness table -- the caller then drops the group exactly as
    `load_fitness_top` makes it. Raises SystemExit when the table EXISTS but its depth cannot be
    resolved: that is a broken invocation (wrong --fragvec-dir, stale matrices), and silently
    falling back to some other number would move the results without saying so.
    """
    if min_last_reads < 1:
        raise ValueError(f"min_last_reads must be >= 1, got {min_last_reads}")
    r = endpoint_round(fitness_dir, group, control)
    if r is None:
        return None

    sample = round_to_sample.get(r)
    if sample is None:
        raise SystemExit(f"[FATAL] {group}: fitness steps end at round R{r} but no sample of this "
                         f"group sits at that round (have {sorted(round_to_sample)}); pass an "
                         f"explicit --min-log2-sum instead of 'auto'")
    long_p = group_long_path(fragvec_dir, group)
    if not long_p.exists():
        raise SystemExit(f"[FATAL] {group}: cannot derive the auto threshold, missing {long_p}; "
                         f"pass an explicit --min-log2-sum instead of 'auto'")
    T = sample_depths(pl.read_parquet(long_p, columns=["sample", "n_reads"])).get(sample)
    if not T:
        raise SystemExit(f"[FATAL] {group}: sample '{sample}' has no reads in {long_p}; "
                         f"pass an explicit --min-log2-sum instead of 'auto'")

    floor = one_read_log2(T, eps)
    # (min_last_reads - 1) reads is the strongest thing that must still FAIL, so the bar sits just
    # above its score. At min_last_reads == 1 that is log2(0 + 1) == 0, i.e. margin alone.
    threshold = math.log2((min_last_reads - 1) * RPM / float(T) / eps + 1.0) + margin
    return threshold, {"group": group, "endpoint_round": r, "endpoint_sample": sample, "T": int(T),
                       "eps": eps, "floor": floor, "margin": margin,
                       "min_last_reads": min_last_reads, "threshold": threshold}


def load_fitness_top(fitness_dir, group: str, control: str, min_log2_sum: float):
    """(vector, log2_sum, fit_rank, n_steps, n_pos_steps, min_step, is_monotone) for `group`,
    filtered to log2_sum >= min_log2_sum.

    `fit_rank` is 1-based by log2_sum descending (ties broken by vector for determinism), assigned
    over the FULL signal set BEFORE the threshold, so a clonotype's rank is its standing among all
    enriched-or-not signal vectors, not just the survivors.

    The shape columns describe the TRAJECTORY that `log2_sum` telescopes away (see the module
    docstring). Only `log2_sum` is thresholded; the rest are carried for inspection:

        n_steps      how many log2_* step columns the table has (constant within a group)
        n_pos_steps  how many of them are > 0 -- the graded version of monotone
        min_step     the worst single step; the margin by which monotonicity holds or fails
        is_monotone  min_step > 0, i.e. the vector gained on the control in EVERY round

    Returns None when the group has no `<group>_vs_<control>` table -- the caller then drops the
    group (there is no fitness readout for it). An empty (0-row) frame means the table exists but no
    vector cleared the threshold.
    """
    p = fitness_table_path(fitness_dir, group, control)
    if not p.exists():
        return None
    df = pl.read_csv(p, separator="\t")
    log_cols = [c for c in df.columns if c.startswith("log2_")]
    if not log_cols:
        raise ValueError(f"{p}: no log2_* step columns to sum")
    ranked = (df.with_columns(
                  pl.sum_horizontal([pl.col(c) for c in log_cols]).alias("log2_sum"),
                  pl.min_horizontal([pl.col(c) for c in log_cols]).alias("min_step"),
                  pl.sum_horizontal([(pl.col(c) > 0).cast(pl.Int32) for c in log_cols])
                    .alias("n_pos_steps"),
                  pl.lit(len(log_cols), dtype=pl.Int32).alias("n_steps"))
              # strictly > 0: a step of exactly 0 is "held level", not "gained on the control"
              .with_columns((pl.col("min_step") > 0).alias("is_monotone"))
              .sort(["log2_sum", "vector"], descending=[True, False])
              .with_row_index("fit_rank", offset=1)
              .select("vector", "log2_sum", "fit_rank", "n_steps", "n_pos_steps",
                      "min_step", "is_monotone"))
    return ranked.filter(pl.col("log2_sum") >= min_log2_sum)


def _selftest() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        # two step columns; log2_sum = their sum. vectors span enriched and depleted.
        # E is the case the shape columns exist for: a HIGH log2_sum reached despite a losing
        # step, i.e. endpoint magnitude and trajectory shape disagreeing.
        # B has a step of exactly 0 -- it must NOT count as positive, nor as monotone.
        pl.DataFrame({
            "vector": ["A", "B", "C", "D", "E"],
            "log2_R1_vs_R0": [3.0, 1.0, -2.0, 0.5, 4.0],
            "log2_R2_vs_R1": [2.0, 0.0, -1.0, -0.5, -0.5],
            "subject_max_reads": [100, 50, 40, 10, 80],
            "empty_present": [True, False, True, True, True],
        }).write_csv(d / "X_vs_empty.log2_step_ratios.tsv", separator="\t")

        # sums: A=5, E=3.5, B=1, D=0, C=-3
        full = load_fitness_top(d, "X", "empty", min_log2_sum=-99)
        assert full.height == 5
        # fit_rank is by log2_sum desc over the FULL set: A,E,B,D,C
        assert full.sort("fit_rank")["vector"].to_list() == ["A", "E", "B", "D", "C"]
        by_v = {r["vector"]: r for r in full.iter_rows(named=True)}
        assert by_v["A"]["log2_sum"] == 5.0

        # shape columns: n_steps constant, n_pos_steps graded, is_monotone == min_step > 0
        assert all(r["n_steps"] == 2 for r in full.iter_rows(named=True))
        assert {v: r["n_pos_steps"] for v, r in by_v.items()} == {"A": 2, "B": 1, "C": 0, "D": 1, "E": 1}
        assert {v: r["min_step"] for v, r in by_v.items()} == {"A": 2.0, "B": 0.0, "C": -2.0,
                                                              "D": -0.5, "E": -0.5}
        assert {v: r["is_monotone"] for v, r in by_v.items()} == {"A": True, "B": False, "C": False,
                                                                 "D": False, "E": False}
        # a step of exactly 0 is "held level", not a gain -- B must fail both tests
        assert by_v["B"]["n_pos_steps"] == 1 and by_v["B"]["is_monotone"] is False
        # E: high log2_sum, NOT monotone -- the magnitude gate and the shape must not imply each other
        assert by_v["E"]["log2_sum"] > by_v["B"]["log2_sum"] and not by_v["E"]["is_monotone"]

        # threshold at 0 keeps A,E,B,D (D==0 included, C dropped); ranks unchanged
        t0 = load_fitness_top(d, "X", "empty", min_log2_sum=0.0)
        assert set(t0["vector"]) == {"A", "B", "D", "E"}, t0["vector"].to_list()
        assert t0.filter(pl.col("vector") == "D")["fit_rank"][0] == 4     # rank from the full set

        # a stricter bar keeps only A -- and does NOT filter on shape (E is dropped on sum, not shape)
        t4 = load_fitness_top(d, "X", "empty", min_log2_sum=4.0)
        assert t4["vector"].to_list() == ["A"]

        # missing table -> None (group dropped), not an error
        assert load_fitness_top(d, "NOPE", "empty", 0.0) is None

        # ------------------------------------------------------------------ #
        # threshold derivation
        # ------------------------------------------------------------------ #
        # Y exists to pin the endpoint rule: its steps stop at R1 (control ran out) while the
        # group still HAS an R2 sample, and the two samples carry deliberately different depths.
        # Keying off the group's max round instead of the last step column would score log2(2),
        # so this catches that mistake rather than merely covering the line.
        pl.DataFrame({"vector": ["A"], "log2_R1_vs_R0": [9.0]}).write_csv(
            d / "Y_vs_empty.log2_step_ratios.tsv", separator="\t")

        def _long(rows):        # {sample: T} -> a long table summing to exactly those depths
            return pl.DataFrame({"vector": ["v"] * len(rows), "sample": list(rows),
                                 "n_reads": [int(t) for t in rows.values()]})

        _long({"X_R1": 500_000, "X_R2": 250_000}).write_parquet(d / "X.fragvec_x_sample.long.parquet")
        _long({"Y_R1": 100_000, "Y_R2": 1_000_000}).write_parquet(d / "Y.fragvec_x_sample.long.parquet")

        # T is summed over the FULL table, not over one row
        assert sample_depths(_long({"S": 7})) == {"S": 7}

        assert endpoint_round(d, "X", "empty") == 2          # numerator of the LAST step column
        assert endpoint_round(d, "Y", "empty") == 1          # ...even though Y has an R2 sample
        assert endpoint_round(d, "NOPE", "empty") is None

        xmap, ymap = {1: "X_R1", 2: "X_R2"}, {1: "Y_R1", 2: "Y_R2"}
        thr, prov = auto_min_log2_sum(d, d, "X", "empty", xmap)
        assert prov["endpoint_sample"] == "X_R2" and prov["T"] == 250_000
        assert abs(thr - (math.log2(1e6 / 250_000 + 1) + AUTO_MARGIN)) < 1e-12   # log2(5) + margin
        assert abs(prov["floor"] - math.log2(5.0)) < 1e-12

        # Y must price off R1 (T=100k -> log2(11)), NOT its deeper R2 (T=1M -> log2(2))
        ythr, yprov = auto_min_log2_sum(d, d, "Y", "empty", ymap)
        assert yprov["endpoint_sample"] == "Y_R1" and yprov["T"] == 100_000
        assert abs(ythr - (math.log2(11.0) + AUTO_MARGIN)) < 1e-12

        # eps and min_last_reads move the bar the way the formula says
        e2, _ = auto_min_log2_sum(d, d, "X", "empty", xmap, eps=2.0)
        assert abs(e2 - (math.log2(1e6 / 250_000 / 2 + 1) + AUTO_MARGIN)) < 1e-12   # log2(3)
        k3, _ = auto_min_log2_sum(d, d, "X", "empty", xmap, min_last_reads=3)
        assert abs(k3 - (math.log2(2 * 1e6 / 250_000 + 1) + AUTO_MARGIN)) < 1e-12   # log2(9)
        k1, _ = auto_min_log2_sum(d, d, "X", "empty", xmap, min_last_reads=1)
        assert abs(k1 - AUTO_MARGIN) < 1e-12                 # log2(0 + 1) == 0, margin alone

        # no fitness table -> None (dropped, as load_fitness_top does); resolvable-but-broken -> loud
        assert auto_min_log2_sum(d, d, "NOPE", "empty", xmap) is None
        for bad_kwargs in ({"round_to_sample": {1: "X_R1"}},              # no sample at R2
                           {"round_to_sample": {2: "X_R9"}}):             # sample not in the table
            try:
                auto_min_log2_sum(d, d, "X", "empty", **bad_kwargs)
            except SystemExit:
                pass
            else:
                raise AssertionError(f"unresolvable depth must be fatal, not silent: {bad_kwargs}")

        # THE property the threshold exists for: single-endpoint-read scores fail, 2-read pass.
        # One step column, so log2_sum == that column and the scores can be written directly.
        T_Z = 250_000
        pl.DataFrame({"vector": ["one_read", "two_reads"],
                      "log2_R1_vs_R0": [math.log2(1e6 / T_Z + 1), math.log2(2e6 / T_Z + 1)]}
                     ).write_csv(d / "Z_vs_empty.log2_step_ratios.tsv", separator="\t")
        _long({"Z_R1": T_Z}).write_parquet(d / "Z.fragvec_x_sample.long.parquet")
        zthr, _ = auto_min_log2_sum(d, d, "Z", "empty", {1: "Z_R1"})
        kept = load_fitness_top(d, "Z", "empty", zthr)["vector"].to_list()
        assert kept == ["two_reads"], f"1-read vector must not survive its own floor: {kept}"

    # The identity the threshold rationale rests on: with EPS added on the ROUND-level rpm scale,
    # the step logs telescope to the endpoint ratio, so no unweighted sum can see the middle rounds.
    EPS = 1.0
    S = [128122.9, 35900.2, 2065.4, 854.0, 229.8, 160.7]      # subject rpm by round
    E = [128122.9, 101596.0, 79576.2, 55615.6, 5080.6, 28065.4]  # control rpm, SAME R0
    steps = [math.log2((S[r] + EPS) / (S[r - 1] + EPS)) - math.log2((E[r] + EPS) / (E[r - 1] + EPS))
             for r in range(1, len(S))]
    endpoint = math.log2((S[-1] + EPS) / (E[-1] + EPS))
    assert abs(sum(steps) - endpoint) < 1e-9, f"telescoping broken: {sum(steps)} != {endpoint}"
    # and the mean is the sum over a constant -> identical ordering, so averaging changes nothing
    assert abs(sum(steps) / len(steps) - endpoint / len(steps)) < 1e-9

    # The inequality the auto threshold rests on: the control's endpoint rpm is >= 0, so the
    # empty-corrected endpoint score can only be LOWER than the subject's own. The 1-read floor is
    # therefore an upper bound for every 1-read vector, whatever the control did.
    for C5 in (0.0, 3.7, 250.0):
        assert math.log2((S[-1] + EPS) / (C5 + EPS)) <= math.log2(S[-1] + EPS) + 1e-12

    print("[INFO] fitness_lib selftest OK  (shape columns, telescoping identity, and the "
          "depth-derived threshold verified)")


if __name__ == "__main__":
    _selftest()
