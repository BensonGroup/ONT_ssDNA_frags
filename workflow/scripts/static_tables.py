#!/usr/bin/env python
"""Analysis tables behind the static F01-F19 figures, for ONE workflow run.

Ported from the cells project's `batch_effect_assess.py --mode within-batch` (its steps t1, t1b, t2,
t4, t5 and t9), keeping only the tables `static_figs.py` draws. One config is one run, so there is
no batch factor: the paired views compare DIFFERENT GROUPS AT THE SAME ROUND, descriptively.

Every input path arrives from the Snakemake rule (`--sheet` carries each sample's files, built from
the config); nothing is globbed. The raw read count is the workflow's own `seqkit stats` count
(`clean_summary.tsv` total_reads) and NanoComp's NanoStats is the raw run summary. The per-read
length/quality profile is its own rule (`seqkit fx2tab`) and goes straight to the figure script.
"""
from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path
from urllib.parse import quote

import numpy as np
import polars as pl

import complexity_lib as cx

OVERLAP_THRESHOLDS = (1, 2, 3, 5, 10)
ABUNDANCE_EDGES = [1, 2, 3, 5, 10, 30, 100, 300, 1000, 10 ** 9]
#: Anchors flank every payload and carry no library identity; their design holds an N10 UMI window.
ANCHORS = ["UMI1_anchor", "UMI1_anchor_rc", "UMI2_anchor", "UMI2_anchor_rc"]
ERROR_SAMPLE_FRAGMENTS = 20_000
BOOTSTRAP = 1000
NAN = float("nan")


# ----------------------------------------------------------------------------------- tsv plumbing

def read_rows(path) -> list[dict]:
    """Rows of a plain TSV, skipping `#` comments; [] when the path is None or absent."""
    if not path or not Path(path).exists():
        return []
    lines = [ln for ln in Path(path).read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    return [dict(zip(lines[0].split("\t"), ln.split("\t"))) for ln in lines[1:]] if lines else []


def _fmt(v, precision: int) -> str:
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "true" if v else "false"
    if isinstance(v, (float, np.floating)):
        # "NaN"/"Inf" exactly: polars reads lower-case "nan" as a string and the column turns to text.
        return "NaN" if v != v else ("Inf" if v > 0 else "-Inf") if np.isinf(v) else f"{v:.{precision}g}"
    return str(v)


def write_tsv(path: Path, rows: list[dict], comments=(), precision: int = 6, columns=()) -> None:
    cols = list(dict.fromkeys([*columns, *(k for r in rows for k in r)]))
    with path.open("w") as fh:
        fh.writelines(f"# {c}\n" for c in comments)
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(_fmt(r.get(c), precision) for c in cols) + "\n")
    print(f"  wrote {len(rows):>7,} rows  {path.name}")


def _num(x) -> float:
    """A missing numeric observation is unavailable, not zero."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return NAN


def _pct(num, den) -> float:
    return 100.0 * num / den if np.isfinite(num) and np.isfinite(den) and den > 0 else NAN


# ------------------------------------------------------------------------------------ statistics

def lin_ccc(x, y) -> float:
    """Lin's concordance correlation: agreement with the identity line, unlike Pearson's r."""
    if x.size < 2:
        return NAN
    mx, my = x.mean(), y.mean()
    sx, sy, sxy = ((x - mx) ** 2).mean(), ((y - my) ** 2).mean(), ((x - mx) * (y - my)).mean()
    denom = sx + sy + (mx - my) ** 2
    return NAN if denom == 0 else 2.0 * sxy / denom


def bootstrap_ci(x, y, n_boot=BOOTSTRAP, seed=0) -> tuple[float, float]:
    """95% percentile bootstrap of Lin's CCC, resampling FEATURES."""
    if x.size < 3:
        return NAN, NAN
    rng = np.random.default_rng(seed)
    vals = np.array([lin_ccc(x[i], y[i]) for i in (rng.integers(0, x.size, x.size) for _ in range(n_boot))])
    vals = vals[np.isfinite(vals)]
    return (float(np.quantile(vals, .025)), float(np.quantile(vals, .975))) if vals.size else (NAN, NAN)


def concordance(x, y, seed=0) -> dict:
    """Pearson / Spearman / Lin's CCC on a pair of log-CPM vectors, CCC with a bootstrap CI."""
    from scipy import stats
    if x.size < 3:
        return {"n": int(x.size), "pearson_r": NAN, "spearman_rho": NAN, "lin_ccc": NAN,
                "ccc_lo": NAN, "ccc_hi": NAN, "status": "insufficient_features"}
    constant = np.ptp(x) == 0 or np.ptp(y) == 0
    lo, hi = bootstrap_ci(x, y, seed=seed)
    return {"n": int(x.size), "pearson_r": NAN if constant else float(np.corrcoef(x, y)[0, 1]),
            "spearman_rho": NAN if constant else float(stats.spearmanr(x, y).statistic),
            "lin_ccc": lin_ccc(x, y), "ccc_lo": lo, "ccc_hi": hi,
            "status": "constant_input" if constant else "computed"}


def cpm_log(mat: np.ndarray) -> np.ndarray:
    """log2 CPM with pseudocount 1, columns = samples."""
    tot = mat.sum(axis=0, keepdims=True).astype(float)
    tot[tot == 0] = 1.0
    return np.log2(1e6 * mat / tot + 1.0)


def clr(mat: np.ndarray, pseudocount: float = 0.5) -> np.ndarray:
    """Centred log-ratio, rows = samples; Euclidean distance on it is Aitchison distance."""
    x = np.asarray(mat, dtype=float) + pseudocount
    log_x = np.log(x / x.sum(axis=1, keepdims=True))
    return log_x - log_x.mean(axis=1, keepdims=True)


def expected_shared(n1: int, n2: int, c_eff: float) -> float:
    """n1*n2/c_eff: a uniform-pool YARDSTICK for shared species, not a test."""
    return n1 * n2 / c_eff if np.isfinite(c_eff) and c_eff > 0 else NAN


# --------------------------------------------------------------------------------- sample model

def comparisons(samples: list[dict]) -> list[dict]:
    """Every different-group, equal-round pair, groups in config order (A = earlier group).

    `k` restarts per group-pair family; it seeds the bootstrap exactly as the original driver did."""
    by_group: dict[str, dict[int, dict]] = {}
    for s in samples:
        by_group.setdefault(s["group"], {})[s["round"]] = s
    out = []
    for ga, gb in combinations(by_group, 2):
        # quote() leaves hyphens alone; escape doubled ones to protect the family separator.
        family = "--".join(quote(g, safe="").replace("--", "%2D%2D") for g in (ga, gb))
        for k, rnd in enumerate(sorted(by_group[ga].keys() & by_group[gb].keys())):
            out.append({"comparison_id": f"{family}__R{rnd}", "comparison_family": family,
                        "group_a": ga, "group_b": gb, "round": rnd, "k": k,
                        "key_a": by_group[ga][rnd]["name"], "key_b": by_group[gb][rnd]["name"]})
    return out


def pair_cols(c: dict) -> dict:
    return {k: c[k] for k in ("comparison_id", "comparison_family", "group_a", "group_b", "round",
                              "key_a", "key_b")}


def overlap_kind(a: dict, b: dict) -> str:
    if a["round"] == 0 or b["round"] == 0:
        return "initial_reference"
    if a["round"] == b["round"] and a["group"] != b["group"]:
        return "same_round_groups"
    return "same_group_cross_round" if a["group"] == b["group"] else "different_group_and_round"


def long_counts(samples: list[dict], cols: list[str]) -> pl.DataFrame:
    """(vector, sample, *cols) for every sample, each read from ITS fragvec group matrix.

    The initial library sits in every group's matrix; the sheet names exactly one per sample, so
    its rows are never multiplied by the number of groups."""
    frames = []
    for pq in dict.fromkeys(s["fragvec_long"] for s in samples):
        names = [s["name"] for s in samples if s["fragvec_long"] == pq]
        frames.append(pl.read_parquet(pq, columns=["vector", "sample", *cols])
                      .filter(pl.col("sample").is_in(names)))
    return pl.concat(frames)


def wide(long: pl.DataFrame, index: str, value: str, keys: list[str]) -> tuple[list[str], np.ndarray]:
    """Pivot to index x sample; a sample with no rows is an observed zero column, not dropped."""
    w = long.pivot(on="sample", index=index, values=value).fill_null(0).sort(index)
    w = w.with_columns([pl.lit(0).alias(k) for k in keys if k not in w.columns])
    return w[index].to_list(), w.select(keys).to_numpy().astype(np.int64)


def vector_order(vectors: list[str]) -> np.ndarray:
    """Payload fragments per vector -- the molecule-length proxy for the stratified agreement."""
    return np.array([sum(f not in ANCHORS for f in v.split("|")) for v in vectors])


# ---------------------------------------------------------------------------------- t1 / t1b

def _overlap_row(a, b, na, nb, shared, ceff, unit) -> dict:
    union = na + nb - shared
    ca, cb = ceff[a["name"]][3], ceff[b["name"]][3]
    pooled = float(np.sqrt(ca * cb)) if np.isfinite(ca) and np.isfinite(cb) else NAN
    exp = expected_shared(na, nb, pooled)
    return {"kind": overlap_kind(a, b), "key_a": a["name"], "key_b": b["name"],
            "group_a": a["group"], "group_b": b["group"], "round_a": a["round"], "round_b": b["round"],
            f"n_{unit}_a": na, f"n_{unit}_b": nb, "shared": shared,
            "jaccard": shared / union if union else NAN,
            "c_eff_a": ca, "c_eff_b": cb, "c_eff_pooled_geom": pooled, "expected_shared": exp,
            "obs_over_exp": shared / exp if exp and np.isfinite(exp) and exp > 0 else NAN,
            "c_eff_a_is_lower_bound": ceff[a["name"]][4], "c_eff_b_is_lower_bound": ceff[b["name"]][4]}


def umi_overlap(out: Path, samples: list[dict]) -> None:
    """Corrected-UMI overlap for every sample pair at every detection threshold.

    A fused UMI is 20 random nt, so a shared one is the same molecule, not chance. Each UMI is
    factorised to an int code once and each pair is a linear-time sorted intersection, instead of
    a umi x sample incidence matrix that would not fit in memory."""
    counts = {s["name"]: pl.read_csv(s["umi_species"], separator="\t")
              .select(pl.col("fused_seq_corrected").alias("umi"), pl.col("n_reads").cast(pl.Int64))
              for s in samples}
    ceff = {k: cx.c_eff(v["n_reads"].to_list()) for k, v in counts.items()}
    long = pl.concat([v.with_columns(pl.lit(k).alias("key")) for k, v in counts.items()])
    codes = long.select("umi").unique().with_row_index("code")
    long = long.join(codes, on="umi", how="left").select("key", "code", "n_reads")
    rows = []
    for thr in OVERLAP_THRESHOLDS:
        sets = {s["name"]: np.sort(long.filter((pl.col("key") == s["name"]) & (pl.col("n_reads") >= thr))
                                   ["code"].to_numpy().astype(np.int64)) for s in samples}
        for i, a in enumerate(samples):
            for b in samples[i + 1:]:
                shared = int(np.intersect1d(sets[a["name"]], sets[b["name"]], assume_unique=True).size)
                rows.append({"threshold_min_reads": thr, **_overlap_row(
                    a, b, sets[a["name"]].size, sets[b["name"]].size, shared, ceff, "umis")})
    write_tsv(out / "umi_overlap.tsv", rows, [
        "Corrected-UMI overlap of every unordered sample pair, including initial R0 (umi/ + umi_initial/).",
        "threshold_min_reads = reads per corrected UMI in BOTH samples; thresholds are sensitivity views.",
        "expected_shared = n_a*n_b/sqrt(c_eff_a*c_eff_b) is a uniform-pool yardstick, not a calibrated null."])


def clonotype_overlap(out: Path, samples: list[dict]) -> None:
    """Fragment-vector overlap, same layout and arithmetic as the UMI table.

    A shared vector is a shared member COMBINATION, not evidence of a shared molecule."""
    keys = [s["name"] for s in samples]
    _, mat = wide(long_counts(samples, ["n_umis"]), "vector", "n_umis", keys)
    ceff = {k: cx.c_eff(mat[mat[:, j] > 0, j].tolist()) for j, k in enumerate(keys)}
    rows = []
    for thr in OVERLAP_THRESHOLDS:
        present = mat >= thr
        for i, a in enumerate(samples):
            for j in range(i + 1, len(samples)):
                shared = int(np.logical_and(present[:, i], present[:, j]).sum())
                rows.append({"unit": "clonotype", "threshold_min_count": thr, **_overlap_row(
                    a, samples[j], int(present[:, i].sum()), int(present[:, j].sum()), shared, ceff,
                    "clonotype")})
    write_tsv(out / "clonotype_overlap.tsv", rows, [
        "Fragment-vector overlap of every unordered sample pair, including initial R0.",
        "threshold_min_count = distinct clones (n_umis) per vector; c_eff is the clonotype pool's."])


# ------------------------------------------------------------------------------------------ t2

def _count(row: dict, col, key: str, metric: str, diags: list) -> float:
    value = _num(row.get(col)) if col else NAN
    if np.isfinite(value) and value >= 0 and value.is_integer():
        return value
    diags.append({"key": key, "invariant": "invalid_or_missing_count", "field": col or "?",
                  "metric": metric, "message": f"{col} must be an observed nonnegative integer; missing is not zero"})
    return NAN


def _stage(stage: str, n_in: float, n_kept: float, key: str, diags: list) -> dict:
    """Stage-local retention/loss against that stage's own input; impossible transitions are NaN."""
    ret = NAN
    if np.isfinite(n_in) and np.isfinite(n_kept) and n_in > 0:
        if 0 <= n_kept <= n_in:
            ret = 100.0 * n_kept / n_in
        else:
            diags.append({"key": key, "invariant": "stage_output_le_input", "field": stage,
                          "metric": f"{stage}_retention_pct", "message": f"{stage} kept {n_kept:g} of {n_in:g}"})
    return {f"{stage}_retention_pct": ret, f"{stage}_loss_pct": 100.0 - ret}


def clean_counts(row: dict, key: str, diags: list) -> dict:
    """Raw / host / length-filter counts from clean_summary.tsv, never repaired.

    Host depletion is the sum of one `<genome>_mapped` column per configured genome. A host or
    length count that exceeds what raw leaves for it invalidates raw rather than being clamped."""
    host_cols = sorted(c for c in row if c.endswith("_mapped"))
    kept_col = next((c for c in row if c.startswith("kept_gt")), None)
    drop_col = next((c for c in row if c.startswith("dropped_le")), None)
    raw = _count(row, "total_reads", key, "raw_reads", diags)
    host = sum(_count(row, c, key, "host_mapped", diags) for c in host_cols) if host_cols else NAN
    kept = _count(row, kept_col, key, "kept_gt_minlen", diags)
    dropped = _count(row, drop_col, key, "dropped_le_minlen", diags)
    if np.isfinite(raw) and (host > raw or kept + dropped > raw - host):
        diags.append({"key": key, "invariant": "known_disjoint_counts_le_raw", "field": "total_reads",
                      "metric": "raw_reads", "message": "host + length outcomes exceed the raw total"})
        raw = NAN
    return {"raw_reads": raw, "host_mapped": host, "reported_unmapped_all": _num(row.get("unmapped_all")),
            "kept_gt_minlen": kept, "dropped_le_minlen": dropped, "pct_dropped_le_minlen": _pct(dropped, raw)}


def read_accounting(out: Path, samples: list[dict], a) -> dict[str, dict]:
    """Per-sample stage counts from each stage's own summary (not recomputed):
    raw -> clean (host + minlen) -> pychopper full length -> coverage filter -> UMI."""
    clean = {r["sample"]: r for r in read_rows(a.clean)}
    prep = {r["Sample"]: r for r in read_rows(a.preprocess)}
    cov = {r["sample"]: r for r in read_rows(a.covfilt)}
    rows, diags = [], []
    for s in samples:
        k = s["name"]
        p, v = prep.get(k, {}), cov.get(k, {})
        u, f = (read_rows(s["umi_summary"]) or [{}])[0], (read_rows(s["fragvec_summary"]) or [{}])[0]
        row = {"key": k, "group": s["group"], "round": s["round"], **clean_counts(clean.get(k, {}), k, diags)}
        fields = [("full_length_reads_in", p, "input_pass"), ("full_length", p, "full_length"),
                  ("rescued", p, "rescued"), ("unclassified", p, "unclassified"),
                  ("usable_with_rescue", p, "usable_with_rescue"),
                  ("umi_reads_in", u, "reads_in"), ("umi_reads_kept", u, "reads_kept"),
                  ("umi_n_species_corrected", u, "n_species_corrected"), ("fragvec_n_vectors", f, "n_vectors")]
        if a.covfilt:            # decode_filter.enabled; without it the stage does not exist
            fields += [("covfilt_reads_in", v, "reads_in"), ("covfilt_reads_kept", v, "reads_kept")]
        for metric, src, col in fields:
            row[metric] = _count(src, col, k, metric, diags)
        for stage, n_in, n_kept in (("clean", "raw_reads", "kept_gt_minlen"),
                                    ("full_length", "full_length_reads_in", "full_length"),
                                    ("covfilt", "covfilt_reads_in", "covfilt_reads_kept"),
                                    ("umi", "umi_reads_in", "umi_reads_kept")):
            row.update(_stage(stage, row.get(n_in, NAN), row.get(n_kept, NAN), k, diags))
        rows.append(row)
    write_tsv(out / "qc_validity.tsv", diags, ["Source accounting inconsistencies; no value is clamped "
              "or substituted."], columns=("key", "invariant", "field", "metric", "message"))
    write_tsv(out / "sample_read_accounting.tsv", rows, [
        "Stage counts from each stage's own summary; raw_reads is clean_summary total_reads (seqkit stats).",
        "Each *_retention_pct / *_loss_pct uses that stage's own input; invalid counts are NaN."], precision=17)
    return {r["key"]: r for r in rows}


# ------------------------------------------------------------------------------------------ t4

def feature_concordance(out: Path, samples: list[dict], comps: list[dict]) -> None:
    """Library-member agreement per same-round pair, on distinct clones carrying each member
    (each (vector, member) counted once), log2 CPM."""
    keys = [s["name"] for s in samples]
    ex = (long_counts(samples, ["n_umis"]).with_columns(pl.col("vector").str.split("|").alias("frag"))
          .explode("frag", empty_as_null=True).filter(~pl.col("frag").is_in(ANCHORS)))
    umis = ex.unique(subset=["sample", "vector", "frag"]).group_by(["sample", "frag"]).agg(pl.col("n_umis").sum())
    _, mat = wide(umis, "frag", "n_umis", keys)
    lc, idx = cpm_log(mat), {k: i for i, k in enumerate(keys)}
    rows = []
    for c in comps:
        ia, ib = idx[c["key_a"]], idx[c["key_b"]]
        keep = (mat[:, ia] > 0) | (mat[:, ib] > 0)
        d = lc[keep, ib] - lc[keep, ia]
        rows.append({**pair_cols(c), "unit": "library_feature", "filter": "present_in_either",
                     **concordance(lc[keep, ia], lc[keep, ib], seed=c["k"]),
                     "mean_abs_log2fc": float(np.abs(d).mean()) if d.size else NAN,
                     "median_log2fc": float(np.median(d)) if d.size else NAN})
    write_tsv(out / "feature_concordance.tsv", rows, [
        "Same-round group-pair library-member concordance, log2 CPM(distinct clones)+1, present in either.",
        "Lin CCC is the agreement statistic; CCC intervals resample features, not biological replicates."])


# ------------------------------------------------------------------------------------------ t5

def vector_analyses(out: Path, samples: list[dict], comps: list[dict]) -> None:
    """Fragment-vector concordance, matched-depth detection, CLR PCA and complexity dependence."""
    from scipy import stats
    keys = [s["name"] for s in samples]
    vectors, mat = wide(long_counts(samples, ["n_umis"]), "vector", "n_umis", keys)
    order, lc, idx = vector_order(vectors), cpm_log(mat), {k: i for i, k in enumerate(keys)}

    rows = []
    for c in comps:
        ia, ib = idx[c["key_a"]], idx[c["key_b"]]
        ca, cb = mat[:, ia], mat[:, ib]
        base = {**pair_cols(c), "unit": "fragment_vector"}
        for label, keep in (("present_in_either", (ca > 0) | (cb > 0)), ("present_in_both", (ca > 0) & (cb > 0)),
                            ("abundant_ge10_either", (ca >= 10) | (cb >= 10))):
            rows.append({**base, "filter": label, "stratum_n_frag": "all",
                         **concordance(lc[keep, ia], lc[keep, ib], seed=c["k"]),
                         "jaccard_presence": float(((ca > 0) & (cb > 0)).sum() / max(((ca > 0) | (cb > 0)).sum(), 1))})
        # fragment-order strata: if the read-length floor reached the counts it shows as a trend here
        abundant = (ca >= 10) | (cb >= 10)
        for kf in range(1, 7):
            sel = abundant & (order == kf)
            stat = (concordance(lc[sel, ia], lc[sel, ib], seed=c["k"] * 10 + kf) if sel.sum() >= 10
                    else {**concordance(np.array([]), np.array([])), "n": int(sel.sum())})
            rows.append({**base, "filter": "abundant_ge10_either", "stratum_n_frag": kf, **stat,
                         "jaccard_presence": NAN})
    write_tsv(out / "vector_concordance.tsv", rows, [
        "Same-round group-pair vector concordance on log2 CPM of distinct clones (n_umis) + 1.",
        "stratum_n_frag: payload fragments per vector (a molecule-length proxy); `all` = unstratified."])

    # detection of A's vectors in B by A-abundance, both thinned without replacement to the
    # shallower clone depth so a low rate cannot be a depth artefact
    rng, det = np.random.default_rng(0), []
    for c in comps:
        ca, cb = mat[:, idx[c["key_a"]]].copy(), mat[:, idx[c["key_b"]]].copy()
        target = int(min(ca.sum(), cb.sum()))
        if ca.sum() > target:
            ca = rng.multivariate_hypergeometric(ca, target, method="marginals")
        if cb.sum() > target:
            cb = rng.multivariate_hypergeometric(cb, target, method="marginals")
        for lo, hi in zip(ABUNDANCE_EDGES[:-1], ABUNDANCE_EDGES[1:]):
            sel = (ca >= lo) & (ca < hi)
            det.append({**pair_cols(c), "a_count_lo": lo, "a_count_hi": hi, "n_vectors_a": int(sel.sum()),
                        "n_detected_b": int((cb[sel] > 0).sum()),
                        "detection_rate": float((cb[sel] > 0).mean()) if sel.any() else NAN,
                        "matched_depth": target, "status": "computed" if sel.any() else "empty_abundance_bin"})
    write_tsv(out / "detection_vs_abundance.tsv", det, [
        "Detection in B of vectors seen in A, binned by A abundance, after multivariate-hypergeometric",
        "thinning of both to the lower clone depth. Directional (A -> B), not a symmetric score."])

    # Aitchison PCA over all samples, vectors with >= 10 clones somewhere
    keep = (mat >= 10).any(axis=1)
    coords, ev, status = np.full((len(keys), 3), NAN), np.full(3, NAN), "insufficient_variation"
    if keep.any():
        x = clr(mat[keep].T)
        u, sv, _ = np.linalg.svd(x - x.mean(axis=0, keepdims=True), full_matrices=False)
        if np.sum(sv ** 2) > 0:
            n = min(3, len(sv))
            coords[:, :n], ev[:n], status = (u * sv)[:, :n], sv[:n] ** 2 / (sv ** 2).sum(), "computed"
    (out / "ordination_info.json").write_text(json.dumps({
        "status": status, "n_samples": len(keys), "n_features": int(keep.sum()),
        "explained_variance": [float(v) if np.isfinite(v) else None for v in ev]}, indent=2) + "\n")
    write_tsv(out / "ordination_coords.tsv", [
        {"key": s["name"], "group": s["group"], "round": s["round"],
         "PC1": coords[i, 0], "PC2": coords[i, 1], "PC3": coords[i, 2]} for i, s in enumerate(samples)],
        ["PCA on CLR-transformed vector counts, all samples; vectors with >=10 clones in any sample."])

    # between-group |log2FC| against clone depth, with one descriptive pooled OLS
    cxr = []
    for c in comps:
        ia, ib = idx[c["key_a"]], idx[c["key_b"]]
        sel = (mat[:, ia] >= 10) | (mat[:, ib] >= 10)
        cxr.append({**pair_cols(c), "clones_a": int(mat[:, ia].sum()), "clones_b": int(mat[:, ib].sum()),
                    "n_vectors_a": int((mat[:, ia] > 0).sum()), "n_vectors_b": int((mat[:, ib] > 0).sum()),
                    "mean_abs_log2fc": float(np.abs(lc[sel, ib] - lc[sel, ia]).mean()) if sel.any() else NAN,
                    "n_vectors_scored": int(sel.sum())})
    lx = np.log10([max(r["clones_a"], 1) for r in cxr])
    ly = np.array([r["mean_abs_log2fc"] for r in cxr])
    ok = np.isfinite(lx) & np.isfinite(ly) & np.array([r["clones_a"] > 0 for r in cxr], dtype=bool)
    fit = stats.linregress(lx[ok], ly[ok]) if ok.sum() > 2 and np.ptp(lx[ok]) > 0 else None
    for r, included in zip(cxr, ok):
        r["fit_included"] = bool(included)
    (out / "complexity_fit.json").write_text(json.dumps({
        "status": "computed" if fit else "insufficient_variation", "n": int(ok.sum()),
        "slope": float(fit.slope) if fit else None, "intercept": float(fit.intercept) if fit else None,
        "r": float(fit.rvalue) if fit else None,
        "comparison_ids": [r["comparison_id"] for r in cxr if r["fit_included"]],
        "interpretation": "descriptive only; rounds and overlapping pairs are not independent replicates",
    }, indent=2) + "\n")
    write_tsv(out / "complexity_dependence.tsv", cxr, [
        "Mean |log2 CPM(B) - log2 CPM(A)| over vectors with >=10 clones in either, against A's clone depth.",
        "fit_included is the mask of the pooled descriptive OLS in complexity_fit.json."])


# ------------------------------------------------------------------------------------------ t9

def fitting_align_ops(ref: str, windows: list[str]) -> tuple[int, int, int, int]:
    """Fitting (free end-gap on the READ) alignment of one reference against many read windows.

    The decoder's start/stop sit within a base or two of the true match, so a global alignment
    would charge that as an indel on every fragment. Vectorised over the windows, which all share
    the reference. Returns (match, mismatch, insertion, deletion) counted against the reference."""
    if not windows:
        return 0, 0, 0, 0
    m, w, n = len(ref), max(len(x) for x in windows), len(windows)
    q = np.full((n, w), 255, dtype=np.uint8)
    for k, s in enumerate(windows):
        if s:
            q[k, :len(s)] = np.frombuffer(s.encode(), dtype=np.uint8)
    r = np.frombuffer(ref.encode(), dtype=np.uint8)
    prev = np.zeros((n, w + 1), dtype=np.int32)            # free leading gap in the read
    ptr = np.zeros((m + 1, w + 1, n), dtype=np.uint8)
    for i in range(1, m + 1):
        cur = np.empty((n, w + 1), dtype=np.int32)
        cur[:, 0] = i                                       # the reference must be consumed
        ptr[i, 0, :] = 1
        sub = prev[:, :-1] + (q != r[i - 1]).astype(np.int32)
        up = prev[:, 1:] + 1
        for j in range(1, w + 1):
            best = np.minimum(np.minimum(sub[:, j - 1], up[:, j - 1]), cur[:, j - 1] + 1)
            cur[:, j] = best
            ptr[i, j, :] = np.where(best == sub[:, j - 1], 0, np.where(best == up[:, j - 1], 1, 2))
        prev = cur
    ends = np.argmin(prev, axis=1)                          # free trailing gap in the read
    ops = [0, 0, 0, 0]
    for k in range(n):
        i, j = m, int(ends[k])
        while i > 0:
            c = ptr[i, j, k]
            if c == 0:
                ops[0 if j >= 1 and q[k, j - 1] == r[i - 1] else 1] += 1
                i, j = i - 1, j - 1
            elif c == 1:
                ops[3] += 1
                i -= 1
            else:
                ops[2] += 1
                j -= 1
    return tuple(ops)


def nanostats(path) -> dict[str, dict[str, str]]:
    """NanoComp's bysample_NanoStats.txt: metrics down the rows, samples across the columns."""
    lines = [ln for ln in Path(path).read_text().splitlines() if ln.strip()]
    cols = lines[0].split("\t")[1:]
    out = {c: {} for c in cols}
    for ln in lines[1:]:
        f = ln.split("\t")
        for c, v in zip(cols, f[1:]):
            out[c][f[0]] = v
    return out


def qc_panel(out: Path, samples: list[dict], acct: dict, a) -> None:
    """ONT run-QC panel: raw summary, decode coverage, depth per UMI/clonotype, identity, error kinds."""
    with open(a.expected, encoding="utf-8-sig", newline="") as fh:   # -sig: a BOM breaks "Name"
        ref_map = {r["Name"]: r["Sequence"].upper() for r in csv.DictReader(fh)}
    # anchors carry an N10 UMI window, which would be scored as error by construction
    ref_map = {k: v for k, v in ref_map.items() if k not in ANCHORS and "N" not in v}
    nano = nanostats(a.nanostats)
    keys = [s["name"] for s in samples]
    _, reads = wide(long_counts(samples, ["n_reads"]), "vector", "n_reads", keys)
    rng, rows = np.random.default_rng(7), []
    for j, s in enumerate(samples):
        k, nn, acc = s["name"], nano.get(s["name"], {}), acct[s["name"]]
        cov = pl.read_parquet(s["coverage"], columns=["fqlen", "covered", "cov"])
        v = reads[:, j][reads[:, j] > 0]
        umi = pl.read_csv(s["umi_species"], separator="\t")["n_reads"].to_numpy()
        fc = (read_rows(s["featurecount_summary"]) or [{}])[0]
        fr = (pl.scan_parquet(s["fragments"]).filter(~pl.col("fragment_id").is_in(ANCHORS))
              .select("read_acc", "fragment_id", "start", "stop", "edit_distance").collect())
        aln = int((fr["stop"].cast(pl.Int64) - fr["start"].cast(pl.Int64)).sum())
        # error decomposition on a bounded sample, re-aligned to the designed sequences
        keep = fr.filter(pl.col("fragment_id").is_in(list(ref_map)))
        if keep.height > ERROR_SAMPLE_FRAGMENTS:
            keep = keep[np.sort(rng.choice(keep.height, ERROR_SAMPLE_FRAGMENTS, replace=False)).tolist()]
        need, seqs = set(keep["read_acc"].to_list()), {}
        with open(s["fused_reads"]) as fh:                  # stop once every wanted read is seen
            remaining = len(need)
            while remaining and (h := fh.readline()):
                sq = fh.readline().rstrip()
                fh.readline(), fh.readline()
                acc_id = h[1:].split(" ", 1)[0]
                if acc_id in need and acc_id not in seqs:
                    seqs[acc_id], remaining = sq, remaining - 1
        tot = np.zeros(4, dtype=np.int64)
        for (frag_id,), grp in keep.group_by("fragment_id"):
            wins = [seqs[r["read_acc"]][max(0, r["start"] - 3):r["stop"] + 3]
                    for r in grp.iter_rows(named=True) if r["read_acc"] in seqs]
            tot += fitting_align_ops(ref_map[frag_id], wins)
        ref_bases = int(tot[0] + tot[1] + tot[3])
        rate = (lambda x: x / ref_bases) if ref_bases else (lambda x: NAN)
        raw, host = acc["raw_reads"], acc["host_mapped"]
        rows.append({
            "key": k, "group": s["group"], "round": s["round"],
            "total_yield_bases": _num(nn.get("number_of_bases")), "n_reads_raw": raw,
            "median_read_quality": _num(nn.get("median_qual")), "mean_read_quality": _num(nn.get("mean_qual")),
            "read_length_n50": _num(nn.get("n50")), "median_read_length": _num(nn.get("median_read_length")),
            "mean_depth_reads_per_clonotype": float(v.mean()) if v.size else NAN,
            "median_depth_reads_per_clonotype": float(np.median(v)) if v.size else NAN,
            "mean_depth_reads_per_umi": float(umi.mean()) if umi.size else NAN,
            "median_depth_reads_per_umi": float(np.median(umi)) if umi.size else NAN,
            "callable_fraction_of_read_bases": float(cov["covered"].sum() / cov["fqlen"].sum()) if cov["fqlen"].sum() else NAN,
            "fragment_coverage_mean": _num(cov["cov"].mean()), "fragment_coverage_median": _num(cov["cov"].median()),
            "library_members_detected": int(pl.scan_parquet(s["decode_featurecount"])
                                            .filter(~pl.col("fragment_id").is_in(ANCHORS))
                                            .select(pl.col("fragment_id").n_unique()).collect().item()),
            "library_members_designed": len(ref_map),
            "host_mapping_rate_pct": _pct(host, raw),
            "payload_assignment_rate_pct": _pct(_num(fc.get("n_reads_with_payload")), _num(fc.get("n_reads"))),
            "identity_from_edit_distance": 1.0 - int(fr["edit_distance"].cast(pl.Int64).sum()) / aln if aln else NAN,
            "error_sampled_ref_bases": ref_bases,
            "mismatch_rate": rate(tot[1]), "insertion_rate": rate(tot[2]), "deletion_rate": rate(tot[3]),
            "total_error_rate": rate(tot[1] + tot[2] + tot[3]), "identity_from_alignment": rate(tot[0]),
        })
        print(f"    {k}: identity={rows[-1]['identity_from_edit_distance']:.4f} "
              f"mism={rows[-1]['mismatch_rate']:.4f} ins={rows[-1]['insertion_rate']:.4f} "
              f"del={rows[-1]['deletion_rate']:.4f}", flush=True)
    write_tsv(out / "ont_qc_panel.tsv", rows, [
        "Raw summary from NanoComp + seqkit; depth = reads per UMI / clonotype (no reference genome);",
        "callable fraction = decoded read bases assigned to a designed fragment; payload assignment is",
        "the assay's analogue of a mapping rate. identity_from_edit_distance is exact over every stored",
        f"non-anchor match; mismatch/insertion/deletion rates re-align up to {ERROR_SAMPLE_FRAGMENTS:,}",
        "fragments per sample to expected_sequences (fitting alignment, per reference base)."])


# --------------------------------------------------------------------------------------- driver

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sheet", required=True, help="JSON list of samples, config order, with their files")
    ap.add_argument("--clean", required=True)
    ap.add_argument("--preprocess", required=True)
    ap.add_argument("--covfilt", default="", help="covfilt/coverage_summary.tsv; empty if the filter is off")
    ap.add_argument("--nanostats", required=True)
    ap.add_argument("--expected", required=True)
    ap.add_argument("--minlen", type=int, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    a = ap.parse_args(argv)
    samples = json.loads(a.sheet)
    comps = comparisons(samples)
    a.outdir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] {len(samples)} samples, {len(comps)} same-round group pairs", flush=True)
    for step in (lambda: umi_overlap(a.outdir, samples), lambda: clonotype_overlap(a.outdir, samples),
                 lambda: feature_concordance(a.outdir, samples, comps),
                 lambda: vector_analyses(a.outdir, samples, comps)):
        step()
    qc_panel(a.outdir, samples, read_accounting(a.outdir, samples, a), a)
    (a.outdir / "run_manifest.json").write_text(json.dumps({
        "samples": [{k: s[k] for k in ("name", "group", "round", "idx")} for s in samples],
        "comparisons": [{k: v for k, v in c.items() if k != "k"} for c in comps],
        "parameters": {"minlen": a.minlen, "overlap_thresholds": list(OVERLAP_THRESHOLDS),
                       "bootstrap_resamples": BOOTSTRAP, "detection_seed": 0,
                       "error_sample_fragments": ERROR_SAMPLE_FRAGMENTS, "error_seed": 7},
    }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
