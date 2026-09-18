"""Within-batch overlap tables, adapted from scripts/within_batch_overlap.py.

Numerical kernels retain the reference's singleton/all tiers, fragment-order
strata, UMI-level sampling without replacement, and sample-label permutations.
Inputs are the workflow's full fragvec counts/read map and clean UMI species.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import itertools
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import complexity_lib as cx                          # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SAMPLE_GROUPS = {}
CLEAN_BASE = "umi"

#: The four anchor pseudo-modules. They flank every payload and carry no library identity, so they
#: are excluded from the order count -- the same rule as `batch_effect_assess.vector_order`.
ANCHORS = frozenset({"UMI1_anchor", "UMI1_anchor_rc", "UMI2_anchor", "UMI2_anchor_rc"})

#: Payload-order strata retained from the reference; pooled includes every payload order.
STRATA = ("<=3", "4", ">=5", "pooled")

TIERS = ("singleton", "all")

#: Detection thresholds for the UMI table, matching `batch_effect_assess.OVERLAP_THRESHOLDS` so the
#: two studies' UMI tables stay directly comparable.
UMI_THRESHOLDS = (1, 2, 3, 5, 10)

#: n_umis bands for the selection-bias curve. The top band is open so no clone count is dropped.
ABUNDANCE_TIERS = (("1", 1, 1), ("2", 2, 2), ("3-4", 3, 4), ("5-9", 5, 9), ("10+", 10, 10 ** 9))

#: Group print order. Rounds sort inside a group, so a sample's position is (group rank, round).
GROUP_ORDER = ("initial", "A549", "A549-SN", "MDA", "MDA-SN", "empty")

STEPS_ALL = ("sizes", "frag", "tiers", "depth", "umi", "infer", "venn")


# ============================================================================ sample model, local

def load_manifest(path: Path) -> list[tuple[str, str, int]]:
    """Config sample order, with the baseline first and rounds ordered within groups."""
    global SAMPLE_GROUPS, GROUP_ORDER
    df = pl.read_csv(path, separator="\t").sort("idx")
    rows = [(r["name"], r["group"], int(r["round"])) for r in df.iter_rows(named=True)]
    SAMPLE_GROUPS = {name: group for name, group, _ in rows}
    if len(SAMPLE_GROUPS) != len(rows) or not rows:
        raise ValueError("Sample metadata must contain unique sample names")
    if sum(group == "initial" for _, group, _ in rows) != 1:
        raise ValueError("Sample metadata must contain exactly one initial library")
    GROUP_ORDER = tuple(["initial", *dict.fromkeys(g for _, g, _ in rows if g != "initial")])
    rank = {g: i for i, g in enumerate(GROUP_ORDER)}
    return sorted(rows, key=lambda row: (rank[row[1]], row[2]))


def fragvec_path(resdir, name, suffix):
    stage = "fragvec_initial" if SAMPLE_GROUPS[name] == "initial" else "fragvec"
    return resdir / stage / name / f"{name}.{suffix}"


def clone_rows(resdir, name):
    # fragvec_clones is thresholded. The read map includes every payload-bearing
    # read; unique vector/UMI pairs are the analysis units, with no source edits.
    return (pl.read_parquet(fragvec_path(resdir, name, "fragvec_reads.parquet"),
                            columns=["vector", "umi"])
            .unique().sort(["vector", "umi"]))


def species_path(resdir, name):
    stage = CLEAN_BASE + ("_initial" if SAMPLE_GROUPS[name] == "initial" else "")
    return resdir / stage / name / f"{name}.umi_species.tsv"


# Empty pair tables remain readable and carry a declared schema.
PAIR_COLUMNS = "key_a key_b group_a group_b round_a round_b pair_class bio_subclass n_a n_b shared union shared_lo95 shared_hi95 jaccard overlap_coef".split()
EMPTY_COLUMNS = {
    "fragvec_overlap.tsv": ["tier", "order_stratum", *PAIR_COLUMNS],
    "umi_overlap.tsv": ["population", "threshold_min_reads", *PAIR_COLUMNS],
    "order_prevalence.tsv": "tier order n_vectors pct_of_union mean_prevalence frac_shared_k_ge2 max_prevalence".split(),
    "abundance_tiers.tsv": "key_a key_b group_a group_b pair_class n_umis_tier_in_a n_vectors_a n_present_in_b p_present_in_b n_singleton_in_b p_singleton_in_b_given_present".split(),
    "depth_matched_overlap.tsv": "depth_m n_reps tier order_stratum key_a key_b group_a group_b pair_class bio_subclass n_a_mean n_b_mean shared_mean shared_sd shared_lo95 shared_hi95 jaccard_mean overlap_coef_mean".split(),
    "class_contrasts.tsv": "order_stratum tier n_pairs contrast_bio_over_bioempty null_median null_p95 n_perm p_one_sided".split(),
    "venn_counts.tsv": "space arm n_initial n_arm shared jaccard initial_only arm_only pct_of_arm pct_of_initial".split(),
}
TABLE_NOTES = {
    "sample_sizes.tsv": "Per-sample richness, singleton counts and UMI depth. C_eff is descriptive; it is not an overlap expectation.",
    "fragvec_overlap.tsv": "Native-depth pairwise overlap by singleton/all tier and fragment-order stratum. Shared counts use exact Poisson intervals.",
    "order_prevalence.tsv": "Observed prevalence by fragment order; orders with fewer than 20 vectors are omitted.",
    "sharing_spectrum.tsv": "Number of vectors carried by k configured samples, per tier and stratum.",
    "abundance_tiers.tsv": "Detection in sample B by abundance tier in A; conditional singleton probability is undefined when no vector is detected.",
    "depth_matched_overlap.tsv": "Sample vector/UMI pairs without replacement, re-derive presence and singleton status, then measure overlap. Poisson limits apply to rounded mean shared counts; shared_sd measures spread across replicates.",
    "umi_overlap.tsv": "Clean corrected-UMI overlap and UMIs associated with singleton fragvecs. No reads are deduplicated or discarded.",
    "class_contrasts.tsv": "Size-adjusted selected/selected sharing divided by selected/empty sharing. One-sided permutations shuffle sample group labels. Undefined contrasts and p-values are NaN.",
    "venn_counts.tsv": "Initial library versus each group's final sampled round, selected from the overlap tables.",
}


def write_tsv(path, rows, comments=None):
    with path.open("w", newline="") as handle:
        for comment in comments or []:
            handle.write("# " + comment + "\n")
        fields = list(rows[0]) if rows else EMPTY_COLUMNS[path.name]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("NaN" if isinstance(value, float) and math.isnan(value) else value)
                             for key, value in row.items()})


def pair_class(ga: str, gb: str) -> str:
    """The within-batch replacement for `matched block`.

    `empty` is the no-template control, so every class involving it is a background measurement.
    The initial library is separated out because it is R0 -- a different kind of comparison from
    one selection round against another.
    """
    if ga == "initial" or gb == "initial":
        other = gb if ga == "initial" else ga
        return "init x empty" if other == "empty" else "init x bio"
    ea, eb = ga == "empty", gb == "empty"
    if ea and eb:
        return "empty x empty"
    if ea or eb:
        return "bio x empty"
    return "bio x bio"


def bio_subclass(ga: str, gb: str) -> str:
    """Sub-structure inside `bio x bio`. `A549` and `A549-SN` are the same cell line, different arm."""
    if ga == gb:
        return "within_group"
    line = lambda g: g.split("-")[0]                                        # noqa: E731
    return "same_line_cross_group" if line(ga) == line(gb) else "cross_line"


# ================================================================================= loading fragvecs

def load_fragvec(resdir: Path, manifest) -> tuple[list[str], np.ndarray, dict]:
    """Factorise full workflow fragvec_counts onto a shared vector index.

    The thresholded fragvec_clones table is never used: singleton vectors below
    min_reads must remain available for every overlap analysis.
    """
    index: dict[str, int] = {}
    per: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, _g, _r in manifest:
        path = fragvec_path(resdir, name, "fragvec_counts.parquet")
        df = pl.read_parquet(path, columns=["vector", "n_umis"])
        vs = df["vector"].to_list()
        nu = df["n_umis"].to_numpy().astype(np.int32)
        codes = np.empty(len(vs), dtype=np.int32)
        for i, v in enumerate(vs):
            c = index.get(v)
            if c is None:
                c = len(index)
                index[v] = c
            codes[i] = c
        o = np.argsort(codes, kind="stable")
        per[name] = (codes[o], nu[o])
    vectors: list[str] = [""] * len(index)
    for v, c in index.items():
        vectors[c] = v
    order = np.array([sum(1 for f in v.split("|") if f not in ANCHORS) for v in vectors],
                     dtype=np.int16)
    return vectors, order, per


def check_singletons(resdir: Path, per: dict) -> None:
    """Check full UMI membership against the workflow's per-vector counts."""
    for name in per:
        counts = pl.read_parquet(fragvec_path(resdir, name, "fragvec_counts.parquet"),
                                 columns=["vector", "n_umis"]).sort("vector")
        actual = clone_rows(resdir, name).group_by("vector").len().sort("vector")
        if (counts["vector"].to_list() != actual["vector"].to_list()
                or counts["n_umis"].to_list() != actual["len"].to_list()):
            raise ValueError(f"{name}: fragvec_counts and full fragvec_reads UMI membership disagree")


def stratum_mask(order_of: np.ndarray, st: str) -> np.ndarray:
    if st == "pooled":
        return np.ones(order_of.shape, dtype=bool)
    if st == "<=3":
        return order_of <= 3
    if st == "4":
        return order_of == 4
    return order_of >= 5


def presence_sets(per: dict, order: np.ndarray) -> dict:
    """{(sample, tier, stratum): sorted code array}. Masking a sorted array keeps it sorted, which
    is what lets every intersection below pass `assume_unique=True`."""
    out = {}
    for name, (codes, nu) in per.items():
        oc = order[codes]
        for tier in TIERS:
            tm = (nu == 1) if tier == "singleton" else np.ones(nu.shape, dtype=bool)
            for st in STRATA:
                out[(name, tier, st)] = codes[tm & stratum_mask(oc, st)]
    return out


# ===================================================================================== statistics

def poisson_ci(k: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact (Garwood) Poisson interval for a count. Used instead of a normal approximation because
    the order>=5 shared counts fall to single digits, where sqrt(k) is meaningless."""
    lo = 0.0 if k == 0 else float(chi2.ppf(alpha / 2, 2 * k) / 2)
    hi = float(chi2.ppf(1 - alpha / 2, 2 * k + 2) / 2)
    return lo, hi


def _overlap_row(a, b, A, B, meta) -> dict:
    sh = int(np.intersect1d(A, B, assume_unique=True).size)
    na, nb = int(A.size), int(B.size)
    un = na + nb - sh
    lo, hi = poisson_ci(sh)
    ga, gb = meta[a][0], meta[b][0]
    return {
        "key_a": a, "key_b": b, "group_a": ga, "group_b": gb,
        "round_a": meta[a][1], "round_b": meta[b][1],
        "pair_class": pair_class(ga, gb),
        "bio_subclass": bio_subclass(ga, gb) if pair_class(ga, gb) == "bio x bio" else "",
        "n_a": na, "n_b": nb, "shared": sh, "union": un,
        "shared_lo95": lo, "shared_hi95": hi,
        "jaccard": (sh / un) if un else float("nan"),
        "overlap_coef": (sh / min(na, nb)) if min(na, nb) else float("nan"),
    }


# ========================================================================================== steps

def step_sizes(out: Path, manifest, vectors, order, per) -> None:
    """Per-sample richness, depth and order composition. The depth spread is the confound every
    other table has to be read against, so it gets its own file rather than a footnote."""
    rows = []
    for name, g, r in manifest:
        codes, nu = per[name]
        oc = order[codes]
        sg = nu == 1
        summ = cx.c_eff(nu.tolist())
        rec = {
            "sample": name, "group": g, "round": r,
            "n_vectors": int(codes.size), "n_singleton": int(sg.sum()),
            "pct_singleton": 100.0 * float(sg.mean()) if sg.size else float("nan"),
            "sum_n_umis": int(nu.sum()), "max_n_umis": int(nu.max()) if nu.size else 0,
            "c_eff_clonotype": float(summ[3]), "c_eff_is_lower_bound": bool(summ[4]),
        }
        for st in ("<=3", "4", ">=5"):
            m = stratum_mask(oc, st)
            rec[f"n_vec_order_{st}"] = int(m.sum())
            rec[f"n_sgl_order_{st}"] = int((m & sg).sum())
        rows.append(rec)
    write_tsv(out / "sample_sizes.tsv", rows, comments=[TABLE_NOTES['sample_sizes.tsv']])
    print(f"  sizes: wrote {len(rows)} rows -> {out / 'sample_sizes.tsv'}")


def step_frag(out: Path, manifest, vectors, order, per) -> None:
    """Native-depth overlap for every pair, in both tiers and all four strata, plus the two
    union-level descriptions (order prevalence, sharing spectrum) that explain its shape."""
    meta = {n: (g, r) for n, g, r in manifest}
    names = [n for n, _, _ in manifest]
    ps = presence_sets(per, order)

    rows = []
    for a, b in itertools.combinations(names, 2):
        for tier in TIERS:
            for st in STRATA:
                rec = _overlap_row(a, b, ps[(a, tier, st)], ps[(b, tier, st)], meta)
                rec = {"tier": tier, "order_stratum": st, **rec}
                rows.append(rec)
    write_tsv(out / "fragvec_overlap.tsv", rows, comments=[TABLE_NOTES['fragvec_overlap.tsv']])
    print(f"  frag: wrote {len(rows)} rows -> {out / 'fragvec_overlap.tsv'}")

    # ---- order prevalence, over the union: why the stratification exists at all ----------------
    orows = []
    for tier in TIERS:
        seen = [ps[(n, tier, "pooled")] for n in names]
        allc = np.concatenate(seen)
        prev = np.bincount(allc, minlength=len(vectors))
        present = prev > 0
        for o in range(1, (int(order.max()) if order.size else 0) + 1):
            m = present & (order == o)
            n_v = int(m.sum())
            if n_v < 20:
                continue
            p = prev[m]
            orows.append({
                "tier": tier, "order": o, "n_vectors": n_v,
                "pct_of_union": 100.0 * n_v / int(present.sum()),
                "mean_prevalence": float(p.mean()),
                "frac_shared_k_ge2": float((p >= 2).mean()),
                "max_prevalence": int(p.max()),
            })
    write_tsv(out / "order_prevalence.tsv", orows, comments=[TABLE_NOTES['order_prevalence.tsv']])
    print(f"  frag: wrote {len(orows)} rows -> {out / 'order_prevalence.tsv'}")

    # ---- sharing spectrum, per tier and stratum -------------------------------------------------
    srows = []
    for tier in TIERS:
        for st in STRATA:
            allc = np.concatenate([ps[(n, tier, st)] for n in names])
            prev = np.bincount(allc, minlength=len(vectors))
            hist = np.bincount(prev[prev > 0], minlength=len(names) + 1)
            tot = int(hist[1:].sum())
            for k in range(1, len(names) + 1):
                srows.append({"tier": tier, "order_stratum": st, "n_samples_carrying": k,
                              "n_vectors": int(hist[k]),
                              "frac_of_observed": (float(hist[k]) / tot) if tot else float("nan")})
    write_tsv(out / "sharing_spectrum.tsv", srows, comments=[TABLE_NOTES['sharing_spectrum.tsv']])
    print(f"  frag: wrote {len(srows)} rows -> {out / 'sharing_spectrum.tsv'}")


def step_tiers(out: Path, manifest, vectors, order, per) -> None:
    """The selection curve the singleton filter induces, measured rather than argued.

    For each ORDERED pair (a, b) and each n_umis band of a: what fraction of a's vectors in that
    band are present in b at all, and among those, what fraction are singletons in b. Together
    these two curves show exactly why `n_umis == 1` in both samples is a pair-dependent filter and
    not a neutral restriction.
    """
    names = [n for n, _, _ in manifest]
    meta = {n: (g, r) for n, g, r in manifest}
    nvec = len(vectors)
    rows = []
    for b in names:
        codes_b, nu_b = per[b]
        dense = np.zeros(nvec, dtype=np.int32)             # n_umis of b, 0 where absent
        dense[codes_b] = nu_b
        for a in names:
            if a == b:
                continue
            codes_a, nu_a = per[a]
            nb_of_a = dense[codes_a]
            for label, lo, hi in ABUNDANCE_TIERS:
                m = (nu_a >= lo) & (nu_a <= hi)
                n_in = int(m.sum())
                if n_in == 0:
                    continue
                nb = nb_of_a[m]
                pres = nb > 0
                n_pres = int(pres.sum())
                rows.append({
                    "key_a": a, "key_b": b,
                    "group_a": meta[a][0], "group_b": meta[b][0],
                    "pair_class": pair_class(meta[a][0], meta[b][0]),
                    "n_umis_tier_in_a": label, "n_vectors_a": n_in,
                    "n_present_in_b": n_pres,
                    "p_present_in_b": n_pres / n_in,
                    "n_singleton_in_b": int((nb == 1).sum()),
                    "p_singleton_in_b_given_present":
                        (float((nb[pres] == 1).mean()) if n_pres else float("nan")),
                })
    write_tsv(out / "abundance_tiers.tsv", rows, comments=[TABLE_NOTES['abundance_tiers.tsv']])
    print(f"  tiers: wrote {len(rows)} rows -> {out / 'abundance_tiers.tsv'}")


def _clone_codes(resdir: Path, name: str, vmap: pl.DataFrame) -> np.ndarray:
    """One vector code per (vector, UMI) row -- the unit hypergeometric rarefaction draws from."""
    df = clone_rows(resdir, name).select("vector")
    if df.is_empty():
        return np.empty(0, dtype=np.int32)
    j = df.join(vmap, on="vector", how="left")
    return j["code"].to_numpy().astype(np.int32)


def step_depth(out: Path, manifest, vectors, order, per, resdir: Path,
               depths: list[int], reps: int, seed: int) -> None:
    """Overlap after matching every sample to a common UMI depth.

    WHY NOT SUBSAMPLE THE PRESENCE SET. Thinning a set of vectors uniformly drops every vector with
    the same probability, but real depth loss drops LOW-ABUNDANCE vectors preferentially -- and the
    low-abundance vectors are exactly the singletons this analysis is about. Uniform set-thinning
    would therefore preserve the very quantity depth matching is supposed to correct.

    So the draw is at the UMI level: m of the sample's (vector, UMI) rows, without replacement,
    and presence and singleton status are recomputed from what survives. That is the same operation
    sequencing less deeply would have performed.
    """
    names = [n for n, _, _ in manifest]
    meta = {n: (g, r) for n, g, r in manifest}
    vmap = pl.DataFrame({"vector": vectors,
                         "code": np.arange(len(vectors), dtype=np.int32)})
    print("  depth: loading UMI rows")
    clones = {n: _clone_codes(resdir, n, vmap) for n in names}
    totals = {n: int(clones[n].size) for n in names}
    for n in names:
        want = int(per[n][1].sum())
        if totals[n] != want:
            raise ValueError(f"{n}: full read map has {totals[n]} rows, fragvec_counts sums to {want}")
    print(f"  depth: UMI rows per sample {min(totals.values()):,} .. {max(totals.values()):,}")

    rng = np.random.default_rng(seed)
    acc: dict[tuple, list[int]] = {}
    for m in depths:
        elig = [n for n in names if totals[n] >= m]
        if len(elig) < 2:
            print(f"  depth: m={m} skipped, only {len(elig)} samples deep enough")
            continue
        print(f"  depth: m={m:,} over {len(elig)} samples x {reps} replicates")
        for rep in range(reps):
            sets = {}
            for n in elig:
                sel = rng.choice(totals[n], size=m, replace=False)
                u, cnt = np.unique(clones[n][sel], return_counts=True)
                oc = order[u]
                for tier in TIERS:
                    tm = (cnt == 1) if tier == "singleton" else np.ones(cnt.shape, dtype=bool)
                    for st in STRATA:
                        sets[(n, tier, st)] = u[tm & stratum_mask(oc, st)]
            for a, b in itertools.combinations(elig, 2):
                for tier in TIERS:
                    for st in STRATA:
                        A, B = sets[(a, tier, st)], sets[(b, tier, st)]
                        sh = int(np.intersect1d(A, B, assume_unique=True).size)
                        acc.setdefault((m, a, b, tier, st), []).append((sh, A.size, B.size))

    rows = []
    for (m, a, b, tier, st), vals in acc.items():
        sh = np.array([v[0] for v in vals], dtype=float)
        na = np.array([v[1] for v in vals], dtype=float)
        nb = np.array([v[2] for v in vals], dtype=float)
        un = na + nb - sh
        ga, gb = meta[a][0], meta[b][0]
        with np.errstate(divide="ignore", invalid="ignore"):
            jac = np.where(un > 0, sh / un, np.nan)
            oc = np.where(np.minimum(na, nb) > 0, sh / np.minimum(na, nb), np.nan)
        lo, hi = poisson_ci(int(round(sh.mean())))
        rows.append({
            "depth_m": m, "n_reps": len(vals), "tier": tier, "order_stratum": st,
            "key_a": a, "key_b": b, "group_a": ga, "group_b": gb,
            "pair_class": pair_class(ga, gb),
            "bio_subclass": bio_subclass(ga, gb) if pair_class(ga, gb) == "bio x bio" else "",
            "n_a_mean": float(na.mean()), "n_b_mean": float(nb.mean()),
            "shared_mean": float(sh.mean()), "shared_sd": float(sh.std(ddof=1)) if len(vals) > 1 else 0.0,
            "shared_lo95": lo, "shared_hi95": hi,
            "jaccard_mean": float(np.nanmean(jac)) if np.isfinite(jac).any() else float("nan"),
            "overlap_coef_mean": float(np.nanmean(oc)) if np.isfinite(oc).any() else float("nan"),
        })
    rows.sort(key=lambda r: (r["depth_m"], r["tier"], r["order_stratum"], r["key_a"], r["key_b"]))
    write_tsv(out / "depth_matched_overlap.tsv", rows, comments=[TABLE_NOTES['depth_matched_overlap.tsv']])
    print(f"  depth: wrote {len(rows)} rows -> {out / 'depth_matched_overlap.tsv'}")


def step_umi(out: Path, manifest, per, resdir: Path) -> None:
    """UMI-space overlap: the molecular-identity control the fragvec numbers must be read against.

    A fused UMI is ~20 random nt, so a shared UMI is the SAME PHYSICAL MOLECULE seen twice, not a
    coincidence (4^20 ~ 1e12). A shared fragvec is only a shared ARCHITECTURE -- two independent
    molecules built from the same modules are the same vector. The gap between the two tables is
    the whole point: it separates "the same designs recur" from "the same material is present".

    Two populations: every corrected UMI in the clean `umi/` layer, and the UMIs of singleton
    fragvecs (for a singleton the fragvec's only UMI IS its `top_umi`).
    """
    names = [n for n, _, _ in manifest]
    meta = {n: (g, r) for n, g, r in manifest}

    frames = []
    for n in names:
        df = (pl.read_csv(species_path(resdir, n), separator="\t")
              .rename({"fused_seq_corrected": "umi"})
              .select(pl.col("umi"), pl.col("n_reads").cast(pl.Int64), pl.lit(n).alias("key")))
        frames.append(df)
    long = pl.concat(frames)
    uni = long.select("umi").unique().with_row_index("code")
    long = long.join(uni, on="umi", how="left").select("key", "code", "n_reads")
    print(f"  umi: {len(names)} samples, {uni.height:,} distinct UMIs")
    del uni, frames

    rows = []
    for thr in UMI_THRESHOLDS:
        codes = {}
        for n in names:
            c = (long.filter((pl.col("key") == n) & (pl.col("n_reads") >= thr))["code"]
                 .to_numpy().astype(np.int64))
            codes[n] = np.sort(c)
        for a, b in itertools.combinations(names, 2):
            r = _overlap_row(a, b, codes[a], codes[b], meta)
            rows.append({"population": "all_umis", "threshold_min_reads": thr, **r})
    del long

    # Each singleton vector has one vector/UMI pair in the full read map.
    sgl = {}
    for n in names:
        singletons = (pl.read_parquet(fragvec_path(resdir, n, "fragvec_counts.parquet"))
                      .filter(pl.col("n_umis") == 1).select("vector"))
        sgl[n] = clone_rows(resdir, n).join(singletons, on="vector", how="inner")["umi"].to_list()
    sidx: dict[str, int] = {}
    scodes = {}
    for n in names:
        arr = np.empty(len(sgl[n]), dtype=np.int64)
        for i, u in enumerate(sgl[n]):
            c = sidx.get(u)
            if c is None:
                c = len(sidx)
                sidx[u] = c
            arr[i] = c
        scodes[n] = np.unique(arr)
    for a, b in itertools.combinations(names, 2):
        r = _overlap_row(a, b, scodes[a], scodes[b], meta)
        rows.append({"population": "singleton_fragvec_umis", "threshold_min_reads": 1, **r})

    write_tsv(out / "umi_overlap.tsv", rows, comments=[TABLE_NOTES['umi_overlap.tsv']])
    print(f"  umi: wrote {len(rows)} rows -> {out / 'umi_overlap.tsv'}")


def step_infer(out: Path, manifest, n_perm: int, seed: int) -> None:
    """Class contrasts with an assumption-free p-value.

    THE STATISTIC. Within one order stratum, the ratio of size-adjusted mean shared count between
    `bio x bio` pairs and `bio x empty` pairs. Size adjustment is the offset log(n_a*n_b), so this
    is a comparison of pair classes at equal pair size, not a raw count comparison.

    THE P-VALUE IS A PERMUTATION OF SAMPLE LABELS, NOT A WALD TEST. Sample pairs are not
    independent -- each sample participates in multiple pairs -- so a model-based standard error understates
    the uncertainty by an unknown factor. Shuffling the GROUP labels across the configured samples and
    recomputing the whole class assignment respects that dependence exactly, because it permutes
    the unit that is actually exchangeable under the null: the sample.
    """
    src = out / "depth_matched_overlap.tsv"
    depth_col = "shared_mean"
    if not src.exists():
        src = out / "fragvec_overlap.tsv"
        depth_col = "shared"
        print("  infer: depth_matched_overlap.tsv absent, falling back to native depth")
    df = pl.read_csv(src, separator="\t", comment_prefix="#", infer_schema_length=100000)
    if df.is_empty():
        df = pl.read_csv(out / "fragvec_overlap.tsv", separator="\t", comment_prefix="#")
        depth_col = "shared"
    if df.is_empty():
        write_tsv(out / "class_contrasts.tsv", [], [TABLE_NOTES["class_contrasts.tsv"]])
        return
    if "depth_m" in df.columns:
        m = int(df["depth_m"].max())
        df = df.filter(pl.col("depth_m") == m)
        print(f"  infer: using depth m={m:,}")
    df = df.filter(pl.col("tier") == "singleton")

    groups = {n: g for n, g, _ in manifest}
    names = [n for n, _, _ in manifest]
    rng = np.random.default_rng(seed)

    def contrast(sub: pl.DataFrame, gmap: dict) -> float:
        ga = np.array([gmap[k] for k in sub["key_a"]])
        gb = np.array([gmap[k] for k in sub["key_b"]])
        cls = np.array([pair_class(x, y) for x, y in zip(ga, gb)])
        sh = sub[depth_col].to_numpy().astype(float)
        na = sub["n_a_mean" if "n_a_mean" in sub.columns else "n_a"].to_numpy().astype(float)
        nb = sub["n_b_mean" if "n_b_mean" in sub.columns else "n_b"].to_numpy().astype(float)
        size = na * nb
        out_ = {}
        for c in ("bio x bio", "bio x empty"):
            m_ = (cls == c) & (size > 0)
            out_[c] = sh[m_].sum() / size[m_].sum() if m_.any() and size[m_].sum() > 0 else np.nan
        if not np.isfinite(out_["bio x empty"]) or out_["bio x empty"] <= 0:
            return float("nan")
        return float(out_["bio x bio"] / out_["bio x empty"])

    rows = []
    for st in STRATA:
        sub = df.filter(pl.col("order_stratum") == st)
        if sub.height == 0:
            continue
        obs = contrast(sub, groups)
        null = np.empty(n_perm)
        for i in range(n_perm):
            perm = list(groups.values())
            rng.shuffle(perm)
            null[i] = contrast(sub, dict(zip(names, perm)))
        good = null[np.isfinite(null)]
        p = (1 + int((good >= obs).sum())) / (1 + good.size) if good.size and np.isfinite(obs) else float("nan")
        rows.append({
            "order_stratum": st, "tier": "singleton", "n_pairs": sub.height,
            "contrast_bio_over_bioempty": obs,
            "null_median": float(np.median(good)) if good.size else float("nan"),
            "null_p95": float(np.percentile(good, 95)) if good.size else float("nan"),
            "n_perm": int(good.size), "p_one_sided": p,
        })
    write_tsv(out / "class_contrasts.tsv", rows, comments=[TABLE_NOTES['class_contrasts.tsv']])
    print(f"  infer: wrote {len(rows)} rows -> {out / 'class_contrasts.tsv'}")


def step_venn(out: Path, manifest) -> None:
    """The R0 library against each arm's final sampled round, in three spaces."""
    if len(manifest) < 2:
        write_tsv(out / "venn_counts.tsv", [], [TABLE_NOTES["venn_counts.tsv"]])
        return
    initial = next((n for n, g, _ in manifest if g == "initial"), None)
    if initial is None:
        raise ValueError("no `initial` sample in the manifest")
    last_round = {g: max(r for _, group, r in manifest if group == g)
                  for g in GROUP_ORDER if g != "initial"}
    arms = [n for n, g, r in manifest if g != "initial" and r == last_round[g]]
    frag = pl.read_csv(out / "fragvec_overlap.tsv", separator="\t", comment_prefix="#",
                       infer_schema_length=100000)
    umi = pl.read_csv(out / "umi_overlap.tsv", separator="\t", comment_prefix="#",
                      infer_schema_length=100000)

    def pick(df, **eq):
        q = df
        for k, v in eq.items():
            q = q.filter(pl.col(k) == v)
        q = q.filter(((pl.col("key_a") == initial) & (pl.col("key_b").is_in(arms)))
                     | ((pl.col("key_b") == initial) & (pl.col("key_a").is_in(arms))))
        return q

    rows = []
    specs = [
        ("singleton fragvec, order>=5", pick(frag, tier="singleton", order_stratum=">=5")),
        ("singleton fragvec, all orders", pick(frag, tier="singleton", order_stratum="pooled")),
        ("UMI", pick(umi, population="all_umis", threshold_min_reads=1)),
    ]
    for space, sub in specs:
        for r in sub.iter_rows(named=True):
            init_is_a = r["key_a"] == initial
            rows.append({
                "space": space,
                "arm": r["key_b"] if init_is_a else r["key_a"],
                "n_initial": r["n_a"] if init_is_a else r["n_b"],
                "n_arm": r["n_b"] if init_is_a else r["n_a"],
                "shared": r["shared"], "jaccard": r["jaccard"],
                "initial_only": (r["n_a"] if init_is_a else r["n_b"]) - r["shared"],
                "arm_only": (r["n_b"] if init_is_a else r["n_a"]) - r["shared"],
                "pct_of_arm": (100.0 * r["shared"] / (r["n_b"] if init_is_a else r["n_a"])
                               if (r["n_b"] if init_is_a else r["n_a"]) else float("nan")),
                "pct_of_initial": (100.0 * r["shared"] / (r["n_a"] if init_is_a else r["n_b"])
                                   if (r["n_a"] if init_is_a else r["n_b"]) else float("nan")),
            })
    write_tsv(out / "venn_counts.tsv", rows, comments=[TABLE_NOTES['venn_counts.tsv']])
    print(f"  venn: wrote {len(rows)} rows -> {out / 'venn_counts.tsv'}")


# =========================================================================================== main

def main(argv=None):
    global CLEAN_BASE, ANCHORS
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--resdir", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--umi-base", choices=("umi", "umicorr"), default="umi")
    ap.add_argument("--depths", default="5000,26000,55000")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--perm", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    depths = sorted(set(int(value) for value in a.depths.split(",")))
    if not depths or min(depths) < 1 or a.reps < 1 or a.perm < 1 or a.seed < 0:
        ap.error("depths, reps and perm must be positive; seed must be nonnegative")
    CLEAN_BASE = a.umi_base
    out = a.outdir
    out.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(a.metadata)
    anchors = set()
    for name, _, _ in manifest:
        summary = pl.read_csv(fragvec_path(a.resdir, name, "fragvec_summary.tsv"), separator="\t")
        if "anchors" in summary.columns:
            for value in summary["anchors"].drop_nulls():
                anchors.update(value.split(";"))
        elif pl.read_parquet(fragvec_path(a.resdir, name, "fragvec_counts.parquet")).height:
            raise ValueError(f"{name}: fragvec summary is missing the anchor vocabulary")
    ANCHORS = frozenset(anchors)
    vectors, order, per = load_fragvec(a.resdir, manifest)
    check_singletons(a.resdir, per)
    step_sizes(out, manifest, vectors, order, per)
    step_frag(out, manifest, vectors, order, per)
    step_tiers(out, manifest, vectors, order, per)
    step_depth(out, manifest, vectors, order, per, a.resdir, depths, a.reps, a.seed)
    step_umi(out, manifest, per, a.resdir)
    step_infer(out, manifest, a.perm, a.seed)
    step_venn(out, manifest)
    accounting = []
    for name, group, rnd in manifest:
        species = pl.read_csv(species_path(a.resdir, name), separator="\t",
                              schema_overrides={"n_reads": pl.Int64})
        accounting.append({"sample": name, "group": group, "round": rnd,
                           "umi_reads_kept": int(species["n_reads"].sum() or 0)})
    write_tsv(out / "sample_read_accounting.tsv", accounting,
              ["Final retained reads: sum n_reads from the clean corrected-UMI species table. No deduplication."])
    (out / "run_manifest.json").write_text(json.dumps({
        "mode": "within-batch", "source": str(a.resdir.resolve()),
        "metadata": str(a.metadata.resolve()), "umi_base": CLEAN_BASE,
        "samples": [{"name": n, "group": g, "round": r} for n, g, r in manifest],
        "depths": depths, "reps": a.reps, "permutations": a.perm, "seed": a.seed,
        "clone_source": "unique vector/UMI pairs from full fragvec_reads.parquet",
        "anchors_excluded_from_payload_order": sorted(ANCHORS),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
