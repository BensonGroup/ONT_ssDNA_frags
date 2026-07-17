# Interactive HTML reports (from scripts/08 and 10). Scripts are unmodified and driven by
# absolute L2F_* env vars.

# One self-contained report per sample.
rule report_sample:
    input:
        fc=f"{OUTDIR}/{{decbase}}/{{sample}}/{{sample}}.featurecount.parquet",
        summ=f"{OUTDIR}/{{decbase}}/{{sample}}/{{sample}}.featurecount.summary.tsv",
        frags=f"{OUTDIR}/{{decbase}}/{{sample}}/{{sample}}.fragments.parquet",
        umis=f"{OUTDIR}/{{decbase}}/{{sample}}/{{sample}}.umis.parquet",
        log=f"{OUTDIR}/{{decbase}}/{{sample}}/{{sample}}.ligateddna.log",
        expected=config["expected_sequences"],
    output:
        f"{OUTDIR}/{{decbase}}/{{sample}}/{{sample}}.featurecount_report.html",
    params:
        stem=lambda wc: os.path.abspath(f"{OUTDIR}/{wc.decbase}/{wc.sample}/{wc.sample}"),
        script=str(SCRIPTS / "build_featurecount_report.py"),
    shell:
        r"""
        L2F_STEM="{params.stem}" pixi run -e default python "{params.script}"
        """


# Per-sample grouping metadata (from the config `samples` table) that the cross-report scripts read
# to order, colour and label columns by `group`/`round` instead of parsing the sample name.
#
# EVERY sample, including the R0 `initial` library. The reports only ever look up names they globbed
# out of the non-initial tree, so the extra row is inert for them -- but it means the baseline's
# group/round come from the config like everyone else's, instead of each report hardcoding its own
# name for it (they disagreed: two said "initial", one said "library"). It also lets
# `rule umi_complexity` read the whole sample table from one file.
#
# The `group` string decides EVERYTHING downstream: `initial` is the R0 baseline (excluded from the
# report columns by report_lib.group_order), `empty` is the PCR-only arm, and every other group is a
# selection arm. There is no second axis -- one field picks the tree, the matrix, the colour and the
# collapse kind.
#
# Schema: name<TAB>group<TAB>round<TAB>idx   -- `idx` is RAW config sample order (not group-major),
# so a group's palette slot follows where it FIRST APPEARS in `samples:`. APPEND-ONLY: see the
# warning in report_lib.load_meta() -- inserting a column before `idx` silently degrades the sort
# key and makes the cell lines interleave by round.
rule cross_sample_metadata:
    output:
        f"{OUTDIR}/{{crossbase}}/sample_metadata.tsv",
    params:
        rows=[(n, str(SAMPLE[n].get("group", "") or ""), SAMPLE[n].get("round"), i)
              for i, n in enumerate(SAMPLES)],
    run:
        with open(output[0], "w") as out:
            out.write("name\tgroup\tround\tidx\n")
            for name, group, rnd, idx in params.rows:
                r = "" if rnd is None else rnd
                out.write(f"{name}\t{group}\t{r}\t{idx}\n")


def cross_inputs(wildcards):
    """Every non-initial sample's artifacts in the requested tree, plus the initial library's when
    the R0 baseline is enabled. The cross script globs L2F_BASE=<crossbase>/*/ (directories only)
    and groups the samples it finds by config `group`; declaring every non-initial sample here
    keeps that glob and the DAG in sync (a stricter subset was declared once, and it raced).
    The baseline lives in the sibling `<crossbase>_initial/` tree — never globbed —
    and is injected separately via L2F_R0_STEM, so the two always move in lockstep."""
    base = wildcards.crossbase
    parts = [".featurecount.parquet", ".featurecount.summary.tsv", ".umis.parquet",
             ".ligateddna.log", ".fragments.parquet"]
    files = []
    for s in NONINITIAL:
        st = f"{OUTDIR}/{base}/{s}/{s}"
        files += [st + p for p in parts]
    if config["report"]["r0_baseline"]:
        st = f"{OUTDIR}/{base}_initial/{INITIAL}/{INITIAL}"
        files += [st + p for p in parts]
    # The coverage panel RENDERS `rule coverage`'s per-read record, so it is a declared input, not
    # just transitively available: rebuilding the report against a stale coverage.parquet would
    # publish a distribution that no longer matches the reads it describes.
    #
    # UNCONDITIONAL, and from stem() -- the raw DECODE tree -- regardless of the coverage filter.
    # Coverage is a decode-step property; measuring it on `base` (the clean umi/ tree, downstream of
    # the coverage filter, the UMI filter AND the jackpot filter) would describe an already-cut set.
    files += [f"{stem(s)}.coverage.parquet" for s in NONINITIAL]
    if config["report"]["r0_baseline"]:
        files.append(f"{stem(INITIAL)}.coverage.parquet")
    return files


# Cross-sample comparison report. L2F_R0_STEM="" disables the R0 baseline in the script;
# L2F_SAMPLES points the script at the config-derived grouping metadata. Parameterized by
# {crossbase} so the same rule serves the raw decode tree and the UMI-filtered one; `rule all`
# requests CROSSBASE (see common.smk).
rule report_cross:
    input:
        cross_inputs,
        meta=f"{OUTDIR}/{{crossbase}}/sample_metadata.tsv",
        expected=config["expected_sequences"],
    output:
        f"{OUTDIR}/{{crossbase}}/cross_sample_featurecount_report.html",
    params:
        base=lambda wc: os.path.abspath(f"{OUTDIR}/{wc.crossbase}"),
        r0=lambda wc: (os.path.abspath(f"{OUTDIR}/{wc.crossbase}_initial/{INITIAL}/{INITIAL}")
                       if config["report"]["r0_baseline"] else ""),
        # The coverage panel's source: the RAW decode tree's per-read record, always. It is the only
        # honest place to measure a decode-step property -- {crossbase} may be the clean umi/ tree,
        # which sits downstream of every filter. L2F_MIN_COVERAGE only draws the cut line, and is "" when
        # the coverage filter is off (then nothing was cut, so there is no line to draw).
        cov_base=os.path.abspath(f"{OUTDIR}/decode"),
        r0_cov=(os.path.abspath(f"{stem(INITIAL)}.coverage.parquet")
                if config["report"]["r0_baseline"] else ""),
        mincov=(DECODE_MIN_COVERAGE if DECODE_FILTER_ON else ""),
        script=str(SCRIPTS / "build_featurecount_report_cross.py"),
    shell:
        r"""
        L2F_BASE="{params.base}" L2F_R0_STEM="{params.r0}" L2F_SAMPLES="{input.meta}" \
        L2F_COV_BASE="{params.cov_base}" L2F_R0_COV="{params.r0_cov}" \
        L2F_MIN_COVERAGE="{params.mincov}" \
            pixi run -e default python "{params.script}"
        """
