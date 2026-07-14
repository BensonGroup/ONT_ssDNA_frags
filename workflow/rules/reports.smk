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
# group/round/role come from the config like everyone else's, instead of each report hardcoding its
# own name for it (they disagreed: two said "initial", one said "library"). It also lets
# `rule umi_complexity` read the whole sample table from one file.
#
# Schema: name<TAB>group<TAB>round<TAB>idx<TAB>role   -- `idx` is config order, so groups appear as
# configured. APPEND-ONLY: see the warning in report_lib.load_meta() -- inserting a column before
# `idx` silently degrades the sort key and makes the cell lines interleave by round.
rule cross_sample_metadata:
    output:
        f"{OUTDIR}/{{crossbase}}/sample_metadata.tsv",
    params:
        rows=[(n, str(SAMPLE[n].get("group", "") or ""), SAMPLE[n].get("round"), i,
               SAMPLE[n]["role"])
              for i, n in enumerate(SAMPLES)],
    run:
        with open(output[0], "w") as out:
            out.write("name\tgroup\tround\tidx\trole\n")
            for name, group, rnd, idx, role in params.rows:
                r = "" if rnd is None else rnd
                out.write(f"{name}\t{group}\t{r}\t{idx}\t{role}\n")


def cross_inputs(wildcards):
    """Every non-initial sample's artifacts in the requested tree, plus the initial library's when
    the R0 baseline is enabled. The cross script globs L2F_BASE=<crossbase>/*/ (directories only)
    and groups the samples it finds by config `group`; declaring every non-initial sample here
    keeps that glob and the DAG in sync (previously only `experimental` samples were declared,
    which raced). The baseline lives in the sibling `<crossbase>_initial/` tree — never globbed —
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
    return files


# Cross-sample comparison report. L2F_R0_STEM="" disables the R0 baseline in the script;
# L2F_SAMPLES points the script at the config-derived grouping metadata. Parameterized by
# {crossbase} so the same rule serves the raw decode tree and the deduplicated one; `rule all`
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
        script=str(SCRIPTS / "build_featurecount_report_cross.py"),
    shell:
        r"""
        L2F_BASE="{params.base}" L2F_R0_STEM="{params.r0}" L2F_SAMPLES="{input.meta}" \
            pixi run -e default python "{params.script}"
        """
