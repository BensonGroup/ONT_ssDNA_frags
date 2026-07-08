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


# Per-sample grouping metadata (from the config `samples` table) that the cross-report script
# reads to order and label columns by `group`/`round` instead of parsing the sample name.
# One row per non-initial sample; `idx` is config order so groups appear as configured.
rule cross_sample_metadata:
    output:
        f"{OUTDIR}/decode/sample_metadata.tsv",
    params:
        rows=[(n, str(SAMPLE[n].get("group", "") or ""), SAMPLE[n].get("round"), i)
              for i, n in enumerate(SAMPLES) if SAMPLE[n]["role"] != "initial"],
    run:
        with open(output[0], "w") as out:
            out.write("name\tgroup\tround\tidx\n")
            for name, group, rnd, idx in params.rows:
                r = "" if rnd is None else rnd
                out.write(f"{name}\t{group}\t{r}\t{idx}\n")


def cross_inputs(wildcards):
    """Every non-initial sample's decode artifacts, plus the initial library's when the R0
    baseline is enabled. The cross script globs L2F_BASE=decode/*/ and groups the samples it
    finds by config `group`; declaring every non-initial sample here keeps that glob and the
    DAG in sync (previously only `experimental` samples were declared, which raced). The
    baseline is injected separately via L2F_R0_STEM."""
    parts = [".featurecount.parquet", ".featurecount.summary.tsv", ".umis.parquet",
             ".ligateddna.log", ".fragments.parquet"]
    files = []
    for s in NONINITIAL:
        st = f"{OUTDIR}/decode/{s}/{s}"
        files += [st + p for p in parts]
    if config["report"]["r0_baseline"]:
        st = f"{OUTDIR}/decode_initial/{INITIAL}/{INITIAL}"
        files += [st + p for p in parts]
    return files


# Cross-sample comparison report. L2F_R0_STEM="" disables the R0 baseline in the script;
# L2F_SAMPLES points the script at the config-derived grouping metadata.
rule report_cross:
    input:
        cross_inputs,
        meta=f"{OUTDIR}/decode/sample_metadata.tsv",
        expected=config["expected_sequences"],
    output:
        f"{OUTDIR}/decode/cross_sample_featurecount_report.html",
    params:
        base=os.path.abspath(f"{OUTDIR}/decode"),
        r0=(os.path.abspath(f"{OUTDIR}/decode_initial/{INITIAL}/{INITIAL}")
            if config["report"]["r0_baseline"] else ""),
        script=str(SCRIPTS / "build_featurecount_report_cross.py"),
    shell:
        r"""
        L2F_BASE="{params.base}" L2F_R0_STEM="{params.r0}" L2F_SAMPLES="{input.meta}" \
            pixi run -e default python "{params.script}"
        """
