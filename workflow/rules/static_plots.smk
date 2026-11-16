# Opt-in print-ready figures (`static_plot: true`) under <outdir>/static_plots/:
#
#   static_fastq_profile (per sample) -- every raw read's length + mean quality, `seqkit fx2tab`
#   static_analysis -> analysis/  THE ANALYSIS SERIES: one figure per workflow step, in run order --
#                      raw sequencing, read funnel, decode QC, raw fragment calls, UMI QC, then the
#                      clean-data analyses (fragment quant, UMI complexity/abundance/sharing, fragvec
#                      quant/trajectories/sharing): 01-12. One message per figure, its data as a TSV.
#   static_tables  -- batch_effect_assess.py (within-batch steps) -> the tables 13-22 draw, and the
#                     sampled error decomposition that figure 04 shows
#   static_figs    -- agreement/  batch_effect_static_figs.py: 13-22, the cross-group agreement set
#                     (overlap, concordance, ordination) that continues the series after fragvec
#   static_reports -- reports/  every Plotly figure of the HTML reports, redrawn for print
#
# Every input is an explicit path derived from the config (samples, their groups, the path helpers in
# common.smk); no script lists or globs a folder.
import json

STATIC = f"{OUTDIR}/static_plots"
STATIC_ON = _cfg_bool(config.get("static_plot"), False, key="static_plot")
if STATIC_ON and not FRAGVEC_ON:
    raise ValueError("config.static_plot: true needs fragvec.enabled: true -- "
                     "the F figures are drawn from the fragvec matrices.")
STATIC_TABLES = [f"{STATIC}/tables/{name}" for name in (
    "umi_overlap.tsv", "clonotype_overlap.tsv", "sample_read_accounting.tsv", "qc_validity.tsv",
    "feature_concordance.tsv", "vector_concordance.tsv", "detection_vs_abundance.tsv",
    "ordination_coords.tsv", "ordination_info.json", "complexity_dependence.tsv",
    "complexity_fit.json", "ont_qc_panel.tsv", "run_manifest.json")]


def static_plot_targets():
    """Optional rule-all outputs, evaluated after the rule modules are loaded."""
    return [f"{STATIC}/analysis", f"{STATIC}/agreement", f"{STATIC}/reports"] if STATIC_ON else []


def _static_sample(s):
    """One sample's files for static_tables. The initial library sits in every group's fragvec
    matrix; it is read from the first group's, so its counts are never multiplied."""
    group = SAMPLE[s]["group"]
    fv_group = GROUPS[0] if group == BASELINE_GROUP else group
    return {
        "umi_species": f"{umi_stem(s)}.umi_species.tsv",
        "umi_summary": f"{umi_stem(s)}.umi_summary.tsv",
        "featurecount_summary": f"{umi_stem(s)}.featurecount.summary.tsv",
        "fragvec_summary": f"{fragvec_stem(s)}.fragvec_summary.tsv",
        "fragvec_long": f"{OUTDIR}/fragvec/{fv_group}.fragvec_x_sample.long.parquet",
        "coverage": f"{stem(s)}.coverage.parquet",
        "fragments": f"{stem(s)}.fragments.parquet",
        "decode_featurecount": f"{stem(s)}.featurecount.parquet",
        "fused_reads": f"{stem(s)}.fused_reads.fq",
    }


def _analysis_sample(s):
    """One sample's files for the analysis series, each from the step that produced it: the raw
    profile, the RAW decode tree (decoder log, coverage, raw fragment calls), the CLEAN end of the
    UMI spine (UMI summary/species, clean fragment calls) and the fragvec branch."""
    group = SAMPLE[s]["group"]
    fv_group = GROUPS[0] if group == BASELINE_GROUP else group
    return {
        "profile": f"{STATIC}/fastq/{s}.len_qual.tsv.gz",
        "decode_log": f"{stem(s)}.ligateddna.log",
        "coverage": f"{stem(s)}.coverage.parquet",
        "decode_fc_summary": f"{stem(s)}.featurecount.summary.tsv",
        "decode_fc": f"{stem(s)}.featurecount.parquet",
        "decode_frags": f"{stem(s)}.fragments.parquet",
        "umi_summary": f"{umi_stem(s)}.umi_summary.tsv",
        "umi_species": f"{umi_stem(s)}.umi_species.tsv",
        "clean_fc_summary": f"{umi_stem(s)}.featurecount.summary.tsv",
        "clean_fc": f"{umi_stem(s)}.featurecount.parquet",
        "fragvec_summary": f"{fragvec_stem(s)}.fragvec_summary.tsv",
        "fragvec_counts": f"{fragvec_stem(s)}.fragvec_counts.parquet",
        "fragvec_long": f"{OUTDIR}/fragvec/{fv_group}.fragvec_x_sample.long.parquet",
    }


rule static_fastq_profile:
    input:
        lambda wc: SAMPLE[wc.sample]["fastq"],
    output:
        f"{STATIC}/fastq/{{sample}}.len_qual.tsv.gz",
    threads: 2
    shell:
        r"""
        pixi run -e default seqkit fx2tab -n -i -l -q -j {threads} {input:q} \
          | cut -f 2,3 | gzip -c > {output:q}
        """


rule static_tables:
    input:
        per_sample=[f for s in SAMPLES for f in _static_sample(s).values()],
        clean=f"{OUTDIR}/clean/clean_summary.tsv",
        preprocess=f"{OUTDIR}/preprocess/batch_summary.tsv",
        covfilt=[f"{OUTDIR}/covfilt/coverage_summary.tsv"] if DECODE_FILTER_ON else [],
        nanostats=f"{OUTDIR}/qc/nanocomp/bysample_NanoStats.txt",
        expected=config["expected_sequences"],
    output:
        STATIC_TABLES,
    params:
        sheet=json.dumps([{"name": s, "group": SAMPLE[s]["group"], "round": SAMPLE[s]["round"], "idx": i,
                           **_static_sample(s)} for i, s in enumerate(SAMPLES)]),
        minlen=config["params"]["minlen"],
        outdir=f"{STATIC}/tables",
        script=str(SCRIPTS / "static_tables.py"),
    threads: 4
    shell:
        r"""
        POLARS_MAX_THREADS={threads} pixi run -e default python {params.script:q} --sheet {params.sheet:q} \
          --clean {input.clean:q} --preprocess {input.preprocess:q} --covfilt "{input.covfilt}" \
          --nanostats {input.nanostats:q} --expected {input.expected:q} --minlen {params.minlen} \
          --outdir {params.outdir:q}
        """


rule static_analysis:
    input:
        per_sample=[f for s in SAMPLES for f in _analysis_sample(s).values()],
        clean=f"{OUTDIR}/clean/clean_summary.tsv",
        preprocess=f"{OUTDIR}/preprocess/batch_summary.tsv",
        covfilt=[f"{OUTDIR}/covfilt/coverage_summary.tsv"] if DECODE_FILTER_ON else [],
        complexity=f"{OUTDIR}/umi/umi_complexity.tsv",
        collapse=f"{OUTDIR}/umi/umi_collapse.tsv",
        jackpot=[f"{OUTDIR}/jackpot/{INITIAL}.umi_blacklist_summary.tsv"] if JACKPOT_ON else [],
        qc_panel=f"{STATIC}/tables/ont_qc_panel.tsv",
        expected=config["expected_sequences"],
    output:
        directory(f"{STATIC}/analysis"),
    params:
        sheet=json.dumps([{"name": s, "group": SAMPLE[s]["group"], "round": SAMPLE[s]["round"], "idx": i,
                           **_analysis_sample(s)} for i, s in enumerate(SAMPLES)]),
        minlen=config["params"]["minlen"],
        mincov=(f"--min-coverage {DECODE_MIN_COVERAGE}" if DECODE_FILTER_ON else ""),
        minreads=FRAGVEC_MIN_READS,
        script=str(SCRIPTS / "static_analysis.py"),
    threads: 4
    shell:
        r"""
        POLARS_MAX_THREADS={threads} pixi run -e default python {params.script:q} --sheet {params.sheet:q} \
          --clean {input.clean:q} --preprocess {input.preprocess:q} --covfilt "{input.covfilt}" \
          --complexity {input.complexity:q} --collapse {input.collapse:q} \
          --jackpot-summary "{input.jackpot}" --qc-panel {input.qc_panel:q} \
          --expected {input.expected:q} --minlen {params.minlen} \
          {params.mincov} --min-reads {params.minreads} --outdir {output:q}
        """


rule static_figs:
    input:
        tables=STATIC_TABLES,
    output:
        directory(f"{STATIC}/agreement"),
    params:
        tables=f"{STATIC}/tables",
        script=str(SCRIPTS / "static_figs.py"),
    shell:
        r"""
        pixi run -e default python {params.script:q} --tables {params.tables:q} --outdir {output:q}
        """


# The reports are exactly the HTML outputs `rule all` declares, so configuration and export cannot
# drift and a stale report left in outdir is never picked up.
rule static_reports:
    input:
        reports=[p for p in rules.all.input if isinstance(p, str) and p.endswith(".html")],
        meta=f"{OUTDIR}/{CROSSBASE}/sample_metadata.tsv",
    output:
        directory(f"{STATIC}/reports"),
    params:
        root=OUTDIR,
        script=str(SCRIPTS / "static_reports.py"),
    shell:
        r"""
        pixi run -e default python {params.script:q} --root {params.root:q} --meta {input.meta:q} \
          --reports {input.reports:q} --outdir {output:q}
        """
