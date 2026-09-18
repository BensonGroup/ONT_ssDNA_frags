import shlex


def static_plot_targets():
    """Optional rule-all outputs, evaluated after the rule modules are loaded."""
    if _cfg_bool(config.get("static_plot"), False, key="static_plot"):
        return [f"{OUTDIR}/static_plots"]
    return []


WITHIN_TABLES = f"{OUTDIR}/within_batch_overlap"
WITHIN_CONFIG = config.get("static_overlap", {}) or {}
WITHIN_DEPTHS = WITHIN_CONFIG.get("depths", [5000, 26000, 55000])
if not isinstance(WITHIN_DEPTHS, list) or not WITHIN_DEPTHS:
    raise ValueError("config.static_overlap.depths must be a nonempty list of positive integers")
WITHIN_DEPTHS = [_cfg_int(value, None, key="static_overlap.depths", lo=1) for value in WITHIN_DEPTHS]

rule within_batch_overlap:
    input:
        counts=[f"{fragvec_stem(s)}.fragvec_counts.parquet" for s in SAMPLES],
        reads=[f"{fragvec_stem(s)}.fragvec_reads.parquet" for s in SAMPLES],
        summaries=[f"{fragvec_stem(s)}.fragvec_summary.tsv" for s in SAMPLES],
        species=[f"{umi_stem(s)}.umi_species.tsv" for s in SAMPLES],
        meta=f"{OUTDIR}/{CROSSBASE}/sample_metadata.tsv",
        code=[str(SCRIPTS / "static_plots/within_batch_overlap.py"), str(SCRIPTS / "complexity_lib.py")],
    output:
        directory(WITHIN_TABLES),
    params:
        source=OUTDIR,
        base=CROSSBASE,
        depths=",".join(map(str, WITHIN_DEPTHS)),
        reps=_cfg_int(WITHIN_CONFIG.get("reps"), 20, key="static_overlap.reps", lo=1),
        perm=_cfg_int(WITHIN_CONFIG.get("permutations"), 500, key="static_overlap.permutations", lo=1),
        seed=_cfg_int(WITHIN_CONFIG.get("seed"), 0, key="static_overlap.seed", lo=0),
        script=str(SCRIPTS / "static_plots/within_batch_overlap.py"),
    threads: 1
    shell:
        r"""
        POLARS_MAX_THREADS={threads} pixi run -e default python {params.script:q} \
          --resdir {params.source:q} --outdir {output:q} --metadata {input.meta:q} \
          --umi-base {params.base:q} --depths {params.depths:q} --reps {params.reps} \
          --perm {params.perm} --seed {params.seed}
        """


# Reuse rule all's declared HTML reports so configuration and export cannot drift.
# Do not glob outdir, which could contain stale reports or the exporter's own index.
rule static_plots:
    input:
        reports=[path for path in rules.all.input
                 if isinstance(path, str) and path.endswith(".html")],
        overlap=([WITHIN_TABLES] if FRAGVEC_ON else []),
        meta=f"{OUTDIR}/{CROSSBASE}/sample_metadata.tsv",
        code=[str(SCRIPTS / "static_plots" / name) for name in (
            "export_reports.py", "html_report_plotting.py", "umi_report_cross_figs.py",
            "palettes.py", "science.mplstyle", "index.css", "index_template.html",
            "render_within_batch.py", "within_batch_overlap_figs.py", "venn_geometry.py")],
    output:
        directory(f"{OUTDIR}/static_plots"),
    params:
        source=OUTDIR,
        within_args=("--within-batch " + shlex.quote(WITHIN_TABLES) if FRAGVEC_ON else ""),
        script=str(SCRIPTS / "static_plots/export_reports.py"),
    threads: 1
    shell:
        r"""
        pixi run -e default python {params.script:q} --source {params.source:q} \
          --outdir {output:q} --metadata {input.meta:q} {params.within_args} --reports {input.reports:q}
        """
