# Per-read feature counts (from scripts/07). The vetted python script is unmodified and driven
# by the L2F_STEM env var pointing at an ABSOLUTE decode stem, so its parents[1] fallback never
# fires and `import ligand_vocab` resolves from the script's own dir.
rule featurecount:
    input:
        frags=f"{OUTDIR}/{{decbase}}/{{sample}}/{{sample}}.fragments.parquet",
        umis=f"{OUTDIR}/{{decbase}}/{{sample}}/{{sample}}.umis.parquet",
    output:
        fc=f"{OUTDIR}/{{decbase}}/{{sample}}/{{sample}}.featurecount.parquet",
        summ=f"{OUTDIR}/{{decbase}}/{{sample}}/{{sample}}.featurecount.summary.tsv",
    params:
        stem=lambda wc: os.path.abspath(f"{OUTDIR}/{wc.decbase}/{wc.sample}/{wc.sample}"),
        script=str(SCRIPTS / "build_featurecounts.py"),
    shell:
        r"""
        L2F_STEM="{params.stem}" pixi run -e default python "{params.script}"
        """


# Optional validation (scripts/07 --check): not in `rule all`. Run `snakemake check_all`.
rule featurecount_check:
    input:
        fc=f"{OUTDIR}/{{decbase}}/{{sample}}/{{sample}}.featurecount.parquet",
        frags=f"{OUTDIR}/{{decbase}}/{{sample}}/{{sample}}.fragments.parquet",
    output:
        f"{OUTDIR}/{{decbase}}/{{sample}}/{{sample}}.featurecount.check.txt",
    params:
        stem=lambda wc: os.path.abspath(f"{OUTDIR}/{wc.decbase}/{wc.sample}/{wc.sample}"),
        script=str(SCRIPTS / "build_featurecounts.py"),
    shell:
        r"""
        L2F_STEM="{params.stem}" pixi run -e default python "{params.script}" --check > "{output}" 2>&1
        """


rule check_all:
    input:
        [f"{stem(s)}.featurecount.check.txt" for s in SAMPLES],
