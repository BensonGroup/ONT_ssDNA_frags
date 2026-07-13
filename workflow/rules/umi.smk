# UMI characterisation branch (mode `correct`): QC-filter + ed<=1 UMI correction, a per-group
# corrected-UMI x sample abundance matrix, and a cross-sample UMI report. This branch is a SIBLING
# of the decode/dedup trees and never feeds featurecount -- the fragment reports are unaffected.
#
#   decode ─► umi_correct (per sample, umi/<s>/) ─► umi_matrix (per group) ─► report_umi_cross
#
# `umi_correct` reads the raw decode artifacts and does its own QC filter (reusing umi_filter_lib),
# so it is independent of `rule dedup`. It keeps every read that passes the filter and relabels its
# fused UMI to the abundant ed<=1 centroid (starcode -d 1 -s). Enabled only when UMI_CORRECT_ON.


def _umi_src_stem(wc):
    """The decode stem feeding a umi/ stem, with the role<->tree guard (mirrors _dedup_src_stem)."""
    want = umibase(wc.sample)                        # role -> umi | umi_initial
    if wc.umibase != want:
        raise ValueError(
            f"sample {wc.sample!r} (role {SAMPLE[wc.sample]['role']!r}) belongs in "
            f"{want}/, not {wc.umibase}/")
    src = "decode_initial" if wc.umibase == "umi_initial" else "decode"
    return f"{OUTDIR}/{src}/{wc.sample}/{wc.sample}"


rule umi_correct:
    input:
        fused=lambda wc: f"{_umi_src_stem(wc)}.fused.parquet",
        umis=lambda wc: f"{_umi_src_stem(wc)}.umis.parquet",
        # fragments drive the payload filter: anchor-only reads (no decoded ligand) are dropped
        frags=lambda wc: f"{_umi_src_stem(wc)}.fragments.parquet",
    output:
        fused=f"{OUTDIR}/{{umibase}}/{{sample}}/{{sample}}.fused.parquet",
        species=f"{OUTDIR}/{{umibase}}/{{sample}}/{{sample}}.umi_species.tsv",
        summary=f"{OUTDIR}/{{umibase}}/{{sample}}/{{sample}}.umi_summary.tsv",
    wildcard_constraints:
        umibase="umi|umi_initial",
    threads: DEDUP_THREADS
    params:
        src_stem=lambda wc: os.path.abspath(_umi_src_stem(wc)),
        dir=lambda wc: os.path.abspath(f"{OUTDIR}/{wc.umibase}/{wc.sample}"),
        maxhp=DEDUP_MAX_HOMOPOLYMER,
        maxbf=DEDUP_MAX_BASE_FRAC,
        script=str(SCRIPTS / "umi_correct.py"),
    shell:
        r"""
        set -euo pipefail
        rm -rf "{params.dir}"
        pixi run -e default python "{params.script}" \
          --src-stem "{params.src_stem}" \
          --dst-dir "{params.dir}" \
          --qc-dir "{params.dir}" \
          --max-homopolymer {params.maxhp} \
          --max-base-frac {params.maxbf} \
          --force
        """


# One corrected-UMI x sample matrix per experimental group: the R0 initial library is the FIRST
# column, then the group's rounds in order (group_samples()). Emits a long parquet + Matrix Market
# + row/col label files. No all-sample matrix.
rule umi_matrix:
    input:
        species=lambda wc: [f"{umi_stem(s)}.umi_species.tsv" for s in group_samples(wc.umigroup)],
    output:
        longp=f"{OUTDIR}/umi/{{umigroup}}.umi_x_sample.long.parquet",
        mtx=f"{OUTDIR}/umi/{{umigroup}}.umi_x_sample.mtx",
        umis=f"{OUTDIR}/umi/{{umigroup}}.umi_x_sample.umis.tsv",
        samples=f"{OUTDIR}/umi/{{umigroup}}.umi_x_sample.samples.tsv",
    wildcard_constraints:
        umigroup="|".join(re.escape(g) for g in GROUPS_EXP),
    params:
        samples=lambda wc: ",".join(group_samples(wc.umigroup)),
        tsvs=lambda wc: ",".join(f"{umi_stem(s)}.umi_species.tsv" for s in group_samples(wc.umigroup)),
        out_prefix=lambda wc: f"{OUTDIR}/umi/{wc.umigroup}.umi_x_sample",
        script=str(SCRIPTS / "build_umi_matrix.py"),
    shell:
        r"""
        pixi run -e default python "{params.script}" \
          --group "{wildcards.umigroup}" \
          --samples "{params.samples}" \
          --species-tsvs "{params.tsvs}" \
          --out-prefix "{params.out_prefix}"
        """


# Cross-sample UMI report: abundance (rank-abundance, D50/D90), species overlap (UpSet per group),
# and validity QC (UMI-group ligand-signature purity vs size, collision null model). Discovers
# experimental samples by globbing umi/*/; the R0 baseline comes from the sibling umi_initial/ tree
# via L2F_R0_STEM (as the featurecount cross report does). Decode fragments feed the purity QC.
def umi_report_inputs(wildcards):
    ins = []
    for s in NONINITIAL:
        ins += [f"{umi_stem(s)}.umi_species.tsv", f"{umi_stem(s)}.umi_summary.tsv",
                f"{umi_stem(s)}.fused.parquet", f"{stem(s)}.fragments.parquet"]
    if config["report"]["r0_baseline"]:
        ins += [f"{umi_stem(INITIAL)}.umi_species.tsv", f"{umi_stem(INITIAL)}.umi_summary.tsv",
                f"{umi_stem(INITIAL)}.fused.parquet", f"{stem(INITIAL)}.fragments.parquet"]
    return ins


rule report_umi_cross:
    input:
        umi_report_inputs,
        meta=f"{OUTDIR}/{CROSSBASE}/sample_metadata.tsv",
    output:
        f"{OUTDIR}/umi/umi_report_cross.html",
    params:
        base=os.path.abspath(f"{OUTDIR}/umi"),
        r0_stem=(os.path.abspath(umi_stem(INITIAL)) if config["report"]["r0_baseline"] else ""),
        decode_base=os.path.abspath(f"{OUTDIR}/decode"),
        r0_frags=(os.path.abspath(f"{stem(INITIAL)}.fragments.parquet")
                  if config["report"]["r0_baseline"] else ""),
        script=str(SCRIPTS / "build_umi_report_cross.py"),
    shell:
        r"""
        L2F_BASE="{params.base}" L2F_R0_STEM="{params.r0_stem}" L2F_SAMPLES="{input.meta}" \
        L2F_DECODE_BASE="{params.decode_base}" L2F_R0_FRAGS="{params.r0_frags}" \
            pixi run -e default python "{params.script}"
        """
