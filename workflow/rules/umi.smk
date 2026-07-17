# The UMI spine: QC-filter + ed<=1 correction (here), then the jackpot filter (rules/jackpot.smk).
#
#   covfilt ─► umi_correct (umicorr/<s>/) ─► jackpot ─► umi/<s>/ ─┬─► umi_matrix ─► report_umi_cross
#                                                                  ├─► umi_complexity (C_eff)
#                                                                  ├─► featurecount (clean fragment report)
#                                                                  └─► fragvec
#
# This is the ONLY UMI stage. It classifies every read from its fused UMI (umi_filter_lib -- the
# dependency-free, unit-tested classifier), drops the garbage, drops anchor-only reads, and relabels
# each survivor's UMI to its abundant ed<=1 centroid (starcode -d 1 -s). NOTHING is collapsed.
#
# Its output tree `umicorr/` is NOT a readout: it still carries the artefact barcodes the jackpot
# stage removes, so a C_eff computed on it would be corrupted (clarification.md:399-403). Only
# jackpot_detect and jackpot_apply read it.


def _umicorr_src_stem(wc):
    """The decode stem feeding a umicorr/ stem, with the group<->tree guard.

    decode_src_stem() resolves to covfilt/ when the coverage filter is on, so the UMI spine starts
    from the same read set the raw fragment report's funnel accounts for."""
    want = umicorrbase(wc.sample)                    # group -> umicorr | umicorr_initial
    if wc.umicorrbase != want:
        raise ValueError(
            f"sample {wc.sample!r} (group {SAMPLE[wc.sample]['group']!r}) belongs in "
            f"{want}/, not {wc.umicorrbase}/")
    return decode_src_stem(wc.sample)


rule umi_correct:
    input:
        fused=lambda wc: f"{_umicorr_src_stem(wc)}.fused.parquet",
        umis=lambda wc: f"{_umicorr_src_stem(wc)}.umis.parquet",
        # fragments drive the payload filter: anchor-only reads (no decoded ligand) are dropped
        frags=lambda wc: f"{_umicorr_src_stem(wc)}.fragments.parquet",
        log=lambda wc: f"{_umicorr_src_stem(wc)}.ligateddna.log",
    output:
        fused=f"{OUTDIR}/{{umicorrbase}}/{{sample}}/{{sample}}.fused.parquet",
        # decode fragment calls carried through UNCHANGED (filtered to kept reads): correction never
        # touches the fragment level, and start/stop are needed for the positional fragment vector
        frags=f"{OUTDIR}/{{umicorrbase}}/{{sample}}/{{sample}}.fragments.parquet",
        # the anchor VOCABULARY and the decode log: featurecount / report_sample find both by stem,
        # so a decode-shaped tree must carry them for the clean fragment report to build here
        umis=f"{OUTDIR}/{{umicorrbase}}/{{sample}}/{{sample}}.umis.parquet",
        log=f"{OUTDIR}/{{umicorrbase}}/{{sample}}/{{sample}}.ligateddna.log",
        species=f"{OUTDIR}/{{umicorrbase}}/{{sample}}/{{sample}}.umi_species.tsv",
        summary=f"{OUTDIR}/{{umicorrbase}}/{{sample}}/{{sample}}.umi_summary.tsv",
        # one row per INPUT read: a dropped read is still here, carrying the class that explains why.
        # Nothing else records that -- it moved here when the separate UMI-filter stage was folded in.
        audit=f"{OUTDIR}/{{umicorrbase}}/{{sample}}/{{sample}}.umi_audit.parquet",
    wildcard_constraints:
        umicorrbase="umicorr|umicorr_initial",
    threads: UMI_THREADS
    params:
        src_stem=lambda wc: os.path.abspath(_umicorr_src_stem(wc)),
        dir=lambda wc: os.path.abspath(f"{OUTDIR}/{wc.umicorrbase}/{wc.sample}"),
        maxhp=UMI_MAX_HOMOPOLYMER,
        maxbf=UMI_MAX_BASE_FRAC,
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


# One corrected-UMI x sample matrix per group (GROUPS = every group but the R0 baseline -- the
# `empty` arm is a PEER and gets its own matrix): the R0 initial library is the FIRST column, then
# the group's rounds in order (group_samples()). Emits a long parquet + Matrix Market + row/col
# label files. No all-sample matrix.
rule umi_matrix:
    input:
        species=lambda wc: [f"{umi_stem(s)}.umi_species.tsv" for s in group_samples(wc.umigroup)],
    output:
        longp=f"{OUTDIR}/umi/{{umigroup}}.umi_x_sample.long.parquet",
        mtx=f"{OUTDIR}/umi/{{umigroup}}.umi_x_sample.mtx",
        umis=f"{OUTDIR}/umi/{{umigroup}}.umi_x_sample.umis.tsv",
        samples=f"{OUTDIR}/umi/{{umigroup}}.umi_x_sample.samples.tsv",
    wildcard_constraints:
        # GROUPS_RE, not `"|".join(GROUPS)`: with no non-baseline groups the latter is the EMPTY regex,
        # which MATCHES THE EMPTY STRING, making `results/umi/.umi_x_sample.mtx` matchable. See common.smk.
        umigroup=GROUPS_RE,
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


# Depth-robust effective complexity: C_eff = 1 / P(two random distinct reads share a molecule), the
# inverse-Simpson complexity of each sample's corrected-UMI pool, plus the COLLAPSE of that
# complexity relative to the R0 library. Unlike a distinct-UMI count -- which just grows with
# sequencing depth -- C_eff is a property of the POOL, so it is comparable across samples whose
# depth spans ~6x here. See complexity_lib.py for the estimator and clarification.md for the theory.
#
# These TSVs are FIRST-CLASS OUTPUTS, not a report internal: `report_umi_cross` renders them and
# recomputes nothing, so a number in the summary table and the same number in a figure cannot drift
# apart. Inspectable without opening the HTML.
#
# Reads only the per-molecule counts `umi_correct` already wrote (.umi_species.tsv). It does NOT
# depend on `umi_matrix` -- nothing in the estimator reads the umi_x_sample.* files.
#
# The INITIAL library is requested UNCONDITIONALLY, unlike report_umi_cross below: `report.r0_baseline`
# is a report DISPLAY toggle, whereas the collapse is DEFINED relative to R0.
rule umi_complexity:
    input:
        species=[f"{umi_stem(s)}.umi_species.tsv" for s in [INITIAL] + NONINITIAL],
        meta=f"{OUTDIR}/{CROSSBASE}/sample_metadata.tsv",
    output:
        tsv=f"{OUTDIR}/umi/umi_complexity.tsv",
        collapse=f"{OUTDIR}/umi/umi_collapse.tsv",
    log:
        f"{OUTDIR}/umi/umi_complexity.log",
    params:
        # NAME=PATH pairs; the script keys group/round off NAME via the metadata TSV
        species=[f"{s}={os.path.abspath(umi_stem(s))}.umi_species.tsv"
                 for s in [INITIAL] + NONINITIAL],
        baseline=INITIAL,
        script=str(SCRIPTS / "effective_complexity.py"),
    shell:
        r"""
        set -euo pipefail
        pixi run -e default python "{params.script}" \
          --species {params.species:q} \
          --meta "{input.meta}" \
          --baseline "{params.baseline}" \
          --out-tsv "{output.tsv}" \
          --out-collapse "{output.collapse}" 2>&1 | tee "{log}"
        """


# Cross-sample UMI report: effective complexity (C_eff trajectory + collapse), abundance
# (rank-abundance, D50/D90), species overlap (UpSet per group), and validity QC (UMI-group
# ligand-signature purity vs size, collision null model). Every figure is coloured by the config
# `group` (cell line). Discovers non-initial samples by globbing umi/*/; the R0 baseline comes from
# the sibling umi_initial/ tree via L2F_R0_STEM (as the featurecount cross report does). Decode
# fragments feed the purity QC.
#
# The complexity TSVs are declared inputs, not just transitively available: the report RENDERS them,
# so rebuilding it against a stale table would silently publish stale numbers.
def umi_report_inputs(wildcards):
    # Fragments for the purity QC come from decode_src_stem(), not stem(): a read the coverage
    # filter dropped must not contribute a ligand signature to a UMI group it is no longer part of.
    ins = []
    for s in NONINITIAL:
        ins += [f"{umi_stem(s)}.umi_species.tsv", f"{umi_stem(s)}.umi_summary.tsv",
                f"{umi_stem(s)}.fused.parquet", f"{decode_src_stem(s)}.fragments.parquet"]
    if config["report"]["r0_baseline"]:
        ins += [f"{umi_stem(INITIAL)}.umi_species.tsv", f"{umi_stem(INITIAL)}.umi_summary.tsv",
                f"{umi_stem(INITIAL)}.fused.parquet",
                f"{decode_src_stem(INITIAL)}.fragments.parquet"]
    return ins


rule report_umi_cross:
    input:
        umi_report_inputs,
        meta=f"{OUTDIR}/{CROSSBASE}/sample_metadata.tsv",
        cplx=f"{OUTDIR}/umi/umi_complexity.tsv",
        collapse=f"{OUTDIR}/umi/umi_collapse.tsv",
    output:
        f"{OUTDIR}/umi/umi_report_cross.html",
    params:
        base=os.path.abspath(f"{OUTDIR}/umi"),
        r0_stem=(os.path.abspath(umi_stem(INITIAL)) if config["report"]["r0_baseline"] else ""),
        decode_base=os.path.abspath(f"{OUTDIR}/{DECODE_SRC_BASE}"),
        r0_frags=(os.path.abspath(f"{decode_src_stem(INITIAL)}.fragments.parquet")
                  if config["report"]["r0_baseline"] else ""),
        script=str(SCRIPTS / "build_umi_report_cross.py"),
    shell:
        r"""
        L2F_BASE="{params.base}" L2F_R0_STEM="{params.r0_stem}" L2F_SAMPLES="{input.meta}" \
        L2F_DECODE_BASE="{params.decode_base}" L2F_R0_FRAGS="{params.r0_frags}" \
        L2F_COMPLEXITY="{input.cplx}" L2F_COLLAPSE="{input.collapse}" \
            pixi run -e default python "{params.script}"
        """
