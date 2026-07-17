# Jackpot filter: DETECT on the R0 library, APPLY everywhere. The end of the UMI spine.
#
#   umicorr/ ─► jackpot_detect (R0 ONLY) ─► blacklist + 2 audit reports
#                     └─► jackpot_apply (every sample) ─► umi/ ─┬─► featurecount (clean fragment report)
#                                                               ├─► umi_matrix, umi_complexity
#                                                               └─► fragvec
#
# A jackpot is a clonotype whose reads collapse onto very few UMIs: thousands of physical copies of
# one construct whose UMI slot failed to randomise, so they pile onto ~one barcode. Ranked by raw
# abundance it looks like a top hit; behaviourally it is an amplification/synthesis artefact.
#
# WHY DETECT-ON-R0 / APPLY-EVERYWHERE, rather than per-sample detection. The ratio n_reads/n_umis
# has two causes: one molecule amplified into thousands of reads (artefact), or a clone genuinely
# enriched by selection (a real hit). Only in the naive R0 library, where every construct starts at
# comparable abundance, is the second cause excluded by construction -- so only there is the
# statistic identifiable. Detecting on a selected round would blacklist true winners. Hence the
# fan-in/fan-out: one blacklist, derived once, applied to all.
#
# The converse limit, worth knowing before over-reading the output: jackpotting is worst when FEW
# templates enter the PCR, so R0 is the LEAST susceptible sample in the study. This catches the
# synthesis/aliquot artefacts already in the pool; it does NOT catch a jackpot generated inside a
# bottlenecked selected round.
#
# This is a FILTER, not a deduplication. clarification.md:667 forbids collapsing -- the typical
# molecule is seen about once and high-count lineages ARE the signal. Dropping a named, audited set
# of artefact barcodes is a different act, and the two audit reports are what carry that burden.
#
# The blacklist is the train_hmm shape: ONE wildcard-free artifact derived from INITIAL, listed as a
# constant input by every per-sample job. JACKPOT_BL is defined in common.smk beside it.


rule jackpot_detect:
    input:
        # the R0 library's CORRECTED tree -- the clonotype table is built from these two via
        # fragvec_lib, NOT read from fragvec/ (which is downstream of this rule; reading it would
        # be a cycle, and is exactly why this step used to be a post-hoc retro-fit)
        frags=f"{umicorr_stem(INITIAL)}.fragments.parquet",
        fused=f"{umicorr_stem(INITIAL)}.fused.parquet",
        umis=f"{stem(INITIAL)}.umis.parquet",     # anchor vocabulary, from raw decode
    output:
        blacklist=JACKPOT_BL,
        summary=f"{OUTDIR}/jackpot/{INITIAL}.umi_blacklist_summary.tsv",
        fv_report=f"{OUTDIR}/jackpot/{INITIAL}.fragvec_jackpot_report.html",
        umi_report=f"{OUTDIR}/jackpot/{INITIAL}.umi_jackpot_report.html",
    params:
        sample=INITIAL,
        src_stem=os.path.abspath(umicorr_stem(INITIAL)),
        umis=os.path.abspath(f"{stem(INITIAL)}.umis.parquet"),
        out_dir=os.path.abspath(f"{OUTDIR}/jackpot"),
        # MUST be fragvec.min_reads: the clone table the blacklist is built from is restricted by
        # that same floor, so an independent one here would silently drop flagged clonotypes.
        minreads=FRAGVEC_MIN_READS,
        ratiomax=JACKPOT_RATIO_MAX,
        robustk=JACKPOT_ROBUST_K,
        script=str(SCRIPTS / "detect_umi_jackpots.py"),
    shell:
        # --force unconditionally: the script refuses to overwrite, and Snakemake reruns always find
        # their own prior outputs. No `rm -rf` -- the dir is shared with nothing per-sample here,
        # but it is flat, so wiping it would take out a rerun's siblings.
        r"""
        set -euo pipefail
        pixi run -e default python "{params.script}" \
          --sample "{params.sample}" \
          --src-stem "{params.src_stem}" \
          --umis "{params.umis}" \
          --out-dir "{params.out_dir}" \
          --min-reads {params.minreads} \
          --ratio-max {params.ratiomax} \
          --robust-k {params.robustk} \
          --force
        """


def _umi_src_stem(wc):
    """The umicorr stem feeding a umi/ stem, with the group<->tree guard (cf. _umicorr_src_stem)."""
    want = umibase(wc.sample)                        # group -> umi | umi_initial
    if wc.umibase != want:
        raise ValueError(
            f"sample {wc.sample!r} (group {SAMPLE[wc.sample]['group']!r}) belongs in "
            f"{want}/, not {wc.umibase}/")
    return umicorr_stem(wc.sample)


# Every sample -- INCLUDING R0. The artefacts are IN the naive library; that is where they were
# found. Leaving them there would corrupt the very baseline every trajectory is measured against.
rule jackpot_apply:
    input:
        fused=lambda wc: f"{_umi_src_stem(wc)}.fused.parquet",
        frags=lambda wc: f"{_umi_src_stem(wc)}.fragments.parquet",
        umis=lambda wc: f"{_umi_src_stem(wc)}.umis.parquet",
        log=lambda wc: f"{_umi_src_stem(wc)}.ligateddna.log",
        summary=lambda wc: f"{_umi_src_stem(wc)}.umi_summary.tsv",
        blacklist=JACKPOT_BL,
    output:
        fused=f"{OUTDIR}/{{umibase}}/{{sample}}/{{sample}}.fused.parquet",
        frags=f"{OUTDIR}/{{umibase}}/{{sample}}/{{sample}}.fragments.parquet",
        umis=f"{OUTDIR}/{{umibase}}/{{sample}}/{{sample}}.umis.parquet",
        log=f"{OUTDIR}/{{umibase}}/{{sample}}/{{sample}}.ligateddna.log",
        species=f"{OUTDIR}/{{umibase}}/{{sample}}/{{sample}}.umi_species.tsv",
        summary=f"{OUTDIR}/{{umibase}}/{{sample}}/{{sample}}.umi_summary.tsv",
    wildcard_constraints:
        umibase="umi|umi_initial",
    params:
        src_stem=lambda wc: os.path.abspath(_umi_src_stem(wc)),
        dir=lambda wc: os.path.abspath(f"{OUTDIR}/{wc.umibase}/{wc.sample}"),
        ed1_flag="--match-ed1" if JACKPOT_MATCH_ED1 else "",
        script=str(SCRIPTS / "apply_jackpot_blacklist.py"),
    shell:
        r"""
        set -euo pipefail
        rm -rf "{params.dir}"
        pixi run -e default python "{params.script}" \
          --src-stem "{params.src_stem}" \
          --dst-dir "{params.dir}" \
          --blacklist "{input.blacklist}" \
          {params.ed1_flag} \
          --force
        """
