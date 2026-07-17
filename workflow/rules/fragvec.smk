# Fragment-vector branch: the TWO-AXIS quantification (abundance + incidence).
#
#   decode ─► fragvec (per sample, fragvec/<s>/) ─► fragvec_matrix (per group) ─► report_fragvec_cross
#
# A SIBLING of the umi/ branch — both hang off `decode` and are independently switchable, so a user
# runs either or both. The fragment vector (a read's sorted PAYLOAD fragment_ids) is the library
# member and is where the power is: ~4-10 reads/vector against ~1.0 reads/UMI. Counting BOTH n_reads
# (abundance) and n_umis (incidence = independent clones) is what separates reproducible selection
# from a single amplified lineage. Enabled by config `fragvec.enabled`.
#
# `fragvec.umi_source` picks the clone label: `corrected` reuses the umi/ branch's ed<=1-corrected UMI
# (requires umi.mode: correct); `raw` reads the decode UMI, so this branch can stand alone.


def _fv_src_stem(wc):
    """The RAW decode stem feeding a fragvec/ stem, with the group<->tree guard (cf. _umi_src_stem).

    Deliberately raw decode, NOT decode_src_stem(): this supplies umis.parquet for the anchor
    VOCABULARY only (anchor_set() takes the distinct anchor_name values), never a read set. A
    vocabulary read off the unfiltered tree cannot shrink just because a coverage filter happened to
    drop every read carrying some anchor."""
    want = fvbase(wc.sample)
    if wc.fvbase != want:
        raise ValueError(
            f"sample {wc.sample!r} (group {SAMPLE[wc.sample]['group']!r}) belongs in "
            f"{want}/, not {wc.fvbase}/")
    src = "decode_initial" if wc.fvbase == "fragvec_initial" else "decode"
    return f"{OUTDIR}/{src}/{wc.sample}/{wc.sample}"


# The per-read clone label (UMI) and the fragment calls BOTH come from the clean `umi/` tree -- the
# end of the UMI spine, after correction and the jackpot filter. That tree carries the fragment calls
# through unchanged (filtered to its read set), so fragvec sees ONE consistent read set and never
# re-derives the payload filter.
#
# There is no `raw` alternative any more. It used to let this branch stand alone on decode/, but
# standing alone now means quantifying clonotypes the pipeline has just declared artefacts, labelled
# with an uncorrected UMI. common.smk rejects the key rather than silently honouring it.
#
# The anchor vocabulary still comes from the decode umis.parquet (the umi/ tree does not re-emit it).
FV_UMI_COL = "fused_seq_corrected"


def _fv_data_stem(s):
    """The stem supplying fragments.parquet + the fused parquet: the clean umi/ tree."""
    return umi_stem(s)


rule fragvec:
    input:
        frags=lambda wc: f"{_fv_data_stem(wc.sample)}.fragments.parquet",
        umis=lambda wc: f"{_fv_src_stem(wc)}.umis.parquet",     # anchors (always from decode)
        umi_fused=lambda wc: f"{_fv_data_stem(wc.sample)}.fused.parquet",
    output:
        counts=f"{OUTDIR}/{{fvbase}}/{{sample}}/{{sample}}.fragvec_counts.parquet",
        clones=f"{OUTDIR}/{{fvbase}}/{{sample}}/{{sample}}.fragvec_clones.parquet",
        subvec=f"{OUTDIR}/{{fvbase}}/{{sample}}/{{sample}}.subvec_counts.parquet",
        reads=f"{OUTDIR}/{{fvbase}}/{{sample}}/{{sample}}.fragvec_reads.parquet",
        summary=f"{OUTDIR}/{{fvbase}}/{{sample}}/{{sample}}.fragvec_summary.tsv",
    wildcard_constraints:
        fvbase="fragvec|fragvec_initial",
    params:
        src_stem=lambda wc: os.path.abspath(_fv_data_stem(wc.sample)),   # fragments + fused live here
        umis=lambda wc: os.path.abspath(f"{_fv_src_stem(wc)}.umis.parquet"),  # anchors (decode)
        dir=lambda wc: os.path.abspath(f"{OUTDIR}/{wc.fvbase}/{wc.sample}"),
        umi_fused=lambda wc: os.path.abspath(f"{_fv_data_stem(wc.sample)}.fused.parquet"),
        umi_col=FV_UMI_COL,
        minreads=FRAGVEC_MIN_READS,
        script=str(SCRIPTS / "build_fragvec.py"),
    shell:
        r"""
        set -euo pipefail
        rm -rf "{params.dir}"
        pixi run -e default python "{params.script}" \
          --src-stem "{params.src_stem}" \
          --umis "{params.umis}" \
          --umi-fused "{params.umi_fused}" \
          --umi-col "{params.umi_col}" \
          --dst-dir "{params.dir}" \
          --min-reads {params.minreads} \
          --force
        """


# One vector x sample matrix per group (GROUPS = every group but the R0 baseline -- the `empty` arm
# is a PEER and gets its own matrix), R0 initial library as the FIRST column. Both axes are emitted
# (.reads.mtx = abundance, .umis.mtx = incidence).
rule fragvec_matrix:
    input:
        counts=lambda wc: [f"{fragvec_stem(s)}.fragvec_counts.parquet"
                           for s in group_samples(wc.fvgroup)],
    output:
        longp=f"{OUTDIR}/fragvec/{{fvgroup}}.fragvec_x_sample.long.parquet",
        reads=f"{OUTDIR}/fragvec/{{fvgroup}}.fragvec_x_sample.reads.mtx",
        umis=f"{OUTDIR}/fragvec/{{fvgroup}}.fragvec_x_sample.umis.mtx",
        vectors=f"{OUTDIR}/fragvec/{{fvgroup}}.fragvec_x_sample.vectors.tsv",
        samples=f"{OUTDIR}/fragvec/{{fvgroup}}.fragvec_x_sample.samples.tsv",
    wildcard_constraints:
        # GROUPS_RE, not `"|".join(GROUPS)`: with no non-baseline groups the latter is the EMPTY regex,
        # which MATCHES THE EMPTY STRING, making `.fragvec_x_sample.reads.mtx` matchable. See common.smk.
        fvgroup=GROUPS_RE,
    params:
        samples=lambda wc: ",".join(group_samples(wc.fvgroup)),
        counts=lambda wc: ",".join(f"{fragvec_stem(s)}.fragvec_counts.parquet"
                                   for s in group_samples(wc.fvgroup)),
        out_prefix=lambda wc: f"{OUTDIR}/fragvec/{wc.fvgroup}.fragvec_x_sample",
        script=str(SCRIPTS / "build_fragvec_matrix.py"),
    shell:
        r"""
        pixi run -e default python "{params.script}" \
          --group "{wildcards.fvgroup}" \
          --samples "{params.samples}" \
          --counts "{params.counts}" \
          --out-prefix "{params.out_prefix}"
        """


def fragvec_report_inputs(wildcards):
    ins = []
    for s in NONINITIAL:
        ins += [f"{fragvec_stem(s)}.fragvec_counts.parquet", f"{fragvec_stem(s)}.fragvec_summary.tsv",
                f"{fragvec_stem(s)}.fragvec_clones.parquet"]   # clone sets -> cross-arm stratification
    if config["report"]["r0_baseline"]:
        ins += [f"{fragvec_stem(INITIAL)}.fragvec_counts.parquet",
                f"{fragvec_stem(INITIAL)}.fragvec_summary.tsv",
                f"{fragvec_stem(INITIAL)}.fragvec_clones.parquet"]
    ins += [f"{OUTDIR}/fragvec/{g}.fragvec_x_sample.long.parquet" for g in GROUPS]
    return ins


rule report_fragvec_cross:
    input:
        fragvec_report_inputs,
        meta=f"{OUTDIR}/{CROSSBASE}/sample_metadata.tsv",
    output:
        f"{OUTDIR}/fragvec/fragvec_report_cross.html",
    params:
        base=os.path.abspath(f"{OUTDIR}/fragvec"),
        r0_stem=(os.path.abspath(fragvec_stem(INITIAL)) if config["report"]["r0_baseline"] else ""),
        minreads=FRAGVEC_MIN_READS,
        script=str(SCRIPTS / "build_fragvec_report_cross.py"),
    shell:
        r"""
        L2F_BASE="{params.base}" L2F_R0_STEM="{params.r0_stem}" L2F_SAMPLES="{input.meta}" \
        L2F_MIN_READS={params.minreads} \
            pixi run -e default python "{params.script}"
        """
