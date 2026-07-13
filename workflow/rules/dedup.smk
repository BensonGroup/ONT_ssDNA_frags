# UMI garbage-filter (+ optional starcode dedup) (from scripts/11), between decode and featurecount.
# Every read is classified from its fused UMI sequence and garbage UMIs are dropped. By DEFAULT that
# is all this stage does — every surviving read is kept (this data must not be UMI-collapsed). In
# config `umi.mode: dedup`, the rule additionally passes `--collapse`, clustering the survivors with
# `starcode -d 1 -s` (spheres, Levenshtein distance 1) and keeping one representative read per
# cluster (highest whole-read base-quality sum, the Picard MarkDuplicates convention). The separate
# ed<=1 UMI *correction* branch (mode `correct`) lives in rules/umi.smk, not here.
#
# The dedup tree is a SIBLING of the decode tree (dedup/ <- decode/, dedup_initial/ <-
# decode_initial/) and carries identical basenames, so it is a drop-in filter: featurecount and
# report_sample run against either tree unchanged. Keeping the initial library in its own tree
# matters — report_cross discovers samples with a `<base>/*/` directory glob, so a flattened tree
# would count the R0 baseline as both a column and the baseline.
#
# QC lands in the SAME directory as the data it describes (`--qc-dir` == `--dst-dir`), matching the
# convention elsewhere in the workflow (`<s>.featurecount_report.html` sits next to
# `<s>.featurecount.parquet`). That is also what keeps this a single rule: Snakemake requires every
# output of a rule to carry the same wildcards, and a shared flat qc/ dir would leave the QC paths
# without a {dedbase} to infer. umi_dedup.py is safe with the two dirs equal — it calls prepare_dst()
# (which wipes) before qc_dir.mkdir() and before the first QC write.
#
# `truncated.fq` is a pass-through symlink into the decode tree that umi_dedup.py writes only when
# the source has one. It is left UNDECLARED (as in `rule decode`): declaring a symlink output
# invites Snakemake's cleanup to follow it and delete the decode tree's real file.
#
# `rm -rf {params.dir}` also removes the featurecount / report_sample outputs that land in this
# same directory later. That is correct: if dedup re-runs, its outputs changed, so those downstream
# jobs must be rebuilt anyway and Snakemake schedules them.

# Basenames the dedup step filters through, in -> out.
_DEDUP_PARTS = [".fused.parquet", ".trimmed.parquet", ".umis.parquet",
                ".fragments.parquet", ".fused_reads.fq", ".ligateddna.log"]


def _dedup_src_stem(wc):
    """The raw decode stem feeding a dedup stem, and the guard that keeps a sample out of the wrong
    tree. `wildcard_constraints` can pin `dedbase` and `sample` independently but cannot express
    "sample must match dedbase"; without this check `results/dedup/<initial>/…` would happily build
    the R0 baseline into the experimental tree, where report_cross's `dedup/*/` glob would then
    count it twice (once as a column, once as the baseline)."""
    want = dedbase(wc.sample)                     # role -> dedup | dedup_initial
    if wc.dedbase != want:
        raise ValueError(
            f"sample {wc.sample!r} (role {SAMPLE[wc.sample]['role']!r}) belongs in "
            f"{want}/, not {wc.dedbase}/")
    src = "decode_initial" if wc.dedbase == "dedup_initial" else "decode"
    return f"{OUTDIR}/{src}/{wc.sample}/{wc.sample}"


rule dedup:
    input:
        **{f"in{i}": (lambda wc, p=p: f"{_dedup_src_stem(wc)}{p}")
           for i, p in enumerate(_DEDUP_PARTS)},
    output:
        # the filtered mirror of the inputs …
        **{f"out{i}": f"{OUTDIR}/{{dedbase}}/{{sample}}/{{sample}}{p}"
           for i, p in enumerate(_DEDUP_PARTS)},
        # … and its QC, beside the data
        counts=f"{OUTDIR}/{{dedbase}}/{{sample}}/{{sample}}.umi_counts.tsv",
        clusters=f"{OUTDIR}/{{dedbase}}/{{sample}}/{{sample}}.umi_clusters.tsv",
        audit=f"{OUTDIR}/{{dedbase}}/{{sample}}/{{sample}}.umi_audit.parquet",
        summary=f"{OUTDIR}/{{dedbase}}/{{sample}}/{{sample}}.umi_dedup_summary.tsv",
    wildcard_constraints:
        dedbase="dedup|dedup_initial",
    threads: DEDUP_THREADS
    params:
        src_stem=lambda wc: os.path.abspath(_dedup_src_stem(wc)),
        dir=lambda wc: os.path.abspath(f"{OUTDIR}/{wc.dedbase}/{wc.sample}"),
        maxhp=DEDUP_MAX_HOMOPOLYMER,
        maxbf=DEDUP_MAX_BASE_FRAC,
        collapse_flag="--collapse" if DEDUP_ON else "",
        script=str(SCRIPTS / "umi_dedup.py"),
    shell:
        # umi_dedup.py refuses a non-empty --dst-dir; Snakemake leaves partial outputs behind on a
        # failed job, so wipe first (cf. train_hmm) and pass --force. `rm -rf` unlinks the stale
        # truncated.fq SYMLINK without touching the decode tree's real file.
        r"""
        set -euo pipefail
        rm -rf "{params.dir}"
        L2F_THREADS={threads} pixi run -e default python "{params.script}" \
          --src-stem "{params.src_stem}" \
          --dst-dir "{params.dir}" \
          --qc-dir "{params.dir}" \
          --threads {threads} \
          --max-homopolymer {params.maxhp} \
          --max-base-frac {params.maxbf} \
          {params.collapse_flag} \
          --force
        """


# Concatenate the per-sample dedup summary rows (parity with clean_summary / pychopper_summary).
# Each row is now a concrete path — dedup_stem() resolves the tree from the sample's role — so the
# rows can be depended on directly. The combined table is a cross-sample artifact, so it sits at the
# tree root next to the cross report, pulling the initial library's row from the sibling
# dedup_initial/ tree exactly as report_cross pulls its R0 baseline. Rows follow config order.
rule umi_dedup_summary:
    input:
        rows=[f"{dedup_stem(s)}.umi_dedup_summary.tsv" for s in SAMPLES],
    output:
        f"{OUTDIR}/dedup/umi_dedup_summary.tsv",
    run:
        with open(output[0], "w") as out:
            for i, p in enumerate(input.rows):
                lines = open(p).read().splitlines()
                if i == 0:
                    out.write(lines[0] + "\n")      # header, once
                for line in lines[1:]:
                    out.write(line + "\n")
