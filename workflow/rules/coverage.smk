# Decode coverage: measure it on decode, then filter on it before every analysis branch.
#
#   decode ─► coverage (<s>.coverage.parquet, IN the decode tree)
#               ├─► covfilt (per sample, covfilt/<s>/) ─► umi_correct ─► jackpot ─► umi/
#               └─► report_cross's coverage panel (renders it; recomputes nothing)
#
# A read's decoded coverage is the fraction of it the decoder explained (covered bases / post-trim
# read length; see coverage_lib.py, the single source of truth for the metric). Reads below config
# `decode_filter.min_coverage` are DROPPED. This is the only read-level drop upstream of the fork,
# which is the whole point: the whole UMI spine starts from the covfilt tree, so every readout
# hanging off its clean end shares one read set.
#
# WHY `rule coverage` IS SEPARATE, AND ALWAYS RUNS. Coverage is a property of the DECODE step, so it
# is measured once, there, whether or not anything filters on it. The report renders that PRE-filter
# record. Folding the measurement into `rule covfilt` would mean the panel had nothing to read when
# the filter is off, and would tempt it into recomputing on whatever tree it happens to hold -- which
# is the clean umi/ tree, DOWNSTREAM of the filter, whose distribution is truncated at the
# threshold. A decode-level metric must never be measured on a UMI-stage output.
#
# The covfilt tree is a SIBLING of the decode tree (covfilt/ <- decode/, covfilt_initial/ <-
# decode_initial/) carrying identical basenames, so it is a drop-in source for the branches exactly
# as umi/ is a drop-in mirror for featurecount. Keeping the initial library in its own tree
# matters for the same reason it does there: report_cross discovers samples with a `<base>/*/` glob,
# so a flattened tree would count the R0 baseline as both a column and the baseline.
#
# When DECODE_FILTER_ON is false, covfilt is never requested -- decode_src_stem() resolves the
# branches straight back to decode/ -- but `rule coverage` still runs, because the panel needs it.
#
# `truncated.fq` is a pass-through symlink into the decode tree, left UNDECLARED for the same reason
# as in `rule decode`: declaring a symlink output invites Snakemake's cleanup to
# follow it and delete the decode tree's real file.
#
# `rm -rf {params.dir}` is required because filter_decode_coverage.py refuses a non-empty --dst-dir;
# Snakemake leaves partial outputs behind on a failed job (cf. rule jackpot_apply).


# Measured INTO the decode tree, beside the fragments it describes -- the same thing `rule
# featurecount` does with .featurecount.parquet. The per-rule `decbase` narrowing is mandatory:
# common.smk's shared alias spans four trees, so without it this rule would also claim
# umi/<s>/<s>.coverage.parquet and collide (cf. the same narrowing on `rule decode`).
rule coverage:
    input:
        frags=f"{OUTDIR}/{{decbase}}/{{sample}}/{{sample}}.fragments.parquet",
        fq=f"{OUTDIR}/{{decbase}}/{{sample}}/{{sample}}.fused_reads.fq",
    output:
        cov=f"{OUTDIR}/{{decbase}}/{{sample}}/{{sample}}.coverage.parquet",
    wildcard_constraints:
        decbase="decode|decode_initial",
    params:
        stem=lambda wc: os.path.abspath(f"{OUTDIR}/{wc.decbase}/{wc.sample}/{wc.sample}"),
        script=str(SCRIPTS / "build_coverage.py"),
    shell:
        r"""
        set -euo pipefail
        pixi run -e default python "{params.script}" --src-stem "{params.stem}"
        """


# Basenames the coverage filter passes through, in -> out.
_COVFILT_PARTS = [".fused.parquet", ".trimmed.parquet", ".umis.parquet",
                  ".fragments.parquet", ".fused_reads.fq", ".ligateddna.log"]


def _covfilt_src_stem(wc):
    """The raw decode stem feeding a covfilt stem, with the group<->tree guard (mirrors
    _umi_src_stem: wildcard_constraints can pin `covbase` and `sample` independently but cannot
    express "sample must match covbase")."""
    want = covbase(wc.sample)                        # group -> covfilt | covfilt_initial
    if wc.covbase != want:
        raise ValueError(
            f"sample {wc.sample!r} (group {SAMPLE[wc.sample]['group']!r}) belongs in "
            f"{want}/, not {wc.covbase}/")
    src = "decode_initial" if wc.covbase == "covfilt_initial" else "decode"
    return f"{OUTDIR}/{src}/{wc.sample}/{wc.sample}"


rule covfilt:
    input:
        **{f"in{i}": (lambda wc, p=p: f"{_covfilt_src_stem(wc)}{p}")
           for i, p in enumerate(_COVFILT_PARTS)},
        # the verdict this rule acts on -- measured upstream, never recomputed here
        cov=lambda wc: f"{_covfilt_src_stem(wc)}.coverage.parquet",
    output:
        # the filtered mirror of the inputs …
        **{f"out{i}": f"{OUTDIR}/{{covbase}}/{{sample}}/{{sample}}{p}"
           for i, p in enumerate(_COVFILT_PARTS)},
        # … and its QC, beside the data
        summary=f"{OUTDIR}/{{covbase}}/{{sample}}/{{sample}}.coverage_summary.tsv",
    wildcard_constraints:
        covbase="covfilt|covfilt_initial",
    params:
        src_stem=lambda wc: os.path.abspath(_covfilt_src_stem(wc)),
        dir=lambda wc: os.path.abspath(f"{OUTDIR}/{wc.covbase}/{wc.sample}"),
        mincov=DECODE_MIN_COVERAGE,
        script=str(SCRIPTS / "filter_decode_coverage.py"),
    shell:
        r"""
        set -euo pipefail
        rm -rf "{params.dir}"
        pixi run -e default python "{params.script}" \
          --src-stem "{params.src_stem}" \
          --dst-dir "{params.dir}" \
          --min-coverage {params.mincov} \
          --force
        """


# Concatenate the per-sample coverage rows (parity with clean_summary / pychopper_summary). Each row
# is a concrete path -- cov_stem() resolves the tree from the sample's group -- so the rows can be
# depended on directly. The combined table sits at the tree root, pulling the initial library's row
# from the sibling covfilt_initial/ tree. Rows follow config order. Schema-agnostic: it copies
# whatever header row 0 carries.
rule covfilt_summary:
    input:
        rows=[f"{cov_stem(s)}.coverage_summary.tsv" for s in SAMPLES],
    output:
        f"{OUTDIR}/covfilt/coverage_summary.tsv",
    run:
        with open(output[0], "w") as out:
            for i, p in enumerate(input.rows):
                lines = open(p).read().splitlines()
                if i == 0:
                    out.write(lines[0] + "\n")      # header, once
                for line in lines[1:]:
                    out.write(line + "\n")
