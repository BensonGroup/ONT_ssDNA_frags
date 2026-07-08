# Decode ligated-DNA constructs with LigandFrags2 (from scripts/06). The `{decbase}` wildcard
# is `decode` for experimental samples and `decode_initial` for the initial library; it is only
# ever requested at the value decbase(sample) returns, so no sample is decoded into both trees.
# The log body reproduces scripts/06 verbatim (esp. the `expected :` line) so ligand_vocab.parse_log
# still recovers the per-sample expected CSV. RAYON handles per-process threading; Snakemake -j
# replaces the old JOBS/xargs fan-out, and the DAG replaces the "skip if exists" guard.
#
# Declared outputs are the ones consumed downstream (fragments/umis/log) plus fused_reads.fq
# (used opportunistically by the cross-report coverage panel). The decoder also writes
# fused.parquet / trimmed.parquet / truncated.fq alongside; those are untracked.
rule decode:
    input:
        fq=f"{OUTDIR}/preprocess/{{sample}}/{{sample}}.full_length.fq",
        bin=ligandfrags_bin,
        expected=config["expected_sequences"],
    output:
        frags=f"{OUTDIR}/{{decbase}}/{{sample}}/{{sample}}.fragments.parquet",
        umis=f"{OUTDIR}/{{decbase}}/{{sample}}/{{sample}}.umis.parquet",
        fused_reads=f"{OUTDIR}/{{decbase}}/{{sample}}/{{sample}}.fused_reads.fq",
        log=f"{OUTDIR}/{{decbase}}/{{sample}}/{{sample}}.ligateddna.log",
    threads: config["threads"]["decode"]
    params:
        stem=lambda wc: f"{OUTDIR}/{wc.decbase}/{wc.sample}/{wc.sample}",
        fmt=config["decode_format"],
    shell:
        r"""
        set -euo pipefail
        export RAYON_NUM_THREADS={threads}
        {{
          echo "== LigatedDNA run =="
          echo "  binary   : {input.bin}"
          echo "  fastq    : {input.fq}"
          echo "  expected : {input.expected}"
          echo "  out stem : {params.stem}"
          echo "  format   : {params.fmt}"
          echo "  started  : $(date -Is)"
          echo
          "{input.bin}" --expected "{input.expected}" --fastq "{input.fq}" --outfile "{params.stem}" --format "{params.fmt}"
          echo
          echo "  finished : $(date -Is)"
        }} 2>&1 | tee "{output.log}"
        """
