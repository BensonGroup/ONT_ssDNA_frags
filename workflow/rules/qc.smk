# Cross-sample read QC with NanoComp, run on the RAW per-sample FASTQs (the same inputs the
# depletion chain starts from — SAMPLE[s]["fastq"]). NanoComp is inherently multi-dataset:
# a single invocation over every sample emits ONE combined report comparing read-length and
# quality distributions across samples — there is no per-sample step and no separate merge.
# This QC branch feeds nothing downstream; it hangs off `rule all` and runs in parallel with
# `clean` (it does not gate processing). `--names` aligns with `--fastq` because both derive
# from SAMPLES in the same order.
# Adapted from scripts/00.run_nanocomp_bysample.sh, which ran NanoComp `--summary` over
# per-barcode sequencing_summary files ordered by exp/barcodes.tsv; here we run `--fastq`
# over the per-sample FASTQs the config already declares, so no extra inputs are needed.
rule nanocomp:
    input:
        fastqs=[SAMPLE[s]["fastq"] for s in SAMPLES],
    output:
        report=f"{OUTDIR}/qc/nanocomp/bysample_NanoComp-report.html",
        stats=f"{OUTDIR}/qc/nanocomp/bysample_NanoStats.txt",
    params:
        names=SAMPLES,
        outdir=f"{OUTDIR}/qc/nanocomp",
    threads: 8
    shell:
        r"""
        pixi run -e default NanoComp --fastq {input.fastqs} --names {params.names} \
          -o "{params.outdir}" -p bysample_ --title "NanoComp (by sample)" --tsv_stats -t {threads}
        """
