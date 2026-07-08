# Pychopper (phmm) orient / full-length-classify / rescue, primers retained (-p), inline
# two-inboard-UMI extraction (-U), per-read primer-position BED (-A) — from scripts/05.
# One job per sample; `{input.patched}` guarantees the pychopper report step exits 0.
rule pychopper:
    input:
        fq=f"{OUTDIR}/clean/{{sample}}.fastq",
        hmm=HMM,
        hmm_db=[f"{HMM}.h3m", f"{HMM}.h3i", f"{HMM}.h3f", f"{HMM}.h3p"],
        cfg=config["pychopper_config"],
        patched=f"{OUTDIR}/.preprocess_patched",
    output:
        full=f"{OUTDIR}/preprocess/{{sample}}/{{sample}}.full_length.fq",
        rescued=f"{OUTDIR}/preprocess/{{sample}}/{{sample}}.rescued.fq",
        unclassified=f"{OUTDIR}/preprocess/{{sample}}/{{sample}}.unclassified.fq",
        len_fail=f"{OUTDIR}/preprocess/{{sample}}/{{sample}}.len_fail.fq",
        stats=f"{OUTDIR}/preprocess/{{sample}}/{{sample}}.stats.tsv",
        report=f"{OUTDIR}/preprocess/{{sample}}/{{sample}}.report.pdf",
        bed=f"{OUTDIR}/preprocess/{{sample}}/{{sample}}.primer_hits.bed",
    log:
        f"{OUTDIR}/preprocess/logs/{{sample}}.log",
    threads: config["threads"]["pychopper"]
    params:
        zmin=config["params"]["zmin"],
    shell:
        r"""
        pixi run -e preprocess pychopper -m phmm -g "{input.hmm}" -c "{input.cfg}" -p -z {params.zmin} -U -t {threads} \
          -A "{output.bed}" -w "{output.rescued}" -u "{output.unclassified}" -l "{output.len_fail}" \
          -S "{output.stats}" -r "{output.report}" \
          "{input.fq}" "{output.full}" > "{log}" 2>&1
        """


# Aggregate the per-sample pychopper stats into batch_summary.tsv (replaces the append loop
# in scripts/05). In-process; no pixi env needed.
rule pychopper_summary:
    input:
        stats=expand(f"{OUTDIR}/preprocess/{{s}}/{{s}}.stats.tsv", s=SAMPLES),
        full=expand(f"{OUTDIR}/preprocess/{{s}}/{{s}}.full_length.fq", s=SAMPLES),
        rescued=expand(f"{OUTDIR}/preprocess/{{s}}/{{s}}.rescued.fq", s=SAMPLES),
        unclassified=expand(f"{OUTDIR}/preprocess/{{s}}/{{s}}.unclassified.fq", s=SAMPLES),
        len_fail=expand(f"{OUTDIR}/preprocess/{{s}}/{{s}}.len_fail.fq", s=SAMPLES),
    output:
        f"{OUTDIR}/preprocess/batch_summary.tsv",
    params:
        samples=SAMPLES,
        fastqs={n: SAMPLE[n]["fastq"] for n in SAMPLES},
    run:
        def fqreads(p):
            try:
                with open(p) as fh:
                    return sum(1 for _ in fh) // 4
            except FileNotFoundError:
                return 0

        def umi_count(p):
            try:
                return sum(1 for line in open(p) if "umi1=" in line)
            except FileNotFoundError:
                return 0

        def pass_reads(p):
            try:
                for line in open(p):
                    f = line.rstrip("\n").split("\t")
                    if len(f) >= 3 and f[1] == "PassReads":
                        return str(int(float(f[2])))
            except FileNotFoundError:
                pass
            return "NA"

        with open(output[0], "w") as out:
            out.write(
                "Sample\tinput\tinput_pass\tfull_length\trescued\tunclassified\tlen_fail"
                "\tusable_with_rescue\twith_both_umi\trescue_yield_x\n"
            )
            for s in params.samples:
                base = f"{OUTDIR}/preprocess/{s}/{s}"
                fl = fqreads(base + ".full_length.fq")
                re_ = fqreads(base + ".rescued.fq")
                un = fqreads(base + ".unclassified.fq")
                lf = fqreads(base + ".len_fail.fq")
                usable = fl + re_
                wu = umi_count(base + ".full_length.fq") + umi_count(base + ".rescued.fq")
                yld = f"{usable / fl:.2f}" if fl > 0 else "NA"
                out.write(
                    f"{s}\t{params.fastqs[s]}\t{pass_reads(base + '.stats.tsv')}"
                    f"\t{fl}\t{re_}\t{un}\t{lf}\t{usable}\t{wu}\t{yld}\n"
                )
