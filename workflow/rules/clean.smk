# Host/contaminant depletion (from scripts/01), generalised to a CHAIN over the config
# `decontaminate` genome list. Stage k aligns the running-clean reads to genome k and keeps
# only the reads UNMAPPED to it (`samtools fastq -f 4`); chained across genomes, a read
# survives only if it is unmapped to EVERY genome. The SAM stream is never stored. A final
# length filter produces clean/<sample>.fastq. Each stage records its in/out read counts in a
# tiny .stats file so clean_summary can report how many reads each genome removed.
rule deplete:
    input:
        reads=deplete_input_reads,
        mmi=deplete_mmi,
    output:
        fq=temp(f"{OUTDIR}/clean/{{sample}}/deplete.{{k}}.fastq"),
        stats=f"{OUTDIR}/clean/{{sample}}/deplete.{{k}}.stats",
        log=f"{OUTDIR}/clean/{{sample}}/deplete.{{k}}.minimap2.log",
    threads: config["threads"]["clean"]
    params:
        gname=lambda wc: GENOMES[int(wc.k)]["name"],
        preset=config["index_preset"],
    shell:
        r"""
        set -euo pipefail
        nin=$(( $(wc -l < "{input.reads}") / 4 ))
        pixi run -e default bash -c '
          set -o pipefail
          minimap2 -ax {params.preset} -Y --secondary=no -t {threads} "{input.mmi}" "{input.reads}" 2>"{output.log}" \
            | samtools fastq -n -f 4 - > "{output.fq}"
        '
        nout=$(( $(wc -l < "{output.fq}") / 4 ))
        printf '%s\t%s\t%s\n' "{params.gname}" "$nin" "$nout" > "{output.stats}"
        """


# Length filter on the fully-depleted reads -> the final clean FASTQ (drops len <= minlen).
rule length_filter:
    input:
        fq=f"{OUTDIR}/clean/{{sample}}/deplete.{LAST_STAGE}.fastq",
    output:
        fq=f"{OUTDIR}/clean/{{sample}}.fastq",
        stats=f"{OUTDIR}/clean/{{sample}}/lenfilt.stats",
    params:
        minlen=config["params"]["minlen"],
    shell:
        r"""
        set -euo pipefail
        nin=$(( $(wc -l < "{input.fq}") / 4 ))
        awk -v minlen={params.minlen} '
            {{ h = $0; getline s; getline p; getline q
               if (length(s) > minlen) print h "\n" s "\n" p "\n" q }}
        ' "{input.fq}" > "{output.fq}"
        kept=$(( $(wc -l < "{output.fq}") / 4 ))
        printf '%s\t%s\n' "$nin" "$kept" > "{output.stats}"
        """


# Aggregate the per-stage counts into a summary with one `<genome>_mapped` column per genome
# (replaces the append loop in scripts/01). In-process; no pixi env needed.
rule clean_summary:
    input:
        deplete=expand(f"{OUTDIR}/clean/{{s}}/deplete.{{k}}.stats", s=SAMPLES, k=range(N_GENOMES)),
        lenfilt=expand(f"{OUTDIR}/clean/{{s}}/lenfilt.stats", s=SAMPLES),
    output:
        f"{OUTDIR}/clean/clean_summary.tsv",
    params:
        samples=SAMPLES,
        gnames=[g["name"] for g in GENOMES],
        minlen=config["params"]["minlen"],
    run:
        gnames = params.gnames
        header = (["sample", "total_reads"]
                  + [f"{g}_mapped" for g in gnames]
                  + ["unmapped_all", f"kept_gt{params.minlen}", f"dropped_le{params.minlen}"])
        with open(output[0], "w") as out:
            out.write("\t".join(header) + "\n")
            for s in params.samples:
                stages = []
                for k in range(len(gnames)):
                    _name, nin, nout = open(
                        f"{OUTDIR}/clean/{s}/deplete.{k}.stats").read().rstrip("\n").split("\t")
                    stages.append((int(nin), int(nout)))
                total = stages[0][0]
                mapped = [str(nin - nout) for nin, nout in stages]
                unmapped_all = stages[-1][1]
                lin, kept = open(
                    f"{OUTDIR}/clean/{s}/lenfilt.stats").read().rstrip("\n").split("\t")
                kept = int(kept)
                dropped = int(lin) - kept
                out.write("\t".join([s, str(total)] + mapped
                                    + [str(unmapped_all), str(kept), str(dropped)]) + "\n")
