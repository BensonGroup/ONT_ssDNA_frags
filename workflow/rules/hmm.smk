# Train the strand-specific primer HMM (hammerpede) from the cleaned initial-library reads
# (from scripts/04). Built once; consumed by every pychopper job. The `rm -rf` of the output
# dir is required because hp_bootstrap.py does os.mkdir(OUTDIR) and errors if it exists; the
# dir is a leaf output owned solely by this rule, so this is safe.
rule train_hmm:
    input:
        reads=f"{OUTDIR}/clean/{INITIAL}.fastq",
        fasta=config["adapters_fasta"],
        patched=f"{OUTDIR}/.preprocess_patched",
    output:
        hmm=HMM,
        h3m=f"{HMM}.h3m",
        h3i=f"{HMM}.h3i",
        h3f=f"{HMM}.h3f",
        h3p=f"{HMM}.h3p",
    params:
        outdir=f"{OUTDIR}/hmm",
        ntrain=config["params"]["ntrain"],
    shell:
        r"""
        set -euo pipefail
        tmp=$(mktemp --suffix=.train.fq); trap 'rm -f "$tmp"' EXIT
        if [ {params.ntrain} -gt 0 ]; then
            head -n $(( {params.ntrain} * 4 )) "{input.reads}" > "$tmp"
        else
            cp "{input.reads}" "$tmp"
        fi
        rm -rf "{params.outdir}"
        pixi run -e preprocess hp_bootstrap.py -f "{input.fasta}" -o "{params.outdir}" "$tmp"
        pixi run -e preprocess hmmpress -f "{output.hmm}"
        """
