# One-time patch of the `preprocess` pixi env (hammerpede "rU", pychopper float(Series)).
# Both bugs make otherwise-successful jobs exit non-zero AFTER writing all outputs, which
# would make Snakemake delete the (complete) outputs. Making this a sentinel rule wires the
# patch into the DAG as a prerequisite of train_hmm and pychopper.
#
# Prerequisite (README): `pixi install -e preprocess` must have run first — the patch edits
# files under .pixi/envs/preprocess/lib/.
rule fix_preprocess_env:
    output:
        touch(f"{OUTDIR}/.preprocess_patched"),
    shell:
        "pixi run -e preprocess fix-preprocess-env"
