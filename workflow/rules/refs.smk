# Conditional reference/tool builds — they only enter the DAG when the target is missing.

# Build the minimap2 index INTO each genome's configured `mmi` path, from its FASTA (local
# path or URL). One rule per genome that declares a `fasta`; Snakemake builds it only if the
# index is missing and reuses it thereafter (and across projects, since the mmi lives at a
# shared reference location) — the DAG equivalent of the scripts' `if [ ! -f "$MMI" ]` guard.
for _g in GENOMES:
    if _g.get("fasta"):

        rule:
            name:
                f"build_index_{_g['name']}"
            input:
                (ancient(_g["fasta"]) if not _is_url(_g["fasta"]) else []),
            output:
                _g["mmi"],
            params:
                src=_g["fasta"],
                preset=config["index_preset"],
                dl=f"{OUTDIR}/index/{_g['name']}.fasta",
            log:
                f"{OUTDIR}/logs/build_index_{_g['name']}.log",
            threads: config["threads"]["clean"]
            shell:
                r"""
                set -euo pipefail
                SRC="{params.src}"
                case "$SRC" in
                  http://*|https://*|ftp://*)
                    echo "[build_index] downloading $SRC" >&2
                    curl -fsSL "$SRC" -o "{params.dl}"
                    SRC="{params.dl}"
                    ;;
                esac
                pixi run -e default minimap2 -x {params.preset} -t {threads} -d "{output}" "$SRC" 2>"{log}"
                """


# Compile the LigandFrags2 decoder from an already-checked-out source tree (no git clone
# here — private repo). Only triggered when ligandfrags.binary is null (see ligandfrags_bin()).
rule build_ligandfrags:
    output:
        f"{OUTDIR}/bin/{config['ligandfrags']['bin_name']}",
    params:
        src=config["ligandfrags"].get("source"),
        manifest=config["ligandfrags"]["manifest_subpath"],
        target=os.path.abspath(f"{OUTDIR}/cargo-target"),
        binname=config["ligandfrags"]["bin_name"],
    shell:
        r"""
        set -euo pipefail
        CARGO_TARGET_DIR="{params.target}" pixi run -e build \
            cargo build --release --manifest-path "{params.src}/{params.manifest}"
        cp "{params.target}/release/{params.binname}" {output}
        """
