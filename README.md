# ONT ligated-DNA selection-screen workflow

A Snakemake workflow for the Nanopore ligated-DNA selection screen: human depletion → primer-HMM
training → pychopper preprocessing → cONcat (new) decode → per-read feature counts → interactive
HTML reports.

```
clean ─┬─────────────► train_hmm ─┐
       │                          ▼
       └────────────────────────► pychopper ──► decode ──► featurecount ─┬─► report_sample
                                                                         └─► report_cross
```

Environments are managed by **pixi** (conda-forge + bioconda); orchestration is Snakemake.
Snakemake is pure Python and architecture-independent — the same workflow runs on `linux-aarch64`
and `linux-64`; only the tools it calls are per-arch (pixi re-solves; the Rust decoder recompiles).

## Prerequisites (once per machine)

1. Install [pixi](https://pixi.sh). If you are **not** on `linux-aarch64`, add your platform to
   `pixi.toml` first, e.g. `platforms = ["linux-aarch64", "linux-64"]`.
2. Create the environments the rules call:
   ```bash
   pixi install -e default      # minimap2, samtools, seqkit, polars, plotly (clean + featurecount + reports)
   pixi install -e preprocess   # pychopper, hmmer, hammerpede         (train_hmm + pychopper)
   pixi install -e build        # rust/cargo                            (only if building LigandFrags2 from source)
   ```
3. Patch the `preprocess` env (idempotent; **re-run after every `pixi install`/clean** of that env):
   ```bash
   pixi run -e preprocess fix-preprocess-env
   ```

   This fixes two unmaintained-upstream bugs (hammerpede `"rU"`, pychopper `float(Series)`) that
   otherwise make successful jobs exit non-zero *after* writing outputs — which would make Snakemake
   delete them. The workflow also runs this via the `fix_preprocess_env` rule (a `.preprocess_patched`
   sentinel that `train_hmm`/`pychopper` depend on), but the env must already be installed.
4. Install Snakemake (e.g. `pipx install snakemake`, or its own pixi/conda env).

## Configure

Edit `config/config.yaml`. Everything is config-driven — point the paths at your data:

- **reference** — a prebuilt minimap2 `mmi`, **or** a `fasta` (local path or `https://` URL) that the
  `build_index` rule downloads + indexes.
- **ligandfrags** — a prebuilt `binary`, **or** `source` (a working tree already checked out to the
  desired branch) that the `build_ligandfrags` rule compiles with cargo. The source https://github.com/BensonGroup/LigatedDNA is private now, so users need to provide the `binary`, **or** `source` path on disk here. Using binary will skip building from source.
  ```yaml
  ligandfrags:
    binary: /abs/path/to/LigandFrags2   # non-empty -> used directly; the build is skipped
    source: null                        # ignored when binary is set
    manifest_subpath: Cargo.toml        # ignored when binary is set
    bin_name: LigandFrags2              # only names the build output; unused with binary
  ```
- **adapters_fasta / pychopper_config / expected_sequences** — per-project; defaults ship in
  `resources/`. The primer HMM is *built each run* from the adapters + the initial-library sample.
- **samples** — the sample table. Exactly one sample has `role: initial` (the R0 baseline); the rest
  are `role: experimental`.

## Run (from the repo root, where `pixi.toml` lives)

```bash
snakemake -n --printshellcmds --configfile /abs/path/to/your_config.yaml          # dry-run: inspect the DAG without executing
snakemake --cores 18 --configfile /abs/path/to/your_config.yaml                   # full run
snakemake --cores 18 <target>          # build one target, e.g. a single sample report:
#   snakemake --cores 8 results/decode/A549_R1/A549_R1.featurecount_report.html
snakemake check_all --cores 4          # optional: validate every featurecount against its raw parquet
```

Three things to know:

- **It merges over the default, doesn't replace it.** The Snakefile loads `config/config.yaml`
  first, then deep-merges your `--configfile` on top. Top-level scalars and *lists* (`samples`,
  `decontaminate`) are replaced by yours; *nested dicts* (`params`, `threads`, `report`,
  `ligandfrags`) merge key-by-key, so any sub-key you omit silently inherits the default — define
  those blocks fully in your config.
- **Relative paths resolve against the working dir, not the config's location.** `outdir: results`
  writes to `./results` in the repo root wherever the config lives, so keep input/resource paths
  **absolute** in an external config. Use `--directory /path` (or set an absolute `outdir`) to put
  outputs elsewhere.
- Scripts and envs are located via the Snakefile itself, so nothing else changes.

## Outputs (under `config.outdir`, default `results/`)


| Path                                                                                                          | What                                                                         |
| --------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `clean/<sample>.fastq`, `clean/clean_summary.tsv`                                                             | human-depleted, length-filtered reads + summary                              |
| `hmm/<adapters>.hmm` (+ hmmpress DB)                                                                          | trained primer HMM                                                           |
| `preprocess/<sample>/<sample>.full_length.fq` (+ rescued/…/stats/report/bed), `preprocess/batch_summary.tsv` | pychopper outputs                                                            |
| `decode/<sample>/…` (experimental), `decode_initial/<initial>/…`                                            | LigandFrags2 decode (fragments/umis parquet, fused_reads.fq, ligateddna.log) |
| `decode/<sample>/<sample>.featurecount.parquet` + `.summary.tsv`                                              | per-read feature counts                                                      |
| `decode/<sample>/<sample>.featurecount_report.html`                                                           | per-sample interactive report                                                |
| `decode/cross_sample_featurecount_report.html`                                                                | cross-sample comparison report                                               |

The initial library decodes into a **separate** `decode_initial/` tree so the cross report's
`decode/*/` sample discovery never double-counts it; it is wired in as the shared R0 baseline via
`report.r0_baseline: true`.