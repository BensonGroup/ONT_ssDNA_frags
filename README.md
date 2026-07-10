# ONT ligated-DNA selection-screen workflow

A Snakemake workflow for the Nanopore ligated-DNA selection screen: genomic depletion → primer-HMM
training → pychopper preprocessing → cONcat (new) decode → UMI dedup → per-read feature counts →
interactive HTML reports.

```
minikow/dorado passed fastq ─┬─► genomic contamination─┐
(zcat per sample)            └─► length filter         └─┬─► train_hmm ─┐
                                                         │              ▼
                                                         └────────────► pychopper ──► cONcat ──► dedup ──► featurecount ─┬─► report_sample
                                                                                       (new)     (new)                   └─► report_cross
```

`dedup` collapses PCR/sequencing duplicates into molecules (see **UMI deduplication** below). It is
on by default; set `umi_dedup.enabled: false` to build the reports from the raw read-level decode
tree instead.

Environments are managed by **pixi** (conda-forge + bioconda); orchestration is Snakemake.
Snakemake is pure Python and architecture-independent — the same workflow runs on `linux-aarch64`
and `linux-64`; only the tools it calls are per-arch (pixi re-solves; the Rust decoder recompiles).

## Set up envs

Install [pixi](https://pixi.sh). If you are **not** on `linux-aarch64`, add your platform to
   `pixi.toml` first, e.g. `platforms = ["linux-aarch64", "linux-64"]`.

Then in this repo root: 

1. Create the environments the rules call:

   ```bash
   pixi install -e default      # minimap2, samtools, seqkit, polars, plotly, starcode (clean + dedup + featurecount + reports)
   pixi install -e preprocess   # pychopper, hmmer, hammerpede         (train_hmm + pychopper)
   pixi install -e build        # rust/cargo                            (only if building LigandFrags2 from source)
   ```
2. Patch the `preprocess` env (idempotent; **re-run after every `pixi install`/clean** of that env):

   ```bash
   pixi run -e preprocess fix-preprocess-env
   ```

   This fixes two unmaintained-upstream bugs (hammerpede `"rU"`, pychopper `float(Series)`) that
   otherwise make successful jobs exit non-zero *after* writing outputs — which would make Snakemake
   delete them. The workflow also runs this via the `fix_preprocess_env` rule (a `.preprocess_patched`
   sentinel that `train_hmm`/`pychopper` depend on), but the env must already be installed.
3. **Snakemake is the orchestrator, in its own isolated env.** It sits *above* the per-rule
   envs, so it lives in a `no-default-feature` `snakemake` environment — pinned in `pixi.lock`,
   with no solver entanglement with the tool envs (the conda package is `snakemake-minimal`, but
   the binary is `snakemake`):

   ```bash
   pixi install -e snakemake
   alias concatsmk='pixi run -e snakemake snakemake'   # convenience prefix, once per shell
   concatsmk --version
   ```

## Your project directory and config

Create four folders in your project dir (outside of this repo):

```
.
|-- config
|   `-- config_cell_preprint.yaml (your config.yaml)
|-- exp
|   `-- expected_seq.csv (your expected sequences)
|-- rawdata (your .fastq, one subdir per sample)
|   |-- HEK293
|   |-- initial
|   `-- RAW
|-- resources (your primers and fwd/bwd config)
|   |-- adapters.fasta
|   `-- pychopper_config.txt
```

Edit `config/config.yaml`. Point the paths at your data:

- **outdir** - abs path to your output dir, likely your project dir/results
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
- **umi_dedup** — the deduplication stage (see below). Omit the block entirely and it stays on with
  the defaults shown here.
  ```yaml
  umi_dedup:
    enabled: true          # false -> reports built from the raw decode tree
    max_homopolymer: 8     # drop a UMI whose longest homopolymer run is >= this
    max_base_frac:   0.7   # drop a UMI where any single base's fraction is >= this
  ```
- **samples** — the sample table. Exactly one sample has `role: initial` (the R0 baseline). See example in this repo config/config.yaml

## UMI deduplication

After decoding, each read carries two UMIs. `dedup` drops garbage UMIs (non-ACGT, low complexity,
or outside the 10–12 nt length window), then collapses the survivors into **molecules** by
clustering the fused UMI sequence with `starcode -d 1 -s` — sphere clustering at Levenshtein
distance 1. One representative read per cluster survives, chosen by highest whole-read base-quality
sum (the Picard MarkDuplicates convention), tie-broken by smallest `read_acc`.

Levenshtein rather than Hamming distance is deliberate: the UMIs are variable length and ONT's
dominant error mode is indels. `starcode` is pinned to a single thread (`-t 1`), which costs ~0.5 s
per sample and makes the raw cluster table byte-reproducible across re-runs.

The stage writes a filtered copy of the sample under **the same basenames** into a sibling tree
(`dedup/` ← `decode/`, `dedup_initial/` ← `decode_initial/`), so `featurecount` and the reports run
against either tree unchanged. The raw `decode/` tree is always built — it is what `dedup` consumes
— so flipping `enabled` reproduces the pre-dedup results without recomputing the decode.

Each sample's dedup QC (`.umi_counts.tsv`, `.umi_clusters.tsv`, `.umi_audit.parquet`, and its
`.umi_dedup_summary.tsv` row) is written **into that sample's own directory**, next to the data it
describes — the same convention as `<sample>.featurecount_report.html`. The concatenated table for
all samples lands at `dedup/umi_dedup_summary.tsv`.

## Run (from the repo root)

Using the `concatsmk` alias from prerequisites step 4 (or the full `pixi run -e snakemake snakemake` prefix):

```bash
concatsmk -n --printshellcmds --configfile /abs/path/to/your_config.yaml         # dry-run: inspect the DAG without executing
concatsmk --cores 18 --configfile /abs/path/to/your_config.yaml                # full run (reads config/config.yaml by default)
concatsmk --cores 8 /abs/path/to/results/dedup/your.featurecount_report.html    # build one target
concatsmk check_all --cores 4        # optional: validate every featurecount against its raw parquet
concatsmk -n --config umi_dedup='{"enabled": false}'   # dry-run the pre-dedup DAG
python3 workflow/scripts/umi_filter_lib.py             # optional: UMI-classification self-test
```

Note the `--` separator when a target follows `--configfile` (which takes several values):
`concatsmk -n --configfile my.yaml -- results/dedup/A549_R1/A549_R1.fragments.parquet`.

## Outputs (under `config.outdir`, default `results/`)


Below, `<tree>` is `dedup` when `umi_dedup.enabled` (the default) and `decode` otherwise; the
initial library lives in the matching `<tree>_initial/`.

| Path                                                                                                          | What                                                                         |
| --------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `clean/<sample>.fastq`, `clean/clean_summary.tsv`                                                             | human-depleted, length-filtered reads + summary                              |
| `hmm/<adapters>.hmm` (+ hmmpress DB)                                                                          | trained primer HMM                                                           |
| `preprocess/<sample>/<sample>.full_length.fq` (+ rescued/…/stats/report/bed), `preprocess/batch_summary.tsv` | pychopper outputs                                                            |
| `decode/<sample>/…` (experimental), `decode_initial/<initial>/…`                                            | LigandFrags2 decode (fragments/umis/fused/trimmed parquet, fused_reads.fq, ligateddna.log) |
| `dedup/<sample>/…` (experimental), `dedup_initial/<initial>/…`                                              | the same artifacts, filtered to one representative read per UMI molecule; `ligateddna.log` gains a provenance block |
| `dedup/<sample>/<sample>.{umi_counts,umi_clusters}.tsv`, `.umi_audit.parquet`, `.umi_dedup_summary.tsv`        | that sample's dedup QC, beside the data it describes: starcode input/output, a one-row-per-input-read audit, and its summary row |
| `dedup/umi_dedup_summary.tsv`                                                                                 | all samples: reads in → molecules out, drop classes, duplicate rate          |
| `<tree>/<sample>/<sample>.featurecount.parquet` + `.summary.tsv`                                              | per-read feature counts                                                      |
| `<tree>/<sample>/<sample>.featurecount_report.html`                                                           | per-sample interactive report                                                |
| `<tree>/cross_sample_featurecount_report.html`                                                                | cross-sample comparison report                                               |

The initial library lands in a **separate** `*_initial/` tree so the cross report's `<tree>/*/`
sample discovery never double-counts it; it is wired in as the shared R0 baseline via
`report.r0_baseline: true`.
