# ONT ligated-DNA selection-screen workflow

A Snakemake workflow for the Nanopore ligated-DNA selection screen: genomic depletion → primer-HMM
training → pychopper preprocessing → cONcat (new) decode → UMI processing (filter / correct / dedup)
→ per-read feature counts → interactive HTML reports.

```
minikow/dorado passed fastq ─┬─► genomic contamination─┐
(zcat per sample)            └─► length filter         └─┬─► train_hmm ─┐
                                                         │              ▼
                                                         └────────────► pychopper ──► cONcat ──► dedup ──► featurecount ─┬─► report_sample
                                                                                       (new)     (new)                   └─► report_cross
```

The UMI stage is set by `umi.mode` (default **`correct`**): `filter` QC-filters garbage UMIs only;
`correct` additionally corrects near-identical UMIs (ed≤1) and runs a separate **UMI characterisation
branch** (per-molecule read counts, a per-group UMI×sample matrix, and a cross-sample UMI report)
without touching the fragment reports; `dedup` collapses PCR/sequencing duplicates into molecules.
This data must not be UMI-collapsed, so collapse is **off** by default. See **UMI processing** below.

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
- **umi** — the UMI stage (see below). Omit the block entirely and it keeps the defaults shown here.
  ```yaml
  umi:
    mode: correct          # filter | correct | dedup   (default correct)
    max_homopolymer: 8     # drop a UMI whose longest homopolymer run is >= this
    max_base_frac:   0.7   # drop a UMI where any single base's fraction is >= this
  ```
- **samples** — the sample table. Exactly one sample has `role: initial` (the R0 baseline). See example in this repo config/config.yaml

## UMI processing (`umi.mode`)

After decoding, each read carries two UMIs concatenated into `fused_seq` (~20–24 nt). `umi.mode`
selects what happens next:

- **`filter`** — drop garbage UMIs only (non-ACGT, low complexity, or outside the 10–12 nt length
  window). Every surviving read is kept.
- **`correct`** (default) — filter, then **correct** each surviving read's UMI to the most-abundant
  ed≤1 variant by clustering with `starcode -d 1 -s` (spheres, Levenshtein distance 1) and relabelling
  members to their sphere's centroid. **Nothing is collapsed** — duplicate molecules keep every one of
  their reads. This additionally runs a separate `umi/` characterisation branch (below); the fragment
  featurecount/reports are unaffected.
- **`dedup`** — filter + correct + **collapse** to one representative read per molecule (highest
  whole-read base-quality sum, the Picard MarkDuplicates convention). The collapsed tree feeds
  featurecount/reports (classic UMI dedup).

Levenshtein rather than Hamming distance is deliberate: the UMIs are variable length and ONT's
dominant error mode is indels. `starcode` is pinned to a single thread (`-t 1`) for byte-reproducible
cluster output (~0.5 s/sample).

**Fragment pipeline.** In every mode the `dedup/` tree (QC-filtered, or collapsed in `dedup` mode)
feeds `featurecount` and the fragment reports — correction never changes fragment counts. Each
sample's dedup QC (`.umi_counts.tsv`, `.umi_clusters.tsv`, `.umi_audit.parquet`, and its
`.umi_dedup_summary.tsv` row) sits in its own directory; the concatenation is `dedup/umi_dedup_summary.tsv`.

**UMI characterisation branch (`correct`).** A separate `umi/` tree (sibling of `decode/`/`dedup/`,
with `umi_initial/` for the R0 library), independent of the fragment reports. A read is kept only if
it passes the UMI QC filter **and** decodes at least one **payload** fragment — *anchor-only* reads
(valid UMI, backbone features only) are **dropped**, so a molecule's read count reflects only reads
that actually decoded a ligand. Anchors come from the umis parquet via `ligand_vocab.anchor_set`
(both orientations), the same authoritative source `build_featurecounts.py` uses; the fragment
pipeline is unaffected and still counts every read. Per sample `umi/<s>/`:

- `<s>.fused.parquet` — one row per kept (payload-bearing) read: `read_acc, strand, fused_seq,
  fused_len, fused_seq_corrected, umi_read_count` (reads sharing this read's corrected UMI here).
- `<s>.umi_species.tsv` — corrected `fused_seq  n_reads` (the matrix substrate).
- `<s>.umi_summary.tsv` — per-sample filter/correction counts (incl. `dropped_no_payload`).

Per experimental **group** (config `group`), one corrected-UMI × sample matrix with the R0 library as
the **first column** (no all-sample matrix): `umi/<group>.umi_x_sample.long.parquet` (`fused_umi,
sample, n_reads`; canonical), `.mtx` (Matrix Market triplets), and `.{umis,samples}.tsv` (row/column
labels). Finally `umi/umi_report_cross.html` — a descriptive QC report (rank-abundance, D50/D90
complexity, per-group species-overlap UpSet, and validity QC: UMI-group ligand-signature purity vs
size and a collision null model). It draws no biological conclusions.

## Run (from the repo root)

Using the `concatsmk` alias from prerequisites step 4 (or the full `pixi run -e snakemake snakemake` prefix):

```bash
concatsmk -n --printshellcmds --configfile /abs/path/to/your_config.yaml         # dry-run: inspect the DAG without executing
concatsmk --cores 18 --configfile /abs/path/to/your_config.yaml                # full run (reads config/config.yaml by default)
concatsmk --cores 8 /abs/path/to/results/dedup/your.featurecount_report.html    # build one target
concatsmk check_all --cores 4        # optional: validate every featurecount against its raw parquet
concatsmk -n --config umi='{"mode": "dedup"}'    # dry-run WITH UMI collapse (classic dedup)
concatsmk -n --config umi='{"mode": "filter"}'   # dry-run filter-only (no correction branch)
python3 workflow/scripts/umi_filter_lib.py             # optional: UMI-classification self-test
```

Note the `--` separator when a target follows `--configfile` (which takes several values):
`concatsmk -n --configfile my.yaml -- results/dedup/A549_R1/A549_R1.fragments.parquet`.

## Outputs (under `config.outdir`, default `results/`)


Below, `<tree>` is `dedup` (the UMI stage always QC-filters, so featurecount/reports run on the
`dedup/` tree; in mode `dedup` it is the collapsed tree); the initial library lives in `dedup_initial/`.

| Path                                                                                                          | What                                                                         |
| --------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `clean/<sample>.fastq`, `clean/clean_summary.tsv`                                                             | human-depleted, length-filtered reads + summary                              |
| `hmm/<adapters>.hmm` (+ hmmpress DB)                                                                          | trained primer HMM                                                           |
| `preprocess/<sample>/<sample>.full_length.fq` (+ rescued/…/stats/report/bed), `preprocess/batch_summary.tsv` | pychopper outputs                                                            |
| `decode/<sample>/…` (experimental), `decode_initial/<initial>/…`                                            | LigandFrags2 decode (fragments/umis/fused/trimmed parquet, fused_reads.fq, ligateddna.log) |
| `dedup/<sample>/…` (experimental), `dedup_initial/<initial>/…`                                              | the same artifacts, QC-filtered (and, with dedup on, reduced to one representative read per UMI molecule); `ligateddna.log` gains a provenance block |
| `dedup/<sample>/<sample>.{umi_counts,umi_clusters}.tsv`, `.umi_audit.parquet`, `.umi_dedup_summary.tsv`        | that sample's UMI QC, beside the data it describes: UMI counts, starcode clusters (empty unless dedup on), a one-row-per-input-read audit, and its summary row |
| `dedup/umi_dedup_summary.tsv`                                                                                 | all samples: reads in → reads kept (molecules out with dedup on), drop classes, duplicate rate |
| `umi/<sample>/<sample>.fused.parquet`, `.umi_species.tsv`, `.umi_summary.tsv` (+ `umi_initial/`)             | (mode `correct`) corrected UMIs over **payload-bearing reads only**: per-read `fused_seq_corrected` + `umi_read_count`, the species→count table, and filter/correction counts |
| `umi/<group>.umi_x_sample.{long.parquet,mtx,umis.tsv,samples.tsv}`                                            | (mode `correct`) per-group corrected-UMI × sample read-count matrix, R0 library as the first column |
| `umi/umi_report_cross.html`                                                                                   | (mode `correct`) cross-sample UMI QC report (abundance, species overlap, validity) |
| `<tree>/<sample>/<sample>.featurecount.parquet` + `.summary.tsv`                                              | per-read feature counts                                                      |
| `<tree>/<sample>/<sample>.featurecount_report.html`                                                           | per-sample interactive report                                                |
| `<tree>/cross_sample_featurecount_report.html`                                                                | cross-sample comparison report                                               |

The initial library lands in a **separate** `*_initial/` tree so the cross report's `<tree>/*/`
sample discovery never double-counts it; it is wired in as the shared R0 baseline via
`report.r0_baseline: true`.
