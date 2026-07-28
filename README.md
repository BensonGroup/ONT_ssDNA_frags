# ONT ligated-DNA selection-screen workflow

A Snakemake workflow for the Nanopore ligated-DNA selection screen: genomic depletion → primer-HMM
training → pychopper preprocessing → cONcat (new) decode → drops reads the decoder barely explained with min_coverage → per-read feature counts. One UMI (QC-filter → ed≤1 correct → jackpot-clean by initial library; not deduplicated) feeds three readouts: per-fragment counts, effective complexity, and fragment-vector quantification. At each step, a html report will be generated. The fitness calculation is defined outside snakemake, as it should be considered per study design. Here it's the log2(exp/emp) at last round.

## Usage

### Set up envs

Install [pixi](https://pixi.sh). If you are **not** on `linux-aarch64`, add your platform to
   `pixi.toml` first, e.g. `platforms = ["linux-aarch64", "linux-64"]`.

Then in this repo root: 

1. Create the environments the rules call:

   ```bash
   pixi install -e default      # minimap2, samtools, seqkit, polars, plotly, starcode (clean + umi + featurecount + reports)
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
- **decode_filter** — the decode coverage filter (see below). Omit the block and it keeps these defaults.
  ```yaml
  decode_filter:
    enabled: true
    min_coverage: 0.9      # drop a read whose decoded coverage is < this
  ```
- **umi** — the UMI stage (see below). Omit the block entirely and it keeps the defaults shown here.
  ```yaml
  umi:
    max_homopolymer: 8     # drop a UMI whose longest homopolymer run is >= this
    max_base_frac:   0.7   # drop a UMI where any single base's fraction is >= this
  ```
- **fragvec** — the fragment-vector branch (see below). A sibling of the `umi` branch; run either or both.
  ```yaml
  fragvec:
    enabled: true
    min_reads: 10          # counts below this are FLAGGED as noise, not dropped
  ```
- **groups** — every group name used in `samples:`, declared once:
  ```yaml
  groups: [initial, A549, MDA]   # add `empty` when you add PCR-only rounds
  ```
  This is the typo guard: a sample whose `group` is not in this list **raises**, instead of silently
  minting a new, half-empty selection arm. A declared group with no samples raises too. It must be
  declared **in the same file that declares `samples:`** — snakemake layers a `--configfile` *on top
  of* the workflow's default config, so an omitted `groups:` would otherwise be silently inherited and
  your samples validated against a list you never wrote. The *order* of this list reaches no report —
  report order and colours come from `samples:` (see below).
- **samples** — the sample table; four keys, no `role`. See **[Writing the sample
  table](#writing-the-sample-table)**.
  - **`group: empty`** = an **empty / PCR-only round**: the same aliquot transfers and PCR cycles as a
    selection round, but *no biological material*. Each transfer is itself a bottleneck, so complexity
    drops even with no selection — and an empty round is the **only** way to separate the collapse
    caused by selection from the collapse caused by the protocol. The three numbers the reports emit,
    with the `kind` label each one carries in `umi_collapse.tsv`:
    ```
    protocol   collapse = c_eff(R0)       / c_eff(empty_Rk)   <- reported ON the empty round
    selection  collapse = c_eff(empty_Rk) / c_eff(real_Rk)    <- the biologically real part
    total               = protocol × selection
    ```
    Every non-initial sample has exactly **one** collapse-vs-R0 row: `protocol` if it is in the `empty`
    group, `total` otherwise — so the empty round is presented **once, under one label**, never as its
    own selection. `selection` rows are extra, and exist for an arm only where a **round-matched** empty
    round does (match the empty round to the real round with the same number of transfers, not to R0).
    An empty round is a **peer** of every cell-line arm: it gets its own group matrix, its own colour,
    its own C_eff trajectory line and its own UpSet panel. It also gives an **empirical null** for
    enrichment fold-change and a data-derived `min_reads`.

### Writing the sample table in the config

Four keys per sample, all required. The constraints are not style — `name` and `group` both become
**paths on disk**, and the cross-reports discover samples by **globbing directories**:

| key | rule | why |
| --- | --- | --- |
| `name` | globally unique; `[A-Za-z0-9][A-Za-z0-9._+-]*` — no `/`, no spaces. (It *may* be the same as a group name — they are different things.) | the sample's **identity**: a wildcard, a **directory**, and the filename stem inside it (`results/umi/<name>/<name>.fused.parquet`). The reports identify a sample **by its directory name**, so a `/` would nest it one level deeper, where the `<tree>/*/` glob never looks — the sample would run to completion and then silently vanish from every report. |
| `group` | must be listed in `groups:`; `[A-Za-z0-9][A-Za-z0-9._+-]*`. Two **reserved** names: `initial` (the R0 baseline, exactly one sample) and `empty` (the PCR-only rounds, at most one per round). Everything else is a selection arm. | it becomes a **filename stem** in the directory the reports glob (`results/umi/`, `results/fragvec/`). A `/` in it would create a *subdirectory* there — sitting in exactly the namespace `BASE.glob("*/")` scans for samples. |
| `round` | an integer, required. R0 = the initial library. | it orders a group's samples within its matrix, and it is what pairs an `empty` round to the real rounds with the **same** transfer/PCR count. |
| `fastq` | absolute path, and **the file must exist**. | snakemake stats every input even on a dry run — a missing path fails with `MissingInputException` before anything runs. (A relative path resolves against the directory you run snakemake in, not against the config.) |

The sample order decides the report color and order:

-  Report colours are assigned in the order a
  group **first appears** in `samples:` (the `groups:` list never reaches a report). 
- **Keep `samples:` grouped** — one group's rounds contiguous, in ascending round order. Nothing
  re-sorts them for you.

## Key rules/steps

## Decode coverage filter (`decode_filter`)

A read's **decoded coverage** is the fraction of it the decoder explained: Σ(stop − start) over its
fragment calls (**anchors included** — they are real decoded bases) ÷ its post-trim length from
`fused_reads.fq`. The decoder's calls never overlap by construction, so the plain sum is exact.
`workflow/scripts/coverage_lib.py` defines it once; the filter and the cross report both call it,
so the number a read is judged on and the number the figure draws cannot drift apart.

Reads below `min_coverage` are **dropped**. A read well short of 1.0 is mostly sequence the decoder
could not resolve, yet it still carries a UMI and a fragment vector and so inflates every downstream
count as if it were clean.

```
decode/  ──►  covfilt/  ──►  umicorr/  ──►  umi/   ──┬─►  featurecount ──►  fragment report [CLEAN]
  (raw,        cov >= 0.9    (corrected)   (jackpot   ├─►  C_eff, UMI matrix/report
   kept)                                    -clean)   └─►  fragvec/ ──► vector matrix/report
   │
   └─►  featurecount  ──►  fragment report [RAW]
```

The raw `decode/` tree is never modified. `covfilt/<s>/<s>.coverage.parquet` records **every**
read's coverage *before* filtering. Per-sample counts land in
`covfilt/coverage_summary.tsv`.

With `enabled: false` the branches read `decode/` directly and no `covfilt/` tree is built.

## UMI processing

After decoding, each read carries two UMIs concatenated into `fused_seq` (~20–24 nt). One spine, no
modes:

1. **QC-filter** — drop garbage UMIs (non-ACGT, low complexity, or outside the 10–12 nt window).
2. **Drop anchor-only reads** — a read carrying nothing but the backbone decoded no ligand, so it
   must not inflate a molecule's read count.
3. **Correct** — relabel each survivor's UMI to the most-abundant ed≤1 variant, by clustering with
   `starcode -d 1 -s` (spheres, Levenshtein distance 1). **Nothing is collapsed** — duplicate
   molecules keep every one of their reads. Output: `umicorr/`.
4. **Jackpot-clean** — see below. Output: `umi/`, the clean end of the spine.

Each sample's `umicorr/<s>/<s>.umi_audit.parquet` is one row per **input** read: a dropped read is
still there, carrying the class that explains why. 

**Readouts.** `umi/` feeds all three: `featurecount` (the clean fragment report), `umi_complexity`
(C_eff) + `umi_matrix` + the cross-sample UMI report, and `fragvec`. They share one read set by
construction.

## Jackpot filter (`jackpot`)

A **jackpot** is a clonotype whose reads collapse onto very few UMIs: thousands of physical copies of
one construct whose UMI slot failed to randomise, so they pile onto ~one barcode. Ranked by raw
abundance it looks like a top hit; behaviourally it is an amplification/synthesis artefact.

**Detection is initial-library only, and the blacklist is applied everywhere.** That asymmetry is
forced. The ratio `n_reads / n_umis` has two possible causes — one molecule amplified into thousands
of reads (artefact), or a clone genuinely enriched by selection (a real hit). Only in R0, where every
construct starts at comparable abundance, is the second cause excluded by construction, so only there
is the statistic identifiable. Detecting on a selected round would blacklist true winners.

```
umicorr_initial/  ──►  jackpot detect  ──►  <R0>.umi_blacklist.tsv  + 2 audit reports
                                              └──►  jackpot apply (every sample)  ──►  umi/
```

`ratio_max = 10` is **cited**: longread_umi `BIN_CLUSTER_RATIO=10` (Karst et al., Nat Methods 2021).
A data-driven robust cut (median + k·MAD of log10 ratio) is computed and reported. `min_reads` reuses `fragvec.min_reads`.

With `enabled: false` the spine ends at `umicorr/` and every readout resolves there instead.

## Fragment-vector quantification (`fragvec`)

| level | depth in this data | usable? |
| --- | --- | --- |
| UMI (clone) | **~1.0 reads**, 96–99.6% singletons | no — you sample ~10⁻⁶ of the molecule pool |
| **fragment vector** | **4–22 reads mean; max up to 67,152** | **yes** |

Per sample and vector it counts **two axes** (the CRISPR-UMI convention, Michlits et al. 2017):

- `n_reads` — **abundance**
- `n_umis` — **incidence**: how many *independent clones* carry the vector
- `max_reads_per_umi` — the **jackpot detector**

`rpm` is the **depth-normalised frequency** (reads per million payload reads), which
is what makes a vector comparable across rounds of different depth — and it is what the trajectories are
built on. It deliberately does *not* scale by the sample's observed richness: Kuai et al.'s
`count × total_diversity / total_reads` uses the **library size**, a design constant identical in every
sample, whereas the *observed* per-sample richness differs per sample (it swings 4.5× here) and is itself
depth-dependent, so using it would distort the very cross-sample comparison it is meant to enable.  Counts below `min_reads` are **flagged, not dropped**. 

Unlike the per-clone UMI matrix (whose R0 column is ~empty, because R0's sequencing never sampled the
clones that later win), the **R0 column here is densely populated**, so R5/R0 enrichment is computable.

### ⚠️ Complexity and sequencing depth

Depth varies ~6× across samples here (94k–551k payload reads). **Every richness-style count — distinct
UMIs, distinct fragment vectors, shared species between samples — grows with depth**, and comparing it across samples of different depth is invalid.

Two depth-robust measures are emitted instead (`workflow/scripts/complexity_lib.py`), and the reports
plot these rather than the raw counts:

- **`c_eff = N(N−1) / Σ nᵢ(nᵢ−1)`** — the effective complexity (inverse-Simpson). It estimates the
  *pool's* read-collision probability, so it does not inflate with depth. Emitted per sample in both
  `umi_summary.tsv` (molecular) and `fragvec_summary.tsv` (vector-level). A sample with **no duplicates
  at all** (an un-bottlenecked `empty` round) yields a 95% **lower bound** — flagged as
  `c_eff_is_lower_bound`.
- **Subsampling** — expected richness at a common depth, `E[distinct] = Σᵢ [1 − (1 − m/N)^nᵢ]`
  (analytic, deterministic). The cross-sample reports subsample every sample to the shallowest one.

Columns that remain depth-dependent (`n_vectors`, `n_vectors_ge_min`, `pct_reads_in_vectors_ge_min`) are
marked ⚠ in the report tables. 

## Outputs (under `config.outdir`, default `results/`)


Below, `<tree>` is `umi` — the clean end of the UMI spine, which every readout reads; the initial
library lives in `umi_initial/`. With `jackpot.enabled: false` it is `umicorr` instead.

| Path                                                                                                          | What                                                                         |
| --------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `clean/<sample>.fastq`, `clean/clean_summary.tsv`                                                             | human-depleted, length-filtered reads + summary                              |
| `hmm/<adapters>.hmm` (+ hmmpress DB)                                                                          | trained primer HMM                                                           |
| `preprocess/<sample>/<sample>.full_length.fq` (+ rescued/…/stats/report/bed), `preprocess/batch_summary.tsv` | pychopper outputs                                                            |
| `decode/<sample>/…` (every non-`initial` sample), `decode_initial/<initial>/…`                                            | LigandFrags2 decode (fragments/umis/fused/trimmed parquet, fused_reads.fq, ligateddna.log) |
| `decode/<sample>/<sample>.coverage.parquet` (+ `decode_initial/`)                                             | every read's decoded coverage, measured on decode and ALWAYS built. The filter selects on it and the cross report renders it; neither recomputes it |
| `covfilt/<sample>/…` (every non-`initial` sample), `covfilt_initial/<initial>/…`, `covfilt/coverage_summary.tsv` | the same artifacts with low-coverage reads dropped, plus the per-sample drop row; `ligateddna.log` gains a provenance block. Absent when `decode_filter.enabled: false` |
| `umicorr/<sample>/…` (+ `umicorr_initial/`)                                                                   | garbage-UMI and anchor-only reads dropped, surviving UMIs corrected to their ed≤1 centroid. NOT a readout — it still carries the jackpot artefacts |
| `umicorr/<sample>/<sample>.umi_audit.parquet`                                                                 | one row per **input** read: its features, its class, and whether it survived. The only per-read record of why a read was dropped |
| `umi/<sample>/…` (+ `umi_initial/`)                                                                           | the clean end of the spine: the same artifacts with the jackpot blacklist applied and every aggregate re-derived. What featurecount, C_eff and fragvec read |
| `jackpot/<initial>.umi_blacklist.tsv`, `.umi_blacklist_summary.tsv`, `.fragvec_jackpot_report.html`, `.umi_jackpot_report.html` | the artefact barcodes the R0 library exposed, the thresholds and provenance, and the two audit reports |
| `umi/<sample>/<sample>.fused.parquet`, `.umi_species.tsv`, `.umi_summary.tsv` (+ `umi_initial/`)             | (mode `correct`) corrected UMIs over **payload-bearing reads only**: per-read `fused_seq_corrected` + `umi_read_count`, the species→count table, and filter/correction counts |
| `umi/<group>.umi_x_sample.{long.parquet,mtx,umis.tsv,samples.tsv}`                                            | (mode `correct`) one per group except `initial` (**`empty` included**): corrected-UMI × sample read-count matrix, R0 library as the first column |
| `umi/umi_report_cross.html`                                                                                   | (mode `correct`) cross-sample UMI QC report (abundance, species overlap, validity) |
| `fragvec/<sample>/<sample>.fragvec_counts.parquet` (+ `.fragvec_clones`, `.subvec_counts`, `.fragvec_summary.tsv`) | (fragvec) per fragment vector: `n_reads` (abundance), `n_umis` (incidence = independent clones), `max_reads_per_umi` (jackpot), `rpm` (depth-normalised frequency — the cross-sample comparable value), `is_noise` |
| `fragvec/<group>.fragvec_x_sample.{long.parquet,reads.mtx,umis.mtx,vectors.tsv,samples.tsv}`                  | (fragvec) one per group except `initial` (**`empty` included**): vector × sample matrices, BOTH axes, R0 first (and genuinely populated) |
| `fragvec/fragvec_report_cross.html`                                                                           | (fragvec) complexity collapse, incidence-vs-abundance jackpot control, enrichment trajectories, cross-arm clone stratification |
| `<tree>/<sample>/<sample>.featurecount.parquet` + `.summary.tsv`                                              | per-read feature counts                                                      |
| `<tree>/<sample>/<sample>.featurecount_report.html`                                                           | per-sample interactive report                                                |
| `<tree>/cross_sample_featurecount_report.html`                                                                | cross-sample comparison report                                               |

The initial library lands in a **separate** `*_initial/` tree so the cross report's `<tree>/*/`
sample discovery never double-counts it; it is wired in as the shared R0 baseline via
`report.r0_baseline: true`.
