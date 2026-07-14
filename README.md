# ONT ligated-DNA selection-screen workflow

A Snakemake workflow for the Nanopore ligated-DNA selection screen: genomic depletion → primer-HMM
training → pychopper preprocessing → cONcat (new) decode → per-read feature counts → interactive HTML
reports, plus two independent analysis branches off `decode`: **UMI processing** (filter / correct /
dedup) and **fragment-vector quantification** (the two-axis abundance + incidence readout).

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

## Scope

**One config = one project.** A project has:

- **one initial library** — `group: initial`, the R0 baseline, exactly one sample;
- **at most one `empty` arm** — `group: empty`, the PCR-only / no-cell rounds, shared as the null for
  *every* experimental arm (one empty round per selection round, not one per cell line);
- **any number of experimental selection arms** — `group: A549`, `group: MDA`, …

A sample's **`group` decides everything**: which tree it is decoded into, whether it forms a group
matrix, how its complexity collapse is labelled. There is no `role` field. See
[Writing the sample table](#writing-the-sample-table).

**This workflow produces diagnostics, not conclusions.** It decodes, QCs, corrects UMIs, quantifies
fragment vectors, and emits **count tables, per-group × sample matrices, and diagnostic reports**. It
deliberately does **not** do project-specific downstream analysis — no hit calling, no enrichment
testing, no structure work. Take the matrices and count tables and do whatever your project needs;
every number in the reports is reproducible from them.

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
- **fragvec** — the fragment-vector branch (see below). A sibling of the `umi` branch; run either or both.
  ```yaml
  fragvec:
    enabled: true
    umi_source: corrected  # corrected (needs umi.mode: correct) | raw (stand-alone)
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

### Writing the sample table

Four keys per sample, all required. The constraints are not style — `name` and `group` both become
**paths on disk**, and the cross-reports discover samples by **globbing directories**:

| key | rule | why |
| --- | --- | --- |
| `name` | globally unique; `[A-Za-z0-9][A-Za-z0-9._+-]*` — no `/`, no spaces. Must not equal any group name. | it becomes a **directory** *and* the filename stem inside it: `results/umi/<name>/<name>.fused.parquet`, `results/dedup/<name>/<name>.featurecount.parquet`. A group writes `results/umi/<group>.umi_x_sample.mtx` into that **same parent directory**, so names and groups share one namespace on disk — and the reports identify a sample *by its directory name*. |
| `group` | must be listed in `groups:`; `[A-Za-z0-9][A-Za-z0-9._+-]*`. Two **reserved** names: `initial` (the R0 baseline, exactly one sample) and `empty` (the PCR-only rounds, at most one per round). Everything else is a selection arm. | it becomes a **filename stem** in the directory the reports glob (`results/umi/`, `results/fragvec/`). A `/` in it would create a *subdirectory* there — sitting in exactly the namespace `BASE.glob("*/")` scans for samples. |
| `round` | an integer, required. R0 = the initial library. | it orders a group's samples within its matrix, and it is what pairs an `empty` round to the real rounds with the **same** transfer/PCR count. |
| `fastq` | absolute path, and **the file must exist**. | snakemake stats every input even on a dry run — a missing path fails with `MissingInputException` before anything runs. (A relative path resolves against the directory you run snakemake in, not against the config.) |

Two rules the code **cannot** check for you, so you have to know them:

- **Append a new group's rows at the END of `samples:`.** Report colours are assigned in the order a
  group **first appears** in `samples:` (the `groups:` list never reaches a report). Insert `empty_R2`
  between `A549_R2` and `MDA_R1` and MDA silently shifts from green to vermillion — in all three
  reports, with no error.
- **Keep `samples:` grouped** — one group's rounds contiguous, in ascending round order. Nothing
  re-sorts them for you.

Everything else **raises at parse time, naming the offending sample**, before any compute: a leftover
`role:` key; a missing, duplicated or badly-formed `name`; an undeclared or missing `group`; a group
declared but unused; a missing or non-integer `round`; a group name with a `/` in it or colliding with
a sample name; two `initial` samples; two `empty` samples in the same round; a `samples:` table whose
file forgets to declare `groups:`. You will never get a silently wrong result from a bad table — you
get a message telling you which row to fix.

## Fragment-vector quantification (`fragvec`) — the two-axis readout

The second, **independent** analysis branch off `decode` (a sibling of the `umi` branch — run either,
or both). A read's **fragment vector** is its sorted set of **payload** `fragment_id`s (anchors excluded;
`_rc` kept distinct, since a reverse-ligated fragment is a different species). That vector **is** the
library member — and it is where the statistical power lives:

| level | depth in this data | usable? |
| --- | --- | --- |
| UMI (clone) | **~1.0 reads**, 96–99.6% singletons | no — you sample ~10⁻⁶ of the molecule pool |
| **fragment vector** | **4–22 reads mean; max up to 67,152** | **yes** |

Per sample and vector it counts **two axes** (the CRISPR-UMI convention, Michlits et al. 2017):

- `n_reads` — **abundance**
- `n_umis` — **incidence**: how many *independent clones* carry the vector
- `max_reads_per_umi` — the **jackpot detector**

The pair is what makes a call trustworthy. 2,530 reads spread over 2,078 distinct clones is
reproducible selection; 2,530 reads from *one* clone is a single amplified lineage and must **not** be
read as enrichment. `rpm` is the **depth-normalised frequency** (reads per million payload reads), which
is what makes a vector comparable across rounds of different depth — and it is what the trajectories are
built on. It deliberately does *not* scale by the sample's observed richness: Kuai et al.'s
`count × total_diversity / total_reads` uses the **library size**, a design constant identical in every
sample, whereas the *observed* per-sample richness differs per sample (it swings 4.5× here) and is itself
depth-dependent, so using it would distort the very cross-sample comparison it is meant to enable. A
*constant* factor only rescales the axis; a *per-sample* one corrupts the trajectory. Counts below
`min_reads` are **flagged, never
dropped** — a vector at 3 reads in R1 and 800 in R5 is exactly the hit you want. Sub-vector marginals
(single fragments, fragment pairs) are emitted as a *lossy denoising view*, never a replacement.

Unlike the per-clone UMI matrix (whose R0 column is ~empty, because R0's sequencing never sampled the
clones that later win), the **R0 column here is densely populated**, so R5/R0 enrichment is computable.

### ⚠️ Complexity and sequencing depth — read this before comparing any count

Depth varies ~6× across samples here (94k–551k payload reads). **Every richness-style count — distinct
UMIs, distinct fragment vectors, shared species between samples — grows with depth**, because the harder
you look the more distinct things you find. Such a count is a property of the *run*, not of the
*library*, and comparing it across samples of different depth is invalid.

> **Rule of thumb: if a number gets bigger when you sequence more, it is not a property of your library.**

Concretely: A549_R1 appears to have **8.3×** more fragment vectors than A549_R5 — but R1 was sequenced
4.5× deeper. Rarefied to a common depth the honest ratio is **2.4×**.

Two depth-robust measures are emitted instead (`workflow/scripts/complexity_lib.py`), and the reports
plot these rather than the raw counts:

- **`c_eff = N(N−1) / Σ nᵢ(nᵢ−1)`** — the effective complexity (inverse-Simpson). It estimates the
  *pool's* read-collision probability, so it does not inflate with depth. Emitted per sample in both
  `umi_summary.tsv` (molecular) and `fragvec_summary.tsv` (vector-level). A sample with **no duplicates
  at all** (an un-bottlenecked `empty` round) yields a 95% **lower bound** — flagged as
  `c_eff_is_lower_bound`.
- **Rarefaction** — expected richness at a common depth, `E[distinct] = Σᵢ [1 − (1 − m/N)^nᵢ]`
  (analytic, deterministic). The cross-sample reports rarefy every sample to the shallowest one.

Columns that remain depth-dependent (`n_vectors`, `n_vectors_ge_min`, `pct_reads_in_vectors_ge_min`) are
marked ⚠ in the report tables. Note also that `min_reads` is a **raw count** applied across that 6× depth
spread, so a vector at the same *frequency* can be "signal" in a deep sample and "noise" in a shallow one;
the principled threshold comes from the `empty` rounds (above).

`workflow/scripts/effective_complexity.py` is a standalone CLI for the same estimator — useful for
running the protocol/selection decomposition on an arbitrary set of samples without invoking Snakemake.

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

Per **group** (config `group` — every group except `initial`, the `empty` arm included: it is a peer
arm, not a special case), one corrected-UMI × sample matrix with the R0 library as the
**first column** (no all-sample matrix): `umi/<group>.umi_x_sample.long.parquet` (`fused_umi,
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
| `decode/<sample>/…` (every non-`initial` sample), `decode_initial/<initial>/…`                                            | LigandFrags2 decode (fragments/umis/fused/trimmed parquet, fused_reads.fq, ligateddna.log) |
| `dedup/<sample>/…` (every non-`initial` sample), `dedup_initial/<initial>/…`                                              | the same artifacts, QC-filtered (and, with dedup on, reduced to one representative read per UMI molecule); `ligateddna.log` gains a provenance block |
| `dedup/<sample>/<sample>.{umi_counts,umi_clusters}.tsv`, `.umi_audit.parquet`, `.umi_dedup_summary.tsv`        | that sample's UMI QC, beside the data it describes: UMI counts, starcode clusters (empty unless dedup on), a one-row-per-input-read audit, and its summary row |
| `dedup/umi_dedup_summary.tsv`                                                                                 | all samples: reads in → reads kept (molecules out with dedup on), drop classes, duplicate rate |
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
