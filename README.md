# ONT ligated-DNA selection-screen workflow

A Snakemake workflow for the Nanopore ligated-DNA selection screen: genomic depletion → primer-HMM
training → pychopper preprocessing → cONcat (new) decode → per-read feature counts → interactive HTML
reports. One UMI spine (QC-filter → ed≤1 correct → jackpot-clean) feeds three readouts: per-fragment
counts, effective complexity, and **fragment-vector quantification** (the two-axis abundance +
incidence readout).

```
minikow/dorado passed fastq ─┬─► genomic contamination─┐
(zcat per sample)            └─► length filter         └─┬─► train_hmm ─┐
                                                         │              ▼
                                                         └────────────► pychopper ──► cONcat ──► covfilt ──► umi_correct ──► jackpot ──► featurecount ─┬─► report_sample
                                                                                       (new)                                                           └─► report_cross
```

`decode_filter.min_coverage` (default **0.9**) drops reads the decoder barely explained — the one
read-level cut upstream of the fork, so every branch below shares a read set. See **Decode coverage
filter** below.

The UMI stage is one spine, with no modes: QC-filter garbage UMIs → correct near-identical UMIs
(ed≤1) → drop the jackpot artefacts the naive R0 library exposes. `umi/` is its clean end, and every
readout reads it. Nothing is ever **collapsed** — this data must not be UMI-deduplicated. See
**UMI processing** and **Jackpot filter** below.

There are **two fragment reports**, one either side of the UMI processing (`decode/` raw and `umi/`
clean); their difference is the read funnel.

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

### Writing the sample table

Four keys per sample, all required. The constraints are not style — `name` and `group` both become
**paths on disk**, and the cross-reports discover samples by **globbing directories**:

| key | rule | why |
| --- | --- | --- |
| `name` | globally unique; `[A-Za-z0-9][A-Za-z0-9._+-]*` — no `/`, no spaces. (It *may* be the same as a group name — they are different things.) | the sample's **identity**: a wildcard, a **directory**, and the filename stem inside it (`results/umi/<name>/<name>.fused.parquet`). The reports identify a sample **by its directory name**, so a `/` would nest it one level deeper, where the `<tree>/*/` glob never looks — the sample would run to completion and then silently vanish from every report. |
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

## Decode coverage filter (`decode_filter`)

A read's **decoded coverage** is the fraction of it the decoder explained: Σ(stop − start) over its
fragment calls (**anchors included** — they are real decoded bases) ÷ its post-trim length from
`fused_reads.fq`. The decoder's calls never overlap by construction, so the plain sum is exact.
`workflow/scripts/coverage_lib.py` defines it once; the filter and the cross report both call it,
so the number a read is judged on and the number the figure draws cannot drift apart.

Reads below `min_coverage` are **dropped**. A read well short of 1.0 is mostly sequence the decoder
could not resolve, yet it still carries a UMI and a fragment vector and so inflates every downstream
count as if it were clean.

This is the **only** read-level drop between decode and the UMI spine — `umi/`
and `fragvec/` all read the `covfilt/` tree, so no branch can disagree with another about which
reads exist:

```
decode/  ──►  covfilt/  ──►  umicorr/  ──►  umi/   ──┬─►  featurecount ──►  fragment report [CLEAN]
  (raw,        cov >= 0.9    (corrected)   (jackpot   ├─►  C_eff, UMI matrix/report
   kept)                                    -clean)   └─►  fragvec/ ──► vector matrix/report
   │
   └─►  featurecount  ──►  fragment report [RAW]
```

The raw `decode/` tree is never modified. `covfilt/<s>/<s>.coverage.parquet` records **every**
read's coverage *before* filtering — it is a first-class output, and the cross report's coverage
panel renders it (drawing the cut as a dotted line) rather than recomputing on the filtered tree,
which would show a distribution mysteriously truncated at the threshold. Per-sample counts land in
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

Levenshtein rather than Hamming distance is deliberate: the UMIs are variable length and ONT's
dominant error mode is indels. `starcode` is pinned to a single thread (`-t 1`) for byte-reproducible
cluster output (~0.5 s/sample). Correction is the only stage that runs it.

There is **no `umi.mode`**. It used to pick between `filter` and `correct`, back when a filter-only
tree fed `featurecount`; featurecount now reads `decode/` and `umi/`, so that tree had no consumer.
`dedup` never applied to this data at all (see `clarification.md`). A stale key is **rejected**, not
ignored.

Each sample's `umicorr/<s>/<s>.umi_audit.parquet` is one row per **input** read: a dropped read is
still there, carrying the class that explains why. Nothing else records that.

**Readouts.** `umi/` feeds all three: `featurecount` (the clean fragment report), `umi_complexity`
(C_eff) + `umi_matrix` + the cross-sample UMI report, and `fragvec`. They share one read set by
construction.

## Jackpot filter (`jackpot`)

A **jackpot** is a clonotype whose reads collapse onto very few UMIs: thousands of physical copies of
one construct whose UMI slot failed to randomise, so they pile onto ~one barcode. Ranked by raw
abundance it looks like a top hit; behaviourally it is an amplification/synthesis artefact.

**Detection is naive-library only, and the blacklist is applied everywhere.** That asymmetry is
forced. The ratio `n_reads / n_umis` has two possible causes — one molecule amplified into thousands
of reads (artefact), or a clone genuinely enriched by selection (a real hit). Only in R0, where every
construct starts at comparable abundance, is the second cause excluded by construction, so only there
is the statistic identifiable. Detecting on a selected round would blacklist true winners.

```
umicorr_initial/  ──►  jackpot detect  ──►  <R0>.umi_blacklist.tsv  + 2 audit reports
                                              └──►  jackpot apply (every sample)  ──►  umi/
```

`ratio_max = 10` is **cited**: longread_umi `BIN_CLUSTER_RATIO=10` (Karst et al., Nat Methods 2021).
A data-driven robust cut (median + k·MAD of log10 ratio) is computed and **reported for
transparency**, never acted on. `min_reads` is not a key — it reuses `fragvec.min_reads`, because
the clone table the blacklist is drawn from is restricted by that same floor.

This runs **before C_eff**, deliberately: the complexity weighting is quadratic, so an artefact in
the top lineage corrupts it more than any biology can (`clarification.md`).

Two limits worth stating, both in the reports:

- The R0 library is the **least** jackpot-susceptible sample in the study (jackpotting is worst when
  few templates enter the PCR). This catches the synthesis/aliquot artefacts already in the pool; it
  does **not** catch a jackpot generated inside a bottlenecked selected round.
- This is a **filter, not a deduplication** — it drops a named, audited set of artefact barcodes.
  High-count lineages are the signal here and are never collapsed.

With `enabled: false` the spine ends at `umicorr/` and every readout resolves there instead.

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
