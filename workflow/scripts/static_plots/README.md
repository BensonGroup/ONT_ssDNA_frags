# Workflow static figures

`export_reports.py` adapts `scripts/batch3_html_static_figs.py`.
`html_report_plotting.py`, `umi_report_cross_figs.py`, and `palettes.py` reuse the
supplied reference implementations. The within-batch producer/renderers and Venn
geometry likewise reuse `within_batch_overlap.py`, `within_batch_overlap_figs.py`,
and `venn_initial_vs_r5_fig.py`. Original files in `scripts/` are unchanged.

Workflow adaptations:

- Explicit report inputs replace recursive discovery and batch-specific defaults.
- Config metadata supplies sample identities and order. UMI facets and axes follow
  the supplied groups, rounds and values.
- PDF/PNG output, captions, source snapshots and trace checks retain the reference
  approach. Combined PDFs use Matplotlib directly.
- The supplied `science.mplstyle` provides fonts and publication sizing. The index
  template and CSS use the shared HTML kit and are bundled locally.

The `static_plots` rule owns `<outdir>/static_plots/`; rerunning it regenerates that
directory. It reads HTML reports and, when `fragvec.enabled` is true, the separate
`within_batch_overlap` rule's tables. No cross-batch figures are included.

## Within-batch tables and figures

The producer retains the reference's singleton/all tiers, fragment-order strata,
Poisson count intervals, sampling of distinct fragvec/UMI pairs without replacement, and sample-label
permutations. It reads full workflow counts and read maps instead of depending on
`clonerank_initial/`. A consistency check compares all distinct vector/UMI pairs
against per-vector `n_umis`; the thresholded `fragvec_clones.parquet` is never used.
Anchor exclusions come from the workflow's fragment-vector summaries.

The numerical kernels produce sample sizes, native and depth-matched overlap,
abundance tiers, prevalence, UMI overlap and class contrasts. Their TSV comments
and figure captions describe the methods without carrying conclusions from the
reference dataset. Undefined contrasts have unavailable p-values. Empty matching
tables fall back to native-depth tables for class contrasts; depth-matched figure
families remain unavailable when no requested depth is eligible.

W10 uses the sum of `n_reads` from each clean corrected-UMI species table. V01 compares
the initial library to every group's final sampled round, using the supplied Venn
geometry. Zero-sized sets have explicit numerical labels. Missing control groups
or empty comparison tables are recorded as unavailable; unexpected rendering
errors fail the export and are recorded in its manifest.

`static_overlap.depths`, `reps`, `permutations`, and `seed` control the reference
analysis. Outputs are `<outdir>/within_batch_overlap/` for tables and
`<outdir>/static_plots/within_batch/` for figures, captions and provenance.
Neither table generation nor rendering modifies source reads or pipeline counts.

## Verification

From the repository root:

```bash
pixi run -e default python -m unittest discover -s workflow/tests -v
```

Tests generate HTML, Parquet, TSV and metadata fixtures under `.codex/scratch/`.
They verify UMI-pair sampling identity at full depth, retained-read counting, metadata
and path handling, all within-batch figure families, and unavailable comparisons.
Snakemake dry runs allow only the static-export and within-batch-table rules with
synthetic inputs; they do not validate or execute upstream sequencing analysis.
No real datasets are opened and no upstream jobs are run.
