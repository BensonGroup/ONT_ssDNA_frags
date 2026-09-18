"""Export the reference within-batch figure families from workflow assessment tables."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import polars as pl

import palettes as P
import within_batch_overlap_figs as W

CAPTIONS = {
    'W01': 'Shared singleton vectors and overlap coefficients by fragment order at the deepest available matched depth. Points are pair-class medians; bars are interquartile ranges.',
    'W02': 'Depth response of singleton-vector sharing and size-adjusted selected/selected versus selected/empty sharing. Undefined ratios remain unavailable.',
    'W03': 'Re-detection and conditional singleton probability in sample B by UMI abundance in A. Curves show pair-class medians and interquartile ranges.',
    'W04': 'Pooled versus high-order overlap coefficients at matched depth, and singleton versus unrestricted Jaccard overlap at native depth. Zeros use labelled log-axis floors.',
    'W05': 'Pairwise shared singleton vectors of fragment order at least five at the deepest available matched depth. Self-comparisons are masked.',
    'W06': 'Shared singleton fragvecs versus shared corrected UMIs for the same sample pairs. These measure architecture identity and barcode identity respectively.',
    'W07': 'Number of configured samples carrying each singleton fragvec, by fragment-order stratum. Zero counts use a labelled plotting floor.',
    'W08': 'Observed vector richness, UMI depth and singleton fraction per sample. These are depth-dependent descriptions.',
    'W09': 'Size-adjusted selected/selected sharing divided by selected/empty sharing. Null bands span the median to the 95th percentile of sample-label permutations; p-values are one-sided.',
    'W10': 'Final retained reads, summed directly from n_reads in the clean corrected-UMI species tables. Every retained read contributes; this is not UMI deduplication.',
    'V01': 'Initial-library overlap with each group at its final sampled round in UMI and singleton-vector spaces. Venn areas use one common scale per row; zero-sized sets are labelled numerically.',
}
REQUIRES = {
    'W01': ('depth_matched_overlap.tsv',),
    'W02': ('depth_matched_overlap.tsv',),
    'W03': ('abundance_tiers.tsv',),
    'W04': ('depth_matched_overlap.tsv', 'fragvec_overlap.tsv'),
    'W05': ('depth_matched_overlap.tsv', 'sample_sizes.tsv'),
    'W06': ('fragvec_overlap.tsv', 'umi_overlap.tsv'),
    'W07': ('sharing_spectrum.tsv',),
    'W08': ('sample_sizes.tsv',),
    'W09': ('class_contrasts.tsv',),
    'W10': ('sample_read_accounting.tsv',),
    'V01': ('venn_counts.tsv',),
}


def render_all(tables: Path, out: Path) -> dict:
    provenance = json.loads((tables / 'run_manifest.json').read_text())
    if provenance.get('mode') != 'within-batch':
        raise ValueError('Expected within-batch assessment tables')
    samples = provenance['samples']
    groups = list(dict.fromkeys(row['group'] for row in samples))
    W.GROUP_ORDER = tuple(groups)
    frames = {name: W._tsv(tables, name) for name in {n for names in REQUIRES.values() for n in names}}
    available_classes = set(frames['fragvec_overlap.tsv']['pair_class'].to_list())
    W.CLASS_ORDER = tuple(c for c in W.CLASS_STYLE if c in available_classes)
    out.mkdir(parents=True, exist_ok=True)
    W.use_style()
    manifest = {'source': str(tables.resolve()), 'analysis': provenance,
                'table_sha256': {name: hashlib.sha256((tables / name).read_bytes()).hexdigest()
                                 for name in frames}, 'figures': []}
    captions = ['# Within-batch figures', '', f'Source tables: `{tables.resolve()}`.', '']
    with PdfPages(out / 'within_batch_figures.pdf') as combined:
        for fid, builder in W.BUILDERS.items():
            record = {'id': fid, 'caption': CAPTIONS[fid]}
            manifest['figures'].append(record)
            empty = [name for name in REQUIRES[fid] if frames[name].is_empty()]
            reason = f'No applicable rows in {", ".join(empty)}' if empty else None
            if fid == 'W08' and not (frames['sample_sizes.tsv']['sum_n_umis'] > 0).any():
                reason = 'No observed vector/UMI pairs for the log-scale richness plot'
            if fid == 'W02' and 'empty' not in groups:
                reason = 'No empty control group for the depth-response contrast'
            if fid == 'W09' and not reason:
                if frames['class_contrasts.tsv'].filter(pl.col('contrast_bio_over_bioempty').is_finite()).is_empty():
                    reason = 'No defined selected/selected versus selected/empty contrast'
            if reason:
                record.update(status='unavailable', reason=reason)
                captions.extend([f'## {fid}', '', reason, ''])
                continue
            try:
                fig, stem = builder(tables)
                fig.canvas.draw()
                if not P.check_min_font(fig):
                    raise ValueError('Figure contains text below 5 pt')
                for ext in ('pdf', 'png'):
                    fig.savefig(out / f'{stem}.{ext}')
                    record[ext] = f'{stem}.{ext}'
                combined.savefig(fig)
                record.update(status='computed', title=stem, minimum_font_check=True)
                captions.extend([f'## {fid}', '', CAPTIONS[fid], '', f'![{fid}]({stem}.png)', ''])
            except Exception as error:
                record.update(status='failed', reason=f'{type(error).__name__}: {error}')
                print(f"ERROR within-batch {fid}: {record['reason']}", flush=True)
            finally:
                plt.close('all')
    manifest['computed'] = sum(r['status'] == 'computed' for r in manifest['figures'])
    manifest['failed'] = sum(r['status'] == 'failed' for r in manifest['figures'])
    manifest['unavailable'] = sum(r['status'] == 'unavailable' for r in manifest['figures'])
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (out / 'figure_legends.md').write_text('\n'.join(captions) + '\n')
    return manifest
