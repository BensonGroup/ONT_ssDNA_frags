#!/usr/bin/env python3
"""Pure UMI-classification helpers for the ligand2frags dedup pipeline.

These are the single source of truth for STEP 1 (garbage-UMI classification). They are
deliberately dependency-free -- NO polars / numpy imports at module top level -- so the
logic can be unit-tested with a plain ``python3`` (no pixi env required), and so the
per-sample driver (``umi_dedup.py``) and any QC script call exactly the same code.

Public API:
    maxrun(seq)                       -> longest run of identical characters (int, 0 for "")
    max_base_frac(seq)                -> max single-character fraction of seq (float, 0.0 for "")
    classify(fused_seq, umi_len_0, umi_len_1, ...) -> one of:
        "keep" | "drop_nonacgt" | "drop_lowcomplexity" | "drop_umilen"

Reporting precedence (a read may satisfy several drop conditions; it is counted once, under
the FIRST that matches, so per-class counts sum to the total dropped):
    drop_nonacgt  >  drop_lowcomplexity  >  drop_umilen

Run the self-test:  python3 workflow/scripts/umi_filter_lib.py
"""
from __future__ import annotations

# --- defaults (mirrored by umi_dedup.py CLI flags --max-homopolymer / --max-base-frac) ---
DEFAULT_MAX_HOMOPOLYMER = 8      # drop if the longest homopolymer run is >= this
DEFAULT_MAX_BASE_FRAC = 0.7      # drop if any single base's fraction is >= this
UMI_LEN_MIN = 10                 # each of the read's two UMI lengths must be within
UMI_LEN_MAX = 12                 # [UMI_LEN_MIN, UMI_LEN_MAX] inclusive

_ACGT = frozenset("ACGT")        # uppercase canonical bases; anything else (n, N, -, ...) drops


def maxrun(seq: str) -> int:
    """Length of the longest run of a single repeated character in ``seq`` (0 for empty)."""
    best = run = 0
    prev = None
    for c in seq:
        run = run + 1 if c == prev else 1
        if run > best:
            best = run
        prev = c
    return best


def max_base_frac(seq: str) -> float:
    """Largest fraction of ``seq`` occupied by any single character (0.0 for empty)."""
    n = len(seq)
    if n == 0:
        return 0.0
    counts: dict[str, int] = {}
    for c in seq:
        counts[c] = counts.get(c, 0) + 1
    return max(counts.values()) / n


# Alias so classify() can call the helper even though it exposes a ``max_base_frac`` keyword
# (the kwarg name matches the --max-base-frac CLI flag; the alias avoids the name shadow).
_max_base_frac = max_base_frac


def classify(
    fused_seq: str,
    umi_len_0: int,
    umi_len_1: int,
    *,
    max_homopolymer: int = DEFAULT_MAX_HOMOPOLYMER,
    max_base_frac: float = DEFAULT_MAX_BASE_FRAC,
    umi_len_min: int = UMI_LEN_MIN,
    umi_len_max: int = UMI_LEN_MAX,
) -> str:
    """Classify one read from its fused UMI sequence and its two UMI lengths.

    Precedence: nonacgt > lowcomplexity > umilen (first match wins).
    """
    # (a) any non-ACGT character (lowercase, N, gap, insertion junk, ...) -> drop.
    if any(c not in _ACGT for c in fused_seq):
        return "drop_nonacgt"
    # (b) low complexity: long homopolymer run OR one base dominates the sequence.
    if maxrun(fused_seq) >= max_homopolymer or _max_base_frac(fused_seq) >= max_base_frac:
        return "drop_lowcomplexity"
    # (c) either UMI length out of the accepted window.
    if not (umi_len_min <= umi_len_0 <= umi_len_max) or not (umi_len_min <= umi_len_1 <= umi_len_max):
        return "drop_umilen"
    return "keep"


def _selftest() -> None:
    # ---- maxrun ----
    assert maxrun("") == 0
    assert maxrun("A") == 1
    assert maxrun("AAAAAAA") == 7          # exactly 7
    assert maxrun("AAAAAAAA") == 8         # exactly 8
    assert maxrun("ACGTACGT") == 1
    assert maxrun("ACAAAGGGGGTT") == 5     # run of 5 G's

    # ---- max_base_frac ----
    assert max_base_frac("") == 0.0
    assert max_base_frac("AAAA") == 1.0
    assert max_base_frac("ACGT") == 0.25
    # 14 A's out of 20, spread so the longest A-run is only 2 -> isolates the base-fraction test.
    seq70 = ("AAG" * 7)[:20]               # "AAGAAGAAGAAGAAGAAGAA": 14 A, 6 G, maxrun 2
    assert len(seq70) == 20 and seq70.count("A") == 14 and maxrun(seq70) == 2
    assert max_base_frac(seq70) == 0.7     # 14/20 == 0.7 exactly (both are the same double)
    seq65 = ("AAG" * 6)[:18] + "AT"        # 18: 12 A / 6 G, + "AT" -> 13 A / 20, maxrun 2
    assert len(seq65) == 20 and seq65.count("A") == 13 and maxrun(seq65) == 2
    assert max_base_frac(seq65) == 0.65

    # ---- classify: pure homopolymer ----
    assert classify("A" * 20, 10, 10) == "drop_lowcomplexity"

    # ---- classify: homopolymer run boundary 7 vs 8 (base fraction kept < 0.7, lengths OK) ----
    run7 = "AAAAAAA" + "CGTACGTAGCTA"      # 19; run of 7 A, no base >= 0.7
    run7 = run7[:20] if len(run7) >= 20 else run7 + "C" * (20 - len(run7))
    assert maxrun(run7) == 7 and max_base_frac(run7) < 0.7
    assert classify(run7, 10, 10) == "keep"
    run8 = "AAAAAAAA" + "CGTACGTAGCT"      # run of 8 A
    assert maxrun(run8) == 8
    assert classify(run8, 10, 10) == "drop_lowcomplexity"

    # ---- classify: base-fraction 0.7 boundary (>= drops), 0.65 keeps ----
    assert classify(seq70, 10, 10) == "drop_lowcomplexity"
    assert classify(seq65, 11, 11) == "keep"

    # ---- classify: lowercase / N -> nonacgt ----
    assert classify("acgtACGTACGTACGTACGT", 10, 10) == "drop_nonacgt"
    assert classify("ACGTNNNNACGTACGTACGT", 10, 10) == "drop_nonacgt"

    # ---- classify: umi_len 9 / 10 / 12 / 13 (benign fused_seq) ----
    ok = "ACGTAGCATGCATGACTGCA"            # 20nt, ACGT, low run, balanced
    assert classify(ok, 10, 12) == "keep"
    assert classify(ok, 12, 12) == "keep"
    assert classify(ok, 10, 10) == "keep"
    assert classify(ok, 9, 11) == "drop_umilen"   # 9 below min
    assert classify(ok, 11, 13) == "drop_umilen"  # 13 above max
    assert classify(ok, 9, 9) == "drop_umilen"
    assert classify(ok, 13, 12) == "drop_umilen"

    # ---- precedence: nonacgt beats lowcomplexity; lowcomplexity beats umilen ----
    assert classify("N" * 20, 5, 5) == "drop_nonacgt"          # also low-complexity & bad len
    assert classify("A" * 20, 5, 5) == "drop_lowcomplexity"    # also bad len

    # ---- custom thresholds are honoured ----
    assert classify(run8, 10, 10, max_homopolymer=9) == "keep"        # raise run threshold
    assert classify(seq65, 10, 10, max_base_frac=0.6) == "drop_lowcomplexity"

    print("umi_filter_lib self-test: ALL PASS")


if __name__ == "__main__":
    _selftest()
