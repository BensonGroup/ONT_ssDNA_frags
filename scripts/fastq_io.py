#!/usr/bin/env python3
"""The single FASTQ reader/writer for the standalone per-study scripts, so `top_vector_reads.py`,
`clone_top_reads.py` and `all_umi_consensus.py` cannot disagree about how a decode record is parsed
or emitted (same reason `fitness_lib` is shared between the fitness scripts).

Two entry points, ONE parser. `iter_fastq_records` streams and yields; `read_fastq_records` is a
``dict()`` over it. The dict is right when the caller needs random access to a few thousand reads;
the iterator is what `all_umi_consensus.py --emit-singletons` needs to copy 3.8M reads straight
back out to disk without first building hundreds of MB of write-once strings.

Both extractors pull sequences out of `results/decode/<sample>/<sample>.fused_reads.fq` by
`read_acc` -- the header's first whitespace token, `<start>:<stop>|<uuid>`. That file is a real
FASTQ, so the QUALITY is already in hand when the record is parsed; the extractors used to drop it
and write FASTA. They now emit FASTQ, and the quality line is copied through verbatim.

Format assumptions, verified on results/decode/A549_R1/A549_R1.fused_reads.fq (468,661 records):
strict 4-line records, a bare `+` on the third line, `len(seq) == len(qual)` everywhere. Nothing
here tolerates a violation -- a wrapped or truncated FASTQ means the wrong file was passed, and
guessing would move the results without saying so.

Why `strand_field` throws most of the header tail away
-------------------------------------------------------
A decode header looks like

    @1:131|2a2ecf4d-... strand=+ umi1=GGCACGTGGT umi2=TGGTGTCGCT

and everything after the read_acc is PYCHOPPER's, passed through untouched by LigandFrags2
(`rule pychopper`, `-U`, workflow/rules/preprocess.smk). Those `umi1=`/`umi2=` are a DIFFERENT,
UNCORRECTED call from the `umi` a clone is keyed on. The pipeline order is

    pychopper -> decode -> umi_correct -> fragvec

so the fq predates correction entirely. The extractors' `umi=` comes from the parquet side
(`fragvec_reads.parquet:umi`, built by `build_fragvec.py --umi-col fused_seq_corrected`, i.e. the
starcode ed<=1 centroid from the `umi/` tree) and is joined to the fq on `read_acc` alone. On the
rank-1 A549 read above the two callers disagree by a one-base window shift:

    pychopper     GGCACGTGGT + TGGTGTCGCT
    LigandFrags2  GGGCACGTGG + ATGGTGTCGC   <- this is what `umi=` reports

Carrying a raw pair that contradicts the UMI the clone is DEFINED by invites keying on the wrong
string, so only `strand=` -- a real property of the emitted sequence -- survives into the output
header.

Run the self-test:  pixi run -e default python scripts/fastq_io.py
"""
from __future__ import annotations

from pathlib import Path

__all__ = ["iter_fastq_records", "read_fastq_records", "strand_field", "write_fastq"]


def iter_fastq_records(fq_path, want=None):
    """Streaming single pass over a 4-line FASTQ, yielding ``(read_acc, header_tail, seq, qual)``
    in file order and holding ONE record at a time.

    `read_fastq_records` materialises a dict of every wanted record, which is right when the
    caller needs random access to a few thousand reads and wrong when it needs to copy millions
    of them straight back out to disk. ``all_umi_consensus.py --emit-singletons`` copies
    3,800,790 singleton reads: that dict would be hundreds of MB of strings each used exactly
    once. Same parsing and the same fatal invariants -- this IS the parser now, and
    `read_fastq_records` is a ``dict()`` over it, so the two can never drift apart.

    ``want`` is any container answering ``in`` -- a set, or the caller's ``read_acc -> value``
    dict, which saves building a second copy of half a million keys. ``None`` means every record.

    Fatal on a truncated record or a seq/qual length mismatch: both mean the input is not the
    decode FASTQ this was pointed at. The truncation check fires for EVERY record, wanted or
    not; the length check only for wanted ones. That asymmetry is deliberate and pinned by the
    selftest -- a truncated file is malformed no matter what you asked for, but paying a
    length check on millions of unwanted records is pure cost.
    """
    fq_path = Path(fq_path)
    with open(fq_path) as fh:
        while True:
            header = fh.readline()
            if not header:
                break
            seq, plus, qual = fh.readline(), fh.readline(), fh.readline()
            if not qual:
                raise SystemExit(f"[FATAL] {fq_path}: truncated record at {header.rstrip()!r} "
                                 f"(expected 4 lines, got fewer)")
            body = header[1:].rstrip("\n")          # header: "@<read_acc> <tail...>"
            acc, _, tail = body.partition(" ")
            if want is not None and acc not in want:
                continue
            seq, qual = seq.rstrip("\n"), qual.rstrip("\n")
            if len(seq) != len(qual):
                raise SystemExit(f"[FATAL] {fq_path}: {acc} has {len(seq)} bases but "
                                 f"{len(qual)} quality values")
            yield acc, tail, seq, qual


def read_fastq_records(fq_path, want: set[str]) -> dict[str, tuple[str, str, str]]:
    """Single streaming pass over a 4-line FASTQ, collected into a dict.

    Returns ``read_acc -> (header_tail, sequence, quality)`` for the headers whose first
    whitespace token is in ``want``. ``header_tail`` is the raw text AFTER that token, returned
    unparsed -- callers decide what of it to keep (see `strand_field`).

    A duplicated ``read_acc`` resolves last-one-wins, as it always has. For a caller that only
    streams its records straight back out, use `iter_fastq_records` and hold nothing.
    """
    return {acc: (tail, seq, qual) for acc, tail, seq, qual in iter_fastq_records(fq_path, want)}


def strand_field(tail: str) -> str:
    """``'strand=+'`` / ``'strand=-'`` pulled out of a decode header tail, ``''`` if absent.

    The rest of the tail is pychopper's raw `umi1=`/`umi2=` -- see the module docstring for why it
    is deliberately not carried into the output.
    """
    for tok in tail.split():
        if tok.startswith("strand="):
            return tok
    return ""


def write_fastq(path, records) -> int:
    """Write 4-line FASTQ records, in the order given. Returns the record count.

    ``records`` is an iterable of ``(header, seq, qual)`` where ``header`` is the full text that
    follows ``@`` (the caller builds it: read_acc plus whatever provenance fields it wants). The
    third line is a bare ``+``, matching what decode itself emits.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as out:
        for header, seq, qual in records:
            if len(seq) != len(qual):
                raise SystemExit(f"[FATAL] {path}: refusing to write {header.split(' ')[0]} -- "
                                 f"{len(seq)} bases but {len(qual)} quality values")
            out.write(f"@{header}\n{seq}\n+\n{qual}\n")
            n += 1
    return n


def _selftest() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        src = d / "in.fq"
        # r2 carries no tail at all; r3 is present in the file but never requested.
        src.write_text(
            "@1:10|aaa strand=+ umi1=ACGTACGTAC umi2=TGCATGCATG\n"
            "ACGTACGT\n+\nIIIIHHHH\n"
            "@1:20|bbb\n"
            "TTTT\n+\n!!!!\n"
            "@1:30|ccc strand=- umi1=GGGGGGGGGG umi2=CCCCCCCCCC\n"
            "GGGGCCCC\n+\n########\n"
        )

        got = read_fastq_records(src, {"1:10|aaa", "1:20|bbb"})
        assert set(got) == {"1:10|aaa", "1:20|bbb"}, got.keys()     # `want` filters, c is skipped

        # the acc/tail split is at the FIRST space, and seq/qual pair with the right read
        tail_a, seq_a, qual_a = got["1:10|aaa"]
        assert tail_a == "strand=+ umi1=ACGTACGTAC umi2=TGCATGCATG"
        assert (seq_a, qual_a) == ("ACGTACGT", "IIIIHHHH")
        assert got["1:20|bbb"] == ("", "TTTT", "!!!!")               # a tail-less header is fine

        # strand_field keeps ONLY strand -- the raw umi pair must not survive it
        assert strand_field(tail_a) == "strand=+"
        assert strand_field(read_fastq_records(src, {"1:30|ccc"})["1:30|ccc"][0]) == "strand=-"
        assert strand_field("") == "" and strand_field("umi1=ACGT umi2=TGCA") == ""
        for t in (tail_a, "strand=- umi1=GGGG umi2=CCCC"):
            assert "umi1=" not in strand_field(t) and "umi2=" not in strand_field(t)

        # round-trip: what write_fastq emits, read_fastq_records reads back identically
        dst = d / "out.fq"
        hdr_a = f"1:10|aaa rank=1 umi=CORRECTED {strand_field(tail_a)}"
        assert write_fastq(dst, [(hdr_a, seq_a, qual_a), ("1:20|bbb rank=2", "TTTT", "!!!!")]) == 2
        assert dst.read_text() == (f"@{hdr_a}\nACGTACGT\n+\nIIIIHHHH\n"
                                   "@1:20|bbb rank=2\nTTTT\n+\n!!!!\n")
        back = read_fastq_records(dst, {"1:10|aaa", "1:20|bbb"})
        assert (back["1:10|aaa"][1], back["1:10|aaa"][2]) == (seq_a, qual_a)
        assert back["1:10|aaa"][0] == "rank=1 umi=CORRECTED strand=+"
        assert write_fastq(d / "empty.fq", []) == 0

        # ------------------------------------------------------------------ #
        # iter_fastq_records: the parser both entry points share
        # ------------------------------------------------------------------ #
        # want=None yields EVERY record, in file order
        every = list(iter_fastq_records(src))
        assert [a for a, _t, _s, _q in every] == ["1:10|aaa", "1:20|bbb", "1:30|ccc"]
        assert every[1] == ("1:20|bbb", "", "TTTT", "!!!!")

        # `want` filters identically to read_fastq_records, and the dict is exactly a dict()
        # over the generator -- this is the assertion that keeps the two from drifting.
        assert {a: (t, s, q) for a, t, s, q in iter_fastq_records(src, {"1:10|aaa", "1:20|bbb"})} \
            == read_fastq_records(src, {"1:10|aaa", "1:20|bbb"})

        # a dict is a legal `want` -- the singleton writer passes its read_acc -> umi map
        # directly rather than building a second copy of the keys
        assert [a for a, *_ in iter_fastq_records(src, {"1:30|ccc": "UMI"})] == ["1:30|ccc"]

        # LAZINESS, which the singleton writer's `wanted.pop(acc)` relies on: mutating `want`
        # between yields takes effect, so a read_acc already emitted cannot be emitted twice.
        dup = d / "dup.fq"
        dup.write_text("@1:10|aaa\nACGT\n+\nIIII\n@1:10|aaa\nTTTT\n+\nJJJJ\n")
        live = {"1:10|aaa": "UMI"}
        emitted = []
        for acc, _t, seq, _q in iter_fastq_records(dup, live):
            if live.pop(acc, None) is None:
                continue
            emitted.append(seq)
        assert emitted == ["ACGT"], emitted     # the second copy is skipped, not double-written

        # a truncated file is fatal even when nothing is wanted -- the check precedes the filter
        trunc_all = d / "truncall.fq"
        trunc_all.write_text("@1:99|zzz\nACGT\n+\n")
        try:
            list(iter_fastq_records(trunc_all, {"nothing"}))
        except SystemExit:
            pass
        else:
            raise AssertionError("a truncated record must be fatal even for an unwanted read")

        # ------------------------------------------------------------------ #
        # the invariants must be LOUD, not silently tolerated
        # ------------------------------------------------------------------ #
        bad_len = d / "badlen.fq"
        bad_len.write_text("@1:10|aaa\nACGTACGT\n+\nIII\n")          # 8 bases, 3 quals
        try:
            read_fastq_records(bad_len, {"1:10|aaa"})
        except SystemExit:
            pass
        else:
            raise AssertionError("a seq/qual length mismatch must be fatal on READ")

        trunc = d / "trunc.fq"
        trunc.write_text("@1:10|aaa\nACGT\n+\n")                     # qual line missing
        try:
            read_fastq_records(trunc, {"1:10|aaa"})
        except SystemExit:
            pass
        else:
            raise AssertionError("a truncated record must be fatal on READ")

        try:
            write_fastq(d / "never.fq", [("1:10|aaa", "ACGT", "II")])
        except SystemExit:
            pass
        else:
            raise AssertionError("a seq/qual length mismatch must be fatal on WRITE")

    print("[INFO] fastq_io selftest OK  (4-line parsing, strand-only tail, round-trip, the "
          "streaming iterator incl. laziness, and the seq/qual + truncation invariants "
          "verified)")


if __name__ == "__main__":
    _selftest()
