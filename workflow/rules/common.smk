# Shared setup: parse config, derive the sample sets, and define the helper /
# input functions the rule modules use. No rules here.
import os
import re
from pathlib import Path

OUTDIR = config["outdir"]

# ---- vendored python scripts (run via `pixi run -e default python <abs path>`) ------
# Absolute so the scripts' sibling `import ligand_vocab` resolves via sys.path[0].
SCRIPTS = Path(workflow.basedir) / "scripts"


# ---- sample table ------------------------------------------------------------------
SAMPLE = {s["name"]: s for s in config["samples"]}
SAMPLES = list(SAMPLE.keys())
EXPERIMENTAL = [n for n, s in SAMPLE.items() if s["role"] == "experimental"]
# Everything except the R0 `initial` decodes into the shared `decode/` tree; this is exactly
# what the cross-sample report globs and groups (by config `group`). `role` beyond `initial`
# does not change processing — all non-initial samples are handled identically.
NONINITIAL = [n for n, s in SAMPLE.items() if s["role"] != "initial"]
_initials = [n for n, s in SAMPLE.items() if s["role"] == "initial"]
if len(_initials) != 1:
    raise ValueError(
        f"config.samples: exactly one sample must have role: initial "
        f"(found {len(_initials)}: {_initials})"
    )
INITIAL = _initials[0]

# The primer HMM name follows the adapters FASTA basename (hammerpede names it that way).
HMM_NAME = Path(config["adapters_fasta"]).stem
HMM = f"{OUTDIR}/hmm/{HMM_NAME}.hmm"

# ---- decontamination genomes (chained depletion in `clean`; see refs.smk / clean.smk) -
GENOMES = config["decontaminate"]
if not GENOMES:
    raise ValueError("config.decontaminate: provide at least one genome")
_gnames = [g["name"] for g in GENOMES]
if len(set(_gnames)) != len(_gnames):
    raise ValueError(f"config.decontaminate: genome names must be unique, got {_gnames}")
for g in GENOMES:
    if not g.get("mmi"):
        raise ValueError(f"config.decontaminate[{g.get('name')!r}]: `mmi` (index path) is required")
N_GENOMES = len(GENOMES)
LAST_STAGE = N_GENOMES - 1     # index of the final depletion stage


def _is_url(p):
    return str(p).startswith(("http://", "https://", "ftp://"))


# ---- typed config accessors ---------------------------------------------------------
# A YAML `configfile:` yields correctly typed values, but `--config key={"flag": false}` does NOT:
# snakemake's parse_config (cli.py) runs a nested value through yaml.BaseLoader, which resolves
# EVERY scalar to a str. So `{"enabled": false}` -> "false" (and `bool("false")` is True, silently
# inverting the flag), and `{"max_homopolymer": 8}` -> "8". Coerce on read; raise on anything
# uninterpretable, so a typo can never quietly enable a stage or reach the shell unvalidated.
def _cfg_bool(v, default, *, key):
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("true", "yes", "on", "1"):
        return True
    if s in ("false", "no", "off", "0"):
        return False
    raise ValueError(f"config.{key}: cannot interpret {v!r} as a boolean (use true/false)")


def _cfg_num(v, default, *, key, cast, lo=None, hi=None):
    """Coerce to `cast`, then bounds-check on the closed interval [lo, hi] (None = unbounded)."""
    if v is None:
        return default
    try:
        n = cast(v)
    except (TypeError, ValueError):
        raise ValueError(f"config.{key}: cannot interpret {v!r} as {cast.__name__}") from None
    if (lo is not None and n < lo) or (hi is not None and n > hi):
        rng = f"[{'-inf' if lo is None else lo}, {'inf' if hi is None else hi}]"
        raise ValueError(f"config.{key}: {n} is outside {rng}")
    return n


def _cfg_int(v, default, *, key, lo=None, hi=None):
    return _cfg_num(v, default, key=key, cast=int, lo=lo, hi=hi)


def _cfg_float(v, default, *, key, lo=None, hi=None):
    return _cfg_num(v, default, key=key, cast=float, lo=lo, hi=hi)


def _cfg_choice(v, default, *, key, choices):
    """Return v lower-cased if it is one of `choices`; `default` when v is None; else raise."""
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in choices:
        return s
    raise ValueError(f"config.{key}: {v!r} is not one of {sorted(choices)}")


wildcard_constraints:
    sample="|".join(re.escape(s) for s in SAMPLES),
    # featurecount / featurecount_check / report_sample serve BOTH the raw decode trees and the
    # deduplicated ones, so `decbase` spans all four. The producers must NOT: `rule decode` pins
    # itself to decode|decode_initial and `rule dedup` to dedup|dedup_initial, both with per-rule
    # constraints. Otherwise decode would also match a dedup/ path -> AmbiguousRuleException.
    decbase="decode|decode_initial|dedup|dedup_initial",
    dedbase="dedup|dedup_initial",                   # the tree `rule dedup` writes into
    umibase="umi|umi_initial",                       # the tree `rule umi_correct` writes into
    crossbase="decode|dedup",                        # which tree report_cross is built over
    k="|".join(str(i) for i in range(N_GENOMES)),   # depletion stage index


# ---- config sanity: the decoder needs one of its two forms -------------------------
if not config["ligandfrags"].get("binary") and not config["ligandfrags"].get("source"):
    raise ValueError("config.ligandfrags: provide either `binary` or `source`")


# ---- path helpers ------------------------------------------------------------------
def decbase(s):
    """Which decode tree a sample lands in (the initial library is kept separate so the
    cross-report's `decode/*/` glob never double-counts it as both column and baseline)."""
    return "decode_initial" if SAMPLE[s]["role"] == "initial" else "decode"


def stem(s):
    """Decode output stem for a sample, e.g. results/decode/A549_R1/A549_R1."""
    return f"{OUTDIR}/{decbase(s)}/{s}/{s}"


# ---- UMI stage: QC-filter (always) + optional ed<=1 correction / collapse ------------
# `umi.mode` selects the behaviour (default `correct`); read defensively (a project config may lack
# the `umi:` key). The processed `dedup/` tree (filtered, or collapsed in `dedup` mode) ALWAYS feeds
# featurecount/reports -- correction never changes fragment counts. Thresholds fall back to the
# umi_filter_lib constants rather than raising KeyError.
UMI = config.get("umi", {}) or {}
UMI_MODE = _cfg_choice(UMI.get("mode"), "correct", key="umi.mode",
                       choices=("filter", "correct", "dedup"))
DEDUP_ON = UMI_MODE == "dedup"            # collapse to one representative read per molecule
UMI_CORRECT_ON = UMI_MODE == "correct"    # run the separate umi/ correction + characterisation branch
UMI_STAGE_ON = True                       # the dedup/ tree (filter or collapse) always feeds featurecount
DEDUP_MAX_HOMOPOLYMER = _cfg_int(UMI.get("max_homopolymer"), 8, key="umi.max_homopolymer", lo=1)
DEDUP_MAX_BASE_FRAC = _cfg_float(UMI.get("max_base_frac"), 0.7,
                                 key="umi.max_base_frac", lo=0.0, hi=1.0)
DEDUP_THREADS = _cfg_int(config["threads"].get("dedup"), 8, key="threads.dedup", lo=1)


def dedbase(s):
    """Which filtered/collapsed tree a sample lands in. Mirrors decbase(): the initial library is
    kept in its own tree so report_cross's `<crossbase>/*/` glob never sees it as both a column and
    the R0 baseline."""
    return "dedup_initial" if SAMPLE[s]["role"] == "initial" else "dedup"


def dedup_stem(s):
    """Filtered/collapsed output stem for a sample, e.g. results/dedup/A549_R1/A549_R1."""
    return f"{OUTDIR}/{dedbase(s)}/{s}/{s}"


def report_stem(s):
    """The stem whose featurecounts / per-sample report `rule all` builds: the processed dedup/
    tree (the UMI stage always filters), so correction never changes fragment counts."""
    return dedup_stem(s) if UMI_STAGE_ON else stem(s)


# Base tree for the cross-sample FRAGMENT report (report_cross is parameterized by {crossbase}); its
# R0 baseline is derived as f"{CROSSBASE}_initial", so the two move in lockstep.
CROSSBASE = "dedup" if UMI_STAGE_ON else "decode"


# ---- UMI correction branch (`umi/` tree, sibling of decode/dedup; see rules/umi.smk) -------------
# Produced only when UMI_CORRECT_ON. Keeps every kept read but relabels its fused UMI to the abundant
# ed<=1 centroid and records the per-molecule read count. The initial library is kept in its own
# `umi_initial/` tree so the cross-sample glob never double-counts it.
def umibase(s):
    return "umi_initial" if SAMPLE[s]["role"] == "initial" else "umi"


def umi_stem(s):
    """UMI-correction output stem, e.g. results/umi/A549_R1/A549_R1."""
    return f"{OUTDIR}/{umibase(s)}/{s}/{s}"


# Experimental groups (config `group`, in config order). Each group gets ONE UMI x sample matrix
# whose column order is the initial library (R0 baseline) first, then that group's samples by round.
GROUPS_EXP = list(dict.fromkeys(SAMPLE[s]["group"] for s in NONINITIAL))


def group_samples(g):
    """Ordered samples for group g's matrix: the R0 initial library first, then g's rounds ascending."""
    members = sorted((s for s in NONINITIAL if SAMPLE[s]["group"] == g),
                     key=lambda s: (SAMPLE[s]["round"], s))
    return [INITIAL] + members


# ---- depletion-chain inputs --------------------------------------------------------
def deplete_input_reads(wildcards):
    """Reads entering depletion stage k: the raw sample FASTQ for k==0, else the survivors
    of the previous stage."""
    k = int(wildcards.k)
    if k == 0:
        return SAMPLE[wildcards.sample]["fastq"]
    return f"{OUTDIR}/clean/{wildcards.sample}/deplete.{k - 1}.fastq"


def deplete_mmi(wildcards):
    """The minimap2 index for depletion stage k (that genome's configured `mmi` path)."""
    return GENOMES[int(wildcards.k)]["mmi"]


def ligandfrags_bin(wildcards):
    """The decoder binary: the configured prebuilt path, else the build_ligandfrags output."""
    return config["ligandfrags"]["binary"] or f"{OUTDIR}/bin/{config['ligandfrags']['bin_name']}"
