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


wildcard_constraints:
    sample="|".join(re.escape(s) for s in SAMPLES),
    decbase="decode|decode_initial",
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
