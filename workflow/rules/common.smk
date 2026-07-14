# Shared setup: parse config, derive the sample sets, and define the helper /
# input functions the rule modules use. No rules here.
import os
import re
from pathlib import Path

OUTDIR = config["outdir"]

# ---- vendored python scripts (run via `pixi run -e default python <abs path>`) ------
# Absolute so the scripts' sibling `import ligand_vocab` resolves via sys.path[0].
SCRIPTS = Path(workflow.basedir) / "scripts"

# ---- the shared group vocabulary ----------------------------------------------------
# `report_lib.py` owns the two reserved group names; this is the SINGLE definition, imported rather
# than restated so the workflow and the reports cannot drift apart.
#
# Loaded by file spec, NOT `sys.path.insert(0, SCRIPTS)`: that would prepend the scripts dir to the
# *snakemake driver's* import path, ahead of the stdlib -- vendor a `workflow/scripts/types.py` one
# day and snakemake breaks three rules deep, far from the cause.
#
# INVARIANT this creates: report_lib is now imported at workflow-PARSE time, under the `snakemake`
# pixi env, which has no plotly and no polars. It must stay stdlib-only or the whole workflow stops
# parsing.
import importlib.util

_spec = importlib.util.spec_from_file_location("report_lib", SCRIPTS / "report_lib.py")
_rl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rl)
BASELINE_GROUP, EMPTY_GROUP = _rl.BASELINE_GROUP, _rl.EMPTY_GROUP


# ---- sample table ------------------------------------------------------------------
# ONE axis decides everything: `group`. It selects the tree a sample decodes into, the matrix it joins,
# its colour in every report, and the collapse row it gets. Two group names are RESERVED:
#
#   initial -- the R0 baseline library. Exactly one sample. Kept in its own `*_initial/` trees so the
#              cross-reports' `<tree>/*/` glob never sees it as both a column AND the baseline.
#   empty   -- the EMPTY / PCR-only rounds: the same transfers and PCR cycles as a selection round, but
#              no biological selection. The null for everything that depends on the pool being
#              bottlenecked (duplication, effective complexity, enrichment fold-change), which is how
#              `rule umi_complexity` splits a collapse into its PROTOCOL and SELECTION parts.
#
# The empty arm is otherwise a PEER of every selection arm: it DOES form its own group matrix, takes
# its own colour, and draws its own C_eff trajectory line. It is special only in the collapse table,
# where its collapse IS the protocol collapse. Every other group name is a selection arm.
#
# Every group used must be declared in config `groups:`, so a typo raises here instead of silently
# minting a new arm. Everything below fails at PARSE time, before any compute, NAMING the offending
# sample: the code rejects a bad config loudly, it never works around one.
#
# A `name` and a `group` are DIFFERENT THINGS and share exactly one constraint: both end up as PATH
# COMPONENTS, so both must be path-safe. That is all they share.
#
#   name  = IDENTITY. A snakemake wildcard, a DIRECTORY, and the filename stem inside it
#           (`results/umi/<name>/<name>.fused.parquet`).
#   group = A LABEL. Which arm the sample belongs to, for computing and colouring -- plus the
#           filename stem of that group's matrix (`results/umi/<group>.umi_x_sample.mtx`).
#
# They do NOT share a namespace, and a group MAY be called the same as a sample. `results/umi/A549/`
# (a directory) and `results/umi/A549.umi_x_sample.mtx` (a file) are different paths and coexist
# happily; snakemake cannot confuse them either, because one output pattern needs a `/<sample>/` infix
# and the other a literal `.umi_x_sample.` infix. There WAS a guard here forbidding the coincidence.
# It was wrong: it rejected a VALID config on grounds of taste, and rejecting a valid config breaks
# the rule below just as surely as accepting an invalid one does.
_NAME_CHARSET = r"[A-Za-z0-9][A-Za-z0-9._+-]*"

# (0) the sample NAME: present, globally unique, path-safe. This runs against config["samples"] (the
#     LIST) and not against SAMPLE (the dict), because the dict comprehension below is last-key-wins:
#     it would silently drop the earlier row and leave the survivor carrying the LATER row's group and
#     fastq -- a sample quietly decoded from another arm's reads and filed into that arm's matrix,
#     UpSet and C_eff trajectory. Once collapsed, the evidence is gone; check first.
#     The charset is not cosmetic either: a name becomes a DIRECTORY and the filename stem inside it,
#     so a '/' nests the sample one level deeper, where the cross-reports' `<tree>/*/` glob can never
#     find it -- the sample would run to completion and then vanish from every report, silently.
_seen = set()
for _s in config["samples"]:
    if "name" not in _s:
        raise ValueError(f"config.samples: every sample needs a `name`; this row has none: {_s!r}")
    _n = _s["name"]
    if not isinstance(_n, str) or not re.fullmatch(_NAME_CHARSET, _n):
        raise ValueError(
            f"config.samples[{_n!r}]: not a valid sample name (must match {_NAME_CHARSET}). "
            f"A name becomes a DIRECTORY and the filename stem inside it "
            f"(results/umi/<name>/<name>.fused.parquet); a '/' or a space nests it where the "
            f"cross-reports' `<tree>/*/` glob never finds it, so it would run and then silently "
            f"disappear from every report."
        )
    if _n in _seen:
        raise ValueError(
            f"config.samples[{_n!r}]: duplicate sample name -- names must be globally unique. A name "
            f"is the dict key, a directory AND a filename stem, so a second row with the same name "
            f"would silently REPLACE the first: the earlier row drops out of the workflow entirely "
            f"and the survivor adopts this row's group, round and fastq. Rename one."
        )
    _seen.add(_n)

SAMPLE = {s["name"]: s for s in config["samples"]}
SAMPLES = list(SAMPLE.keys())

# (1) MIGRATION GUARD -- non-negotiable. `role` is gone. Without this, an old row
#     `{group: A549, round: 5, role: control}` parses FINE and `role` is silently ignored, drawing the
#     empty round as a second A549 R5: a duplicate point on the C_eff trajectory and a full member of
#     A549's UpSet. That is the exact bug the reserved `empty` group kills; a stale key must not be
#     able to resurrect it.
for _n, _s in SAMPLE.items():
    if "role" in _s:
        raise ValueError(
            f"config.samples[{_n!r}]: 'role' was removed -- the group decides everything. "
            f"Reserved groups: {BASELINE_GROUP}, {EMPTY_GROUP}. Delete the role key "
            f"(a control round is now group: {EMPTY_GROUP})."
        )

# (2) the declared group list: required, a non-empty list, no duplicates, and it must name the
#     baseline. This is what makes a typo in a sample's `group` an error rather than a new arm.
#
#     `"groups" not in config` is NOT enough on its own. The Snakefile's `configfile:` directive makes
#     this repo's config/config.yaml the BASE config, and snakemake MERGES a --configfile on top of it
#     -- a top-level key the CLI file omits SURVIVES from the base. So a project config that forgets
#     `groups:` would silently inherit `[initial, A549, MDA]` from a file it never wrote, and the typo
#     guard would be validating its samples against someone else's declaration. `groups:` must
#     therefore be declared BY THE SAME FILE THAT DECLARES `samples:` -- which also leaves a pure
#     overlay (`--configfile tweak.yaml` carrying only `umi:`) legitimately alone, since its samples
#     and its groups then both come from the base.
_MSG_GROUPS_REQUIRED = (
    f"list every group name used in `samples:`, e.g. groups: [{BASELINE_GROUP}, A549, MDA]. "
    f"Reserved names: {BASELINE_GROUP} (the R0 baseline, exactly one sample) and {EMPTY_GROUP} "
    f"(the PCR-only / no-cell rounds). Declaring the groups is what makes a typo in a sample's "
    f"`group` raise instead of silently minting a new, half-empty arm."
)
if "groups" not in config:
    raise ValueError(f"config.groups: required -- {_MSG_GROUPS_REQUIRED}")

# Merge precedence: the `configfile:` directive's file(s) first, then each --configfile on top. The
# LAST file declaring `samples:` is the one whose table is actually in play; it must declare `groups:`.
_cli_cfgs = [str(p) for p in (getattr(workflow, "overwrite_configfiles", None) or [])]
_all_cfgs = [str(p) for p in workflow.configfiles if str(p) not in _cli_cfgs] + _cli_cfgs
_owner = None
if _all_cfgs:
    import yaml
    for _p in _all_cfgs:
        try:
            _doc = yaml.safe_load(Path(_p).read_text()) or {}
        except OSError:
            continue          # snakemake already loaded it; only a workdir change can land here
        if isinstance(_doc, dict) and "samples" in _doc:
            _owner = (_p, _doc)
if _owner is not None and "groups" not in _owner[1]:
    raise ValueError(
        f"config.groups: required in {_owner[0]} -- the file that declares `samples:` must declare "
        f"`groups:` too. {_MSG_GROUPS_REQUIRED} (It is NOT enough that some other config in the merge "
        f"chain has one: snakemake layers a --configfile ON TOP of the workflow's default config, so "
        f"an omitted `groups:` would be silently inherited and your samples validated against a group "
        f"list you never wrote.)"
    )

DECLARED_GROUPS = config["groups"]
if not isinstance(DECLARED_GROUPS, list) or not DECLARED_GROUPS:
    raise ValueError(f"config.groups: must be a non-empty list of group names, got {DECLARED_GROUPS!r}")
if len(set(DECLARED_GROUPS)) != len(DECLARED_GROUPS):
    raise ValueError(f"config.groups: group names must be unique, got {DECLARED_GROUPS}")
if BASELINE_GROUP not in DECLARED_GROUPS:
    raise ValueError(
        f"config.groups: must contain the reserved baseline group {BASELINE_GROUP!r} "
        f"(the R0 initial library), got {DECLARED_GROUPS}"
    )

# (4) a group name becomes a FILENAME STEM (`results/umi/<group>.umi_x_sample.mtx`) in the very
#     directory the cross-reports glob for samples, so it must be path-safe: no '/', no spaces. A '/'
#     would instead create a SUBDIRECTORY there -- sitting in exactly the namespace `BASE.glob("*/")`
#     scans -- which the reports would then discover as a PHANTOM SAMPLE.
#
#     A group name MAY equal a sample name. That is not a collision: one is a directory and the other
#     a filename stem, and no lookup in this codebase is keyed by both. See the note above _NAME_CHARSET.
for _g in DECLARED_GROUPS:
    if not isinstance(_g, str) or not re.fullmatch(_NAME_CHARSET, _g):
        raise ValueError(
            f"config.groups: {_g!r} is not a valid group name (must match {_NAME_CHARSET}). "
            f"A group name becomes a filename stem in the directory the cross-reports glob for "
            f"samples; a '/' would create a subdirectory there instead, in exactly the namespace "
            f"`BASE.glob(\"*/\")` scans."
        )

# (3) every sample's group must be declared, and (6) `round` is required and an int (`group_samples()`
#     sorts on it, and a str/None round would TypeError deep inside the DAG build instead of here).
for _n, _s in SAMPLE.items():
    if "group" not in _s:
        raise ValueError(
            f"config.samples[{_n!r}]: `group` is required and must be one of config.groups "
            f"({DECLARED_GROUPS})."
        )
    if _s["group"] not in DECLARED_GROUPS:
        raise ValueError(
            f"config.samples[{_n!r}].group: {_s['group']!r} is not declared in config.groups "
            f"({DECLARED_GROUPS}). Fix the typo, or add the group to `groups:`."
        )
    if "round" not in _s:
        raise ValueError(
            f"config.samples[{_n!r}]: `round` is required (an integer; R0 = the initial library)."
        )
    if not isinstance(_s["round"], int) or isinstance(_s["round"], bool):
        raise ValueError(
            f"config.samples[{_n!r}].round: {_s['round']!r} is not an integer "
            f"(rounds order a group's samples and pair an `{EMPTY_GROUP}` round to the real round with "
            f"the same transfer/PCR count)."
        )

# (5) exactly one R0 baseline. One config = one project = one initial library.
_initials = [n for n, s in SAMPLE.items() if s["group"] == BASELINE_GROUP]
if len(_initials) != 1:
    raise ValueError(
        f"config.samples: exactly one sample must have group: {BASELINE_GROUP} "
        f"(the R0 baseline), found {len(_initials)}: {_initials}"
    )
INITIAL = _initials[0]

# Everything except the R0 baseline decodes into the shared `decode/` tree -- exactly what the
# cross-sample reports glob and then group by config `group`. Beyond `initial` vs not, the group does
# NOT change processing: every non-baseline sample (the empty arm included) is handled identically.
NONINITIAL = [n for n, s in SAMPLE.items() if s["group"] != BASELINE_GROUP]
EMPTIES = [n for n, s in SAMPLE.items() if s["group"] == EMPTY_GROUP]   # 0 or more rounds

# (7) two empty rounds at the same round would each be a candidate null for the same real round --
#     the pairing would be arbitrary. Hoisted here from effective_complexity.py (which KEEPS its own
#     runtime check as the backstop for the off-workflow `--fused` path).
_by_round = {}
for _n in EMPTIES:
    _r = SAMPLE[_n]["round"]
    if _r in _by_round:
        raise ValueError(
            f"config.samples[{_n!r}]: a second group: {EMPTY_GROUP} sample at round {_r} "
            f"(already {_by_round[_r]!r}). At most one empty round per round -- it is the null that "
            f"every experimental sample at that round is compared against."
        )
    _by_round[_r] = _n

# (8) a declared group with no samples is a self-contradicting config. RAISE, do not silently filter:
#     the payoff is that GROUPS below needs no membership filter at all.
for _g in DECLARED_GROUPS:
    if not any(s["group"] == _g for s in SAMPLE.values()):
        raise ValueError(
            f"config.groups: {_g!r} is declared but no sample is in it. Remove it from `groups:`, "
            f"or add its samples."
        )

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
    fvbase="fragvec|fragvec_initial",                # the tree `rule fragvec` writes into
    crossbase="decode|dedup",                        # which tree report_cross is built over
    k="|".join(str(i) for i in range(N_GENOMES)),   # depletion stage index


# ---- config sanity: the decoder needs one of its two forms -------------------------
if not config["ligandfrags"].get("binary") and not config["ligandfrags"].get("source"):
    raise ValueError("config.ligandfrags: provide either `binary` or `source`")


# ---- path helpers ------------------------------------------------------------------
def decbase(s):
    """Which decode tree a sample lands in (the initial library is kept separate so the
    cross-report's `decode/*/` glob never double-counts it as both column and baseline)."""
    return "decode_initial" if SAMPLE[s]["group"] == BASELINE_GROUP else "decode"


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
    return "dedup_initial" if SAMPLE[s]["group"] == BASELINE_GROUP else "dedup"


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
    return "umi_initial" if SAMPLE[s]["group"] == BASELINE_GROUP else "umi"


def umi_stem(s):
    """UMI-correction output stem, e.g. results/umi/A549_R1/A549_R1."""
    return f"{OUTDIR}/{umibase(s)}/{s}/{s}"


# Every group except the baseline gets a matrix and a report line -- `empty` INCLUDED. It is a PEER:
# special only in the collapse table, where its collapse IS the protocol collapse. Each group gets ONE
# matrix whose column order is the initial library (R0 baseline) first, then that group's rounds
# ascending. Declared order, and no membership filter: guard (8) already proved every declared group
# has samples.
GROUPS = [g for g in DECLARED_GROUPS if g != BASELINE_GROUP]

# `"|".join([])` is the EMPTY regex -- which MATCHES THE EMPTY STRING, so a rule constrained by it
# would accept `results/umi/.umi_x_sample.mtx`. Reachable from a legitimate baseline-only
# `groups: [initial]`. Emit a never-matching pattern instead. Use this for any {group} wildcard.
GROUPS_RE = "|".join(re.escape(g) for g in GROUPS) if GROUPS else r"(?!)"


def group_samples(g):
    """Ordered samples for group g's matrix: the R0 initial library first, then g's rounds ascending."""
    members = sorted((s for s in NONINITIAL if SAMPLE[s]["group"] == g),
                     key=lambda s: (SAMPLE[s]["round"], s))
    return [INITIAL] + members


# ---- fragment-vector branch (`fragvec/` tree; see rules/fragvec.smk) -----------------------------
# The second, INDEPENDENT analysis branch off decode (sibling of the umi/ branch -- run either or both).
# A read's fragment vector (its sorted PAYLOAD fragment_ids) is the library member, and unlike the UMI
# it carries real depth (~4-10 reads/vector vs ~1.0 reads/UMI). Counts two axes per vector: n_reads
# (abundance) and n_umis (incidence = independent clones), so a jackpot (many reads, one clone) is
# distinguishable from reproducible selection (many reads, many clones).
FRAGVEC = config.get("fragvec", {}) or {}
FRAGVEC_ON = _cfg_bool(FRAGVEC.get("enabled"), True, key="fragvec.enabled")
FRAGVEC_UMI_SOURCE = _cfg_choice(FRAGVEC.get("umi_source"), "corrected", key="fragvec.umi_source",
                                 choices=("corrected", "raw"))
FRAGVEC_MIN_READS = _cfg_int(FRAGVEC.get("min_reads"), 10, key="fragvec.min_reads", lo=1)
if FRAGVEC_ON and FRAGVEC_UMI_SOURCE == "corrected" and not UMI_CORRECT_ON:
    raise ValueError(
        "config.fragvec.umi_source: 'corrected' needs the UMI correction branch "
        "(set umi.mode: correct, or use fragvec.umi_source: raw)")


def fvbase(s):
    return "fragvec_initial" if SAMPLE[s]["group"] == BASELINE_GROUP else "fragvec"


def fragvec_stem(s):
    """Fragment-vector output stem, e.g. results/fragvec/A549_R1/A549_R1."""
    return f"{OUTDIR}/{fvbase(s)}/{s}/{s}"


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
