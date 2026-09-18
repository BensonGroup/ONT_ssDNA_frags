#!/usr/bin/env python
"""Run the workflow with Snakemake arguments and optional --static-plot exports.

Run from the repository root in the snakemake Pixi environment.
"""
from pathlib import Path
import subprocess
import sys


def snakemake_args(argv):
    """Translate the workflow flag without losing existing --config overrides."""
    args = list(argv)
    enabled = "--static-plot" in args
    args = [arg for arg in args if arg != "--static-plot"]
    if enabled:
        normalized = []
        for arg in args:
            if arg.startswith("--config="):
                normalized.extend(["--config", arg.split("=", 1)[1]])
            else:
                normalized.append(arg)
        args = normalized
        config_positions = [i for i, arg in enumerate(args) if arg in ("--config", "-C")]
        if config_positions:
            start = config_positions[-1] + 1
            end = start
            while end < len(args) and not args[end].startswith("-"):
                end += 1
            overrides = [arg for arg in args[start:end] if not arg.startswith("static_plot=")]
            args[start:end] = [*overrides, "static_plot=true"]
        else:
            args.extend(["--config", "static_plot=true"])
    return ["--snakefile", str(Path(__file__).with_name("Snakefile")), *args]


def main():
    if "--help" in sys.argv[1:] or "-h" in sys.argv[1:]:
        print("Workflow option: --static-plot writes PDF/PNG figures to <outdir>/static_plots/.",
              flush=True)
    return subprocess.call([sys.executable, "-m", "snakemake", *snakemake_args(sys.argv[1:])])


if __name__ == "__main__":
    raise SystemExit(main())
