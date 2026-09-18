#!/usr/bin/env python
"""Export explicitly selected workflow HTML figures as PDF and PNG.

Adapted from scripts/batch3_html_static_figs.py. Reuses its extraction,
fingerprinting, rendering, captions and provenance; never reads sequencing data.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from datetime import datetime, timezone
import gzip
import hashlib
from html import escape, unescape
import json
import os
from pathlib import Path
import re
import traceback

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import html_report_plotting as R

HERE = Path(__file__).resolve().parent


def extract(path: Path, relative: str) -> tuple[str, list[dict]]:
    source = path.read_text(encoding="utf-8")
    title = re.search(r"<title[^>]*>(.*?)</title>", source, re.S | re.I)
    report_title = unescape(title[1]).strip() if title else path.stem
    headings = list(re.finditer(r"<h[1-6]\b[^>]*>(.*?)</h[1-6]>", source, re.S | re.I))
    positions = [h.start() for h in headings]
    decoder = json.JSONDecoder()
    plots = []
    for match in re.finditer(r'Plotly\.newPlot\(\s*"', source):
        pos = match.end() - 1
        args = []
        for i in range(3):
            value, pos = decoder.raw_decode(source, pos)
            args.append(value)
            if i < 2:
                while source[pos].isspace():
                    pos += 1
                if source[pos] != ",":
                    raise ValueError(f"Unexpected newPlot argument at {path}:{pos}")
                pos += 1
                while source[pos].isspace():
                    pos += 1
        div_id, data, layout = args
        if not isinstance(data, list) or not isinstance(layout, dict):
            raise ValueError(f"Unsupported Plotly arguments in {path}")
        if layout.get("updatemenus") or layout.get("sliders"):
            raise ValueError(f"Interactive states require explicit expansion: {path}")
        heading_i = bisect_right(positions, match.start()) - 1
        heading = unescape(headings[heading_i][1]) if heading_i >= 0 else report_title
        if not layout.get("title", {}).get("text"):
            xlabel = R.axis_title(layout.get("xaxis", {}))
            ylabel = R.axis_title(layout.get("yaxis", {}))
            if "fragment copies" in xlabel:
                heading = "Fragment multiplicity"
            elif "edit distance" in xlabel:
                heading = "Fragment edit distance"
            elif "% of read length" in xlabel:
                heading = "Fragment coverage per read"
            elif data and data[0].get("name") in {"reads decoded", "reads retained"}:
                heading = ("Reads retained after filtering" if data[0]["name"] == "reads retained"
                           else "Reads decoded and reads with fragments")
            if ".featurecount_report.html" in relative:
                sample = path.name.split(".featurecount_report.html")[0]
                heading = sample + " · " + R.label(heading)
        elif relative.startswith("qc/") and layout["title"]["text"] == "NanoComp (by sample)":
            heading = R.axis_title(layout.get("yaxis", {})) + " by sample"
        plot = {"id": div_id, "data": data,
                "layout": {k: v for k, v in layout.items() if k != "template"},
                "heading": R.label(heading), "report_title": R.label(report_title),
                "source": relative, "source_plot": len(plots) + 1}
        plots.append(plot)
    return R.label(report_title), plots


def fingerprint(plot):
    # Div ids and report wrappers differ between NanoComp's standalone files and
    # its summary; the complete data and layout, including units, must agree.
    payload = {k: plot[k] for k in ("data", "layout")}
    payload["display_title"] = R.title_of(plot)
    payload["umi_renderer"] = plot.get("source") == "umi/umi_report_cross.html"
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def slug(text, limit=75):
    result = re.sub(r"[^A-Za-z0-9_-]+", "_", R.label(text)).strip("_")
    return result[:limit].rstrip("_") or "figure"


def source_points(plot):
    result = 0
    for trace in plot["data"]:
        if trace.get("type") == "heatmap":
            result += R.array(trace["z"]).size
        else:
            result += R.array(trace.get("y", trace.get("x", []))).size
    return result


def write_indexes(out, reports, unique, failures, metadata, within=None):
    figures = [view for record in unique.values() for view in record.get("views", [])]
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Explicit workflow analysis and QC reports",
        "metadata": {"source": str(metadata.resolve()),
                     "sha256": hashlib.sha256(metadata.read_bytes()).hexdigest()},
        "combined_pdf": "all_analysis_qc_figures.pdf" if figures else None,
        "terminology": {"payload": "fragment", "clonotype": "fragvec", "clone": "UMI",
                        "unique sequence": "construct"},
        "report_count": len(reports), "source_plot_instances": sum(len(r.get("plots", [])) for r in reports),
        "unique_source_plots": len(unique), "static_figure_count": len(figures),
        "duplicate_plot_instances": sum(len(r.get("plots", [])) for r in reports) - len(unique),
        "failures": failures,
        "within_batch": within,
        "reports": reports, "unique_plots": list(unique.values()),
    }
    (out / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    md = ["# Static analysis and QC figures", "",
          f"{len(reports)} reports · {summary['source_plot_instances']} plot instances · "
          f"{len(unique)} unique source plots · {len(figures)} static figures.", "",
          ("[All figures (PDF)](all_analysis_qc_figures.pdf) · " if figures else "")
          + "[Browsable index](index.html) · [Provenance manifest](manifest.json)", "",
          "Uses the shared Science figure style, editable PDF text and 600-dpi PNGs. "
          "Display terminology: fragvec and UMI. Original source labels remain in the archived data. "
          "Separate panels may be exported separately for readability. Exact duplicate plots share files.", "",
          "Histogram binning and violin density estimation are documented beside their figures. "
          "All other charts draw the embedded values directly; no pipeline analysis is rerun.", ""]
    template = (HERE / "index_template.html").read_text()
    css = (HERE / "index.css").read_text()
    body = [template.replace("<!-- INLINE_STYLE -->", "<style>" + css + "</style>"),
            "<h1>Static analysis and QC figures</h1>",
            f"<p class='subtitle'>{len(reports)} reports · {len(unique)} unique source plots · {len(figures)} static figures.</p>",
            "<p class='nav'>" + ("<a href='all_analysis_qc_figures.pdf'>All figures (PDF)</a> · " if figures else "")
            + "<a href='manifest.json'>Provenance manifest</a></p>",
            "<p class='callout note'>600-dpi PNGs and editable PDF text. Terminology: fragvec and UMI. "
            "Only the explicitly selected workflow reports are exported.</p>"]
    for report in reports:
        title = report["title"]
        md.extend([f"## {title}", "", f"Source: `{report['source']}`.", ""])
        body.append(f"<details><summary>{escape(title)} <small>({len(report.get('plots', []))} plots)</small></summary>")
        body.append(f"<p><small>{escape(report['source'])}</small></p>")
        body.append("<div class='figs'>")
        for entry in report.get("plots", []):
            record = unique[entry["fingerprint"]]
            for view in record.get("views", []):
                md.append(f"- {view['title']}: [PNG]({view['png']}) · [PDF]({view['pdf']})")
                body.append(f"<figure><a href='{escape(view['png'])}'><img loading='lazy' src='{escape(view['png'])}' alt='{escape(view['title'])}'></a>"
                            f"<figcaption>{escape(view['title'])} · <a href='{escape(view['pdf'])}'>PDF</a></figcaption></figure>")
        md.append("")
        body.append("</div></details>")
    if within is not None:
        md.extend(["## Within-batch overlap", "", "[Manifest](within_batch/manifest.json) · [Captions](within_batch/figure_legends.md)", ""])
        body.append("<h2>Within-batch overlap</h2><p class='nav'><a href='within_batch/manifest.json'>Manifest</a> · <a href='within_batch/figure_legends.md'>Captions</a></p>")
        if within["computed"]:
            md.append("[Within-batch figures (PDF)](within_batch/within_batch_figures.pdf)")
            body.append("<p><a href='within_batch/within_batch_figures.pdf'>Within-batch figures (PDF)</a></p>")
        body.append("<div class='figs'>")
        for record in within["figures"]:
            if record["status"] == "computed":
                png, pdf = "within_batch/" + record["png"], "within_batch/" + record["pdf"]
                md.extend([f"- {record['id']}: [PNG]({png}) · [PDF]({pdf})", ""])
                body.append(f"<figure><a href='{escape(png)}'><img loading='lazy' src='{escape(png)}' alt='{escape(record['id'])}'></a>"
                            f"<figcaption>{escape(record['caption'])} · <a href='{escape(pdf)}'>PDF</a></figcaption></figure>")
            else:
                message = f"{record['id']}: {record['status']} — {record['reason']}"
                md.extend([message, ""])
                body.append("<p class='callout note'>" + escape(message) + "</p>")
        body.append("</div>")
    if failures:
        md.extend(["## Export failures", "", *[f"- {f['source']}: {f['error']}" for f in failures], ""])
        body.append("<h2>Export failures</h2><pre>" + escape(json.dumps(failures, indent=2)) + "</pre>")
    (out / "index.md").write_text("\n".join(md) + "\n")
    (out / "index.html").write_text("\n".join(body) + "\n</body></html>\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", type=Path, required=True, help="workflow output root")
    parser.add_argument("--outdir", type=Path, help="default: <source>/static_plots")
    parser.add_argument("--metadata", type=Path, required=True, help="config-derived sample_metadata.tsv")
    parser.add_argument("--reports", type=Path, nargs="+", required=True,
                        help="explicit report paths, relative to the current working directory")
    parser.add_argument("--within-batch", type=Path, help="optional within-batch assessment table directory")
    args = parser.parse_args(argv)
    source = Path(os.path.abspath(args.source))
    out = Path(os.path.abspath(args.outdir or source / "static_plots"))
    paths = list(dict.fromkeys(Path(os.path.abspath(path)) for path in args.reports))
    if source.resolve().is_relative_to(out.resolve()):
        parser.error("export directory must not contain or equal the source directory")
    for path in paths:
        if not path.is_relative_to(source) or path.resolve().is_relative_to(out.resolve()):
            parser.error(f"report must be below --source and outside the export directory: {path}")
        if not path.is_file():
            parser.error(f"missing report: {path}")
    R.U.configure(args.metadata)
    out.mkdir(parents=True, exist_ok=True)
    plt.style.use(str(HERE / "science.mplstyle"))
    reports, unique, failures = [], {}, []
    within = None
    if args.within_batch:
        from render_within_batch import render_all
        within = render_all(args.within_batch, out / "within_batch")
        failures.extend({"source": f"within_batch/{record['id']}", "error": record["reason"]}
                        for record in within["figures"] if record["status"] == "failed")
    with PdfPages(out / "all_analysis_qc_figures.pdf") as combined:
        combined.infodict()["Title"] = "Static analysis and QC figures"
        for path in paths:
            relative = str(path.relative_to(source))
            try:
                title, plots = extract(path, relative)
            except Exception as error:
                failures.append({"source": relative, "error": str(error)})
                continue
            report = {"source": relative, "resolved_source": str(path.resolve()), "title": title,
                      "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "plots": []}
            reports.append(report)
            figdir = out / path.relative_to(source).with_suffix("") / "figures"
            figdir.mkdir(parents=True, exist_ok=True)
            captions = [f"# {title}", "", f"Source: `{path.resolve()}`", "",
                        "Display terminology: fragvec and UMI. Source arrays and original labels are preserved in source_plot_data.json.gz.", ""]
            with gzip.open(figdir / "source_plot_data.json.gz", "wt", encoding="utf-8") as archive:
                json.dump(plots, archive)
            for plot in plots:
                digest = fingerprint(plot)
                report["plots"].append({"source_plot": plot["source_plot"], "fingerprint": digest})
                if digest in unique:
                    continue
                record = {"fingerprint": digest, "source": relative, "source_plot": plot["source_plot"],
                          "source_div_id": plot["id"], "source_traces": len(plot["data"]),
                          "source_points": int(source_points(plot)), "title": R.title_of(plot), "views": []}
                unique[digest] = record
                covered = set()
                try:
                    for view_i, (fig, view_name, trace_ids, notes) in enumerate(R.render(plot), 1):
                        R.finish(fig)
                        stem = f"F{plot['source_plot']:02d}_{view_i:02d}_{slug(view_name)}"
                        pdf, png = figdir / f"{stem}.pdf", figdir / f"{stem}.png"
                        fig.savefig(pdf)
                        fig.savefig(png)
                        combined.savefig(fig)
                        covered.update(trace_ids)
                        view = {"title": R.title_of(plot) + (f" · {R.label(view_name)}" if view_name not in ("plot", "histogram", "violin") else ""),
                                "pdf": str(pdf.relative_to(out)), "png": str(png.relative_to(out)),
                                "trace_indices": trace_ids, "notes": [R.label(n) for n in notes], "minimum_font_check": True}
                        record["views"].append(view)
                        captions.extend([f"## {stem}", "", view["title"], "", *view["notes"], "",
                                         f"![{view['title']}]({stem}.png)", ""])
                        plt.close(fig)
                    if not record["views"]:
                        raise ValueError("Plot contains no renderable traces")
                    if covered != set(range(len(plot["data"]))):
                        raise ValueError(f"Unrendered source traces: {sorted(set(range(len(plot['data']))) - covered)}")
                    record["trace_coverage_verified"] = True
                except Exception as error:
                    failures.append({"source": relative, "source_plot": plot["source_plot"],
                                     "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()})
                    plt.close("all")
                    print(f"ERROR {relative} plot {plot['source_plot']}: {error}", flush=True)
            (figdir / "figure_legends.md").write_text("\n".join(captions) + "\n")
            print(f"{relative}: {len(plots)} source plots", flush=True)
    summary = write_indexes(out, reports, unique, failures, args.metadata, within)
    print(f"Reports {summary['report_count']} · unique plots {summary['unique_source_plots']} · "
          f"figures {summary['static_figure_count']} · failures {len(failures)}", flush=True)
    print(f"Index: {out / 'index.html'}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
