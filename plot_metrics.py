"""
Plot histograms from a CSV produced by get_metrics.py / get_distance.py
(or any CSV with a mesh_id column and one or more numeric columns).

Behavior per numeric column:
  - --threshold name=value: values >= threshold collapse into one red bar on
    the right; below it is split into --nbins equal-width bins. A sibling text
    file <col>_above_threshold.txt lists the mesh_ids in that red bin.
  - --threshold-low name=value: values <= threshold collapse into one red bar
    on the left, with <col>_below_threshold.txt listing those mesh_ids.
  - You can pass both for the same column to clip both tails.
  - Without any threshold, you get a plain --nbins equal-width histogram.

Examples:
    python plot_metrics.py -i metrics.csv -o plots \\
        --threshold min_angle=30 radius_ratio=0.6 --nbins 30
    python plot_metrics.py -i distance.csv -o plots --threshold hausdorff=0.01
    python plot_metrics.py -i metrics.csv -o plots --threshold-low num_faces=100
    python plot_metrics.py -i metrics.csv -o plots          # no thresholds, plain hists
"""
import argparse
import os

import numpy as np
import pandas as pd


def _set3_color(index):
    """Pick a color from Plotly's Set3 qualitative palette by integer index."""
    from plotly.colors import qualitative
    palette = qualitative.Set3
    return palette[index % len(palette)]


def parse_thresholds(items):
    out = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Bad --threshold entry '{item}', expected name=value")
        name, val = item.split("=", 1)
        out[name.strip()] = float(val)
    return out


def _layout(fig, title):
    fig.update_layout(
        title=title,
        xaxis_title=title,
        yaxis_title="Frequency",
        bargap=0,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Times New Roman, serif", size=14),
        margin=dict(l=40, r=40, t=50, b=40),
        yaxis=dict(showgrid=True, gridcolor="lightgray", ticksuffix="%"),
    )


def _save(fig, out_path, fmt, width, height):
    if fmt == "html":
        fig.write_html(out_path)
    else:
        fig.write_image(out_path, format=fmt, width=width, height=height)


def _add_stats_overlay(fig, values, value_suffix=""):
    """Mean/median dashed lines (gray, fine vs coarse dash) + a stats box."""
    if values.size == 0:
        return
    mean_v = float(np.mean(values))
    median_v = float(np.median(values))
    line_color = "#555555"
    fig.add_vline(
        x=mean_v, line_dash="dot", line_color=line_color, line_width=1.5,
    )
    fig.add_vline(
        x=median_v, line_dash="longdash", line_color=line_color, line_width=1.5,
    )
    fig.add_annotation(
        xref="paper", yref="paper", x=0.98, y=0.98,
        xanchor="right", yanchor="top",
        showarrow=False, align="left",
        text=(
            f"mean (·····) = {mean_v:.4g}{value_suffix}<br>"
            f"median (—  —) = {median_v:.4g}{value_suffix}"
        ),
        bgcolor="white", bordercolor="black", borderwidth=1, borderpad=4,
        font=dict(family="Times New Roman, serif", size=12, color=line_color),
    )


def plot_plain_hist(values, nbins, title, out_path, fmt, width, height,
                    bar_color="#80B1D3", as_percent=False):
    import plotly.graph_objs as go

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        print(f"  no data for {title}, skipping")
        return

    if as_percent:
        values = values * 100.0

    fig = go.Figure(
        go.Histogram(
            x=values, nbinsx=nbins, marker_color=bar_color, marker_line_width=0,
            histnorm="percent",
        )
    )
    _layout(fig, title)
    _add_stats_overlay(fig, values, value_suffix="%" if as_percent else "")
    if as_percent:
        fig.update_xaxes(ticksuffix="%")
    _save(fig, out_path, fmt, width, height)
    print(f"  wrote {out_path} ({values.size} values, no threshold)")


def plot_threshold_hist(values, threshold, nbins, title, out_path, fmt, width, height,
                        bar_color="#80B1D3", as_percent=False, low_threshold=None):
    """Histogram with optional collapsed red bars on either side.

    `threshold` (high): values >= threshold collapse into a single red bar on the right.
    `low_threshold`: values <= low_threshold collapse into a single red bar on the left.
    Either or both may be set. The interior bins span (low_threshold, threshold).
    """
    import plotly.graph_objs as go

    values = np.asarray(values, dtype=float)
    finite_mask = np.isfinite(values)
    values = values[finite_mask]
    if values.size == 0:
        print(f"  no data for {title}, skipping")
        return

    raw_high = threshold
    raw_low = low_threshold
    if as_percent:
        values = values * 100.0
        if threshold is not None:
            threshold = threshold * 100.0
        if low_threshold is not None:
            low_threshold = low_threshold * 100.0

    if low_threshold is not None and threshold is not None:
        middle = values[(values > low_threshold) & (values < threshold)]
    elif threshold is not None:
        middle = values[values < threshold]
    elif low_threshold is not None:
        middle = values[values > low_threshold]
    else:
        middle = values

    above_count = int((values >= threshold).sum()) if threshold is not None else 0
    below_count = int((values <= low_threshold).sum()) if low_threshold is not None else 0

    if middle.size:
        v_lo = float(middle.min())
        v_hi = float(middle.max())
    else:
        v_lo = float(values.min()) if values.size else 0.0
        v_hi = float(values.max()) if values.size else 1.0

    lower_edge = low_threshold if low_threshold is not None else v_lo
    upper_edge = threshold if threshold is not None else v_hi
    if upper_edge <= lower_edge:
        upper_edge = lower_edge + max(1.0, abs(lower_edge))

    edges = np.linspace(lower_edge, upper_edge, nbins + 1)
    counts, _ = np.histogram(middle, bins=edges)

    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    bar_x = list(centers)
    bar_y_counts = list(counts.astype(int))
    bar_w = list(widths)
    colors = [bar_color] * len(bar_x)

    extra_w = float(np.mean(widths)) if widths.size else 1.0

    if below_count > 0 and low_threshold is not None:
        bar_x.insert(0, low_threshold - extra_w / 2.0)
        bar_y_counts.insert(0, below_count)
        bar_w.insert(0, extra_w)
        colors.insert(0, "#FB8072")

    if above_count > 0 and threshold is not None:
        bar_x.append(threshold + extra_w / 2.0)
        bar_y_counts.append(above_count)
        bar_w.append(extra_w)
        colors.append("#FB8072")

    total = float(values.size) if values.size > 0 else 1.0
    bar_y = [100.0 * c / total for c in bar_y_counts]

    fig = go.Figure(
        go.Bar(x=bar_x, y=bar_y, width=bar_w, marker_color=colors, marker_line_width=0)
    )
    _layout(fig, title)
    _add_stats_overlay(fig, values, value_suffix="%" if as_percent else "")
    if as_percent:
        fig.update_xaxes(ticksuffix="%")
    _save(fig, out_path, fmt, width, height)
    print(
        f"  wrote {out_path} ("
        f"<= {raw_low}: {below_count}, "
        f"middle: {int(counts.sum())}, "
        f">= {raw_high}: {above_count})"
    )


ID_COL_CANDIDATES = ("mesh_id", "model")


def detect_id_col(df):
    for c in ID_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None


def write_above_threshold_list(df, id_col, col, threshold, out_path):
    mask = df[col] >= threshold
    sub = df.loc[mask, [id_col, col]].copy()
    sub = sub.sort_values(col, ascending=False)
    with open(out_path, "w") as f:
        for mid, val in zip(sub[id_col].tolist(), sub[col].tolist()):
            f.write(f"{mid}\t{val}\n")
    print(f"  wrote {out_path} ({len(sub)} models with {col} >= {threshold})")


def write_below_threshold_list(df, id_col, col, threshold, out_path):
    mask = df[col] <= threshold
    sub = df.loc[mask, [id_col, col]].copy()
    sub = sub.sort_values(col, ascending=True)
    with open(out_path, "w") as f:
        for mid, val in zip(sub[id_col].tolist(), sub[col].tolist()):
            f.write(f"{mid}\t{val}\n")
    print(f"  wrote {out_path} ({len(sub)} models with {col} <= {threshold})")


STAGE_DEFS = [
    ("Stage 1+2", "patch_tracing_sum_s", 0),
    ("Stage 3+4", "align_fuse_s",        2),
    ("Stage 5",   "remesh_s",            4),
]


def _filter_finished(df):
    if "n_runs_unfinished" in df.columns:
        before = len(df)
        df = df[df["n_runs_unfinished"] == 0]
        print(f"  filtered unfinished: kept {len(df)}/{before} rows")
    return df


def _label_for_path(path):
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    if "abc" in stem:
        return "ABC"
    if "assem" in stem or "fusion" in stem:
        return "Fusion"
    return os.path.splitext(os.path.basename(path))[0]


def plot_stage_runtimes(rows, agg_label, out_path, fmt, width, height, color_seed=0):
    """rows: list of (dataset_label, {col: aggregated_value}) preserving caller order.

    Renders one horizontal stacked bar per dataset, segmented by STAGE_DEFS.
    color_seed shifts the Set3 palette indices for the stage colors.
    """
    import plotly.graph_objs as go

    labels = [r[0] for r in rows]
    fig = go.Figure()
    for stage_label, col, palette_idx in STAGE_DEFS:
        xs = [float(r[1].get(col, 0.0)) for r in rows]
        fig.add_trace(go.Bar(
            name=stage_label,
            y=labels, x=xs, orientation="h",
            marker_color=_set3_color(palette_idx + color_seed),
            marker_line_width=0,
            text=[f"{v:.0f}" for v in xs],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(family="Times New Roman, serif", size=14, color="white"),
            cliponaxis=False,
        ))
    fig.update_layout(
        barmode="stack",
        bargap=0.6,
        title=f"{agg_label} runtime per stage (s)",
        xaxis_title="seconds",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Times New Roman, serif", size=14),
        margin=dict(l=80, r=180, t=50, b=40),
        xaxis=dict(showgrid=True, gridcolor="lightgray"),
        legend=dict(traceorder="normal"),
    )
    tight_height = 105 * len(rows) + 195
    _save(fig, out_path, fmt, width, tight_height)
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-i", type=str, nargs="+", required=True,
                        help="Input CSV(s). Pass multiple to plot e.g. metrics.csv and metrics_counts.csv together. "
                             "If a single CSV is passed and a sibling <stem>_counts.csv exists, it is auto-included.")
    parser.add_argument("-o", type=str, default="plots", help="Output folder.")
    parser.add_argument(
        "--threshold",
        nargs="*",
        default=[],
        help="Optional per-metric high thresholds (values >= collapse into a red right bar), "
             "e.g. min_angle=30 radius_ratio=0.6 hausdorff=0.01",
    )
    parser.add_argument(
        "--threshold-low",
        nargs="*",
        default=[],
        help="Optional per-metric low thresholds (values <= collapse into a red left bar), "
             "e.g. num_faces=100",
    )
    parser.add_argument("--nbins", type=int, default=30, help="Equal-width bins below threshold (or total bins for plain hist).")
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg", "html"],
        help="Output format. png/pdf/svg need kaleido (pip install --upgrade kaleido).",
    )
    parser.add_argument("--width", type=int, default=1000, help="Image width in px (ignored for html).")
    parser.add_argument("--height", type=int, default=600, help="Image height in px (ignored for html).")
    parser.add_argument("--color-seed", type=int, default=0,
                        help="Offset into the Plotly Set3 palette; consecutive metrics cycle through Set3 starting from this index.")
    parser.add_argument("--percent-axes", nargs="*", default=["hausdorff"],
                        help="Column names to render with both axes in %%. Values are multiplied by 100 on x; "
                             "y becomes percent of total. Default: hausdorff. Pass empty to disable.")
    parser.add_argument("--stage-bar", action="store_true",
                        help="Also emit stage_runtime_mean.<fmt> and stage_runtime_median.<fmt>: "
                             "horizontal stacked bars per input CSV using "
                             "patch_tracing_sum_s (Stage 1+2), align_fuse_s (Stage 3+4), remesh_s (Stage 5). "
                             "Rows with n_runs_unfinished > 0 are excluded from the aggregates.")
    parser.add_argument("--stage-bar-only", action="store_true",
                        help="Implies --stage-bar and skips the per-column histograms.")
    parser.add_argument("--dataset-labels", nargs="*", default=None,
                        help="Per-input dataset labels (must match --i order). "
                             "If omitted, labels are inferred from filename: *abc* -> ABC, "
                             "*assem*/*fusion* -> Fusion, else the filename stem.")
    args = parser.parse_args()

    thresholds = parse_thresholds(args.threshold)
    low_thresholds = parse_thresholds(args.threshold_low)
    os.makedirs(args.o, exist_ok=True)

    inputs = list(args.i)
    if len(inputs) == 1:
        stem, ext = os.path.splitext(inputs[0])
        sibling = f"{stem}_counts{ext or '.csv'}"
        if not stem.endswith("_counts") and os.path.isfile(sibling):
            print(f"Auto-including sibling counts CSV: {sibling}")
            inputs.append(sibling)

    percent_cols = set(args.percent_axes or [])
    want_stage_bar = args.stage_bar or args.stage_bar_only
    skip_hist = args.stage_bar_only

    if args.dataset_labels is not None and len(args.dataset_labels) != len(inputs):
        parser.error(
            f"--dataset-labels has {len(args.dataset_labels)} entries but "
            f"-i has {len(inputs)} inputs"
        )

    stage_rows_mean = []
    stage_rows_median = []

    for idx, csv_path in enumerate(inputs):
        print(f"Reading {csv_path}")
        df = pd.read_csv(csv_path)

        if want_stage_bar:
            label = (args.dataset_labels[idx] if args.dataset_labels
                     else _label_for_path(csv_path))
            stage_df = _filter_finished(df)
            missing = [c for _, c, _ in STAGE_DEFS if c not in stage_df.columns]
            if missing:
                print(f"  skipping stage-bar for {csv_path}: missing columns {missing}")
            else:
                mean_agg = {c: float(stage_df[c].mean()) for _, c, _ in STAGE_DEFS}
                median_agg = {c: float(stage_df[c].median()) for _, c, _ in STAGE_DEFS}
                stage_rows_mean.append((label, mean_agg))
                stage_rows_median.append((label, median_agg))

        if skip_hist:
            continue

        id_col = detect_id_col(df)
        numeric_cols = [
            c for c in df.columns
            if c != id_col and pd.api.types.is_numeric_dtype(df[c])
        ]
        color_idx = args.color_seed
        for col in numeric_cols:
            out_path = os.path.join(args.o, f"{col}.{args.format}")
            bar_color = _set3_color(color_idx)
            color_idx += 1
            as_percent = col in percent_cols
            high = thresholds.get(col)
            low = low_thresholds.get(col)
            if high is not None or low is not None:
                plot_threshold_hist(
                    df[col].values, high, args.nbins, col, out_path,
                    args.format, args.width, args.height,
                    bar_color=bar_color, as_percent=as_percent,
                    low_threshold=low,
                )
                if id_col is None:
                    print(f"  skipping above/below-threshold list for {col}: "
                          f"no id column found (expected one of {ID_COL_CANDIDATES})")
                else:
                    if high is not None:
                        list_path = os.path.join(args.o, f"{col}_above_threshold.txt")
                        write_above_threshold_list(df, id_col, col, high, list_path)
                    if low is not None:
                        list_path = os.path.join(args.o, f"{col}_below_threshold.txt")
                        write_below_threshold_list(df, id_col, col, low, list_path)
            else:
                plot_plain_hist(
                    df[col].values, args.nbins, col, out_path,
                    args.format, args.width, args.height,
                    bar_color=bar_color, as_percent=as_percent,
                )

    if want_stage_bar and stage_rows_mean:
        mean_path = os.path.join(args.o, f"stage_runtime_mean.{args.format}")
        median_path = os.path.join(args.o, f"stage_runtime_median.{args.format}")
        plot_stage_runtimes(stage_rows_mean, "Mean", mean_path,
                            args.format, args.width, args.height,
                            color_seed=args.color_seed)
        plot_stage_runtimes(stage_rows_median, "Median", median_path,
                            args.format, args.width, args.height,
                            color_seed=args.color_seed)
