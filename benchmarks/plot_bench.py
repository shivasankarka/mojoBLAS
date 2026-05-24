"""
Usage (from repo root):
  python benchmarks/plot_bench.py \\
      --mojo-json benchmarks/mojo_l1_results.json \\
                  benchmarks/mojo_l2_results.json \\
                  benchmarks/mojo_l3_results.json \\
      --c-json    benchmarks/c_bench_results.json \\
      --out-prefix benchmarks/bench_plot
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

mpl.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "text.usetex": False,
        "axes.edgecolor": "black",
        "axes.linewidth": 1.1,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "legend.frameon": True,
        "legend.fancybox": False,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.7",
    }
)


# FLOPs formula per operation
def flops(op: str, n: int) -> float:
    op = op.lower()
    if op in ("axpy", "dot"):
        return 2.0 * n
    if op in ("scal", "copy", "swap", "rot", "rotm"):
        return float(n)
    if op in ("nrm2", "sum", "asum"):
        return 2.0 * n
    if op in ("rotg", "rotmg"):
        return 1.0
    if op in ("gemv", "gemv_trans"):
        return 2.0 * n * n
    if op in ("symv", "spmv", "sbmv", "gbmv"):
        return 2.0 * n * n
    if op in ("syr", "trmv", "trsv", "tpmv", "tpsv", "tbmv", "tbsv"):
        return float(n * n)
    if op in ("syr2", "ger", "spr", "spr2"):
        return 2.0 * n * n
    if op in ("gemm", "symm"):
        return 2.0 * n * n * n
    if op in ("syrk", "trmm", "trsm"):
        return float(n * n * n)
    if op in ("syr2k",):
        return 2.0 * n * n * n
    return 0.0


LIB_STYLE = {
    "mojo": dict(color="steelblue", marker="o", ls="-", lw=2.0, ms=5),
    "accelerate": dict(color="darkorange", marker="s", ls="--", lw=1.8, ms=5),
    "openblas": dict(color="mediumseagreen", marker="^", ls=":", lw=1.5, ms=5),
}
LIB_LABEL = {
    "mojo": "mojoBLAS",
    "accelerate": "Accelerate",
    "openblas": "OpenBLAS",
}
LIB_ORDER = ["mojo", "accelerate", "openblas"]

LEVEL1_OPS = ["axpy", "scal", "dot", "nrm2", "sum", "rotm", "rotmg"]
LEVEL2_OPS = [
    "gemv",
    "gemv_trans",
    "symv",
    "syr",
    "syr2",
    "trmv",
    "trsv",
    "spmv",
    "tpmv",
    "tpsv",
    "tbmv",
    "tbsv",
    "spr",
    "spr2",
]
LEVEL3_OPS = ["gemm", "symm", "syrk", "syr2k", "trmm", "trsm"]


# Data loading and aggregation


def load_all(mojo_paths, c_path):
    records = []
    for p in mojo_paths:
        records.extend(json.loads(Path(p).read_text()).get("results", []))
    if c_path:
        records.extend(json.loads(Path(c_path).read_text()).get("results", []))
    return records


def build_series(records):
    """
    Returns:
      gf_series[op][lib] = sorted [(n, gflops), ...]
      ms_series[op][lib] = sorted [(n, ms), ...]
    """
    gf_raw = defaultdict(lambda: defaultdict(dict))
    ms_raw = defaultdict(lambda: defaultdict(dict))

    for r in records:
        op = r.get("op", "").lower()
        lib = r.get("lib", "").lower()
        n = int(r.get("n", 0))
        sec = float(r.get("avg_seconds") or 0)
        if sec <= 0 or n <= 0:
            continue
        fp = flops(op, n)
        if fp <= 0:
            continue
        gf = fp / sec / 1e9
        ms = sec * 1000

        if n not in gf_raw[op][lib] or gf > gf_raw[op][lib][n]:
            gf_raw[op][lib][n] = gf
            ms_raw[op][lib][n] = ms

    def to_sorted(raw):
        return {
            op: {lib: sorted(nmap.items()) for lib, nmap in libs.items()}
            for op, libs in raw.items()
        }

    return to_sorted(gf_raw), to_sorted(ms_raw)


# Plotting helpers


def _make_legend_handles(libs_present):
    handles = []
    for lib in LIB_ORDER:
        if lib in libs_present:
            style = LIB_STYLE.get(
                lib, dict(color="gray", marker="x", ls="-", lw=1.5, ms=5)
            )
            handles.append(plt.Line2D([0], [0], label=LIB_LABEL.get(lib, lib), **style))
    return handles


def _plot_one_ax(ax, op, series, ylabel, log_y=False):
    """Plots all libraries for one op onto ax."""
    op_data = series.get(op, {})
    libs_here = [l for l in LIB_ORDER if l in op_data]
    for lib in libs_here:
        pts = op_data[lib]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        style = LIB_STYLE.get(lib, dict(color="gray", marker="x", ls="-", lw=1.5, ms=5))
        ax.plot(xs, ys, label=LIB_LABEL.get(lib, lib), **style)

    ax.set_title(op, fontsize=10, fontweight="bold")
    ax.set_xlabel("n", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, which="major", ls="--", lw=0.4, alpha=0.6)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x):,}" if x >= 1000 else str(int(x)))
    )
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=7, loc="best")

    return set(libs_here)


def plot_group(gf_series, ops, title_prefix, out_gf):
    present = [op for op in ops if op in gf_series]
    if not present:
        return

    ncols = min(len(present), 4)
    nrows = math.ceil(len(present) / ncols)
    all_libs = set()

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 3.8 * nrows), squeeze=False
    )
    for idx, op in enumerate(present):
        ax = axes[idx // ncols][idx % ncols]
        libs = _plot_one_ax(ax, op, gf_series, "GFlops/s  (higher=faster)", log_y=False)
        all_libs |= libs
    for idx in range(len(present), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    handles = _make_legend_handles(all_libs)
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.suptitle(f"{title_prefix}", fontsize=13, fontweight="bold", y=1.05)
    fig.tight_layout()
    fig.savefig(out_gf, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_gf}")


def main():
    ap = argparse.ArgumentParser(description="Plot mojoBLAS benchmark results")
    ap.add_argument("--mojo-json", nargs="+", required=True)
    ap.add_argument("--c-json", required=True)
    ap.add_argument("--out-prefix", default="benchmarks/bench_plot")
    args = ap.parse_args()

    print("Loading benchmark results...")
    records = load_all(args.mojo_json, args.c_json)
    print(f"  {len(records)} records")

    gf_series, ms_series = build_series(records)
    print(f"  ops: {sorted(gf_series)}")

    p = args.out_prefix
    print("\nLevel 1 (vector ops)...")
    plot_group(gf_series, LEVEL1_OPS, "Level 1 - Vector ops", f"{p}_level1.png")

    print("Level 2 (matrix-vector ops)...")
    plot_group(gf_series, LEVEL2_OPS, "Level 2 - Matrix-vector ops", f"{p}_level2.png")

    print("Level 3 (matrix-matrix ops)...")
    plot_group(gf_series, LEVEL3_OPS, "Level 3 - Matrix-matrix ops", f"{p}_level3.png")

    print("\nAll done.")


if __name__ == "__main__":
    main()
