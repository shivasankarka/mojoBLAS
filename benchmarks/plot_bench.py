import argparse
import json
from collections import defaultdict

import matplotlib.pyplot as plt

# TODO: Cross check this later.
OP_BYTES_PER_ELEM = {
    "axpy": 3.0,
    "scal": 2.0,
    "dot": 2.0,
    "nrm2": 1.0,
    "sum": 1.0,
    "gemv": 2.0,
    "gemv_trans": 2.0,
    "trmv": 1.0,
    "trsv": 1.0,
    "symv": 2.0,
    "syr": 1.0,
    "syr2": 1.0,
    "gemm": 2.0,
    "syrk": 1.0,
    "syr2k": 2.0,
    "symm": 2.0,
    "trmm": 2.0,
    "trsm": 2.0,
}

LEVEL1_OPS = ["axpy", "scal", "dot", "nrm2", "sum"]
LEVEL2_OPS = ["gemv", "gemv_trans", "trmv", "trsv", "symv", "syr", "syr2"]
LEVEL3_OPS = ["gemm", "syrk", "syr2k", "symm", "trmm", "trsm"]


def load_results(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("results", [])


def parse_size_map(text):
    if not text:
        return {}
    out = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        name, val = item.split(":")
        out[name.strip()] = int(val.strip())
    return out


def build_series(results, elem_sizes):
    series = defaultdict(lambda: defaultdict(list))
    for row in results:
        lib = row["lib"]
        op = row["op"]
        n = int(row["n"])
        if "avg_seconds" in row:
            avg_seconds = float(row["avg_seconds"])
        elif "avg_ns" in row:
            avg_seconds = float(row["avg_ns"]) * 1e-9
        else:
            continue
        gb_per_s = avg_seconds
        series[op][lib].append((n, gb_per_s))
    for op in series:
        for lib in series[op]:
            series[op][lib].sort(key=lambda t: t[0])
    return series


def plot_level(series, out_path, title, ops):
    n_ops = len(ops)
    ncols = min(n_ops, 4)
    nrows = (n_ops + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=False)
    if nrows == 1 and ncols == 1:
        axes = [axes]
    elif nrows == 1:
        axes = list(axes)
    else:
        axes = axes.flatten()

    for idx, op in enumerate(ops):
        ax = axes[idx]
        libs = series.get(op, {})
        for lib, points in libs.items():
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            ax.plot(xs, ys, marker="o", label=lib)
        ax.set_title(op)
        ax.set_xlabel("n")
        ax.set_ylabel("avg_seconds")
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax.set_xscale("log")
        ax.set_xlim([min(xs) * 0.8, max(xs) * 1.2])

    for idx in range(len(ops), len(axes)):
        axes[idx].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        all_libs = set()
        for op in ops:
            for lib in series.get(op, {}):
                all_libs.add(lib)
        fig.legend(handles, labels, loc="upper center", ncol=len(all_libs))
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=200)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mojo-json",
        required=True,
        nargs="+",
        help="One or more mojo benchmark JSON files",
    )
    parser.add_argument("--c-json", required=True)
    parser.add_argument("--out-prefix", default="bench_plot")
    parser.add_argument(
        "--elem-sizes",
        default="mojo:4,accelerate:8,openblas:8",
        help="comma-separated lib:bytes entries",
    )
    args = parser.parse_args()

    elem_sizes = parse_size_map(args.elem_sizes)

    results = []
    for path in args.mojo_json:
        results.extend(load_results(path))
    results.extend(load_results(args.c_json))

    series = build_series(results, elem_sizes)

    plot_level(
        series,
        f"{args.out_prefix}_level1.png",
        "BLAS Level 1 (avg_seconds)",
        LEVEL1_OPS,
    )
    plot_level(
        series,
        f"{args.out_prefix}_level2.png",
        "BLAS Level 2 (avg_seconds)",
        LEVEL2_OPS,
    )
    plot_level(
        series,
        f"{args.out_prefix}_level3.png",
        "BLAS Level 3 (avg_seconds)",
        LEVEL3_OPS,
    )


if __name__ == "__main__":
    main()
