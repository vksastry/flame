import re
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt


MODELS = {
    "Transformer": "transformer",
    "GLA": "gla",
    "Gated DeltaNet": "gated_deltanet",
    "DeltaNet": "delta_net",
    "Mamba2": "mamba2",
    "KDA": "kda",
}

STYLE_MAP = {
    "Transformer": "--",
    "GLA": "-.",
    "Gated DeltaNet": ":",
    "DeltaNet": (0, (5, 1)),
    "Mamba2": (0, (3, 1, 1, 1)),
    "KDA": (0, (1, 1)),
}

FONT_SIZE = 12
TITLE_SIZE = 16
LABEL_SIZE = 14
LEGEND_SIZE = 12
SEQ_ORDER = [16, 128, 1024, 16384]

RUN_PATTERN = re.compile(r"Running\s+(\w+)_340M\s+with\s+(\d+)")
STEP_PATTERN = re.compile(
    r"step:\s*(\d+)\s+.*?memory:\s*([0-9.]+)GiB.*?tps:\s*([0-9,]+)",
    re.IGNORECASE,
)


def _parse_benchmark_log(log_path: Path) -> dict[str, dict[int, dict[str, float]]]:
    text = log_path.read_text(errors="ignore")
    runs = [
        (m.start(), m.group(1), int(m.group(2)))
        for m in RUN_PATTERN.finditer(text)
    ]
    runs.append((len(text), "", -1))

    data: dict[str, dict[int, dict[str, float]]] = {}
    for idx in range(len(runs) - 1):
        start, model, seq = runs[idx]
        end = runs[idx + 1][0]
        chunk = text[start:end]
        steps = list(STEP_PATTERN.finditer(chunk))
        if not steps:
            continue
        last = steps[-1]
        tps = float(last.group(3).replace(",", ""))
        mem = float(last.group(2))
        data.setdefault(model, {})[seq] = {"tps": tps, "memory": mem}
    return data


def _plot_series(
    series: dict[str, dict[int, float]],
    title: str,
    ylabel: str,
    output_path: Path,
    font_size: int = FONT_SIZE,
    title_size: int = TITLE_SIZE,
    label_size: int = LABEL_SIZE,
    legend_size: int = LEGEND_SIZE,
    line_width: float = 2.0,
) -> None:
    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.titlesize": title_size,
            "axes.labelsize": label_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": legend_size,
        }
    )

    plt.figure(figsize=(7.2, 4.6))
    x_labels = ["16", "128", "1K", "16K"]
    for label, data in series.items():
        if not data:
            continue
        xs = [data.get(seq) for seq in SEQ_ORDER]
        if any(v is None for v in xs):
            continue
        values = [float(cast(float, v)) for v in xs]
        linestyle = STYLE_MAP.get(label, "--")
        plt.plot(range(len(SEQ_ORDER)), values, linewidth=line_width, linestyle=linestyle, label=label)

    plt.title(title)
    plt.xlabel("Sequence length")
    plt.ylabel(ylabel)
    plt.xticks(range(len(SEQ_ORDER)), x_labels)
    plt.grid(True, alpha=0.2)
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def generate_scaling_plots(log_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = _parse_benchmark_log(log_path)

    tps_series: dict[str, dict[int, float]] = {}
    mem_series: dict[str, dict[int, float]] = {}
    for label, key in MODELS.items():
        model_data = raw.get(key, {})
        if not model_data:
            continue
        tps_series[label] = {seq: model_data[seq]["tps"] for seq in model_data if seq in SEQ_ORDER}
        mem_series[label] = {seq: model_data[seq]["memory"] for seq in model_data if seq in SEQ_ORDER}

    _plot_series(
        tps_series,
        "Tokens/sec vs. Sequence Length",
        "Tokens/sec",
        output_dir / "throughput_scaling.pdf",
    )
    _plot_series(
        mem_series,
        "Memory vs. Sequence Length",
        "Memory (GiB)",
        output_dir / "memory_scaling.pdf",
    )


if __name__ == "__main__":
    generate_scaling_plots(
        Path("run_logs/benchmarks_all_model_inp.txt"),
        Path("paper/figures"),
    )
    print("Wrote updated plots to paper/figures/throughput_scaling.pdf and memory_scaling.pdf")
