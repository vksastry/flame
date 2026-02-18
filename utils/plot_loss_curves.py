import re
from pathlib import Path

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

TRAIN_PATTERN = re.compile(r"step:\s*(\d+)\s+.*?loss:\s*([0-9.]+)")
EVAL_PATTERN = re.compile(r"\[eval\]\s*step\s*(\d+)\s*val_loss=([0-9.]+)")


def _select_log(model_dir: Path, min_step: int) -> Path | None:
    logs = sorted(model_dir.glob("*.log"))
    selected = None
    for log in logs:
        text = log.read_text(errors="ignore")
        steps = [int(m.group(1)) for m in TRAIN_PATTERN.finditer(text)]
        if steps and min(steps) <= 1 and max(steps) >= min_step:
            selected = log
            break
    if not selected:
        for log in logs:
            text = log.read_text(errors="ignore")
            steps = [int(m.group(1)) for m in TRAIN_PATTERN.finditer(text)]
            if steps and max(steps) >= min_step:
                selected = log
                break
    return selected if selected else (logs[-1] if logs else None)


def _parse_steps(log_path: Path, pattern: re.Pattern, max_step: int) -> dict[int, float]:
    text = log_path.read_text(errors="ignore")
    data: dict[int, float] = {}
    for match in pattern.finditer(text):
        step = int(match.group(1))
        if step > max_step:
            continue
        if step not in data:
            data[step] = float(match.group(2))
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
    for label, data in series.items():
        if not data:
            continue
        steps = sorted(data.keys())
        values = [data[s] for s in steps]
        linestyle = STYLE_MAP.get(label, "--")
        plt.plot(steps, values, linewidth=line_width, linestyle=linestyle, label=label)

    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.2)
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def generate_loss_plots(
    logs_root: Path,
    output_dir: Path,
    max_step: int = 200,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_data: dict[str, dict[int, float]] = {}
    eval_data: dict[str, dict[int, float]] = {}
    selected_logs: dict[str, str] = {}

    for label, dir_name in MODELS.items():
        log = _select_log(logs_root / dir_name, min_step=max_step)
        if not log:
            continue
        selected_logs[label] = log.name
        train_data[label] = _parse_steps(log, TRAIN_PATTERN, max_step=max_step)
        eval_data[label] = _parse_steps(log, EVAL_PATTERN, max_step=max_step)

    _plot_series(
        train_data,
        "Training Loss (Steps 1–200)",
        "Loss",
        output_dir / "loss_train_200steps.pdf",
    )
    _plot_series(
        eval_data,
        "Evaluation Loss (Steps 1–200)",
        "Validation Loss",
        output_dir / "loss_eval_200steps.pdf",
    )

    return selected_logs


if __name__ == "__main__":
    logs_root = Path("final_logs")
    output_dir = Path("paper/figures")
    selected = generate_loss_plots(logs_root, output_dir, max_step=200)
    print("Selected logs:")
    for label, name in selected.items():
        print(label, name)
    print("Wrote updated plots to paper/figures/loss_train_200steps.pdf and loss_eval_200steps.pdf")
