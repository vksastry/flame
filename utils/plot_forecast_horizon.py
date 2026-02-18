import re
from pathlib import Path

import matplotlib.pyplot as plt


STYLE_MAP = {
    "Transformer": "--",
    "GLA": "-.",
    "Persistence": (0, (5, 1)),
    "Climatology": (0, (1, 1)),
}

FONT_SIZE = 12
TITLE_SIZE = 16
LABEL_SIZE = 14
LEGEND_SIZE = 12

EVAL_PATTERN = re.compile(
    r"\[eval\]\s*step\s*(\d+)\s*val_loss=([0-9.]+)\s*val_acc=([0-9.]+)\s*"
    r"persist_rmse=([0-9.]+)\s*persist_mae=([0-9.]+)\s*persist_acc=([0-9.]+)\s*"
    r"climo_rmse=([0-9.]+)\s*climo_mae=([0-9.]+)\s*climo_acc=([0-9.]+)"
)
PER_PATTERN = re.compile(
    r"\[eval\]\s*per-timestep rmse:\s*val\s*t0=([0-9.]+)\s*t1=([0-9.]+)\s*t2=([0-9.]+)\s*t3=([0-9.]+)\s*\|\s*"
    r"persist\s*t0=([0-9.]+)\s*t1=([0-9.]+)\s*t2=([0-9.]+)\s*t3=([0-9.]+)"
)


def _extract_step_metrics(log_path: Path, step_target: int) -> tuple[float | None, list[float] | None, list[float] | None]:
    text = log_path.read_text(errors="ignore")
    climo_rmse = None
    for match in EVAL_PATTERN.finditer(text):
        if int(match.group(1)) == step_target:
            climo_rmse = float(match.group(7))
            break

    per_vals = None
    for match in PER_PATTERN.finditer(text):
        per_vals = (
            [float(match.group(1)), float(match.group(2)), float(match.group(3)), float(match.group(4))],
            [float(match.group(5)), float(match.group(6)), float(match.group(7)), float(match.group(8))],
        )
    if per_vals is None:
        return climo_rmse, None, None
    return climo_rmse, per_vals[0], per_vals[1]


def generate_forecast_horizon_plot(
    transformer_log: Path,
    gla_log: Path,
    output_path: Path,
    step_target: int = 5000,
) -> None:
    climo_t, val_t, persist = _extract_step_metrics(transformer_log, step_target)
    climo_g, val_g, _ = _extract_step_metrics(gla_log, step_target)

    if val_t is None or val_g is None or persist is None:
        raise RuntimeError("Missing per-timestep RMSE data in logs.")

    climo = climo_t if climo_t is not None else climo_g
    horizons = ["t0", "t1", "t2", "t3"]

    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "axes.labelsize": LABEL_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": LEGEND_SIZE,
        }
    )

    plt.figure(figsize=(7.2, 4.6))
    series = [
        ("Transformer", val_t),
        ("GLA", val_g),
        ("Persistence", persist),
    ]
    for label, values in series:
        linestyle = STYLE_MAP.get(label, "--")
        plt.plot(horizons, values, linewidth=2, linestyle=linestyle, marker="o", label=label)

    if climo is not None:
        linestyle = STYLE_MAP.get("Climatology", "--")
        plt.plot(horizons, [climo] * len(horizons), linewidth=2, linestyle=linestyle, label="Climatology")

    plt.title("Per-timestep RMSE at Step 5000 (T_out=4)")
    plt.xlabel("Forecast horizon")
    plt.ylabel("RMSE (m)")
    plt.grid(True, alpha=0.2)
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    generate_forecast_horizon_plot(
        Path("multi_step_logs/transformer/transformer_340M-in8-out4-2026-02-16_23-01-04.log"),
        Path("multi_step_logs/gla/gla_340M-in8-out4-2026-02-17_01-31-28.log"),
        Path("paper/figures/forecast_horizon_rmse_step5000.pdf"),
    )
    print("Wrote updated plot to paper/figures/forecast_horizon_rmse_step5000.pdf")
