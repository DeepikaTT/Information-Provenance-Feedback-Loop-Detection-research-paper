# ─────────────────────────────────────────────
# visualize.py
# Reads results/pipeline_log.jsonl and plots
# cumulative vs. step GDS drift across generations.
# ─────────────────────────────────────────────

import json
import matplotlib.pyplot as plt
import os


def load_log(path: str = "results/pipeline_log.jsonl"):
    """
    Loads all logged runs. Since pipeline.py appends to this
    file every time you run it, this may contain multiple
    experiment runs back to back. We split them into separate
    "runs" whenever the generation number resets to 1.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No log found at {path}. Run pipeline.py first."
        )

    with open(path, "r") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    runs = []
    current_run = []
    for row in rows:
        if row["generation"] == 1 and current_run:
            runs.append(current_run)
            current_run = []
        current_run.append(row)
    if current_run:
        runs.append(current_run)

    return runs


def plot_runs(runs, save_path: str = "results/figures/gds_drift.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6))

    for i, run in enumerate(runs, start=1):
        gens = [r["generation"] for r in run]
        cumulative = [r["gds_cumulative"] for r in run]
        step = [r["gds_step"] for r in run]

        ax.plot(
            gens, cumulative,
            marker="o", linestyle="-",
            label=f"Run {i} — cumulative", linewidth=2,
        )
        ax.plot(
            gens, step,
            marker="x", linestyle="--",
            label=f"Run {i} — step", alpha=0.6,
        )

    ax.set_xlabel("Generation")
    ax.set_ylabel("GDS (Generational Drift Score)")
    ax.set_title("Information Drift Across Generations")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")

    plt.show()


if __name__ == "__main__":
    runs = load_log()
    print(f"Loaded {len(runs)} run(s) from log.")
    plot_runs(runs)