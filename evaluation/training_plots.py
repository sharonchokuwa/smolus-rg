import json
import logging
from pathlib import Path
from typing import Optional
import matplotlib.ticker as mticker

logger = logging.getLogger(__name__)

def load_trainer_state(cfg: dict):
    """
    Load trainer_state.json from the experiment output directory.
    """
    output_dir = Path(cfg["experiment"]["output_dir"])
    candidate  = output_dir / "trainer_state.json"

    if not candidate.exists():
        # Try the latest checkpoint
        checkpoints = sorted(output_dir.glob("checkpoint-*"), key=_ckpt_step)
        if checkpoints:
            candidate = checkpoints[-1] / "trainer_state.json"
            logger.info(f"Root trainer_state.json not found — using {candidate}")

    if not candidate.exists():
        raise FileNotFoundError(
            f"trainer_state.json not found in {output_dir}.\n"
            "Make sure training has run at least one checkpoint save."
        )

    with open(candidate, encoding="utf-8") as f:
        state = json.load(f)

    logger.info(
        f"Loaded trainer_state.json: {len(state.get('log_history', []))} entries, "
        f"global_step={state.get('global_step', '?')}"
    )
    return state


def _ckpt_step(path: Path):
    """Extract step number from checkpoint-N directory name."""
    try:
        return int(path.name.split("-")[-1])
    except ValueError:
        return 0


def parse_log_history(state: dict):
    """
    Split the log_history list into separate time series.
    """
    train_steps   = []
    train_loss    = []
    train_lr      = []
    train_grad    = []

    eval_steps    = []
    eval_loss     = []
    eval_entropy  = []
    eval_accuracy = []
    eval_tokens   = []

    for entry in state.get("log_history", []):
        step = entry.get("step", 0)

        if "loss" in entry:                    # training log entry
            train_steps.append(step)
            train_loss.append(entry["loss"])
            if "learning_rate" in entry:
                train_lr.append(entry["learning_rate"])
            if "grad_norm" in entry:
                train_grad.append(entry["grad_norm"])

        if "eval_loss" in entry:               # evaluation log entry
            eval_steps.append(step)
            eval_loss.append(entry["eval_loss"])
            if "eval_entropy" in entry:
                eval_entropy.append(entry["eval_entropy"])
            if "eval_mean_token_accuracy" in entry:
                eval_accuracy.append(entry["eval_mean_token_accuracy"] * 100)
            if "eval_num_tokens" in entry:
                eval_tokens.append(entry["eval_num_tokens"])

    return {
        "train_steps":   train_steps,
        "train_loss":    train_loss,
        "train_lr":      train_lr,
        "train_grad":    train_grad,
        "eval_steps":    eval_steps,
        "eval_loss":     eval_loss,
        "eval_entropy":  eval_entropy,
        "eval_accuracy": eval_accuracy,
        "eval_tokens":   eval_tokens,
        # Metadata
        "best_metric":   state.get("best_metric"),
        "best_step":     _best_step(state),
        "global_step":   state.get("global_step", 0),
        "total_epochs":  state.get("epoch", 0),
    }


def _best_step(state: dict):
    """Extract the step of the best checkpoint from its directory name."""
    ckpt = state.get("best_model_checkpoint", "")
    if ckpt:
        try:
            return int(Path(ckpt).name.split("-")[-1])
        except (ValueError, IndexError):
            pass
    return None


# Plot style
_TRAIN_COLOUR  = "#2563EB" 
_EVAL_COLOUR   = "#DC2626"   
_ACCENT_COLOUR = "#16A34A"   
_LR_COLOUR     = "#7C3AED"  
_ENTROPY_COLOUR= "#EA580C"   
_GRID_COLOUR   = "#E5E7EB"   
_BG_COLOUR     = "#FAFAFA"
_TEXT_COLOUR   = "#111827"


def _apply_style(ax, title: str, xlabel: str, ylabel: str):
    ax.set_title(title, fontsize=12, fontweight="bold",
                 color=_TEXT_COLOUR, pad=10)
    ax.set_xlabel(xlabel, fontsize=10, color=_TEXT_COLOUR)
    ax.set_ylabel(ylabel, fontsize=10, color=_TEXT_COLOUR)
    ax.set_facecolor(_BG_COLOUR)
    ax.grid(True, color=_GRID_COLOUR, linewidth=0.8, linestyle="--", alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(_GRID_COLOUR)
    ax.spines["bottom"].set_color(_GRID_COLOUR)
    ax.tick_params(colors=_TEXT_COLOUR, labelsize=9)
    ax.legend(fontsize=9, framealpha=0.9,
              edgecolor=_GRID_COLOUR, facecolor="white")


def _mark_best(ax, best_step: Optional[int], ymin: float, ymax: float):
    """Draw a vertical dashed line at the best checkpoint step."""
    if best_step is not None:
        ax.axvline(
            x=best_step, color=_ACCENT_COLOUR, linestyle=":",
            linewidth=1.5, label=f"Best ckpt (step {best_step})", alpha=0.8,
        )

def plot_loss_curves(ax, series: dict) -> None:
    """Plot training loss and validation loss on the same axes."""
    if series["train_steps"]:
        ax.plot(series["train_steps"], series["train_loss"],
                color=_TRAIN_COLOUR, linewidth=1.8,
                label="Training loss", alpha=0.9)

    if series["eval_steps"]:
        ax.plot(series["eval_steps"], series["eval_loss"],
                color=_EVAL_COLOUR, linewidth=2.0,
                marker="o", markersize=4,
                label="Validation loss", alpha=0.9)

        # Annotate the final validation loss
        ax.annotate(
            f"{series['eval_loss'][-1]:.4f}",
            xy=(series["eval_steps"][-1], series["eval_loss"][-1]),
            xytext=(8, -14), textcoords="offset points",
            fontsize=8, color=_EVAL_COLOUR,
        )

    _mark_best(ax, series["best_step"],
               ymin=0, ymax=max(series["train_loss"] or [1]))
    _apply_style(ax, "Loss", "Step", "Loss")
    ax.set_ylim(bottom=0)


def plot_learning_rate(ax, series: dict):
    """Plot the learning rate schedule."""
    if not series["train_lr"]:
        ax.text(0.5, 0.5, "No LR data", transform=ax.transAxes,
                ha="center", va="center", color=_TEXT_COLOUR)
        _apply_style(ax, "Learning Rate Schedule", "Step", "LR")
        return

    ax.plot(series["train_steps"][:len(series["train_lr"])],
            series["train_lr"],
            color=_LR_COLOUR, linewidth=1.8, label="Learning rate")

    # Annotate peak LR
    peak_lr = max(series["train_lr"])
    peak_step = series["train_steps"][series["train_lr"].index(peak_lr)]
    ax.annotate(
        f"{peak_lr:.2e}",
        xy=(peak_step, peak_lr),
        xytext=(8, 6), textcoords="offset points",
        fontsize=8, color=_LR_COLOUR,
    )

    _mark_best(ax, series["best_step"], ymin=0, ymax=peak_lr)
    _apply_style(ax, "Learning Rate Schedule", "Step", "Learning Rate")
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))


def plot_token_accuracy(ax, series: dict):
    """Plot mean token accuracy on validation set."""
    if not series["eval_accuracy"]:
        ax.text(0.5, 0.5, "No accuracy data", transform=ax.transAxes,
                ha="center", va="center", color=_TEXT_COLOUR)
        _apply_style(ax, "Mean Token Accuracy (Val)", "Step", "Accuracy (%)")
        return

    ax.plot(series["eval_steps"], series["eval_accuracy"],
            color=_ACCENT_COLOUR, linewidth=2.0,
            marker="o", markersize=4, label="Token accuracy (%)")

    # Annotate final value
    ax.annotate(
        f"{series['eval_accuracy'][-1]:.2f}%",
        xy=(series["eval_steps"][-1], series["eval_accuracy"][-1]),
        xytext=(8, -14), textcoords="offset points",
        fontsize=8, color=_ACCENT_COLOUR,
    )

    _mark_best(ax, series["best_step"], ymin=0, ymax=100)
    _apply_style(ax, "Mean Token Accuracy (Val)", "Step", "Accuracy (%)")

    ymin = max(0, min(series["eval_accuracy"]) - 2)
    ax.set_ylim(bottom=ymin, top=100)


def plot_entropy(ax, series: dict):
    if not series["eval_entropy"]:
        ax.text(0.5, 0.5, "No entropy data", transform=ax.transAxes,
                ha="center", va="center", color=_TEXT_COLOUR)
        _apply_style(ax, "Output Entropy (Val)", "Step", "Entropy")
        return

    ax.plot(series["eval_steps"], series["eval_entropy"],
            color=_ENTROPY_COLOUR, linewidth=2.0,
            marker="o", markersize=4, label="Entropy")

    ax.annotate(
        f"{series['eval_entropy'][-1]:.4f}",
        xy=(series["eval_steps"][-1], series["eval_entropy"][-1]),
        xytext=(8, -14), textcoords="offset points",
        fontsize=8, color=_ENTROPY_COLOUR,
    )

    _mark_best(ax, series["best_step"], ymin=0, ymax=max(series["eval_entropy"]))
    _apply_style(ax, "Output Entropy (Val)", "Step", "Entropy")
    ax.set_ylim(bottom=0)


def _build_dashboard(series: dict, cfg: dict):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(14, 10), facecolor="white")   # was (14, 9)
    fig.patch.set_facecolor("white")

    #  Experiment title 
    exp_name  = cfg["experiment"]["name"]
    model_size = cfg.get("model_size", "?")
    peft_mode  = cfg["peft"].get("mode", "full")
    precision  = cfg["training"].get("precision", "bf16")
    best_val   = f"{series['best_metric']:.4f}" if series["best_metric"] else "N/A"
    epochs     = f"{series['total_epochs']:.1f}"

    subtitle_line1 = (
        f"Model: SmolVLM-{model_size}  │  PEFT: {peft_mode}  │  "
        f"Precision: {precision}"
    )
    subtitle_line2 = (
        f"Epochs: {epochs}  │  Best val loss: {best_val}  │  "
        f"Total steps: {series['global_step']}"
    )

    fig.suptitle(
        exp_name,
        fontsize=14, fontweight="bold", color=_TEXT_COLOUR,
        y=0.985,                          # main title 
    )
    fig.text(
        0.5, 0.958,                       
        subtitle_line1,
        ha="center", fontsize=9, color="#6B7280",
    )
    fig.text(
        0.5, 0.938,                       # second subtitle line
        subtitle_line2,
        ha="center", fontsize=9, color="#6B7280",
    )
    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        hspace=0.42, wspace=0.32,
        left=0.07, right=0.97,
        top=0.89,                        
        bottom=0.07,
    )

    ax_loss = fig.add_subplot(gs[0, 0])
    ax_lr   = fig.add_subplot(gs[0, 1])
    ax_acc  = fig.add_subplot(gs[1, 0])
    ax_ent  = fig.add_subplot(gs[1, 1])

    plot_loss_curves(ax_loss, series)
    plot_learning_rate(ax_lr, series)
    plot_token_accuracy(ax_acc, series)
    plot_entropy(ax_ent, series)

    return fig


def plot_training_progress(
    cfg:          dict,
    clearml_task  = None,
    show:         bool = False,
):
    """
    Load training history and save all plots to {output_dir}/plots/.
    """
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend; safe in Colab
    import matplotlib.pyplot as plt

    #  Load data 
    state  = load_trainer_state(cfg)
    series = parse_log_history(state)

    output_dir = Path(cfg["experiment"]["output_dir"])
    plots_dir  = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    saved = {}

    #  Dashboard (all panels) 
    fig = _build_dashboard(series, cfg)
    path = plots_dir / "training_dashboard.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    saved["dashboard"] = str(path)
    logger.info(f"Saved: {path}")

    if show:
        plt.show()
    plt.close(fig)

    individual = {
        "loss":     (plot_loss_curves,    "Loss",              "loss_curves.png"),
        "lr":       (plot_learning_rate,  "Learning Rate",     "learning_rate.png"),
        "accuracy": (plot_token_accuracy, "Token Accuracy",    "token_accuracy.png"),
        "entropy":  (plot_entropy,        "Output Entropy",    "entropy.png"),
    }

    for key, (plot_fn, title, filename) in individual.items():
        fig, ax = plt.subplots(1, 1, figsize=(7, 4.5), facecolor="white")
        plot_fn(ax, series)
        fig.tight_layout()
        path = plots_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        saved[key] = str(path)
        logger.info(f"Saved: {path}")
        if show:
            plt.show()
        plt.close(fig)

    if clearml_task is not None:
        _upload_plots_to_clearml(clearml_task, saved)

    print(f"\nTraining plots saved to: {plots_dir}")
    for name, path in saved.items():
        print(f"  {name:<12} : {Path(path).name}")

    _print_training_table(series)
    return saved


def _upload_plots_to_clearml(task, saved: dict[str, str]):
    """Upload each saved plot as a ClearML debug image."""
    try:
        cm_logger = task.get_logger()
        for name, path in saved.items():
            cm_logger.report_image(
                title="Training Progress",
                series=name,
                local_path=path,
                iteration=0,
            )
        logger.info(f"ClearML: uploaded {len(saved)} training plots.")
    except Exception as e:
        logger.warning(f"ClearML plot upload failed: {e}")


def _print_training_table(series: dict):
    """Print a summary table of the eval metrics history."""
    if not series["eval_steps"]:
        return

    headers = ["Step", "Train Loss", "Val Loss", "Entropy", "Accuracy (%)"]
    widths  = [8, 12, 12, 10, 14]

    header_row = "  ".join(f"{h:<{w}}" for h, w in zip(headers, widths))
    print(f"\n{'─'*len(header_row)}")
    print(header_row)
    print(f"{'─'*len(header_row)}")

    for i, step in enumerate(series["eval_steps"]):
        # Find the matching train loss (closest step at or before this eval step)
        train_loss_at_step = "—"
        for j, ts in enumerate(series["train_steps"]):
            if ts <= step:
                train_loss_at_step = f"{series['train_loss'][j]:.6f}"

        val_loss = f"{series['eval_loss'][i]:.6f}"  if i < len(series["eval_loss"])     else "—"
        entropy  = f"{series['eval_entropy'][i]:.6f}" if i < len(series["eval_entropy"]) else "—"
        accuracy = f"{series['eval_accuracy'][i]:.4f}" if i < len(series["eval_accuracy"]) else "—"

        row = [str(step), train_loss_at_step, val_loss, entropy, accuracy]
        print("  ".join(f"{v:<{w}}" for v, w in zip(row, widths)))

    print(f"{'─'*len(header_row)}")
    if series["best_metric"] is not None:
        print(f"  Best val loss: {series['best_metric']:.6f} "
              f"(step {series['best_step']})")
    print()
