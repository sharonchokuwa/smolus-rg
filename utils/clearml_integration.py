"""
ClearML experiment tracking integration
"""

import logging
import os
from typing import Optional
import time as _time   # alias to avoid shadowing any local 'time' variables

logger = logging.getLogger(__name__)

def clearml_init(cfg: dict):
    if cfg["training"].get("report_to") != "clearml":
        logger.info("ClearML not enabled (report_to != 'clearml'). Skipping init.")
        return None

    try:
        from clearml import Task
    except ImportError:
        logger.warning(
            "clearml package not installed. Run: pip install clearml\n"
            "Continuing without ClearML tracking."
        )
        return None

    clearml_cfg = cfg.get("clearml", {})
    project_name = clearml_cfg.get("project_name", "SmolVLM-Medical-Reports")
    task_name    = clearml_cfg.get("task_name",    cfg["experiment"]["name"])

    os.environ["CLEARML_PROJECT"] = project_name
    os.environ["CLEARML_TASK"]    = task_name

    log_model = clearml_cfg.get("log_model", False)
    os.environ["CLEARML_LOG_MODEL"] = str(log_model).upper()

    logger.info(f"Initialising ClearML Task: {project_name} / {task_name}")

    task = Task.init(
        project_name=project_name,
        task_name=task_name,
        reuse_last_task_id=False,
        auto_connect_frameworks={"transformers": True},
    )

    _log_config_as_hyperparams(task, cfg)
    logger.info(f"ClearML Task ID: {task.task_id}")
    logger.info(f"View at: {task.get_output_log_web_page()}")
    return task

def _log_config_as_hyperparams(task, cfg: dict):
    exp = cfg.get("experiment", {})
    task.connect({
        "name":  exp.get("name", ""),
        "seed":  exp.get("seed", 42),
    }, name="Experiment")

    model = cfg.get("model", {})
    task.connect({
        "architecture":  model.get("architecture", ""),
        "init_strategy": model.get("init_strategy", ""),
        "checkpoint":    model.get("checkpoints", {}).get(
                             model.get("init_strategy", ""), ""),
    }, name="Model")

    active_ds = cfg.get("csv_dataset", {}) if cfg.get("dataset_format") == "raw" else cfg.get("dataset", {})
    task.connect({
        "format":      cfg.get("dataset_format", ""),
        "instruction": active_ds.get("instruction", ""),
        "val_split":   active_ds.get("val_split", 0.1),
        "max_images":  str(active_ds.get("max_images_per_sample", "unlimited")),
    }, name="Dataset")

    lora = cfg.get("peft", {}).get("lora", {})
    task.connect({
        "mode":              cfg.get("peft", {}).get("mode", "lora"),
        "r":                 lora.get("r", 8),
        "lora_alpha":        lora.get("lora_alpha", 8),
        "lora_dropout":      lora.get("lora_dropout", 0.1),
        "target_modules":    str(lora.get("target_modules", [])),
        "use_dora":          lora.get("use_dora", False),
        "init_lora_weights": lora.get("init_lora_weights", "lora"),
    }, name="LoRA")

    tr = cfg.get("training", {})
    task.connect({
        "num_train_epochs":             tr.get("num_train_epochs", 3),
        "per_device_train_batch_size":  tr.get("per_device_train_batch_size", 2),
        "gradient_accumulation_steps":  tr.get("gradient_accumulation_steps", 8),
        "effective_batch_size":         tr.get("per_device_train_batch_size", 2)
                                        * tr.get("gradient_accumulation_steps", 8),
        "learning_rate":                tr.get("learning_rate", 1e-4),
        "warmup_steps":                 tr.get("warmup_steps", 50),
        "lr_scheduler_type":            tr.get("lr_scheduler_type", "cosine"),
        "weight_decay":                 tr.get("weight_decay", 0.01),
        "precision":                    tr.get("precision", "bf16"),
        "optim":                        tr.get("optim", "adamw_torch_fused"),
    }, name="Training")


def log_dataset_stats(task, train_size: int, val_size: int):
    if task is None:
        return
    task.get_logger().report_single_value("dataset/train_samples", train_size)
    task.get_logger().report_single_value("dataset/val_samples",   val_size)
    logger.info(f"ClearML: logged dataset sizes (train={train_size}, val={val_size})")


def close_task(task):
    if task is None:
        return
    task.close()
    logger.info("ClearML task closed.")


class ExperimentTimer:
    """
    Lightweight wall-clock timer for experiment phase tracking.
    """

    def __init__(self):
        self._phases:   dict[str, dict] = {}   # phase_name {start, end, duration_s}
        self._active:   Optional[str]   = None
        self._exp_start = _time.perf_counter()
        self._exp_wall  = _time.time()         # human-readable absolute start

    def start(self, phase: str):
        """Begin timing a named phase."""
        if self._active is not None:
            logger.warning(
                f"Timer: starting '{phase}' but '{self._active}' is still running. "
                f"Stopping '{self._active}' first."
            )
            self.stop(self._active)
        self._phases[phase] = {"start": _time.perf_counter(), "end": None, "duration_s": None}
        self._active = phase
        logger.info(f"[timer] {phase}")

    def stop(self, phase: Optional[str] = None):
        """
        Stop timing a phase. If phase is None, stops the currently active phase.
        Returns the duration in seconds.
        """
        phase = phase or self._active
        if phase is None:
            logger.warning("Timer: stop() called but no phase is active.")
            return 0.0
        if phase not in self._phases:
            logger.warning(f"Timer: phase '{phase}' was never started.")
            return 0.0

        end = _time.perf_counter()
        duration = end - self._phases[phase]["start"]
        self._phases[phase]["end"]        = end
        self._phases[phase]["duration_s"] = round(duration, 3)
        self._active = None
        logger.info(f"[timer] {phase} : {_fmt_duration(duration)}")
        return duration

    def elapsed(self):
        """Total seconds since ExperimentTimer was created."""
        return _time.perf_counter() - self._exp_start

    def summary(self):
        # All values are in seconds.
        result = {}
        for phase, data in self._phases.items():
            result[f"duration_s_{phase}"] = data["duration_s"]

        result["duration_s_total"]      = round(self.elapsed(), 3)
        result["experiment_start_utc"]  = _time.strftime(
            "%Y-%m-%d %H:%M:%S", _time.gmtime(self._exp_wall)
        )
        return result

    def print_summary(self):
        print(f"\n{'='*45}")
        print(f"  Experiment Timing Summary")
        print(f"{'='*45}")
        for phase, data in self._phases.items():
            d = data["duration_s"]
            s = _fmt_duration(d) if d is not None else "still running"
            print(f"  {phase:<28} {s:>12}")
        print(f"  {'─'*40}")
        print(f"  {'TOTAL':<28} {_fmt_duration(self.elapsed()):>12}")
        print(f"{'='*45}\n")


def _fmt_duration(seconds: Optional[float]):
    if seconds is None:
        return "N/A"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s   = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def log_timing_to_clearml(task, timer: ExperimentTimer):
    if task is None:
        return
    summary = timer.summary()
    cm_logger = task.get_logger()
    for key, value in summary.items():
        if isinstance(value, (int, float)):
            cm_logger.report_single_value(f"timing/{key}", value)
    logger.info("ClearML: timing summary logged.")