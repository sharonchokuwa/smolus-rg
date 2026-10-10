"""
Handles checkpoint management, resume-from-checkpoint, and
Google Drive backup for Colab training sessions.
"""

import json
import shutil
import logging
from pathlib import Path
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

logger = logging.getLogger(__name__)

BEST_MODEL_DIR  = "best_model"
LAST_CKPT_DIR   = "last_checkpoint"

def find_resume_checkpoint(output_dir: str):
    """
    Return the checkpoint path to pass to trainer.train(resume_from_checkpoint=).
    Priority order:
      1. last_checkpoint/  — saved by BestAndLastCheckpointCallback; has full
                             optimizer + scheduler state; preferred for resuming.
      2. checkpoint-N/     — fallback if the callback was not used or the session
                             crashed before last_checkpoint/ was written.
      3. None              — start from scratch.
    """
    out = Path(output_dir)

    last = out / LAST_CKPT_DIR
    if last.exists() and (last / "trainer_state.json").exists():
        logger.info(f"Resuming from last_checkpoint: {last}")
        return str(last)

    # Fallback: numbered checkpoint dirs written by HF Trainer
    numbered = sorted(
        [d for d in out.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[-1]),
    ) if out.exists() else []

    if numbered:
        latest = str(numbered[-1])
        logger.info(f"Resuming from latest numbered checkpoint: {latest}")
        return latest

    logger.info("No checkpoint found — starting from scratch.")
    return None

# Keep the old name as an alias so existing callers don't break
find_latest_checkpoint = find_resume_checkpoint

def should_resume(cfg: dict):
    if not cfg["checkpointing"].get("auto_resume", True):
        return None
    return find_resume_checkpoint(cfg["experiment"]["output_dir"])


def load_best_model_path(output_dir: str):
    p = Path(output_dir) / BEST_MODEL_DIR
    return str(p) if p.exists() else None

class BestAndLastCheckpointCallback(TrainerCallback):
    """
    Maintains exactly two checkpoints throughout training:
        best_model/       — copy of the checkpoint with the lowest eval_loss seen
                            so far 
        last_checkpoint/  — copy of the most recently saved checkpoint-N
    """

    def __init__(
        self,
        output_dir: str,
        metric_name: str = "eval_loss",
        higher_is_better: bool = False,
        drive_backup_dir: str | None = None,
    ):
        self.output_dir       = Path(output_dir)
        self.metric_name      = metric_name
        self.higher_is_better = higher_is_better
        self.drive_dir        = Path(drive_backup_dir) if drive_backup_dir else None
        self.best_metric      = float("inf") if not higher_is_better else float("-inf")
        self._best_step       = None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.drive_dir:
            self.drive_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"BestAndLastCheckpointCallback initialised | "
            f"metric={metric_name} | higher_is_better={higher_is_better} | "
            f"drive_backup={'yes' if drive_backup_dir else 'no'}"
        )

    def _is_better(self, value: float):
        if self.higher_is_better:
            return value > self.best_metric
        return value < self.best_metric

    def _atomic_copy(self, src: Path, dst: Path):
        tmp = dst.parent / (dst.name + "_tmp")
        try:
            if tmp.exists():
                shutil.rmtree(tmp)
            shutil.copytree(src, tmp)
            if dst.exists():
                shutil.rmtree(dst)
            tmp.rename(dst)
            return True
        except Exception as e:
            logger.error(f"Atomic copy {src} : {dst} failed: {e}")
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)
            return False

    def _mirror_to_drive(self, name: str):
        if not self.drive_dir:
            return
        src = self.output_dir / name
        dst = self.drive_dir  / name
        if src.exists():
            ok = self._atomic_copy(src, dst)
            if ok:
                logger.info(f"Drive backup updated: {dst}")

    def _checkpoint_dir(self, step: int):
        return self.output_dir / f"checkpoint-{step}"

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        src = self._checkpoint_dir(state.global_step)
        if not src.exists():
            logger.warning(f"on_save: expected {src} not found — skipping last_checkpoint update.")
            return

        dst = self.output_dir / LAST_CKPT_DIR
        ok  = self._atomic_copy(src, dst)
        if ok:
            logger.info(f"last_checkpoint updated: checkpoint-{state.global_step}")
            self._mirror_to_drive(LAST_CKPT_DIR)

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict | None = None,
        **kwargs,
    ):
        if not metrics:
            return

        value = metrics.get(self.metric_name)
        if value is None:
            logger.debug(f"on_evaluate: metric '{self.metric_name}' not in metrics dict — skipping.")
            return

        if not self._is_better(value):
            logger.info(
                f"Eval {self.metric_name}={value:.6f} (best so far: {self.best_metric:.6f}) "
                f"— no improvement, best_model unchanged."
            )
            return

        src = self._checkpoint_dir(state.global_step)
        if not src.exists():
            # Find the newest checkpoint-N that does exist
            candidates = sorted(
                [d for d in self.output_dir.iterdir()
                 if d.is_dir() and d.name.startswith("checkpoint-")],
                key=lambda d: int(d.name.split("-")[-1]),
            )
            if not candidates:
                logger.warning("on_evaluate: no checkpoint-N found to copy to best_model/.")
                return
            src = candidates[-1]
            logger.info(f"on_evaluate: using {src.name} (checkpoint-{state.global_step} not present).")

        dst = self.output_dir / BEST_MODEL_DIR
        ok  = self._atomic_copy(src, dst)
        if ok:
            try:
                _m = kwargs.get("model")
                from peft import PeftModel
                if isinstance(_m, PeftModel):
                    from full_component_io import save_non_lora_trainables
                    save_non_lora_trainables(_m, str(dst))
            except Exception as _e:
                logger.warning(f"Could not save non_lora_trainables to best_model/: {_e}")
            old_best        = self.best_metric
            self.best_metric = value
            self._best_step  = state.global_step
            logger.info(
                f"★ New best {self.metric_name}={value:.6f} "
                f"(was {old_best:.6f}) at step {state.global_step} "
                f": best_model/ updated."
            )
            self._mirror_to_drive(BEST_MODEL_DIR)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Log a summary when training finishes."""
        best_dir = self.output_dir / BEST_MODEL_DIR
        last_dir = self.output_dir / LAST_CKPT_DIR
        logger.info("=" * 60)
        logger.info("BestAndLastCheckpointCallback — training complete")
        logger.info(f"  best_model/      : {best_dir} (step {self._best_step}, {self.metric_name}={self.best_metric:.6f})")
        logger.info(f"  last_checkpoint/ : {last_dir}")
        if self.drive_dir:
            logger.info(f"  Drive backup     : {self.drive_dir}")
        logger.info("=" * 60)

    def summary(self):
        return {
            "best_metric_name":  self.metric_name,
            "best_metric_value": self.best_metric,
            "best_step":         self._best_step,
            "best_model_dir":    str(self.output_dir / BEST_MODEL_DIR),
            "last_ckpt_dir":     str(self.output_dir / LAST_CKPT_DIR),
        }


class DriveBackupCallback(TrainerCallback):
    def __init__(self, source_dir: str, backup_dir: str, enabled: bool = True):
        self.source_dir = Path(source_dir)
        self.backup_dir = Path(backup_dir)
        self.enabled    = enabled
        if enabled:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"DriveBackupCallback: checkpoints : {backup_dir}")

    def on_save(self, args, state, control, **kwargs):
        if not self.enabled:
            return
        checkpoint_name = f"checkpoint-{state.global_step}"
        src = self.source_dir / checkpoint_name
        dst = self.backup_dir / checkpoint_name
        if src.exists():
            try:
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
                logger.info(f"Checkpoint backed up to Drive: {dst}")
            except Exception as e:
                logger.error(f"Drive backup failed for {checkpoint_name}: {e}")
        else:
            logger.warning(f"Expected checkpoint not found: {src}")



def build_callbacks(cfg: dict):
    """
    Build the trainer callback list from config.
    Returns a list containing BestAndLastCheckpointCallback and
    DriveBackupCallback (only when checkpointing.drive_backup is True and
    BestAndLastCheckpointCallback is not already handling Drive mirroring).
    """
    train_cfg   = cfg["training"]
    ckpt_cfg    = cfg["checkpointing"]
    output_dir  = cfg["experiment"]["output_dir"]
    drive_backup = ckpt_cfg.get("drive_backup", False)
    drive_dir    = ckpt_cfg.get("drive_backup_dir") if drive_backup else None

    callbacks = [
        BestAndLastCheckpointCallback(
            output_dir        = output_dir,
            metric_name       = train_cfg.get("metric_for_best_model", "eval_loss"),
            higher_is_better  = train_cfg.get("greater_is_better", False),
            drive_backup_dir  = drive_dir,   # None if drive_backup=False
        )
    ]
    return callbacks
