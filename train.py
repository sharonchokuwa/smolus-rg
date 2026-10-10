import argparse
import logging
import os
import sys
import random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.loader import load_config, get_active_checkpoint_id, make_output_dirs
from data.dataset import load_split_datasets
from models.loader import load_model_and_processor
from training.trainer import build_trainer, run_training, save_final_model
from evaluation.metrics import build_compute_metrics, GenerationExamplesCallback
from utils.checkpointing import should_resume, build_callbacks
from utils.clearml_integration import clearml_init, log_dataset_stats, close_task

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False

def parse_overrides(overrides: list[str]):
    parsed: dict = {}
    for override in overrides:
        key_path, _, value = override.partition("=")
        key_path = key_path.strip()
        if value.lower() in ("true", "false"):
            value = value.lower() == "true"
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass  
        parsed[key_path] = value
        logger.info(f"Override parsed: {key_path} = {value!r}")
    return parsed                               


def main():
    parser = argparse.ArgumentParser(description="SmolVLM Medical Report Finetuning")
    parser.add_argument("--config",     required=True, help="Path to experiment YAML config")
    parser.add_argument("--drive_root", default=None,  help="Override Google Drive root path")
    parser.add_argument("--override",   action="append", default=[],
                        help="Override config values")
    args = parser.parse_args()

    overrides = parse_overrides(args.override) if args.override else None
    cfg = load_config(args.config, drive_root=args.drive_root, overrides=overrides)

    make_output_dirs(cfg)
    set_seed(cfg["experiment"]["seed"])

    logger.info(f"\n{'='*55}")
    logger.info(f"Experiment : {cfg['experiment']['name']}")
    logger.info(f"Model      : {cfg['model']['architecture']} ({cfg['model']['init_strategy']})")
    active_ds_cfg = cfg['dataset'] if cfg['dataset_format'] == 'structured' else cfg['csv_dataset']
    logger.info(f"Dataset    : {cfg['dataset_format']}")    
    logger.info(f"PEFT mode  : {cfg['peft']['mode']}")
    logger.info(f"{'='*55}\n")

    clearml_task = clearml_init(cfg)

    logger.info("Loading datasets...")
    train_ds, val_ds = load_split_datasets(cfg)
    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    log_dataset_stats(clearml_task, len(train_ds), len(val_ds))

    logger.info("Loading model and processor...")
    model, processor, lora_config = load_model_and_processor(cfg)

    callbacks = build_callbacks(cfg)
    callbacks.append(
        GenerationExamplesCallback(
            model=model,
            processor=processor,
            val_dataset=val_ds,
            cfg=cfg,
        )
    )

    compute_metrics = build_compute_metrics(processor, cfg)

    trainer = build_trainer(
        model=model,
        processor=processor,
        train_dataset=train_ds,
        val_dataset=val_ds,
        lora_config=lora_config,
        cfg=cfg,
        callbacks=callbacks,
        compute_metrics=compute_metrics,
    )

    resume_from = should_resume(cfg)
    result = run_training(trainer, resume_from=resume_from)
    save_final_model(trainer, cfg)

    logger.info("Running final evaluation...")
    eval_results = trainer.evaluate()
    logger.info(f"Final eval results: {eval_results}")

    close_task(clearml_task)

    logger.info(f"\nExperiment '{cfg['experiment']['name']}' complete.")
    return result


if __name__ == "__main__":
    main()
