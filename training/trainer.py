"""
Assembles and runs the SFTTrainer.
"""

import logging
import shutil
import torch
from pathlib import Path

from transformers import TrainingArguments
from utils.checkpointing import load_best_model_path  
from trl import SFTTrainer, SFTConfig
logger = logging.getLogger(__name__)


def build_sft_config(cfg: dict):
    train_cfg = cfg["training"]
    exp_cfg   = cfg["experiment"]
    precision = train_cfg["precision"]

    return SFTConfig(
        output_dir=exp_cfg["output_dir"],
        run_name=exp_cfg["name"],
        report_to=train_cfg.get("report_to", "none"),
        logging_steps=train_cfg["logging_steps"],
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 0),
        dataloader_pin_memory=True,          
        dataloader_persistent_workers=train_cfg.get("dataloader_num_workers", 0) > 0,
        tf32=True,                           
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg["warmup_ratio"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        max_grad_norm=train_cfg["max_grad_norm"],
        optim=train_cfg["optim"],
        bf16=(precision == "bf16"),
        fp16=(precision == "fp16"),
        eval_strategy=train_cfg["eval_strategy"],
        eval_steps=train_cfg["eval_steps"],
        save_strategy=train_cfg["save_strategy"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        load_best_model_at_end=train_cfg["load_best_model_at_end"],
        metric_for_best_model=train_cfg["metric_for_best_model"],
        greater_is_better=train_cfg["greater_is_better"],
        max_length=train_cfg.get("max_seq_length", None),
        label_smoothing_factor=train_cfg.get("label_smoothing_factor", 0.1),   
        seed=cfg["experiment"]["seed"],
        remove_unused_columns=False,
    )

def make_collate_fn(processor, assistant_only_loss: bool = False):
    def _image_marker_ids(tokenizer):
        ids = set()
        img_tok = getattr(processor, "image_token", "<image>")
        tid = tokenizer.convert_tokens_to_ids(img_tok)
        if tid is not None and tid >= 0:
            ids.add(tid)
        for t in tokenizer.additional_special_tokens:
            if ("image" in t) or ("row_" in t) or t.strip("<>").startswith("global-img"):
                tid = tokenizer.convert_tokens_to_ids(t)
                if tid is not None and tid >= 0:
                    ids.add(tid)
        return torch.tensor(sorted(ids))
    
    image_marker_ids = _image_marker_ids(processor.tokenizer)

    image_token_id = processor.tokenizer.additional_special_tokens_ids[
        processor.tokenizer.additional_special_tokens.index("<image>")
    ]

    def collate_fn(examples):
        images = [ex["images"] for ex in examples]

        # Full conversation text: user turn + assistant report
        full_texts = [
            processor.apply_chat_template(
                ex["messages"],
                add_generation_prompt=False,
                tokenize=False,
            )
            for ex in examples
        ]

        full_batch = processor(
            text=full_texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )

        labels = full_batch["input_ids"].clone()

        assert processor.tokenizer.pad_token_id != processor.tokenizer.eos_token_id, (
            "pad_token_id == eos_token_id — collator would mask EOS and break stopping. "
            "Check models/loader.py pad-token setup."
        )
        labels[labels == processor.tokenizer.pad_token_id] = -100
        labels[labels == image_token_id] = -100

        if assistant_only_loss:
            prompt_texts = [
                processor.apply_chat_template(
                    [ex["messages"][0]],     # user turn only
                    add_generation_prompt=True,
                    tokenize=False,
                )
                for ex in examples
            ]
            prompt_batch = processor(
                text=prompt_texts,
                images=images,
                return_tensors="pt",
                padding=True,
            )
            assert prompt_batch["input_ids"].shape[1] <= full_batch["input_ids"].shape[1], \
             "prompt_batch longer than full_batch — label masking boundary is invalid"
            for i in range(len(examples)):
                prompt_len = int(
                    (prompt_batch["input_ids"][i] != processor.tokenizer.pad_token_id).sum()
                )
                labels[i, :prompt_len] = -100

        full_batch["labels"] = labels
        return full_batch

    return collate_fn


def build_trainer(
    model,
    processor,
    train_dataset,
    val_dataset,
    lora_config,
    cfg: dict,
    callbacks: list,
    compute_metrics=None,
):
    sft_config = build_sft_config(cfg)
    assistant_only = cfg["training"].get("assistant_only_loss", False)
    logger.info(f"Loss mode: {'assistant-only (report tokens)' if assistant_only else 'full-sequence'}")

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor,
        peft_config=None,         # model is already wrapped by apply_peft in loader.py
        callbacks=callbacks,
        compute_metrics=None,
        data_collator=make_collate_fn(processor, assistant_only_loss=assistant_only),
    )

    logger.info(f"SFTTrainer ready | train={len(train_dataset)} | val={len(val_dataset)}")
    return trainer


def run_training(trainer: SFTTrainer, resume_from: str | None = None):
    """
    Execute training with optional checkpoint resume.
    """
    if resume_from:
        logger.info(f"Resuming training from: {resume_from}")
    else:
        logger.info("Starting training from scratch.")

    result = trainer.train(resume_from_checkpoint=resume_from)

    logger.info(f"Training complete.")
    logger.info(f"  Global steps : {result.global_step}")
    logger.info(f"  Training loss: {result.training_loss:.4f}")
    return result

def save_final_model(trainer: SFTTrainer, cfg: dict):
    """Save best_model/ weights as the final model."""
    output_dir = cfg["experiment"]["output_dir"]
    final_dir  = str(Path(output_dir) / "final_model")

    best_path = load_best_model_path(output_dir)
    if best_path:
        if Path(final_dir).exists():
            shutil.rmtree(final_dir)
        shutil.copytree(best_path, final_dir)
        logger.info(f"final_model/ :best_model/ ({best_path})")
    else:
        trainer.save_model(final_dir)
        logger.warning(
            "best_model/ not found — saved trainer.model as final_model/. "
            "This may be the last step, not the best eval checkpoint."
        )
    logger.info(f"Final model saved to: {final_dir}")