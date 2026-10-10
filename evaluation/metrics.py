import logging
import torch
import random
import numpy as np
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

logger = logging.getLogger(__name__)

def _safe_import_metrics():
    """Import evaluation libraries with helpful error messages."""
    try:
        import evaluate
        return evaluate
    except ImportError:
        raise ImportError(
            "Run: pip install evaluate rouge_score sacrebleu"
        )

def build_compute_metrics(processor, cfg: dict):
    """
    This function is passed as trainer.compute_metrics.
    """
    evaluate = _safe_import_metrics()
    eval_cfg = cfg.get("evaluation", {})

    bleu_metric  = evaluate.load("sacrebleu") if eval_cfg.get("compute_bleu", True) else None
    rouge_metric = evaluate.load("rouge")     if eval_cfg.get("compute_rouge", True) else None

    tokenizer = processor.tokenizer

    def compute_metrics(eval_pred):
        try:
            predictions, labels = eval_pred

            if predictions is None or labels is None:
                return {}

            if isinstance(predictions, tuple):
                predictions = predictions[0]

            predictions = predictions[:, :-1, :]  
            labels      = labels[:, 1:]           

            predicted_ids = np.argmax(predictions, axis=-1)

            # Mask non-response positions (-100 in labels = prompt or padding)
            response_mask = labels != -100
            predicted_ids = np.where(response_mask, predicted_ids, tokenizer.pad_token_id)

            # Replace -100 with pad token for decoding
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

            decoded_preds = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_preds  = [p.strip() for p in decoded_preds]
            decoded_labels = [l.strip() for l in decoded_labels]

            results = {}

            if rouge_metric:
                rouge = rouge_metric.compute(
                    predictions=decoded_preds,
                    references=decoded_labels,
                    use_stemmer=True,
                )
                results["rouge1"] = rouge["rouge1"]
                results["rouge2"] = rouge["rouge2"]
                results["rougeL"] = rouge["rougeL"]

            if bleu_metric:
                bleu = bleu_metric.compute(
                    predictions=decoded_preds,
                    references=[[l] for l in decoded_labels],
                )
                results["bleu"] = bleu["score"] 

            return results

        except Exception as e:
            logger.warning(f"compute_metrics failed (returning empty): {e}")
            return {}

    return compute_metrics


class GenerationExamplesCallback(TrainerCallback):
    """
    At each evaluation step, generate reports for a small number of
    validation samples and log them for qualitative inspection.
    """

    def __init__(self, model, processor, val_dataset, cfg: dict):
        self.model     = model
        self.processor = processor
        self.cfg       = cfg

        eval_cfg       = cfg.get("evaluation", {})
        n_samples      = eval_cfg.get("generation_samples", 0)
        self.max_new_tokens = eval_cfg.get("max_new_tokens", 512)
        self.enabled = n_samples > 0
        if not self.enabled:
            self.samples = []
        else:
            rng = random.Random(cfg["experiment"]["seed"])
            indices = rng.sample(range(len(val_dataset)), k=min(n_samples, len(val_dataset)))
            self.samples = [val_dataset[i] for i in indices]

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not self.enabled:
            return
        logger.info(f"\n{'='*60}")
        logger.info(f"GENERATION EXAMPLES — Step {state.global_step}")
        logger.info(f"{'='*60}")

        self.model.eval()
        device = next(self.model.parameters()).device

        # Re-enable KV cache for generation (disabled during training)
        _was_use_cache = self.model.config.use_cache     
        self.model.config.use_cache = True               

        for i, sample in enumerate(self.samples):
            try:
                # Apply chat template to the user turn only
                messages = sample["messages"]
                user_turn = [messages[0]]  # Just the human turn
                content = messages[1]["content"]
                reference = content[0]["text"] if isinstance(content, list) else content

                text = self.processor.apply_chat_template(user_turn, add_generation_prompt=True, tokenize=False)
                inputs = self.processor(text=[text], images=[sample["images"]], return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                print("In metrics.py: on_evaluate()")
                print("pixel_values:", tuple(inputs["pixel_values"].shape), "| n_images:", len(sample["images"]))

                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                    )

                input_len = inputs["input_ids"].shape[1]
                generated_text = self.processor.tokenizer.decode(
                    generated_ids[0][input_len:],
                    skip_special_tokens=True,
                )

                logger.info(f"\n--- Sample {i+1} ---")
                logger.info(f"REFERENCE:\n{reference[:400]}")
                logger.info(f"GENERATED:\n{generated_text[:400]}")

            except Exception as e:
                logger.error(f"Generation failed for sample {i}: {e}")

        self.model.config.use_cache = _was_use_cache 
        self.model.train()
        logger.info(f"{'='*60}\n")
