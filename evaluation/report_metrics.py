import logging
import re
import warnings
from typing import Optional
from xml.parsers.expat import model

import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports — each metric group loads only what it needs
def _load_evaluate():
    try:
        import evaluate
        return evaluate
    except ImportError:
        raise ImportError("pip install evaluate rouge_score sacrebleu")


def _load_nltk_meteor():
    try:
        import nltk

        _NLTK_RESOURCES = {
            "wordnet":   "corpora/wordnet",
            "punkt":     "tokenizers/punkt",
            "punkt_tab": "tokenizers/punkt_tab",
            "omw-1.4":   "corpora/omw-1.4",
        }
        for resource, data_path in _NLTK_RESOURCES.items():
            try:
                nltk.data.find(data_path)
            except LookupError:
                try:
                    nltk.download(resource, quiet=True)
                except Exception as e:
                    logger.debug(f"NLTK resource '{resource}' not downloadable: {e}")

        from nltk.translate.meteor_score import meteor_score
        from nltk.tokenize import word_tokenize
        return meteor_score, word_tokenize
    except ImportError:
        raise ImportError("pip install nltk")

def _load_cider():
    try:
        from pycocoevalcap.cider.cider import Cider
        return Cider
    except ImportError:
        logger.warning(
            "pycocoevalcap not installed — CIDEr will be skipped.\n"
            "Install with: pip install pycocoevalcap"
        )
        return None

def _load_bertscore():
    try:
        import bert_score
        return bert_score
    except ImportError:
        raise ImportError("pip install bert_score")

def compute_bleu(predictions: list[str], references: list[str]):
    from pycocoevalcap.bleu.bleu import Bleu

    refs_dict = {str(i): [r] for i, r in enumerate(references)}
    hyps_dict = {str(i): [p] for i, p in enumerate(predictions)}
    scorer = Bleu(4)
    scores, _ = scorer.compute_score(refs_dict, hyps_dict)

    return {
        "BLEU-1": round(scores[0] * 100, 4),
        "BLEU-2": round(scores[1] * 100, 4),
        "BLEU-3": round(scores[2] * 100, 4),
        "BLEU-4": round(scores[3] * 100, 4),
    }

def compute_rouge(predictions: list[str], references: list[str]):
    from pycocoevalcap.rouge.rouge import Rouge

    refs_dict = {str(i): [r] for i, r in enumerate(references)}
    hyps_dict = {str(i): [p] for i, p in enumerate(predictions)}
    scorer = Rouge()
    score, _ = scorer.compute_score(refs_dict, hyps_dict)
    return {"ROUGE-L": round(score * 100, 4)}

def compute_meteor(predictions: list[str], references: list[str]):
    from pycocoevalcap.meteor.meteor import Meteor

    refs_dict = {str(i): [r] for i, r in enumerate(references)}
    hyps_dict = {str(i): [p] for i, p in enumerate(predictions)}

    scorer = Meteor()
    score, _ = scorer.compute_score(refs_dict, hyps_dict)
    return {"METEOR": round(score * 100, 4)}


def compute_cider(predictions: list[str], references: list[str]):
    Cider = _load_cider()
    if Cider is None:
        return {"CIDEr": None}

    refs_dict  = {str(i): [r] for i, r in enumerate(references)}
    hyps_dict  = {str(i): [p] for i, p in enumerate(predictions)}

    cider = Cider()
    score, _ = cider.compute_score(refs_dict, hyps_dict)

    return {"CIDEr": round(score, 4)}


def _load_spice():
    """
    Lazy-load SPICE from pycocoevalcap, which wraps the original
    peteanderson80/SPICE Java implementation (spice-1.0.jar). SPICE requires:
      - Java 8+ on PATH, and
      - the Stanford CoreNLP models the jar downloads on first use.
    Returns the Spice class, or None (with a clear message) if unavailable.
    """
    try:
        from pycocoevalcap.spice.spice import Spice
        return Spice
    except Exception as e:
        logger.warning(
            "SPICE unavailable — pycocoevalcap.spice import failed. SPICE needs "
            "Java 8+ and the SPICE jar/CoreNLP models. SPICE will be skipped.\n"
            f"Reason: {e}"
        )
        return None
    
def compute_spice(predictions: list[str], references: list[str]):
    """
    SPICE — Semantic Propositional Image Caption Evaluation (Anderson et al. 2016).
    """
    Spice = _load_spice()
    if Spice is None:
        return {"SPICE": None}
 
    refs_dict = {str(i): [r] for i, r in enumerate(references)}
    hyps_dict = {str(i): [p] for i, p in enumerate(predictions)}
 
    try:
        scorer = Spice()
        score, per_caption = scorer.compute_score(refs_dict, hyps_dict)
    except Exception as e:
        logger.warning(
            f"SPICE computation failed (is Java installed and CoreNLP available?): {e}"
        )
        return {"SPICE": None}
 
    out = {"SPICE": round(float(score), 4)}
 
    # semantic sub-scores (the paper's Object / Relation / Attribute F).
    try:
        import numpy as np
        for sub in ("Object", "Relation", "Attribute"):
            vals = [
                s[sub]["f"] for s in per_caption
                if isinstance(s.get(sub), dict)
                and s[sub].get("f") is not None
                and s[sub]["f"] == s[sub]["f"]          # drop NaN
            ]
            if vals:
                out[f"SPICE-{sub}"] = round(float(np.mean(vals)), 4)
    except Exception:
        pass
 
    return out

# Semantic similarity
def compute_bertscore(
    predictions: list[str],
    references: list[str],
    model_type: str = "bert-base-multilingual-cased",
    batch_size: int = 8,
    device: Optional[str] = None,
):
    bert_score = _load_bertscore()

    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        P, R, F1 = bert_score.score(
            cands=predictions,
            refs=references,
            model_type=model_type,
            lang="en",
            rescale_with_baseline=False,   
            device=device,
            batch_size=batch_size,
            verbose=False,
        )

    return {
        "BERTScore-P":  round(P.mean().item()  * 100, 4),
        "BERTScore-R":  round(R.mean().item()  * 100, 4),
        "BERTScore-F1": round(F1.mean().item() * 100, 4),
    }


# Clinical efficacy metrics
# Entity vocabulary 
# These terms are drawn from Li et.al's (2024) KMVE paper's Table V (clinical entity extraction for US reports).
_LIVER_FINDINGS = [
    "liver", "capsule", "echogenicity", "vein", "kidney", "intrahepatic duct", "bile duct", "gallbladder", "margin", "pancreas",
    "pancreatic duct", "lesion", "spleen", "CDFI", "nodule",
]

_THYROID_FINDINGS = [
    "thyroid gland", "glandular tissue", "echogenicity", "lesion", "CDFI", "lymph node", "border", "shape", "nodule", "left lobe",
    "right lobe", "margin"
]

_BREAST_FINDINGS = [
    "breast", "gland", "Colour Doppler flow", "CDFI", "axilla", "echogenicity", "nodule", "lymph node", "mammary duct",
    "lesion", "subcutaneous fat layer", "tumour"
]

_ANATOMY_ENTITIES = {
    "liver":   _LIVER_FINDINGS,
    "thyroid": _THYROID_FINDINGS,
    "breast":  _BREAST_FINDINGS,
    "default": _LIVER_FINDINGS,   # fallback
}

def _extract_entities(text: str, patterns: list[str]):
    """
    Extract clinical entity mentions from text using regex or literal patterns.
    Patterns that start with a backslash (e.g. r"\\d+\\s*(?:mm|cm|DS)") are
    treated as regular expressions and used directly with re.search().
    All other patterns are treated as literal strings and escaped with
    re.escape() before matching.
    Returns a set of matched pattern strings (not raw spans) so that
    precision/recall are computed over entity types, not character positions.
    """

    found = set()
    text_lower = text.lower()
    for pattern in patterns:
        is_regex  = len(pattern) > 0 and ord(pattern[0]) == 92   # chr(92) == backslash
        safe_pattern = pattern if is_regex else re.escape(pattern)
        if re.search(safe_pattern, text_lower):
            found.add(pattern)
    return found

def _entity_vector(text: str, patterns: list[str]):
    text_lower = text.lower()                                                      
    vec = []                                                                         
    for pattern in patterns:                                                        
        is_regex = len(pattern) > 0 and ord(pattern[0]) == 92   
        safe_pattern = pattern if is_regex else re.escape(pattern.lower())          
        vec.append(1 if re.search(safe_pattern, text_lower) else 0)                 
    return vec                                                                   
 
def compute_clinical_efficacy(
    predictions: list[str],
    references:  list[str],
    anatomy: str = "liver",
):
    """
    Clinical-efficacy metrics computed with scikit-learn as per Li et.al's (2024) KMVE paper.
    """
    from sklearn.metrics import (                                                 
        precision_score, recall_score, f1_score, accuracy_score,                    
    )                                                                              
    import numpy as np                                                             
    patterns = _ANATOMY_ENTITIES.get(anatomy.lower(), _ANATOMY_ENTITIES["default"])
 
    Y_pred = np.array([_entity_vector(p, patterns) for p in predictions])           
    Y_true = np.array([_entity_vector(r, patterns) for r in references])            
 
    return {
        "CE-Accuracy":  round(accuracy_score(Y_true, Y_pred) * 100, 4),  
        "CE-Precision": round(precision_score(Y_true, Y_pred, average="micro", zero_division=0) * 100, 4),  
        "CE-Recall":    round(recall_score(Y_true, Y_pred,    average="micro", zero_division=0) * 100, 4),  
        "CE-F1":        round(f1_score(Y_true, Y_pred,        average="micro", zero_division=0) * 100, 4),  
    }

def evaluate_reports(
    predictions:     list[str],
    references:      list[str],
    cfg:             Optional[dict] = None,
    anatomy:         str = "liver",
    bertscore_model: str = "bert-base-multilingual-cased",
    compute_groups:  Optional[list[str]] = None,
):
    """
    Compute all report generation metrics and return a results dict.
    """
    if cfg is not None:
        anatomy = cfg.get("anatomy", anatomy)

    if not predictions or not references:
        raise ValueError("predictions and references must both be non-empty lists.")
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions ({len(predictions)}) and references ({len(references)}) "
            "must have the same length."
        )

    predictions = [str(p).strip() for p in predictions]
    references  = [str(r).strip() for r in references]

    all_groups = {"bleu", "rouge", "meteor", "cider", "bertscore", "clinical", "spice"}
    groups = set(compute_groups) if compute_groups else all_groups

    results = {}

    #  BLEU 
    if "bleu" in groups:
        logger.info("Computing BLEU-1/2/3/4...")
        try:
            results.update(compute_bleu(predictions, references))
        except Exception as e:
            logger.warning(f"BLEU failed: {e}")
            results.update({"BLEU-1": None, "BLEU-2": None, "BLEU-3": None, "BLEU-4": None})

    #  ROUGE 
    if "rouge" in groups:
        logger.info("Computing ROUGE-L...")
        try:
            results.update(compute_rouge(predictions, references))
        except Exception as e:
            logger.warning(f"ROUGE failed: {e}")
            results["ROUGE-L"] = None

    #  METEOR 
    if "meteor" in groups:
        logger.info("Computing METEOR...")
        try:
            results.update(compute_meteor(predictions, references))
        except Exception as e:
            logger.warning(f"METEOR failed: {e}")
            results["METEOR"] = None

    #  CIDEr 
    if "cider" in groups:
        logger.info("Computing CIDEr...")
        try:
            results.update(compute_cider(predictions, references))
        except Exception as e:
            logger.warning(f"CIDEr failed: {e}")
            results["CIDEr"] = None

    #  BERTScore 
    if "bertscore" in groups:
        logger.info(f"Computing BERTScore (model={bertscore_model})...")
        try:
            results.update(compute_bertscore(
                predictions, references, model_type=bertscore_model
            ))
        except Exception as e:
            logger.warning(f"BERTScore failed: {e}")
            results.update({"BERTScore-P": None, "BERTScore-R": None, "BERTScore-F1": None})

    #  Clinical Efficacy 
    if "clinical" in groups:
        logger.info(f"Computing Clinical Efficacy (anatomy={anatomy})...")
        try:
            results.update(compute_clinical_efficacy(predictions, references, anatomy))
        except Exception as e:
            logger.warning(f"Clinical efficacy failed: {e}")
            results.update({"CE-Precision": None, "CE-Recall": None, "CE-F1": None, "CE-Accuracy": None})

    #  SPICE — scene-graph semantic F-score (Anderson et al. 2016) 
    if "spice" in groups:
        logger.info("Computing SPICE (Java/CoreNLP)...")
        try:
            results.update(compute_spice(predictions, references))
        except Exception as e:
            logger.warning(f"SPICE failed: {e}")
            results["SPICE"] = None

    return results

# Column order
_DISPLAY_ORDER = [
    "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4",
    "ROUGE-L",
    "METEOR",
    "CIDEr",
    "BERTScore-P", "BERTScore-R", "BERTScore-F1",
    "SPICE", "SPICE-Object", "SPICE-Relation", "SPICE-Attribute",
    "CE-Accuracy", "CE-Precision", "CE-Recall", "CE-F1",
]

def print_metrics_table(results: dict, split_name: str = "Eval"):
    ordered = [k for k in _DISPLAY_ORDER if k in results]
    extras  = [k for k in results if k not in _DISPLAY_ORDER]
    keys    = ordered + extras

    col_w = max(len(k) for k in keys) + 2
    val_w = 10

    print(f"\n┌{'─'*(col_w)}┬{'─'*val_w}┐")
    print(f"│ {'Metric':<{col_w-2}} │ {split_name:>{val_w-2}} │")
    print(f"├{'─'*(col_w)}┼{'─'*val_w}┤")

    for k in keys:
        v = results[k]
        if v is None:
            val_str = "  N/A"
        elif k == "CIDEr":
            val_str = f"{v:>8.4f} "
        else:
            val_str = f"{v:>7.2f}% "
        print(f"│ {k:<{col_w-2}} │ {val_str:>{val_w-1}} │")

    print(f"└{'─'*(col_w)}┴{'─'*val_w}┘\n")


def metrics_to_clearml(task, results: dict, split_name: str = "val"):
    """
    Log all metrics to ClearML as scalars
    """
    if task is None:
        return
    logger_cm = task.get_logger()
    for metric_name, value in results.items():
        if value is not None:
            logger_cm.report_scalar(
                title="Report Metrics",
                series=f"{split_name}/{metric_name}",
                value=value,
                iteration=0,
            )


def save_metrics(
    results:    dict,
    cfg:        dict,
    split_name: str = "val",
):
    """
    Write metrics to both CSV and JSON inside the experiment output directory.
    """
    import csv
    import json
    from datetime import datetime
    from pathlib import Path

    output_dir = Path(cfg["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    csv_path   = output_dir / f"metrics_{split_name}.csv"
    json_path  = output_dir / f"metrics_{split_name}.json"

    json_payload = {
        "experiment":   cfg["experiment"]["name"],
        "split":        split_name,
        "anatomy":      cfg.get("anatomy", "unknown"),
        "model_size":   cfg.get("model_size", "unknown"),
        "seed":         cfg["experiment"].get("seed", 42),
        "timestamp":    timestamp,
        "metrics":      {
            k: (round(v, 6) if isinstance(v, float) else v)
            for k, v in results.items()
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2, ensure_ascii=False)

    ordered_keys = [k for k in _DISPLAY_ORDER if k in results]
    extra_keys   = [k for k in results          if k not in _DISPLAY_ORDER]
    all_keys     = ordered_keys + extra_keys

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Header row
        writer.writerow(["experiment", "split", "anatomy", "timestamp"] + all_keys)

        # Single data row — one experiment result per file
        row = [
            cfg["experiment"]["name"],
            split_name,
            cfg.get("anatomy", "unknown"),
            timestamp,
        ]
        for k in all_keys:
            v = results[k]
            row.append("" if v is None else round(v, 6) if isinstance(v, float) else v)

        writer.writerow(row)

    logger.info(f"Metrics saved : {csv_path}")
    logger.info(f"Metrics saved : {json_path}")
    return str(csv_path), str(json_path)

# Dataset-level runner (for notebook integration)
def run_evaluation_on_dataset(
    model,
    processor,
    dataset:         list[dict],
    cfg:             dict,
    split_name:      str = "val",
    max_new_tokens:  int = 512,
    batch_size:      int = 1,
    compute_groups:  Optional[list[str]] = None,
    clearml_task     = None,
):
    """
    Generate reports for every sample in `dataset`, then compute all metrics,
    """
    import torch

    device = next(model.parameters()).device
    model.eval()

    _was_use_cache = model.config.use_cache             
    model.config.use_cache = True 

    predictions = []
    references  = []

    logger.info(f"Generating reports for {len(dataset)} {split_name} samples...")

    _orig_padding_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "left"

    eos_id = processor.tokenizer.eos_token_id
    pad_id = processor.tokenizer.pad_token_id

    try:
        for i in range(0, len(dataset), batch_size):
            batch_texts, batch_images, batch_refs = [], [], []
            for j in range(i, min(i + batch_size, len(dataset))):
                sample    = dataset[j]
                content   = sample["messages"][1]["content"]
                reference = content[0]["text"] if isinstance(content, list) else content
                batch_refs.append(reference)

                user_turn = [sample["messages"][0]]
                batch_texts.append(processor.apply_chat_template(
                    user_turn, add_generation_prompt=True, tokenize=False))
                batch_images.append(sample["images"])

            n_rows = len(batch_texts)                 
            references.extend(batch_refs)

            try:
                inputs = processor(text=batch_texts, images=batch_images,
                                return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                        eos_token_id=eos_id, pad_token_id=pad_id)
                input_len = inputs["input_ids"].shape[1]
                for k in range(n_rows):
                    predictions.append(processor.tokenizer.decode(
                        generated_ids[k][input_len:], skip_special_tokens=True).strip())
            except Exception as e:
                logger.warning(f"Batch generation failed at i={i}: {e}")
                predictions.extend([""] * n_rows)

            assert len(predictions) == len(references), (
                f"pred/ref drift at i={i}: {len(predictions)} vs {len(references)}")
    finally:
        processor.tokenizer.padding_side = _orig_padding_side

    n_failures = predictions.count("")
    if n_failures:
        logger.warning(f"{n_failures}/{len(dataset)} samples failed generation — metrics may be deflated")

    model.config.use_cache = _was_use_cache 
    model.train()

    anatomy = cfg.get("anatomy", "liver")
    bertscore_model = cfg.get("evaluation", {}).get(                               
        "bertscore_model", "bert-base-multilingual-cased"                         
    )     
    results = evaluate_reports(
        predictions=predictions,
        references=references,
        cfg=cfg,
        anatomy=anatomy,
        bertscore_model=bertscore_model,                                           
        compute_groups=compute_groups,
    )
 
    print_metrics_table(results, split_name=split_name.capitalize())

    csv_path, json_path = save_metrics(results, cfg, split_name=split_name)
    print(f"Results written to:\n  {csv_path}\n  {json_path}")

    if clearml_task is not None:
        metrics_to_clearml(clearml_task, results, split_name=split_name)

    return results
