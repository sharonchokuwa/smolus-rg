import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

try:
    from openpyxl import load_workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
except ImportError:
    sys.exit("pip install openpyxl")

# Column maps Experiments sheet: columns A-W
EXP_AUTO_COLS = {
    "B": "exp_id",          "C": "exp_name",       "E": "config_file",
    "F": "timestamp",       "G": "clearml_task_id",
    "I": "model_size",      "J": "init_strategy",  "K": "peft_mode",
    "L": "precision",       "M": "optim",
    "N": "dataset_format",  "O": "anatomy",        "P": "group_images",
    "Q": "num_images",      "R": "assistant_loss",
    "S": "gpu",             "T": "epochs",         "U": "batch_size",
    "V": "grad_accum",
}

# Results sheet column layout:
# A=Timestamp  B=ExpID  C=ExpName  D=ModelSize  E=PEFTMode  F=Anatomy  G=ConfigFile
# H=BLEU-1  I=BLEU-2  J=BLEU-3  K=BLEU-4
# L=ROUGE-L                        
# M=METEOR  N=CIDEr
# O=BERTScore-P  P=BERTScore-R  Q=BERTScore-F1
# R=CE-Accuracy  S=CE-Precision  T=CE-Recall  U=CE-F1   
# V=Split
RESULTS_METRIC_MAP = {
    "BLEU-1":        "H",
    "BLEU-2":        "I",
    "BLEU-3":        "J",
    "BLEU-4":        "K",
    "ROUGE-L":       "L",   
    "METEOR":        "M",
    "CIDEr":         "N",
    "BERTScore-P":   "O",
    "BERTScore-R":   "P",
    "BERTScore-F1":  "Q",
    "CE-Accuracy":   "R",  
    "CE-Precision":  "S",
    "CE-Recall":     "T",
    "CE-F1":         "U",
}

# Efficiency sheet: A-AA
EFF_COL_MAP = {
    "A": "timestamp",         "B": "exp_id",              "C": "exp_name",
    "D": "device_name",       "E": "peft_mode",           "F": "precision",
    "G": "total_params",      "H": "trainable_params",    "I": "trainable_pct",
    "J": "disk_size_mb",
    "K": "preprocess_ms_mean","L": "ttft_ms_mean",        "M": "generation_ms_mean",
    "N": "e2e_ms_mean",       "O": "e2e_ms_std",          "P": "device_name",
    "Q": "tokens_per_sec",    "R": "images_per_sec",      "S": "output_tokens_mean",
    "T": "n_timed_runs",
    "U": "vram_total_mb",     "V": "vram_idle_mb",        "W": "vram_peak_inference_mb",
    "X": "vram_utilisation_pct", "Y": "cpu_ram_peak_mb",
    "Z": "bleu4_per_sec",     "AA": "latency_per_billion_params_ms",
}

def _col_to_int(col: str):
    result = 0
    for ch in col.upper():
        result = result * 26 + (ord(ch) - ord('A') + 1)
    return result

def _load_json(path):
    p = Path(path) if path else None
    if not p or not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)

def _next_row(ws, id_col=1, start=4):
    for r in range(start, ws.max_row + 2):
        if ws.cell(row=r, column=id_col).value is None:
            return r
    return ws.max_row + 1

def _exists(ws, ts, ts_col=1, split=None, split_col=None, start=4):
    for r in range(start, ws.max_row + 1):
        if str(ws.cell(row=r, column=ts_col).value or "") == ts:
            if split is None:
                return r
            if str(ws.cell(row=r, column=split_col).value or "") == split:
                return r
    return None

def _shade(ws, row, n):
    thin = Side(style="thin", color="CCCCCC")
    brd  = Border(left=thin, right=thin, top=thin, bottom=thin)
    bg   = "F5F5F5" if row % 2 == 0 else "FFFFFF"
    for ci in range(1, n + 1):
        c = ws.cell(row=row, column=ci)
        raw = c.fill.fgColor.rgb if c.fill and c.fill.fgColor else "00000000"
        if raw in ("00000000", "FFFFFFFF", "00FFFFFF"):
            c.fill = PatternFill("solid", fgColor=bg)
        c.border = brd

def _extract_cfg_fields(cfg: dict, gpu: str, overrides: dict = None):
    if not cfg:
        return {}
    training = cfg.get("training", {})
    peft     = cfg.get("peft", {})
    dsf      = cfg.get("dataset_format", "raw")
    ds       = cfg.get("dataset", {}) if dsf == "structured" else cfg.get("csv_dataset", {})

    fields = {
        "model_size":     cfg.get("model_size", ""),
        "init_strategy":  cfg.get("model", {}).get("init_strategy", ""),
        "peft_mode":      peft.get("mode", "full"),
        "precision":      training.get("precision", ""),
        "optim":          training.get("optim", ""),
        "dataset_format": dsf,
        "anatomy":        cfg.get("anatomy", ""),
        "group_images":   str(ds.get("group_images", "false")),
        "num_images":     str(ds.get("max_images_per_sample", 1)),
        "assistant_loss": str(training.get("assistant_only_loss", False)),
        "gpu":            gpu,
        "epochs":         str(training.get("num_train_epochs", "")),
        "batch_size":     str(training.get("per_device_train_batch_size", "")),
        "grad_accum":     str(training.get("gradient_accumulation_steps", "")),
    }

    # Apply runtime overrides so the tracker reflects what was used
    _override_field_map = {
        "training.assistant_only_loss": "assistant_loss",
        "training.num_train_epochs":    "epochs",
        "training.learning_rate":       None,   # not in tracker columns — skip
        "csv_dataset.group_images":     "group_images",
        "csv_dataset.max_images_per_sample": "num_images",
        "peft.mode":                    "peft_mode",
        "training.precision":           "precision",
        "training.optim":               "optim",
        "model.init_strategy":          "init_strategy",
        "anatomy":                      "anatomy",
        "dataset_format":               "dataset_format",
        "experiment.name":              None,   # handled separately as exp_name
    }
    for dotted_key, value in (overrides or {}).items():
        flat_key = _override_field_map.get(dotted_key)
        if flat_key:
            fields[flat_key] = str(value)
    return fields

def ingest_experiment(ws, args, cfg, runtime_overrides):
    if _exists(ws, args.timestamp, ts_col=6, start=5):
        logger.info("  Experiments: timestamp already exists.")
        return
    row = _next_row(ws, id_col=2, start=5)
    while ws.cell(row=row, column=2).value is not None:
        row += 1
    fields = {
        "exp_id": args.exp, "exp_name": args.exp_name,
        "config_file": Path(args.config).name if args.config else "",
        "timestamp": args.timestamp, "clearml_task_id": args.clearml_task_id or "",
        **_extract_cfg_fields(cfg, args.gpu, overrides=runtime_overrides),
    }

    for col_letter, key in EXP_AUTO_COLS.items():
        ws.cell(row=row, column=_col_to_int(col_letter), value=fields.get(key, ""))
    ws.cell(row=row, column=23, value="Completed")  # W = Status default
    _shade(ws, row, 23)
    ws.row_dimensions[row].height = 20
    logger.info(f"  Experiments: appended row {row}")

def ingest_results(ws, args, data, split, cfg):
    if _exists(ws, args.timestamp, ts_col=1, split=split, split_col=22, start=4):
        logger.info(f"  Results [{split}]: already exists — skipped.")
        return
    row = _next_row(ws, id_col=1, start=4)
    while ws.cell(row=row, column=1).value is not None:
        row += 1
    ws.cell(row=row, column=1,  value=args.timestamp)
    ws.cell(row=row, column=2,  value=args.exp)
    ws.cell(row=row, column=3,  value=args.exp_name)
    ws.cell(row=row, column=4,  value=cfg.get("model_size", "") if cfg else "")
    ws.cell(row=row, column=5,  value=cfg.get("peft", {}).get("mode", "") if cfg else "")
    ws.cell(row=row, column=6,  value=cfg.get("anatomy", "") if cfg else "")
    ws.cell(row=row, column=7,  value=Path(args.config).name if args.config else "")
    metrics = data.get("metrics", data)
    for name, col_letter in RESULTS_METRIC_MAP.items():
        val = metrics.get(name)
        if val is not None:
            c = ws.cell(row=row, column=_col_to_int(col_letter), value=round(float(val), 4))
            c.number_format = "0.0000"
    ws.cell(row=row, column=22, value=split)  
    _shade(ws, row, 22)
    ws.row_dimensions[row].height = 18
    logger.info(f"  Results [{split}]: appended row {row}")

def ingest_efficiency(ws, args, data):
    if _exists(ws, args.timestamp, ts_col=1, start=4):
        logger.info("  Efficiency: already exists — skipped.")
        return
    row = _next_row(ws, id_col=1, start=4)
    while ws.cell(row=row, column=1).value is not None:
        row += 1
    full = {**data, "exp_id": args.exp, "exp_name": args.exp_name, "timestamp": args.timestamp}
    for col_letter, key in EFF_COL_MAP.items():
        val = full.get(key)
        if val is not None:
            c = ws.cell(row=row, column=_col_to_int(col_letter), value=val)
            if isinstance(val, float):
                c.number_format = "0.00"
    _shade(ws, row, 27)
    ws.row_dimensions[row].height = 18
    logger.info(f"  Efficiency: appended row {row}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp",             required=True)
    p.add_argument("--exp-dir",         required=True)
    p.add_argument("--tracker",         required=True)
    p.add_argument("--config",          default=None)
    p.add_argument("--timestamp",       default=None)
    p.add_argument("--clearml-task-id", default="")
    p.add_argument("--gpu",             default="")
    p.add_argument("--exp-name",        default=None)
    p.add_argument("--dry-run",         action="store_true")
    p.add_argument("--overrides", default="{}",
               help="JSON dict of runtime overrides applied to this experiment")
    args = p.parse_args()

    try:
        runtime_overrides = json.loads(args.overrides)
    except (json.JSONDecodeError, AttributeError):
        runtime_overrides = {}

    if not args.timestamp:
        args.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cfg = None
    if args.config and Path(args.config).exists():
        try:
            import yaml
            with open(args.config, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load config: {e}")

    if not args.exp_name:
        if cfg:
            raw = cfg.get("experiment", {}).get("name", args.exp)
            raw = raw.replace("{model_size}", str(cfg.get("model_size", "")))
            raw = raw.replace("{model.init_strategy}",
                  str(cfg.get("model", {}).get("init_strategy", "")))
            args.exp_name = raw
        else:
            args.exp_name = Path(args.exp_dir).name

    if not args.gpu:
        try:
            import torch
            if torch.cuda.is_available():
                args.gpu = torch.cuda.get_device_name(0)
        except ImportError:
            pass

    exp_dir = Path(args.exp_dir)
    val_m   = _load_json(exp_dir / "metrics_val.json")
    test_m  = _load_json(exp_dir / "metrics_test.json")
    eff_d   = _load_json(exp_dir / "efficiency_test.json")

    logger.info(f"\nIngesting {args.exp} | {args.exp_name} | {args.timestamp}")

    if not Path(args.tracker).exists():
        sys.exit(f"Tracker not found: {args.tracker}")

    wb = load_workbook(args.tracker)

    if " Experiments" in wb.sheetnames:
        ingest_experiment(wb[" Experiments"], args, cfg, runtime_overrides)
    if " Results" in wb.sheetnames:
        if val_m:
            ingest_results(wb[" Results"], args, val_m,  "val",  cfg)
        if test_m:
            ingest_results(wb[" Results"], args, test_m, "test", cfg)
    if "⚡ Efficiency" in wb.sheetnames and eff_d:
        ingest_efficiency(wb[" Efficiency"], args, eff_d)

    if args.dry_run:
        logger.info("\n[dry-run] NOT saved.")
    else:
        wb.save(args.tracker)
        logger.info(f"\n Saved: {args.tracker}")

if __name__ == "__main__":
    main()
