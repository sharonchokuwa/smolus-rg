"""
One-time patient-aware train/val/test split at the record level.
"""

import argparse
import csv
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def _extract_patient_id(image_field):
    """Derive patient ID from filename. Convention: <patient_id>_<view>.ext"""
    filename = image_field[0] if isinstance(image_field, list) else image_field
    stem = Path(filename).stem
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return stem

def _get_image_field(record: dict):
    """Extract the image filename field regardless of format."""
    return record.get("image", record.get("image_path", ""))

def _load_records(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".json":
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    if suffix == ".csv":
        with open(path, encoding="utf-8-sig", newline="") as f:
            reader  = csv.DictReader(f)
            records = []
            skipped = 0
            for row in reader:
                image_path = (row.get("image_path") or row.get("image") or "").strip()
                annotation = (row.get("annotation") or row.get("report") or "").strip()
                if not image_path or not annotation:
                    skipped += 1
                    continue
                records.append(dict(row))
            if skipped:
                logger.warning(
                    f"split.py: skipped {skipped} rows with empty image_path or annotation. "
                )
            return records
    raise ValueError(f"Unsupported file type: {suffix}")

def _write_records(records: list[dict], path: Path):
    suffix = path.suffix.lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
    elif suffix == ".csv":
        if not records:
            path.write_text("")
            return
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    logger.info(f"  Wrote {len(records)} records: {path}")

# Core split logic 
def patient_aware_split(
    records: list[dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"

    # Group records by patient ID
    patient_buckets: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        img_field  = _get_image_field(record)
        patient_id = _extract_patient_id(img_field)
        patient_buckets[patient_id].append(record)

    # Shuffle patient IDs deterministically
    patient_ids = sorted(patient_buckets.keys())
    rng = random.Random(seed)
    rng.shuffle(patient_ids)

    n = len(patient_ids)
    n_test  = max(1, round(n * test_ratio))
    n_val   = max(1, round(n * val_ratio))
    n_train = n - n_val - n_test

    if n_train < 1:
        raise ValueError(
            f"Too few patients ({n}) to create train/val/test splits. "
            f"Need at least 10 patients for a 7:1:2 ratio."
        )

    train_ids = set(patient_ids[:n_train])
    val_ids   = set(patient_ids[n_train:n_train + n_val])
    test_ids  = set(patient_ids[n_train + n_val:])

    train_records = [r for pid in train_ids  for r in patient_buckets[pid]]
    val_records   = [r for pid in val_ids    for r in patient_buckets[pid]]
    test_records  = [r for pid in test_ids   for r in patient_buckets[pid]]

    return train_records, val_records, test_records

def run_split(
    input_path: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
    stem: str = None,     
):
    """
    Load a single source file, split it, and write train/val/test files.
    Returns the three output paths so callers can update configs programmatically.
    """
    src = Path(input_path)
    out = Path(output_dir)
    base = stem or src.stem          # "liver_reports"
    ext  = src.suffix                # ".csv" or ".json"

    train_path = out / f"{base}_train{ext}"
    val_path   = out / f"{base}_val{ext}"
    test_path  = out / f"{base}_test{ext}"

    logger.info(f"Loading: {src}")
    records = _load_records(src)
    logger.info(f"Total records: {len(records)}")

    patient_set = {_extract_patient_id(_get_image_field(r)) for r in records}
    logger.info(f"Unique patients: {len(patient_set)}")

    # Validate filename convention before splitting.
    # Convention: <patient_id>_<view_number>.ext
    non_conforming = []
    for record in records:
        img_field = _get_image_field(record)
        filename  = img_field[0] if isinstance(img_field, list) else img_field
        stem      = Path(filename).stem
        parts     = stem.rsplit("_", 1)
        if not (len(parts) == 2 and parts[1].isdigit()):
            non_conforming.append(filename)

    if non_conforming:
        logger.warning(
            f"{len(non_conforming)} filenames do not follow the <patient_id>_<N>.ext "
            f"convention. Patient IDs will use the full stem, which may merge unrelated "
            f"records or fail to separate views from the same patient.\n"
            f"First 5 non-conforming: {non_conforming[:5]}\n"
            f"Review the extracted patient IDs below before accepting the split."
        )

    # Log all unique patient IDs so they can be visually verified
    all_patient_ids = sorted({_extract_patient_id(_get_image_field(r)) for r in records})
    logger.info(f"Extracted {len(all_patient_ids)} unique patient IDs.")
    logger.info(f"First 10: {all_patient_ids[:10]}")
    logger.info(f"Last 10:  {all_patient_ids[-10:]}")
    if len(all_patient_ids) == len(records):
        logger.warning(
            "Number of patient IDs equals number of records — every record is treated as "
            "a separate patient. If patients have multiple images, the split is incorrect. "
            "Verify the filename convention."
        )

    train_records, val_records, test_records = patient_aware_split(
        records, train_ratio=train_ratio, val_ratio=val_ratio,
        test_ratio=test_ratio, seed=seed,
    )

    total = len(train_records) + len(val_records) + len(test_records)
    logger.info(f"\nSplit summary (by patient, {seed=}):")
    logger.info(f"  train : {len(train_records):>5} records  ({100*len(train_records)/total:.1f}%)")
    logger.info(f"  val   : {len(val_records):>5} records  ({100*len(val_records)/total:.1f}%)")
    logger.info(f"  test  : {len(test_records):>5} records  ({100*len(test_records)/total:.1f}%)")

    _verify_no_leakage(train_records, val_records, test_records)     # Verify no patient appears in multiple splits

    _write_records(train_records, train_path)
    _write_records(val_records,   val_path)
    _write_records(test_records,  test_path)

    logger.info("\nDone. Update the config to use these paths:")
    logger.info(f"  train_path: {train_path}")
    logger.info(f"  val_path:   {val_path}")
    logger.info(f"  test_path:  {test_path}")
    return train_path, val_path, test_path


def _verify_no_leakage(train: list[dict], val: list[dict], test: list[dict]):
    """Assert that no patient ID appears in more than one split."""
    train_patients = {_extract_patient_id(_get_image_field(r)) for r in train}
    val_patients   = {_extract_patient_id(_get_image_field(r)) for r in val}
    test_patients  = {_extract_patient_id(_get_image_field(r)) for r in test}

    tv = train_patients & val_patients
    tt = train_patients & test_patients
    vt = val_patients   & test_patients

    if tv or tt or vt:
        raise RuntimeError(
            f"Data leakage detected!\n"
            f"  train ∩ val  : {tv}\n"
            f"  train ∩ test : {tt}\n"
            f"  val   ∩ test : {vt}"
        )
    logger.info(" No patient overlap between splits.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patient-aware train/val/test split")
    parser.add_argument("--input",      required=True, help="Source file (.csv or .json)")
    parser.add_argument("--output-dir", required=True, help="Directory to write split files")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio",   type=float, default=0.1)
    parser.add_argument("--test-ratio",  type=float, default=0.2)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--stem",        default=None,
                        help="Output filename stem (default: input filename stem)")
    args = parser.parse_args()
    run_split(
        input_path=args.input,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        stem=args.stem,
    )