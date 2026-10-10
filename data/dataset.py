import json
import csv
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional
from datasets import Dataset

from PIL import Image

logger = logging.getLogger(__name__)

def _load_records(path: str):
    # returns a list of dict
    suffix = Path(path).suffix.lower()

    if suffix == ".json":
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    if suffix == ".csv":
        with open(path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            records = []
            for row in reader:
                image_path = (row.get("image_path") or "").strip()
                annotation = (row.get("annotation") or "").strip()
                if not image_path or not annotation:
                    continue
                records.append({"image_path": image_path, "annotation": annotation})
            logger.info(f"Loaded {len(records)} records from {path}")
            return records

    raise ValueError(f"Unsupported dataset file type: {suffix}")

def _load_image(image_path: str):
    p = Path(image_path)
    if not p.exists():
        logger.warning(f"Image not found: {image_path}")
        return None
    try:
        return Image.open(p).convert("RGB")
    except Exception as e:
        logger.warning(f"Failed to load image {image_path}: {e}")
        return None

def _build_messages(images: list, instruction: str, report_text: str):
    user_content = [{"type": "image", "image": img} for img in images]
    user_content.append({"type": "text", "text": instruction})
    return [
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": [{"type": "text", "text": report_text}]},
    ]

def _extract_patient_id(image_field):
    """
    Derive patient ID from a filename or list of filenames.
    Convention: <patient_id>_<view_number>.png
    Returns a str
    """
    if isinstance(image_field, list):
        filename = image_field[0] if image_field else ""
    else:
        filename = image_field

    stem = Path(filename).stem
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return stem

def _parse_structured_record(record: dict, image_dir: str, instruction: str):
    raw_image_field = record.get("image", record.get("image_path", ""))
    report_text = record.get("report", "").strip()

    if not report_text:
        logger.warning(f"Skipping record with empty report: {record.get('id','?')}")
        return None

    if isinstance(raw_image_field, str):
        image_filenames = [raw_image_field]
    elif isinstance(raw_image_field, list):
        image_filenames = raw_image_field
    else:
        logger.warning(f"Unexpected image field type: {type(raw_image_field)}")
        return None

    images = [_load_image(str(Path(image_dir) / f)) for f in image_filenames]
    images = [img for img in images if img is not None]

    if not images:
        logger.warning(f"No valid images for record {record.get('id','?')}, skipping.")
        return None

    return {
        "images":     images,
        "messages":   _build_messages(images, instruction, report_text),
        "_image_ref": image_filenames[0],   # first filename — used for patient-aware split
    }


def _parse_raw_record(record: dict, image_dir: str, instruction: str):
    image_filename = record.get("image", record.get("image_path", ""))
    report_text = record.get("annotation", record.get("report", "")).strip()

    if not report_text or not image_filename:
        return None

    image_filenames = image_filename if isinstance(image_filename, list) else [image_filename]

    images = [_load_image(str(Path(image_dir) / f)) for f in image_filenames]
    images = [img for img in images if img is not None]

    if not images:
        return None

    return {
        "images":     images,
        "messages":   _build_messages(images, instruction, report_text),
        "_image_ref": image_filenames[0],   # first filename — used for patient-aware split
    }

def group_by_patient(records: list[dict], dataset_format: str):
    """
    Group records belonging to the same patient into one multi-image sample.
    group=True: one sample per patient, multiple images per sample.
    group=False: one sample per image.
    Returns a list of dicts
    """
    if dataset_format == "structured":
        image_fn  = lambda r: r.get("image", r.get("image_path", ""))
        report_fn = lambda r: r.get("report", "")
        annot_fn  = lambda r: r.get("_meta", {}).get("raw_annotation", r.get("report", ""))
    else:
        image_fn  = lambda r: r.get("image_path", r.get("image", ""))
        report_fn = lambda r: r.get("annotation", "")
        annot_fn  = lambda r: r.get("annotation", "")

    groups = defaultdict(lambda: {"images": [], "report": "", "annotation": ""})

    for record in records:
        img_field  = image_fn(record)
        patient_id = _extract_patient_id(img_field)
        if not patient_id:
            logger.warning(f"Could not extract patient ID from: {img_field!r}, skipping.")
            continue
        if isinstance(img_field, list):
            groups[patient_id]["images"].extend(img_field)
        else:
            groups[patient_id]["images"].append(img_field)
        current_report = groups[patient_id]["report"]
        new_report     = report_fn(record)
        if not current_report:
            groups[patient_id]["report"]     = new_report
            groups[patient_id]["annotation"] = annot_fn(record)
        elif new_report and new_report.strip() != current_report.strip():
            # Two records for the same patient have different annotations.
            logger.warning(
                f"Patient '{patient_id}' has conflicting annotations across views. "
                f"Using the first annotation. Review the dataset for annotation consistency.\n"
                f"  Kept    : {current_report[:80]!r}\n"
                f"  Discarded: {new_report[:80]!r}"
            )

    def _sort_key(fname: str):
        parts = Path(fname).stem.rsplit("_", 1)
        return int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 0

    grouped = []
    for patient_id, gd in groups.items():
        sorted_imgs = sorted(gd["images"], key=_sort_key)
        grouped.append({
            "patient_id": patient_id,
            "image":      sorted_imgs,
            "image_path": sorted_imgs[0],
            "report":     gd["report"],
            "annotation": gd["annotation"],
        })

    logger.info(
        f"Grouped {len(records)} records: {len(grouped)} patient samples "
        f"(avg {len(records)/max(len(grouped),1):.1f} images/patient)"
    )
    return grouped


def pair_by_patient(
    records: list[dict],
    dataset_format: str,
    max_images: int = 2,
):
    """
    Group raw CSV records by patient ID into multi-image paired samples.
    returns a list of dicts
    """
    if dataset_format == "structured":
        image_fn  = lambda r: r.get("image", r.get("image_path", ""))
        report_fn = lambda r: r.get("report", r.get("annotation", ""))
    else:
        image_fn  = lambda r: r.get("image_path", r.get("image", ""))
        report_fn = lambda r: r.get("annotation", r.get("report", ""))

    def _sort_key(fname: str):
        parts = Path(fname).stem.rsplit("_", 1)
        return int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 0

    # Group filenames by patient ID
    patient_files:  dict[str, list[str]] = defaultdict(list)
    patient_report: dict[str, str]       = {}

    for record in records:
        img = image_fn(record)
        pid = _extract_patient_id(img)
        if not pid:
            logger.warning(f"Could not extract patient ID from {img!r}, skipping.")
            continue
        patient_files[pid].append(img)
        if pid not in patient_report:
            patient_report[pid] = report_fn(record)

    paired = []
    for pid, files in patient_files.items():
        sorted_files = sorted(files, key=_sort_key)[:max_images]   # cap at max_images
        # Return a record in raw-CSV shape with image_path as a list
        paired.append({
            "image_path": sorted_files,               
            "annotation": patient_report.get(pid, ""),
        })

    single = sum(1 for p in paired if len(p["image_path"]) == 1)
    multi  = len(paired) - single

    logger.info(
        f"Paired {len(records)} records : {len(paired)} samples "
        f"({multi} multi-image, {single} single-image)"
    )
    return paired

# Patient-aware train/val split. Split by patient ID - returns a tuple
def make_train_val_split(dataset: list, val_split: float = 0.1, seed: int = 42):
    patient_ids = sorted({_extract_patient_id(s["_image_ref"]) for s in dataset})

    rng = random.Random(seed)
    rng.shuffle(patient_ids)

    n_val        = max(1, round(len(patient_ids) * val_split))
    val_patients = set(patient_ids[:n_val])

    train, val = [], []
    for sample in dataset:
        pid = _extract_patient_id(sample.pop("_image_ref"))  # strip here
        (val if pid in val_patients else train).append(sample)

    logger.info(
        f"Patient-aware split: "
        f"{len(patient_ids) - n_val} train patients / {n_val} val patients: "
        f"{len(train)} train samples / {len(val)} val samples"
    )
    return train, val


def load_and_prepare_dataset(
    path: str,
    image_dir: str,
    dataset_format: str,
    instruction: str,
    group_images: bool = True,
    max_images_per_sample: Optional[int] = None,
    seed: int = 42,
):
    """
    Load a dataset file and return a plain Python list of samples.
    Each sample: {"images": [PIL, ...], "messages": [...], "_image_ref": str}
    The _image_ref key is stripped by make_train_val_split or load_split_datasets.
    Returns a list
    """
    records = _load_records(path)
    logger.info(f"Loaded {len(records)} records from {path}")

    if group_images == "pair":
        max_imgs_for_pairing = max_images_per_sample or 2
        records = pair_by_patient(records, dataset_format, max_images=max_imgs_for_pairing)
        # After pairing, max_images_per_sample truncation is already applied above
        # so skip it below to avoid double-truncation
        max_images_per_sample = None
    elif group_images is True or group_images == "true":
        records = group_by_patient(records, dataset_format)

    if max_images_per_sample is not None:
        capped = 0
        for r in records:
            img_field = r.get("image", r.get("image_path", ""))
            if isinstance(img_field, list) and len(img_field) > max_images_per_sample:
                key = "image" if "image" in r else "image_path"
                r[key] = img_field[:max_images_per_sample]
                capped += 1
        if capped == 0:
            logger.warning(
                f"max_images_per_sample={max_images_per_sample} is configured but had no effect. "
                f"This setting only truncates multi-image grouped records. "
                f"With group_images=False and raw CSV format, each record has exactly one image. "
                f"Use group_images='pair' to create 2-image samples from paired views."
            )

    parser = _parse_structured_record if dataset_format == "structured" else _parse_raw_record
    normalised, skipped = [], 0

    for record in records:
        parsed = parser(record, image_dir, instruction)
        if parsed is not None:
            normalised.append(parsed)
        else:
            skipped += 1

    logger.info(f"Parsed {len(normalised)} samples ({skipped} skipped)")

    if not normalised:
        raise ValueError("No valid samples after parsing. Check image_dir and paths.")

    # _image_ref is present here; caller strips it
    # return normalised
    return Dataset.from_list(normalised)


def load_split_datasets(cfg: dict):
    dataset_cfg = cfg["dataset"] if cfg["dataset_format"] == "structured" else cfg["csv_dataset"]

    format_     = dataset_cfg["format"]
    instruction = dataset_cfg["instruction"]
    image_dir   = dataset_cfg["image_dir"]
    group       = dataset_cfg.get("group_images", False)
    max_imgs    = dataset_cfg.get("max_images_per_sample")
    seed        = cfg["experiment"]["seed"]

    assert dataset_cfg["format"] == cfg["dataset_format"], \
    (f"dataset block format '{dataset_cfg['format']}' conflicts with "
     f"top-level dataset_format '{cfg['dataset_format']}'. "
     f"These must match or the wrong parser will be selected.")

    train_path = dataset_cfg["train_path"]
    val_path   = dataset_cfg.get("val_path")

    val_path_valid = (
        val_path
        and isinstance(val_path, str)
        and "{" not in val_path
        and Path(val_path).is_file()
    )

    if not val_path_valid:
        raise FileNotFoundError(
            f"val_path '{val_path}' does not exist or was not interpolated.\n"
            f"Run data/split.py once to create the split files, then update the config."
        )

    train_ds = load_and_prepare_dataset(
        path=train_path,
        image_dir=image_dir,
        dataset_format=format_,
        instruction=instruction,
        group_images=group,
        max_images_per_sample=max_imgs,
        seed=seed,
    )
    for s in train_ds: s.pop("_image_ref", None)

    val_ds = load_and_prepare_dataset(
        path=val_path,
        image_dir=image_dir,
        dataset_format=format_,
        instruction=instruction,
        group_images=group,
        max_images_per_sample=max_imgs,
        seed=seed,
    )
    for s in val_ds: s.pop("_image_ref", None)

    logger.info(f"Loaded {len(train_ds)} train / {len(val_ds)} val samples from pre-split files.")
    return train_ds, val_ds


def cap_majority_class(dataset: list, max_fraction: float = 0.05,
                       max_count: int = None, seed: int = 42):
    """
    Hard-cap how many times any single report can appear in training.
    Args:
        dataset:      List of training samples.
        max_fraction: Cap each report at this fraction of total dataset size.
        max_count:    Absolute cap. Overrides max_fraction if both given.
        seed:         Random seed for which samples to keep when capping.
    Returns: A list of capped, shuffled dataset. Does not touch val_ds.
    """
    import random
    import logging
    from collections import defaultdict

    logger = logging.getLogger(__name__)
    rng    = random.Random(seed)

    # Group by report text
    buckets = defaultdict(list)
    for sample in dataset:
        content = sample['messages'][1]['content']
        text    = content[0]['text'] if isinstance(content, list) else content
        buckets[text].append(sample)

    total    = len(dataset)
    hard_cap = max_count if max_count is not None else max(1, int(max_fraction * total))

    capped_ds, n_capped, n_removed = [], 0, 0
    for text, samples in buckets.items():
        if len(samples) > hard_cap:
            kept = rng.sample(samples, hard_cap)
            n_removed += len(samples) - hard_cap
            n_capped  += 1
        else:
            kept = samples
        capped_ds.extend(kept)

    rng.shuffle(capped_ds)

    logger.info(
        f"cap_majority_class: {n_capped} report(s) capped at {hard_cap} "
        f"(max_fraction={max_fraction}). "
        f"Dataset: {total} - {len(capped_ds)} samples "
        f"({100*n_removed/total:.1f}% removed)"
    )
    return Dataset.from_list(capped_ds) 
    # return capped_ds