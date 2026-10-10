# Effect of Pretrained Initialisation and Component-Wise Tuning on Lightweight VLMs for Ultrasound Report Generation

Official implementation for:

> **Effect of Pretrained Initialisation and Component-Wise Tuning on Lightweight VLMs for Ultrasound Report Generation**  
> Sharon Mlotshwa, Pramit Saha, Jamal Bentahar, Muzammal Naseer and Mohammad Alsharid  
> *MICCAI MIRASOL, 2026*

- **Paper:** Coming soon!
- **Code:** [Repository](https://github.com/sharonchokuwa/smolus-rg)
- **Dataset:** 
    - Original Dataset: https://github.com/LijunRio/Ultrasound-Report-Generation
    - English Translated version:  https://github.com/TongxinWang0405/BMI702_project
- **SmolUS Checkpoint:** [Model URL](https://drive.google.com/drive/folders/1qLoygrWCQ7crM92g4yTGMSldeynmn97Q?usp=sharing)


## Abstract

Ultrasound report generation (US-RG) can ease sonographers’ documentation burden, but it is a challenging task. Practical clinical deployment requires US-RG lightweight models, but progress is constrained by the scarcity of open-source ultrasound image-text datasets. Under these low-data and resource constrained conditions, we study how the choice of a lightweight US-RG model’s pretrained initialisation and component-level (vision/language model) finetuning affects how it performs on multi-anatomy datasets. Our findings suggest that: (i) domain-specific ultrasound pretraining may not be necessary for lightweight US-RG as general biomedical initialisation and an initialisation from web-scale pretraining perform comparably; (ii) given the limited data, scaling the model yields little gains, whereas parameter-efficient adaptation of both the vision and language model is more promising; (iii) task-specific finetuning irrespective of the initialisation source weights outperforms zero-shot evaluation and (iv) lightweight models suitable for on-device deployment can be used for US-RG, but the gap between the semantic similarity and accurate identification of clinically relevant findings shows the need for methods that improve clinical factuality.



## Overview

We present an empirical study on the effect of initialisation and component-level tuning for lightweight ultrasound report generation and we find that under limited data, parameter-efficient task-specific finetuning matters more than domain-specific pretraining or model scaling.

The findings of this work suggest that, for US report generation:
1. Domain-specificity (ultrasound-specific pretraining) may not be necessary for model performance improvement in lightweight US-RG models;
2. Given the limited US dataset size, increasing model scale, does not guarantee performance gains, rather improving the parameter-efficient adaptation method on both the vision model and language model (LM) may be more promising
3. Task-specific finetuning irrespective of the initialisation source weights outperforms zero-shot evaluation
4. Lightweight models are suitable for on-device deployment, but the gap between semantic similarity evaluation scores and accurate identification of clinically relevant findings shows the need for methods that improve clinical factuality

### Repository Structure

```text
.
├── config/                      # YAML configs + config loader/interpolation
│   ├── full_ft.yaml
│   ├── lora_ft.yaml
│   └── loader.py
│
├── data/                        # dataset loading + patient-aware splitting
│   ├── dataset.py
│   └── split.py
│
├── evaluation/                  # NLG, clinical metrics + plots
│   ├── metrics.py
│   ├── report_metrics.py
│   └── training_plots.py
│   
│
├── models/                      # model + processor loading, PEFT setup
│   └── loader.py
│
├── notebooks/
│   └── smolus_rg_training.ipynb # full training pipeline for Colab
│
├── tools/
│   └── ingest_results.py        # ingests results into the tracker
│
├── training/
│   └── trainer.py               # SFTTrainer assembly + collator
│
├── utils/                       # checkpointing, ClearML, weight verification
│   ├── checkpointing.py
│   └── clearml_integration.py
│
└── verify_weights.py           
└── full_component_io.py
```


## Hardware

All fine-tuning experiments were conducted on **Google Colab** using an **NVIDIA L4 GPU (24 GB VRAM)**.
The only exception is **SmolUS** (our ultrasound-pretrained initialization), which was pretrained on an
**NVIDIA A100 GPU (40 GB VRAM)** due to its larger memory footprint.

CPU-only inference (used for the edge-deployment efficiency profiling) was also run on Colab.


## Dataset

> **The dataset is not included in this repository.** Obtain it from the original sources described in
> the paper.

Once obtained, the data must be split into patient-level train/val/test files before training. Use the
provided split script, which guarantees no patient appears in more than one split:

```bash
python data/split.py \
    --input   /path/to/<anatomy>_reports.csv \
    --output-dir /path/to/data \
    --train-ratio 0.7 --val-ratio 0.1 --test-ratio 0.2 --seed 42
```

This produces `<anatomy>_reports_train.csv`, `_val.csv`, and `_test.csv`. Repeat for each anatomy
(liver, thyroid, breast). Update the `train_path` / `val_path` / `test_path` / `image_dir` fields in the
config to point at these files. 

## Environment

Install dependencies:

```bash
- Python 3.13.15 (Google Colab runtime)
- torch 2.11.0+cu130 (pre-installed on Colab)
- GPU: NVIDIA L4 (24 GB)

Install packages with `pip install -r requirements.txt`.
```



## How to run

The full training pipeline is driven from the Colab notebook:

**Steps:**

1. Ensure the split data files and images are on Drive at the paths referenced in the config.
2. Choose the experiment(s) to run by editing the experiment queue near in the notebook —
   each entry sets the config, the initialization strategy, the adaptation method (M1–M9), and the
   anatomy via overrides. 
3. Run all cells. Each experiment loads the config, stages images to local disk, finetunes, evaluates
   on val and test, and writes checkpoints and metrics to `output_dir`.


### Configuration

Experiment behaviour is controlled by the YAML files in `config/` (`lora_ft.yaml`, `full_ft.yaml`).
Key fields:

- `model.init_strategy` — initialization: `biomedica`, `huggingface`, `ultrasound` (SmolUS), or `random`
- `peft.method` — component-wise adaptation preset (M1–M9)
- `model_size` — `256M`, `500M`, or `2.2B`
- `anatomy` — `liver`, `thyroid`, `breast`, or `all` (combined)

These can be set in the YAML directly or overridden per-experiment in the notebook's queue.

## Citation
Coming soon!


