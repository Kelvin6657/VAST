# VAST
### View-Aware Semi-Supervised Multi-Task Learning Framework for Fetal Cardiac Ultrasound

Research implementation for view-aware joint segmentation and multi-label
classification in fetal cardiac ultrasound.

**Qiujin Wang · Wenzheng Huang · Jiachen Yuan · Zhengnong Chen · Silvia Spinato · Netzahualcoyotl Hernandez-Cruz**

[English](README.md) | [简体中文](README_zh-CN.md)  
[Method](#method) · [Reproduction](#reproduction) · [Repository structure](#repository-structure) · [Citation](#citation)

## Manuscript

| Item | Details |
|---|---|
| Title | View-Aware Semi-Supervised Multi-Task Learning Framework for Fetal Cardiac Ultrasound |
| Citation status | Manuscript; final publication metadata pending |
| Citation files | [BibTeX](citations/VAST.bib), [CITATION.cff](CITATION.cff) |

The provisional citation below can be used until the final venue, year and DOI
are available. No publication or acceptance status is asserted here.

## Overview

Standard fetal cardiac ultrasound views depict different anatomical structures.
A segmentation class or classification label can therefore be meaningful in one
view and ineligible in another. VAST combines the acquired-view metadata with
a shared segmentation/classification network to constrain supervision and
pseudo-label learning across these views.

The framework has three components:

- **C1 — View-conditioned anatomical masking:** predefined view eligibility
  constrains segmentation logits and classification losses.
- **C2 — View-aware semi-supervised consistency learning:** predictions from
  weakly augmented unlabeled images provide confidence-filtered targets for
  strong augmentation, same-view CutMix and feature-perturbation branches.
- **C3 — Parameter-efficient encoder adaptation:** LoRA residuals adapt query
  and value projections of a frozen Swin encoder, while a UNETR decoder and
  classification head learn the two tasks.

## Method

![VAST architecture: labeled supervision and view-aware unlabeled consistency](assets/1.png)

**Figure 1.** Overview of the proposed semi-supervised multi-path training framework. The model is optimized with three complementary components: (C1) view-conditioned anatomical masking on labeled data, (C2) view-aware semi-supervised consistency learning on unlabeled data, and (C3) parameter-efficient encoder adaptation.


### Shared network

The model uses an input adapter, a Swin encoder, a UNETR segmentation decoder
and a multi-label classification head. It produces 15 segmentation logits
including background and seven classification logits. The classification head
uses pooled bottleneck features. The original encoder parameters are frozen;
low-rank query/value residuals are enabled in the LoRA stages.

### Supervised and unlabeled paths

Labeled images contribute segmentation cross-entropy/Dice and masked binary
classification losses. Unlabeled images have weak and strong augmentations.
Weak predictions are detached and confidence filtered before supervising the
strong predictions. The complete model additionally uses same-view CutMix
and feature perturbation. All branches use shared network parameters; the
implementation does not maintain a separate EMA teacher.

### Progressive configurations

| Stage | SSL | Training view mask | LoRA | Same-view CutMix | Feature perturbation |
|---|---|---|---|---|---|
| 0 — Supervised baseline | No | No | No | No | No |
| 1 — + SSL | Yes | No | No | No | No |
| 2 — + view masking | Yes | Yes | No | No | No |
| 3 — + LoRA | Yes | Yes | Yes | No | No |
| 4 — Complete VAST | Yes | Yes | Yes | Yes | Yes |

The final transition adds CutMix and feature perturbation jointly. Known-view
segmentation constraints and classification eligibility are used during
validation in every stage; the view-mask ablation changes training-time use.

## Reproduction

### 1. Install the environment

Use Python 3.10 with a compatible Linux/NVIDIA environment. The requirements
specify PyTorch 2.5.1, torchvision 0.20.1, MONAI 1.4.0 and CUDA 12.1 wheels.

```bash
conda create -n vast python=3.10 -y
conda activate vast
python -m pip install -r requirements.txt
```

Use a supported PyTorch/CUDA build for the target machine and prepare each run
in its actual environment. Environment/source changes after preparation require
a new protocol and new run directories.

### 2. Configure local paths

```bash
cp configs/paths.example.sh configs/paths.sh
```

Set `PROJECT_ROOT` in `configs/paths.sh` to your private input workspace. The
default is `private_workspace/` inside the repository, which is ignored by Git.
`PYTHON_BIN` selects the interpreter and `VAST_DEVICE` defaults to `cuda:0`.
Optional `CONDA_SH` and `CONDA_ENV` configure activation for batch jobs.

### 3. Prepare the dataset and encoder

Obtain the FETUS data and compatible encoder through their original providers.
Create your own authorized local split lists and use this layout:

```text
PROJECT_ROOT/
├── data/
│   ├── train_labeled.json
│   ├── train_unlabeled.json
│   ├── valid.json
│   ├── images/
│   └── labels/
└── pretrained_weights/
    └── echocare_encoder.pth
```

Each JSON file is a list of records. `image` gives an image HDF5 path beginning
with `data/`, relative to `PROJECT_ROOT`. Labeled/validation records also need
`label`, the corresponding target HDF5 path. Unlabeled records use `image` only.
No original sample records or split lists are distributed.

| HDF5 input | Required fields |
|---|---|
| Image | `image`: H×W×3 RGB array; `view`: integer array of shape (1,), with 1=4CH, 2=LVOT, 3=RVOT, 4=3VT |
| Target | `mask`: aligned H×W integer mask, classes 0–14; `label`: binary classification vector of shape (7,) |

The encoder architecture and numeric anatomy definitions must match the task.
Patient/scan separation must be established from the source metadata; image-ID
disjointness alone does not establish patient independence.

```bash
bash scripts/01_prepare.sh
```

Preparation validates input paths, mask dimensions and split overlap, then
records source hashes, split order, environment and encoder identity locally.
Original private splits and weights are not included, so the repository alone
does not reconstruct the original experimental population.

### 4. Run smoke checks

```bash
bash scripts/02_smoke.sh
```

This runs one real optimizer update and validation for each of the five
configurations. It needs the local data, encoder weights and GPU. All five
statuses must be `smoke_passed` before formal training.

### 5. Train

```bash
# Single complete-VAST run
bash scripts/03_train.sh 4 42
```

For the complete progressive comparison, start with fresh run directories and
use the following instead of rerunning an existing single configuration:

```bash
for stage in 0 1 2 3 4; do
  for seed in 42 123 2026; do
    bash scripts/03_train.sh "$stage" "$seed"
  done
done
```

The protocol uses 120 epochs, 256×256 inputs, batch size 8, AdamW, polynomial
learning-rate decay, bf16 AMP and gradient clipping. Existing stage/seed output
directories are deliberately preserved and cause a new run to stop.

### 6. Re-evaluate a checkpoint

```bash
bash scripts/04_evaluate.sh VAST/runs/stage4_seed42/best.pth
```

Or call the evaluation CLI directly:

```bash
python eval/evaluate.py \
  --checkpoint VAST/runs/stage4_seed42/best.pth \
  --project-root /path/to/private_workspace \
  --device cuda:0
```

This re-evaluates a recorded best checkpoint on its original prepared internal
validation split. It requires the adjacent `manifest.json` and matching local
`VAST/prepared/` protocol, source and environment. The complete model state is
loaded strictly. Results are written to a new `evaluation.json` beside the
checkpoint, or to a new path supplied with `--output`.

The output also compares fresh metrics with the checkpoint's saved reference.
Finite differences are reported with a match flag and are not silently discarded.

Segmentation uses Dice and NSD with a two-pixel tolerance; classification uses
Macro-F1 at threshold 0.5 over the view-eligible slots. Checkpoint selection uses
`(Dice% + NSD%)/2 + 100*Macro-F1`. Re-evaluating this split is not independent
test-set or external validation.

### 7. Aggregate a full sequence

```bash
bash scripts/05_aggregate.sh
```

The aggregator checks local checkpoint/protocol records and writes per-seed
values, stage means, sample standard deviations and adjacent-stage increments.
Missing configurations are marked incomplete instead of receiving a partial
three-seed mean. Run outputs remain in ignored local directories.

### Slurm

Activate the environment, prepare the inputs, and submit from the repository root:

```bash
mkdir -p logs
sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION --qos=YOUR_QOS \
  --chdir="$PWD" --export=ALL train/slurm_smoke.sh
# After all five smoke checks pass:
sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION --qos=YOUR_QOS \
  --chdir="$PWD" --export=ALL train/slurm_train.sh
```

The templates use one GPU per task and contain no institution-specific account,
partition or QOS. Array index = `stage*3 + seed_index`, with seed order
42, 123, 2026. Set concurrency, memory and time limits for your cluster.

## Repository structure

```text
VAST/
├── assets/                  # framework diagram shown above
├── citations/               # provisional manuscript BibTeX
├── configs/                 # portable local-path template
├── scripts/                 # prepare, smoke, train, evaluate and aggregate
├── train/                   # generic Slurm launchers
├── eval/                    # standalone checkpoint evaluation
├── VAST/                    # five-stage training and shared evaluation logic
│   └── runtime/             # model, dataset, augmentation, loss and metric modules
├── tests/                   # synthetic software tests
├── CITATION.cff
├── requirements.txt
├── README.md
└── README_zh-CN.md
```

## Data and checkpoints

Datasets, real split lists, sample IDs, experiment outputs and trained weights
are not distributed. Users obtain them under the original access terms and
configure local paths. Generated protocols, local configurations, runs,
predictions and results are excluded from Git.

## Implementation notes

The implementation applies feature perturbation to labeled/weak feature
branches. Strong spatial augmentations are sampled independently without
corresponding spatial warping of the weak segmentation target; this behavior
is preserved in the current training implementation. Method changes require
a new experiment version rather than reuse of an existing protocol.

Synthetic software tests can be run after installing the environment:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

These checks use synthetic tensors/records and do not require study data.

## Citation

If you use this work, please cite the following provisional manuscript entry.
Final publication metadata will replace it when available.

```bibtex
@unpublished{wangVAST,
  title  = {View-Aware Semi-Supervised Multi-Task Learning Framework for Fetal Cardiac Ultrasound},
  author = {Wang, Qiujin and Huang, Wenzheng and Yuan, Jiachen and Chen, Zhenzhong and Spinato, Silvia and Hernandez-Cruz, Netzahualcoyotl},
  note   = {Unpublished manuscript; publication details to be updated},
  url    = {https://github.com/Kelvin6657/VAST}
}
```

The same entry is provided in [citations/VAST.bib](citations/VAST.bib), with
machine-readable software/manuscript metadata in [CITATION.cff](CITATION.cff).

## Acknowledgements

This implementation builds on the FETUS baseline, an EchoCare-style SwinUNETR
backbone, MONAI and PyTorch. We thank the dataset and software maintainers.
Third-party dependencies retain their own licenses. A project-wide software
license has not yet been specified for this repository.
