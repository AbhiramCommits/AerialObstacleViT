# AerialObstacleViT

**Obstacle and landing-zone perception for low-altitude drone delivery — a
perception benchmark for frozen foundation features versus open-vocabulary
vision-language models.**

A delivery UAV descending into an urban or semi-urban scene has to answer two
questions at once, from a single nadir or near-nadir frame: *what is down there
that I must not hit* (roofs, trees, water, people, vehicles) and *where can I
actually put down* (flat, clear, obstacle-free ground). That is a dense
prediction problem on imagery whose viewpoint — straight down, from 30–120 m —
almost never appears in the web-scale datasets that pretrained vision models
learn from. This repository asks whether that mismatch can be absorbed by a
*frozen* foundation backbone and a cheap task head, or whether a generic
open-vocabulary vision-language model can already answer the question zero-shot.

**Research question.** For obstacle and landing-zone perception on drone imagery,
how do frozen self-supervised ViT foundation-model features (DINOv2) compare to
open-vocabulary VLM zero-shot detection (OWL-ViT / CLIP), and how does that
trade-off change under label scarcity?

The benchmark therefore measures three things on one shared taxonomy, one shared
train/val/test split and one shared metric implementation:

1. **Task-trained DINOv2.** `facebook/dinov2-small` frozen, with a linear probe,
   a small conv decoder and an anchor-free FCOS-lite box head trained on top.
2. **Open-vocabulary zero-shot.** `google/owlvit-base-patch32` and a
   `openai/clip-vit-base-patch32` dense sliding-window classifier, driven only by
   text prompts — no training, no prompt tuning.
3. **The label-scarcity axis.** The same adaptation modes swept across 1 %, 5 %,
   10 %, 25 % and 100 % of the training labels, because the practical question on
   a new airframe and a new city is not "which model is best" but "which model is
   best when I have forty labelled frames".

Everything beyond that — hard-case mining, failure-mode clustering, ONNX export
and latency — exists to answer the follow-up question: *when it fails, how does
it fail, and can it still run on board?*

---

## 1. Architecture

```
                      ┌─────────────────────────────┐
                      │  Semantic Drone Dataset      │   (real, if reachable)
                      │  ─ or ─                      │
   scripts/fetch_data │  synthetic nadir generator   │   (deterministic fallback)
   ─────────────────► │  + canonical 8-class remap   │
                      └──────────────┬───────────────┘
                                     │  image / mask / boxes / landing-zone
                                     ▼
                      ┌─────────────────────────────┐
                      │  frozen DINOv2-small  ViT    │   22.1 M params, 14 px patch
                      │  forward_features / cls_token│   0 trainable in probe mode
                      └───────┬─────────────┬───────┘
                              │             │
              patch tokens    │             │   CLS token (embeddings)
                              ▼             ▼
        ┌───────────────────────────┐   ┌──────────────────────────────┐
        │  task heads               │   │  hard-case mining (Step 5)    │
        │   • LinearSegHead (probe) │   │   entropy / margin / MC-drop  │
        │   • ConvSegHead (decoder) │   │   kNN OOD vs train embeddings │
        │   • BoxHead (FCOS-lite)   │   │   KMeans + auto-tagging       │
        └──────────┬────────────────┘   └──────────────────────────────┘
                   │
                   │        ┌──────────────────────────────────────────┐
                   │        │  zero-shot branch (Step 3)                │
                   │        │   OWL-ViT: text queries -> boxes          │
                   │        │   CLIP: sliding window -> dense labels    │
                   │        │   prompt ensembling over templates        │
                   │        └──────────────────┬───────────────────────┘
                   ▼                           ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  eval/metrics.py   mIoU · per-class IoU · pixel acc           │
        │                    landing-zone IoU/F1 · COCO mAP@0.5/[.5:.95]│
        │                    rare-class mAP (measured, not assumed)     │
        └──────────────────────────┬───────────────────────────────────┘
                                   ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  export + latency (Step 6)                                    │
        │   TorchScript · ONNX opset 17 (dynamic batch) · parity checks  │
        │   eager vs TorchScript vs ONNX Runtime, batch 1 & 8, 2 sizes   │
        │   -> onboard feasibility vs a stated 10 Hz requirement         │
        └──────────────────────────────────────────────────────────────┘
```

### Module layout

| Path | Responsibility |
|---|---|
| `aerial_vit/config.py` | Dataclass configs (`RunCfg`, `DataCfg`, `BackboneCfg`, `ModelCfg`, `TrainCfg`, `EvalCfg`, `AblationCfg`) loaded from `configs/*.yaml` |
| `aerial_vit/data/synthetic.py` | Deterministic procedural nadir-scene generator with pixel-perfect masks, instance boxes, occlusion statistics, and glare/motion-blur/haze/altitude degradations |
| `aerial_vit/data/semantic_drone.py` | Canonical-layout dataset, 20+ class → 8-class taxonomy remap, landing-zone derivation, data manifest, stratified label sub-sampling |
| `aerial_vit/data/transforms.py` | Resize/crop/normalise (ImageNet stats) plus flip, colour jitter and random-scale (altitude) augmentation, box-aware |
| `aerial_vit/models/dinov2_backbone.py` | Frozen DINOv2 wrapper: `forward_features`, `cls_token`, `freeze`/`unfreeze`, trainable-parameter accounting, no-download presets |
| `aerial_vit/models/heads.py` | `LinearSegHead` (probe), `ConvSegHead` (decoder), `BoxHead` (FCOS-lite), `AerialPerceptionModel`, segmentation and detection losses |
| `aerial_vit/models/lora.py` | Native `LoRALinear` + PEFT backend, injection/merge, trainable-vs-total accounting |
| `aerial_vit/models/openvocab.py` | OWL-ViT and CLIP zero-shot wrappers with multi-template prompt ensembling and canonical label mapping |
| `aerial_vit/eval/metrics.py` | Confusion-matrix mIoU, landing-zone IoU/F1, COCO mAP via pycocotools, rare-class restriction |
| `aerial_vit/eval/uncertainty.py` | Predictive entropy, margin, MC-dropout variance, kNN OOD distance, stable top-K ranking |
| `aerial_vit/eval/hardcases.py` | Scene statistics, silhouette-selected KMeans, failure-mode auto-tagging, per-cluster metric deltas |
| `aerial_vit/train_engine.py` | One training loop + one evaluation path shared by `train`, `eval`, `ablate` and hard-case mining |
| `aerial_vit/ablation.py` / `aerial_vit/reporting.py` | Sweep grid + aggregation + figures; report tables rendered from artifacts only |
| `aerial_vit/export.py` / `aerial_vit/benchmark.py` | TorchScript/ONNX export with parity checks; warmup-corrected p50/p95 latency benchmarking |
| `scripts/*.py` | Thin Typer shims — `python scripts/train.py …` is the same code path as `aerialvit train …` |

---

## 2. How to run

```bash
# 1. environment (Python 3.11)
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev]"

# 2. data: the real Semantic Drone Dataset if reachable, else the deterministic
#    synthetic fallback. The active source is recorded in artifacts/data_manifest.json
python scripts/fetch_data.py --config configs/full.yaml --force
# or point at a local copy / URL:
#   python scripts/fetch_data.py --config configs/full.yaml --src /path/to/semantic_drone

# 3. train / evaluate the task model
aerialvit train --config configs/full.yaml --mode linear --run-name full_linear
aerialvit eval  --config configs/full.yaml --run-name full_linear

# 4. open-vocabulary zero-shot baselines + comparison table
aerialvit zeroshot --config configs/full.yaml --method both --task-run full_linear

# 5. label-scarcity ablation (linear / LoRA r4 / LoRA r16 / full x 1..100 % labels)
aerialvit ablate --config configs/ablate.yaml

# 6. hard-case mining, failure-mode clustering, targeted retrain
aerialvit mine-hard-cases --config configs/ablate.yaml --run-name full_linear

# 7. export + latency
aerialvit export       --config configs/full.yaml --run-name full_linear
aerialvit bench-latency --config configs/full.yaml --run-name full_linear --iterations 50

# 8. tests, coverage, and the report tables
aerialvit quality
aerialvit make-report

# everything, in order, from scratch:
bash scripts/run_all.sh
```

Adapters: `--mode linear` (frozen backbone, linear probe), `--mode lora_r16`
(frozen backbone + rank-16 LoRA on q/k/v/o projections), `--mode full`
(everything trainable). `--label-fraction 0.05` trains on 5 % of the training
split, stratified so rare classes survive.

### Docker (CPU-only image)

```bash
docker build -t aerialvit .
docker run --rm aerialvit --help
docker run --rm -v "$PWD/artifacts:/app/artifacts" aerialvit \
    fetch-data --config configs/smoke.yaml
```

### What was actually run for the numbers below

{{table:provenance}}

| | |
|---|---|
| device used for every measured number | `{{provenance.device}}` |
| python / torch | `{{provenance.python}}` / `{{provenance.torch}}` |
| commit at generation time | `{{provenance.git_commit}}` |
| generated | `{{provenance.date}}` |

> If the Semantic Drone Dataset could not be reached from the build machine, the
> manifest records `source: synthetic` and every number below is measured on the
> deterministic procedural fallback. **No number in this README is estimated.**

---

## 3. Results

All tables in this section are rendered from `artifacts/` by
`python scripts/make_report.py` into `reports/tables/*.md`, and are included here
verbatim. [reports/RESULTS.md](reports/RESULTS.md) contains the same content.

{{table:headline}}

### 3.1 Foundation features vs open-vocabulary VLMs

{{table:foundation_vs_tasktrained}}

### 3.2 Label scarcity: linear probe vs LoRA vs full fine-tune

{{table:ablation_miou}}

Landing-zone quality under the same sweep:

{{table:ablation_landing_zone}}

Detection mAP@0.5:

{{table:ablation_map50}}

Adaptation cost (the reason any of this matters on a drone):

{{table:ablation_params}}

![mIoU vs label budget](reports/figures/ablation_miou_vs_labels.png)

![trainable parameters vs mIoU](reports/figures/params_vs_miou.png)

### 3.3 Hard cases and failure modes

{{table:hard_cases}}

![hard-case clusters](reports/figures/hard_case_clusters.png)

Representative hardest frames -- ground truth on the left, predictions on the right:

![hard-case montage](reports/figures/hard_case_montage.png)

Targeted retrain on the mined hard set:

{{table:hardcase_retrain}}

### 3.4 Latency and onboard feasibility

{{table:latency}}

![latency](reports/figures/latency.png)

### 3.5 Test suite

{{table:quality}}

---

## 4. Limitations and next steps

**Honest scope.** This is a single-machine benchmark, and the numbers above were
measured on the device named in the provenance table — one consumer SoC, CPU and
Apple MPS, **no CUDA GPU and no embedded target**. Latency figures in particular
will not transfer unchanged to a Jetson or a Qualcomm flight controller.

**Dataset scale.** The repository targets the Semantic Drone Dataset, which is a
few hundred high-resolution nadir frames from a handful of European cities. If
that dataset was not reachable during the build, `artifacts/data_manifest.json`
records `source: synthetic`, and the results describe a procedural stand-in:
geometrically faithful (pixel-perfect masks, real occlusion statistics, explicit
glare/blur/haze/altitude variation) but **not a substitute for real
imagery**. Conclusions about absolute accuracy should be treated as
domain-shaped, not domain-validated.

**Detection is the weak axis, by construction.** The smallest canonical classes —
`person` and `vehicle` — are on the order of 6–15 px at 224 px input, which is at
or below the 14 px ViT patch and only marginally above the stride-7 detection
level. The per-class AP breakdown in the artifact JSON shows this directly, and
Section 3.3 quantifies it as a named failure cluster rather than hiding it in an
average.

**What is deliberately not modelled.**

* No temporal or multi-view fusion — every frame is scored independently, so a
  video stream gets no benefit from motion parallax or temporal smoothing.
* No real flight data, no GPS/IMU, no stable-camera assumptions; the synthetic
  altitude-shift augmentation is a crude proxy for ground-sample-distance change.
* No depth, no LiDAR, no terrain model — "landable" is inferred from a single
  image plus a class taxonomy, which cannot detect a low-contrast wire, a
  shallow puddle, or a slope.
* Single-node, single-process training; no distributed or sharded runs.

**Next steps that the code is already shaped for.**

1. Point `--src` at the real dataset (or any aerial segmentation set with a
   colour-coded `class_dict.csv`) and re-run `scripts/run_all.sh` — the taxonomy
   remap in `configs/taxonomy.yaml` is the only thing that needs editing.
2. Add a feature-pyramid level (stride 3.5) or a tiled high-resolution inference
   mode to attack the small-object cluster that Section 3.3 measures.
3. Swap the frozen backbone for a satellite/aerial-pretrained checkpoint
   (e.g. a SatMAE- or DOFA-style encoder) to test whether the frozen-feature
   advantage is about self-supervision or about *domain-matched* self-supervision.
4. Distill the best adaptation mode into the ONNX graph and re-run
   `bench-latency` against an embedded target with a real 10 Hz budget.
5. Calibrate the uncertainty scores (temperature scaling on the validation
   split) so the entropy ranking in `mine-hard-cases` becomes a usable
   abstention signal rather than only a mining heuristic.

---

*All numbers reproduced by `scripts/run_all.sh` on {{provenance.device}}, {{provenance.date}}.
Tables are generated from `artifacts/` by `scripts/make_report.py`; this README is
assembled from those same table files by `scripts/build_readme.py`, so the two
cannot drift apart. See [LICENSE](LICENSE) (MIT).*
