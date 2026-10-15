"""``mine-hard-cases`` -- rank, cluster, auto-tag, and retrain on failure modes.

    aerialvit mine-hard-cases --config configs/ablate.yaml

Pipeline
--------
1. Score every **validation** frame with four uncertainty signals (predictive
   entropy, margin, MC-dropout variance) plus a kNN out-of-distribution distance
   against the *training* embeddings.
2. Take the top-K hardest validation frames and embed them with the frozen
   DINOv2 CLS token.
3. KMeans, with ``k`` chosen by silhouette score.
4. Auto-tag each cluster from measurable image/geometry statistics (glare,
   occlusion, small-object, altitude scale, haze/low contrast, density).
5. Assign every val+test frame to the nearest cluster centroid, then report
   per-cluster mIoU / landing-zone IoU / mAP@0.5 on the held-out **test** split
   next to the split average, plus the delta.
6. Retrain with the mined hard validation frames up-weighted and re-measure
   per-cluster on the same test split.

Mining on *validation* frames and measuring on the untouched *test* split is
deliberate: mining the test split and retraining on it would be label leakage.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import typer

from aerial_vit.checkpoint import load_checkpoint, load_checkpoint_model
from aerial_vit.commands._common import ConfigOpt, SetOpt, load_cli_config
from aerial_vit.config import FIGURES_DIR, RESULTS_DIR, Config
from aerial_vit.data.semantic_drone import build_datasets
from aerial_vit.data.transforms import denormalize
from aerial_vit.eval.hardcases import (
    cluster_embeddings,
    cluster_metric_table,
    scene_statistics,
    tag_clusters,
)
from aerial_vit.eval.metrics import (
    ConfusionMatrix,
    coco_map,
    landing_zone_counts,
    seg_metrics_from_confusion,
)
from aerial_vit.eval.uncertainty import (
    frame_uncertainty,
    knn_ood_scores,
    mc_dropout_stats,
    topk_uncertainty,
)
from aerial_vit.train_engine import make_loader
from aerial_vit.utils.logging import get_logger
from aerial_vit.utils.seed import device_name, resolve_device, seed_everything

app = typer.Typer(add_completion=False, help="Uncertainty-based hard-case mining and retraining.")
log = get_logger("aerialvit.hardcases")

OBJ_CLASSES = (3, 4, 6, 7)  # roof, tree, person, vehicle


# --------------------------------------------------------------------------- #
# feature + uncertainty collection
# --------------------------------------------------------------------------- #


@torch.no_grad()
def collect_features(model, loader, device) -> tuple[np.ndarray, list[str]]:
    """Frozen DINOv2 CLS embeddings for every frame."""
    model.eval()
    feats, names = [], []
    for images, _, _, metas in loader:
        f = model.backbone.cls_token(images.to(device))
        feats.append(f.float().cpu().numpy())
        names.extend(m.get("name", str(i)) for i, m in enumerate(metas))
    return (np.concatenate(feats) if feats else np.zeros((0, 0))), names


@torch.no_grad()
def collect_split(
    cfg: Config, model, ds, device, use_mc_dropout: bool = True
) -> list[dict[str, Any]]:
    """Per-frame posteriors, uncertainty, features and predictions for one split."""
    loader = make_loader(ds, 1, False, 0, cfg.run.seed)
    records: list[dict[str, Any]] = []
    for img_id, (images, masks, boxes, metas) in enumerate(loader):
        img = images.to(device)
        out = model(
            img,
            return_detections=model.box_head is not None,
            score_thresh=cfg.eval.score_thresh,
            nms_iou=cfg.eval.nms_iou,
            max_dets=cfg.eval.max_dets,
        )
        probs = out["seg_logits"][0].softmax(0).float().cpu().numpy()
        unc = frame_uncertainty(probs)
        if use_mc_dropout:
            mc = mc_dropout_stats(model, img, n_samples=cfg.eval.mc_dropout_samples)
            unc["mc_variance"] = float(mc["mc_variance_mean"][0])
        else:
            unc["mc_variance"] = 0.0
        det = out["detections"][0]
        rec = {
            "name": metas[0].get("name", str(img_id)),
            "split": metas[0].get("split", "?"),
            "probs": probs,
            "uncertainty": unc,
            "feature": model.backbone.cls_token(img)[0].float().cpu().numpy(),
            "image": denormalize(images[0]),
            "gt_mask": masks[0].numpy().astype(np.uint8),
            "gt_boxes": boxes[0][:, :4].numpy().astype(np.float32)
            if boxes[0].numel()
            else np.zeros((0, 4), np.float32),
            "gt_labels": boxes[0][:, 4].numpy().astype(int)
            if boxes[0].numel()
            else np.zeros((0,), int),
            "pred_mask": out["seg_logits"][0].argmax(0).cpu().numpy().astype(np.uint8),
            "det_pred": {
                "image_id": img_id,
                "boxes": det["boxes"].cpu().numpy(),
                "scores": det["scores"].cpu().numpy(),
                "labels": det["labels"].cpu().numpy().astype(int),
            },
            "det_gt": {
                "image_id": img_id,
                "width": int(masks.shape[-1]),
                "height": int(masks.shape[-2]),
                "boxes": boxes[0][:, :4].numpy().astype(np.float32)
                if boxes[0].numel()
                else np.zeros((0, 4), np.float32),
                "labels": boxes[0][:, 4].numpy().astype(int)
                if boxes[0].numel()
                else np.zeros((0,), int),
            },
        }
        records.append(rec)
    return records


def frame_metric(rec: dict[str, Any], cfg: Config) -> dict[str, float]:
    """Per-frame segmentation metrics (mAP is only meaningful over a set)."""
    cm = ConfusionMatrix(cfg.data.num_classes)
    cm.update(rec["pred_mask"], rec["gt_mask"])
    m = seg_metrics_from_confusion(cm, cfg.data.classes)
    c = landing_zone_counts(rec["pred_mask"], rec["gt_mask"], cfg.data.landable_indices)
    lz = c["tp"] / max(1, c["tp"] + c["fp"] + c["fn"])
    return {"miou": m["miou"], "landing_zone_iou": round(float(lz), 6)}


def subset_coco(records: list[dict[str, Any]], cfg: Config, prefix: str) -> dict[str, Any]:
    """COCO mAP computed over exactly these frames (valid for a cluster subset)."""
    if not records:
        return {"map_50": None, "map": None}
    gts, preds = [], []
    for i, r in enumerate(records):
        g = dict(r["det_gt"])
        g["image_id"] = i
        p = dict(r["det_pred"])
        p["image_id"] = i
        gts.append(g)
        preds.append(p)
    res = coco_map(
        preds,
        gts,
        cfg.data.classes,
        num_classes=cfg.data.num_classes,
        prefix=prefix,
        max_dets=cfg.eval.max_dets,
    )
    return {"map_50": res["map_50"], "map": res["map"]}


def subset_seg(records: list[dict[str, Any]], cfg: Config) -> dict[str, float]:
    cm = ConfusionMatrix(cfg.data.num_classes)
    lz = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for r in records:
        cm.update(r["pred_mask"], r["gt_mask"])
        c = landing_zone_counts(r["pred_mask"], r["gt_mask"], cfg.data.landable_indices)
        for k in lz:
            lz[k] += c[k]
    m = seg_metrics_from_confusion(cm, cfg.data.classes)
    tp, fp, fn = lz["tp"], lz["fp"], lz["fn"]
    return {
        "miou": m["miou"],
        "pixel_accuracy": m["pixel_accuracy"],
        "landing_zone_iou": round(tp / max(1, tp + fp + fn), 6),
    }


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #


def figure_clusters(features, labels, k_scores, path, title: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.asarray(features, np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    pcs = x @ vt[:2].T if vt.shape[0] >= 2 else np.stack([x[:, 0], np.zeros(len(x))], axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), dpi=140)
    labels = np.asarray(labels)
    cmap = plt.get_cmap("tab10")
    for cid in sorted(set(labels.tolist())):
        sel = labels == cid
        axes[0].scatter(
            pcs[sel, 0], pcs[sel, 1], s=34, color=cmap(cid % 10), label=f"cluster {cid}", alpha=0.85
        )
    axes[0].set_xlabel("PC 1")
    axes[0].set_ylabel("PC 2")
    axes[0].set_title(title)
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.25)

    if k_scores:
        ks = sorted(k_scores)
        axes[1].plot(ks, [k_scores[k] for k in ks], marker="o", color="#1f77b4")
        best_k = max(k_scores, key=lambda kk: k_scores[kk])
        axes[1].scatter(
            [best_k],
            [k_scores[best_k]],
            s=90,
            facecolor="none",
            edgecolor="red",
            zorder=5,
            label=f"chosen k={best_k}",
        )
        axes[1].set_xlabel("k")
        axes[1].set_ylabel("silhouette")
        axes[1].set_title("k selection")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.25)
    else:
        axes[1].axis("off")
    fig.tight_layout()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p)
    plt.close(fig)
    return p


def figure_montage(records: list[dict[str, Any]], cfg: Config, path: Path, cols: int = 4) -> Path:
    from aerial_vit.utils.viz import draw_boxes, label_panel, montage, overlay_mask, save_image

    panels = []
    for r in records:
        left = overlay_mask(r["image"], r["gt_mask"], 0.45)
        right = overlay_mask(r["image"], r["pred_mask"], 0.45)
        if r["gt_boxes"].shape[0]:
            left = draw_boxes(left, r["gt_boxes"], r["gt_labels"])
        d = r["det_pred"]
        if len(d["boxes"]):
            right = draw_boxes(right, d["boxes"][:12], d["labels"][:12], d["scores"][:12])
        panels.append(label_panel(left, f"{r['name']} GT"))
        panels.append(label_panel(right, "pred"))
    img = montage(panels, cols=cols * 2, pad=5)
    save_image(path, img)
    return path


# --------------------------------------------------------------------------- #
# main routine
# --------------------------------------------------------------------------- #


def run_mining(
    cfg: Config,
    checkpoint: str | None = None,
    top_k: int | None = None,
    retrain: bool = True,
) -> dict[str, Any]:
    seed_everything(cfg.run.seed, cfg.run.deterministic)
    device = resolve_device(cfg.run.device)
    top_k = int(top_k or cfg.eval.top_k_hard)

    ckpt = checkpoint or str(Path(cfg.run.out_dir) / "runs" / cfg.run.name / "best.pt")
    model, info, ckpt_data = load_checkpoint_model(ckpt, cfg=cfg, device=device)
    log.info("loaded %s (mode=%s) on %s", ckpt, ckpt_data.get("mode"), device_name(device))

    train_ds, val_ds, test_ds = build_datasets(cfg, label_fraction=1.0, seed=cfg.run.seed)

    log.info("extracting train embeddings for kNN OOD (%d frames)", len(train_ds))
    train_feats, _ = collect_features(
        model, make_loader(train_ds, 8, False, 0, cfg.run.seed), device
    )

    log.info("scoring validation split (%d frames)", len(val_ds))
    val_recs = collect_split(cfg, model, val_ds, device)
    log.info("scoring test split (%d frames)", len(test_ds))
    test_recs = collect_split(cfg, model, test_ds, device)

    # --- kNN OOD against the training manifold ----------------------------- #
    for recs in (val_recs, test_recs):
        feats = np.stack([r["feature"] for r in recs])
        ood = knn_ood_scores(train_feats, feats, k=5)
        for r, o in zip(recs, ood, strict=False):
            r["uncertainty"]["knn_ood"] = float(o)

    # --- rank -------------------------------------------------------------- #
    val_scores = {r["name"]: r["uncertainty"]["entropy_mean"] for r in val_recs}
    ranked = topk_uncertainty(val_scores, k=len(val_scores))
    hard_names = [n for n, _ in ranked[:top_k]]
    by_name = {r["name"]: r for r in val_recs}
    hard_recs = [by_name[n] for n in hard_names if n in by_name]

    # --- cluster ----------------------------------------------------------- #
    hard_feats = np.stack([r["feature"] for r in hard_recs]) if hard_recs else np.zeros((0, 0))
    clustered = cluster_embeddings(hard_feats, tuple(cfg.eval.cluster_k_range), seed=cfg.run.seed)
    hard_labels = clustered["labels"]
    log.info(
        "clustered %d hard frames into k=%s (silhouette %s)",
        len(hard_recs),
        clustered["k"],
        clustered["silhouette"],
    )

    # --- scene statistics + auto tags ------------------------------------- #
    stats = []
    for r in hard_recs:
        st = scene_statistics(r["image"], r["gt_boxes"], r["gt_mask"], r["probs"])
        st["name"] = r["name"]
        stats.append(st)
    areas = np.array([s["median_instance_area"] for s in stats], np.float64)
    med = float(np.median(areas[areas > 0])) if (areas > 0).any() else 1.0
    for s in stats:
        s["altitude_ratio"] = float(s["median_instance_area"] / med) if med > 0 else 1.0
    cluster_info = tag_clusters(stats, hard_labels, top_n=2)
    for cid, info_c in cluster_info.items():
        info_c["frames"] = [
            r["name"] for r, lb in zip(hard_recs, hard_labels, strict=False) if int(lb) == cid
        ]

    # --- assign every val+test frame to the nearest hard-case centroid ---- #
    assignments: dict[str, int] = {}
    if len(hard_feats) and clustered["k"] >= 1:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=clustered["k"], n_init=10, random_state=cfg.run.seed).fit(hard_feats)
        for recs in (val_recs, test_recs):
            f = np.stack([r["feature"] for r in recs])
            lab = km.predict(f)
            for r, lb in zip(recs, lab, strict=False):
                assignments[r["name"]] = int(lb)
        # the mined frames themselves must keep the cluster they were fit in
        for r, lb in zip(hard_recs, hard_labels, strict=False):
            assignments[r["name"]] = int(lb)

    for cid in cluster_info:
        cluster_info[cid]["test_frames"] = [
            r["name"] for r in test_recs if assignments.get(r["name"]) == cid
        ]

    # --- per-cluster metrics on the held-out test split -------------------- #
    for r in test_recs:
        r["metrics"] = frame_metric(r, cfg)
    split_test = subset_seg(test_recs, cfg)
    split_coco = subset_coco(test_recs, cfg, "hardcases_split")
    split_metrics = {
        "miou": split_test["miou"],
        "landing_zone_iou": split_test["landing_zone_iou"],
        "map_50": split_coco["map_50"],
    }

    # cluster_metric_table aggregates frame-level metrics; COCO mAP needs a set
    # of frames, so it is filled in per cluster afterwards.
    per_frame_metrics = {r["name"]: r["metrics"] for r in test_recs}
    per_cluster_rows = cluster_metric_table(
        cluster_info, per_frame_metrics, split_metrics, metric_keys=("miou", "landing_zone_iou")
    )
    for row in per_cluster_rows:
        cid = row["cluster"]
        names = set(cluster_info[cid].get("test_frames", []))
        row["n_frames"] = len(names)
        members = [r for r in test_recs if r["name"] in names]
        det = subset_coco(members, cfg, f"hardcases_c{cid}") if members else {"map_50": None}
        row["map_50"] = det.get("map_50")
        row["split_map_50"] = split_metrics["map_50"]
        row["delta_map_50"] = (
            None
            if (det.get("map_50") is None or split_metrics["map_50"] is None)
            else round(det["map_50"] - split_metrics["map_50"], 6)
        )

    write_cluster_csv(per_cluster_rows)

    # --- figures ----------------------------------------------------------- #
    fig_paths: list[str] = []
    if len(hard_feats):
        p = figure_clusters(
            hard_feats,
            hard_labels,
            clustered.get("k_scores", {}),
            FIGURES_DIR / "hard_case_clusters.png",
            f"hard-case clusters (k={clustered['k']}, silhouette={clustered['silhouette']})",
        )
        fig_paths.append(str(p))
        reps = []
        for cid in sorted(cluster_info):
            members = [r for r in hard_recs if assignments.get(r["name"]) == cid]
            reps.extend(members[:2])
        if reps:
            fig_paths.append(str(figure_montage(reps, cfg, FIGURES_DIR / "hard_case_montage.png")))

    # --- retrain on the mined hard set ------------------------------------ #
    retrain_info = {"performed": False}
    if retrain and hard_recs:
        retrain_info = run_hardcase_retrain(
            cfg, ckpt, hard_names, test_recs, assignments, cluster_info, split_metrics
        )

    result = {
        "checkpoint": ckpt,
        "mode": ckpt_data.get("mode"),
        "device": device_name(device),
        "n_val_frames": len(val_recs),
        "n_test_frames": len(test_recs),
        "top_k": top_k,
        "ranking_key": "entropy_mean",
        "hard_frames": hard_names,
        "hard_frame_uncertainty": {
            n: by_name[n]["uncertainty"] for n in hard_names if n in by_name
        },
        "uncertainty_summary": summarize_uncertainty(val_recs, test_recs),
        "clustering": {
            "k": clustered["k"],
            "silhouette": clustered["silhouette"],
            "k_scores": clustered.get("k_scores", {}),
            "labels": [int(v) for v in hard_labels],
        },
        "clusters": {str(k): v for k, v in cluster_info.items()},
        "per_cluster": per_cluster_rows,
        "split_metrics": split_metrics,
        "figures": fig_paths,
        "retrain": retrain_info,
        "config": {
            "backbone": cfg.backbone.name,
            "seg_head": cfg.model.seg_head,
            "mode": ckpt_data.get("mode"),
            "mc_dropout_samples": cfg.eval.mc_dropout_samples,
            "cluster_k_range": list(cfg.eval.cluster_k_range),
            "seed": cfg.run.seed,
        },
    }
    out = RESULTS_DIR / "hard_cases.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=float) + "\n")
    log.info("wrote %s", out)
    return result


def summarize_uncertainty(val_recs, test_recs) -> dict[str, Any]:
    def agg(recs, key):
        vals = np.array([r["uncertainty"].get(key, np.nan) for r in recs], np.float64)
        vals = vals[~np.isnan(vals)]
        return (
            {"mean": round(float(vals.mean()), 6), "p95": round(float(np.percentile(vals, 95)), 6)}
            if vals.size
            else {}
        )

    out: dict[str, Any] = {}
    for split, recs in (("val", val_recs), ("test", test_recs)):
        out[split] = {
            k: agg(recs, k)
            for k in ("entropy_mean", "margin_mean", "confidence_mean", "mc_variance", "knn_ood")
        }
    return out


def write_cluster_csv(rows: list[dict[str, Any]]) -> Path:
    import csv

    fields = [
        "cluster",
        "tag",
        "tags",
        "n_frames",
        "miou",
        "split_miou",
        "delta_miou",
        "landing_zone_iou",
        "split_landing_zone_iou",
        "delta_landing_zone_iou",
        "map_50",
        "split_map_50",
        "delta_map_50",
    ]
    p = RESULTS_DIR / "per_cluster_metrics.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    return p


# --------------------------------------------------------------------------- #
# targeted retrain
# --------------------------------------------------------------------------- #


def _train_with_names(
    cfg: Config,
    checkpoint: str,
    train_names: list[str],
    run_name: str,
    root: Path,
    backup: dict,
    mode: str | None = None,
):
    """Warm-start from ``checkpoint`` and train on exactly ``train_names``.

    The split file is patched for the duration of the call and always restored,
    so a crash cannot leave the repository with a corrupted split.
    """
    from aerial_vit.data.semantic_drone import write_splits
    from aerial_vit.train_engine import train_model

    c = copy.deepcopy(cfg)
    c.run.name = run_name
    if mode:
        # keep the adaptation mode of the checkpoint we are warm-starting from,
        # otherwise the "retrain" would silently change what is being adapted
        c.train.mode = mode
    run_dir = Path(c.run.out_dir) / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(c.run.device)

    try:
        patched = dict(backup)
        patched["train"] = list(train_names)
        write_splits(root, patched)
        train_ds, val_ds, _ = build_datasets(c, label_fraction=1.0, seed=c.run.seed)
        tr = make_loader(train_ds, c.train.batch_size, True, c.data.num_workers, c.run.seed)
        va = make_loader(val_ds, c.eval.batch_size, False, c.data.num_workers, c.run.seed)
        model, _, _ = load_checkpoint_model(checkpoint, cfg=c, mode=c.train.mode, device=device)
        summary = train_model(model, tr, va, c, device, run_dir, c.train.mode)
    finally:
        write_splits(root, backup)
    return model, summary


def run_hardcase_retrain(
    cfg: Config,
    checkpoint: str,
    hard_names: list[str],
    test_recs: list[dict[str, Any]],
    assignments: dict[str, int],
    cluster_info: dict[int, dict[str, Any]],
    split_metrics: dict[str, float],
) -> dict[str, Any]:
    """Retrain with the mined hard frames up-weighted, **with a control run**.

    Two runs are trained identically -- same warm-start checkpoint, same number
    of epochs, same base training set -- except that one additionally sees the
    mined hard frames repeated ``repeat`` times.  Reporting the control is what
    separates "the hard examples helped" from "training for longer helped",
    which a naive before/after comparison would silently conflate.

    The test split is never trained on: the hard frames come from *validation*,
    and every ``after`` number is re-measured on the untouched test split.
    """
    repeat = 3
    manifest = json.loads((Path(cfg.run.out_dir) / "data_manifest.json").read_text())
    base_train = list(manifest["splits"]["train"])
    val_names = set(manifest["splits"]["val"])
    extra = [n for n in hard_names if n in val_names]
    augmented = base_train + extra * repeat
    root = Path(manifest["root"])
    backup = json.loads((root / "splits.json").read_text())

    log.info(
        "retrain: %d base train frames (+ %d hard val frames x%d = %d) with a control run",
        len(base_train),
        len(extra),
        repeat,
        len(augmented),
    )

    ckpt_mode = load_checkpoint(checkpoint).get("mode") or cfg.train.mode
    log.info("retraining in adaptation mode %r (from the checkpoint)", ckpt_mode)
    _, control_summary = _train_with_names(
        cfg, checkpoint, base_train, f"{cfg.run.name}_hardctl", root, backup, mode=ckpt_mode
    )
    _, hard_summary = _train_with_names(
        cfg, checkpoint, augmented, f"{cfg.run.name}_hardretrain", root, backup, mode=ckpt_mode
    )

    device = resolve_device(cfg.run.device)
    _, _, test_ds = build_datasets(cfg, label_fraction=1.0, seed=cfg.run.seed)
    control_model, _, _ = load_checkpoint_model(
        control_summary["checkpoint"], cfg=cfg, device=device
    )
    control_recs = collect_split(cfg, control_model, test_ds, device)
    hard_model, _, _ = load_checkpoint_model(hard_summary["checkpoint"], cfg=cfg, device=device)
    after_recs = collect_split(cfg, hard_model, test_ds, device)

    def _overall(recs, prefix):
        seg = subset_seg(recs, cfg)
        det = subset_coco(recs, cfg, prefix)
        return {
            "miou": seg["miou"],
            "landing_zone_iou": seg["landing_zone_iou"],
            "map_50": det["map_50"],
        }

    def _delta(a, b):
        keys = ("miou", "map_50", "landing_zone_iou")
        return {
            k: (None if (a.get(k) is None or b.get(k) is None) else round(a[k] - b[k], 6))
            for k in keys
        }

    before = _overall(test_recs, "hardretrain_before")
    control = _overall(control_recs, "hardretrain_control")
    after = _overall(after_recs, "hardretrain_after")

    per_cluster = []
    for cid in sorted(cluster_info):
        names = set(cluster_info[cid].get("test_frames", []))
        bx = [r for r in test_recs if r["name"] in names]
        cx = [r for r in control_recs if r["name"] in names]
        ax = [r for r in after_recs if r["name"] in names]
        b_seg = subset_seg(bx, cfg) if bx else {"miou": None}
        c_seg = subset_seg(cx, cfg) if cx else {"miou": None}
        a_seg = subset_seg(ax, cfg) if ax else {"miou": None}
        b_map = subset_coco(bx, cfg, f"hc_before_c{cid}").get("map_50") if bx else None
        c_map = subset_coco(cx, cfg, f"hc_control_c{cid}").get("map_50") if cx else None
        a_map = subset_coco(ax, cfg, f"hc_after_c{cid}").get("map_50") if ax else None
        per_cluster.append(
            {
                "cluster": cid,
                "tag": cluster_info[cid]["tag"],
                "n_frames": len(ax),
                "miou_before": b_seg.get("miou"),
                "miou_control": c_seg.get("miou"),
                "miou_after": a_seg.get("miou"),
                "delta_miou": None
                if (b_seg.get("miou") is None or a_seg.get("miou") is None)
                else round(a_seg["miou"] - b_seg["miou"], 6),
                "delta_miou_vs_control": None
                if (c_seg.get("miou") is None or a_seg.get("miou") is None)
                else round(a_seg["miou"] - c_seg["miou"], 6),
                "map_50_before": b_map,
                "map_50_control": c_map,
                "map_50_after": a_map,
                "delta_map_50": None
                if (b_map is None or a_map is None)
                else round(a_map - b_map, 6),
                "delta_map_50_vs_control": None
                if (c_map is None or a_map is None)
                else round(a_map - c_map, 6),
            }
        )

    result = {
        "performed": True,
        "repeat": repeat,
        "n_hard_frames_added": len(extra),
        "base_train_size": len(base_train),
        "augmented_train_size": len(augmented),
        "mode": ckpt_mode,
        "epochs_per_run": cfg.train.epochs,
        "train_seconds_control": control_summary["train_seconds"],
        "train_seconds_hard": hard_summary["train_seconds"],
        "trainable_params": int(sum(p.numel() for p in hard_model.parameters() if p.requires_grad)),
        "before": before,
        "control": control,
        "after": after,
        "delta": _delta(after, before),
        "delta_vs_control": _delta(after, control),
        "per_cluster": per_cluster,
        "test_split_unchanged": True,
        "note": (
            "hard frames are mined from the validation split; the test split is never used for "
            "training. 'control' is the same warm-start + same epochs trained WITHOUT the hard "
            "frames, so delta_vs_control isolates the effect of the hard examples themselves."
        ),
    }
    out = RESULTS_DIR / "hardcase_retrain.json"
    out.write_text(json.dumps(result, indent=2, default=float) + "\n")
    log.info(
        "retrain: control mIoU=%s, hard-frame run mIoU=%s, delta_vs_control=%s",
        control["miou"],
        after["miou"],
        result["delta_vs_control"]["miou"],
    )
    _ = (assignments, split_metrics)
    return result


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


@app.command()
def main(
    config: str = ConfigOpt,
    checkpoint: str = typer.Option(None, "--checkpoint", help="Checkpoint to mine with."),
    top_k: int = typer.Option(None, "--top-k", help="Number of hardest frames to keep."),
    run_name: str = typer.Option(
        None, "--run-name", help="Model run to mine with (default: config)."
    ),
    no_retrain: bool = typer.Option(False, "--no-retrain", help="Skip the targeted retrain."),
    set_: list[str] = SetOpt,
) -> None:
    """Mine, cluster, tag and (optionally) retrain on hard cases."""
    cfg = load_cli_config(config, set_)
    if run_name:
        cfg.run.name = run_name
    res = run_mining(cfg, checkpoint, top_k, retrain=not no_retrain)
    typer.echo(f"hard frames    : {len(res['hard_frames'])} (top_k={res['top_k']})")
    typer.echo(
        f"clusters       : k={res['clustering']['k']} silhouette={res['clustering']['silhouette']}"
    )
    for r in res["per_cluster"]:
        typer.echo(
            f"  cluster {r['cluster']:<2} {r['tag']:<28} test n={r['n_frames']:<3} "
            f"mIoU={r['miou']} (split {r['split_miou']}, Δ{r['delta_miou']})  mAP@0.5={r['map_50']} (Δ{r['delta_map_50']})"
        )
    rt = res.get("retrain", {})
    if rt.get("performed"):
        typer.echo(
            f"retrain        : before mIoU {rt['before']['miou']} -> control {rt['control']['miou']} "
            f"-> +hard-frames {rt['after']['miou']}  (Δ vs control {rt['delta_vs_control']['miou']})"
        )
    typer.echo(f"figures        : {', '.join(res['figures'])}")


if __name__ == "__main__":  # pragma: no cover
    app()
