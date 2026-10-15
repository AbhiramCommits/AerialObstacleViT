"""Per-frame uncertainty and out-of-distribution scores.

Four complementary signals, because they fail differently:

``predictive entropy``   ``H(p)`` over the dense class posterior -- peaks when the
                         model is torn between classes (usually at object
                         boundaries and on rare classes).
``margin``               ``p(top1) - p(top2)`` -- small when two classes compete.
``MC-dropout variance``  variance of the *mean* class probability over ``N``
                         stochastic forward passes (dropout kept active in the
                         heads only).  Captures model uncertainty rather than
                         data uncertainty.
``kNN OOD distance``     distance from the frame's frozen DINOv2 CLS embedding to
                         its k-th nearest *training* embedding.  A training-free
                         signal that flags frames from outside the training
                         manifold -- the honest way to ask "is this scene even
                         represented in my labels?".

All functions are pure and take already-computed posteriors/features, so they are
cheap to unit test.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = [
    "predictive_entropy",
    "normalized_entropy",
    "margin",
    "frame_uncertainty",
    "mc_dropout_stats",
    "knn_ood_scores",
    "topk_uncertainty",
]

EPS = 1e-12


# --------------------------------------------------------------------------- #
# per-pixel / per-frame scores
# --------------------------------------------------------------------------- #


def predictive_entropy(probs: np.ndarray | torch.Tensor, dim: int = 0) -> np.ndarray | torch.Tensor:
    """Shannon entropy along ``dim`` (natural log)."""
    if isinstance(probs, torch.Tensor):
        return -(probs * (probs + EPS).log()).sum(dim=dim)
    p = np.asarray(probs, np.float64)
    return -(p * np.log(p + EPS)).sum(axis=dim)


def normalized_entropy(probs: np.ndarray | torch.Tensor, dim: int = 0):
    """Entropy divided by ``log(C)`` so the score is in ``[0, 1]``."""
    c = probs.shape[dim]
    h = predictive_entropy(probs, dim=dim)
    return h / max(EPS, float(np.log(c)) if c > 1 else 1.0)


def margin(probs: np.ndarray | torch.Tensor, dim: int = 0):
    """Top-1 minus top-2 posterior along ``dim``."""
    if isinstance(probs, torch.Tensor):
        top2 = probs.topk(2, dim=dim).values
        return top2.select(dim, 0) - top2.select(dim, 1)
    p = np.asarray(probs, np.float64)
    s = np.sort(p, axis=dim)
    hi = np.take(s, -1, axis=dim)
    lo = np.take(s, -2, axis=dim)
    return hi - lo


def frame_uncertainty(probs: np.ndarray | torch.Tensor) -> dict[str, float]:
    """Aggregate a ``(C, H, W)`` posterior into per-frame scalars.

    Higher ``entropy_mean`` / ``entropy_normalized`` and lower ``margin_mean`` /
    ``confidence_mean`` all mean "less certain".
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().float().cpu().numpy()
    p = np.asarray(probs, np.float64)
    ent = predictive_entropy(p, dim=0)
    nent = normalized_entropy(p, dim=0)
    mar = margin(p, dim=0)
    conf = p.max(axis=0)
    return {
        "entropy_mean": float(np.mean(ent)),
        "entropy_p95": float(np.percentile(ent, 95)),
        "entropy_normalized": float(np.mean(nent)),
        "margin_mean": float(np.mean(mar)),
        "confidence_mean": float(np.mean(conf)),
        "confidence_p05": float(np.percentile(conf, 5)),
    }


# --------------------------------------------------------------------------- #
# MC dropout
# --------------------------------------------------------------------------- #


@torch.no_grad()
def mc_dropout_stats(model, images: torch.Tensor, n_samples: int = 8) -> dict[str, np.ndarray]:
    """Run ``n_samples`` stochastic forward passes with dropout active in the heads.

    Returns the mean class posterior, its per-pixel variance, and the per-frame
    mean variance (the "MC-dropout variance" ranking signal).
    """
    was_training = model.training
    model.eval()
    # enable dropout only in the task heads; the frozen backbone stays in eval
    # mode so "frozen features" remain exactly frozen.
    head_modules = [
        m
        for n, m in model.named_modules()
        if n.startswith(("seg.", "box_head.")) and isinstance(m, torch.nn.Dropout2d)
    ]
    for m in head_modules:
        m.train()

    accum = None
    accum_sq = None
    for _ in range(max(1, int(n_samples))):
        out = model(images)
        probs = out["seg_logits"].softmax(dim=1).float()
        accum = probs if accum is None else accum + probs
        accum_sq = probs * probs if accum_sq is None else accum_sq + probs * probs

    n = max(1, int(n_samples))
    mean = accum / n
    var = (accum_sq / n - mean * mean).clamp(min=0)

    for m in head_modules:
        m.eval()
    if was_training:
        model.train()

    return {
        "mean_probs": mean.cpu().numpy(),
        "variance": var.cpu().numpy(),
        "mc_variance_mean": var.mean(dim=(1, 2, 3)).cpu().numpy(),
    }


# --------------------------------------------------------------------------- #
# kNN out-of-distribution
# --------------------------------------------------------------------------- #


def knn_ood_scores(
    train_features: np.ndarray,
    query_features: np.ndarray,
    k: int = 5,
    metric: str = "cosine",
    block: int = 256,
) -> np.ndarray:
    """Mean distance to the ``k`` nearest training embeddings.

    Embeddings are L2-normalised first when ``metric='cosine'``, which makes the
    score a pure direction difference and independent of feature magnitude.
    """
    train = np.asarray(train_features, np.float64)
    query = np.asarray(query_features, np.float64)
    if train.ndim == 1:
        train = train[None]
    if query.ndim == 1:
        query = query[None]
    if train.size == 0:
        return np.full(len(query), np.nan)

    if metric == "cosine":
        train = train / np.clip(np.linalg.norm(train, axis=1, keepdims=True), EPS, None)
        query = query / np.clip(np.linalg.norm(query, axis=1, keepdims=True), EPS, None)
        # cosine distance = 1 - dot
        sim = query @ train.T
        dist = 1.0 - sim
    else:
        dist = np.zeros((len(query), len(train)), np.float64)
        for i in range(0, len(query), block):
            chunk = query[i : i + block]
            dist[i : i + block] = np.linalg.norm(chunk[:, None, :] - train[None, :, :], axis=2)

    k = int(min(max(1, k), train.shape[0]))
    part = np.partition(dist, k - 1, axis=1)[:, :k]
    return part.mean(axis=1)


# --------------------------------------------------------------------------- #
# ranking
# --------------------------------------------------------------------------- #


def topk_uncertainty(
    scores: dict[str, float] | np.ndarray | list[float],
    names=None,
    k: int = 10,
    descending: bool = True,
):
    """Rank frames by an uncertainty score; returns ``[(name, score), ...]``.

    ``scores`` may be either a ``{name: score}`` mapping or an array aligned with
    ``names``.  Ties are broken by frame name, so the ranking is byte-stable
    across re-runs -- which is what makes the mined hard set reproducible.
    """
    if isinstance(scores, dict):
        names = list(scores)
        values = np.asarray([scores[n] for n in names], np.float64)
    else:
        values = np.asarray(scores, np.float64).reshape(-1)
        names = list(names) if names is not None else [str(i) for i in range(len(values))]
    if len(names) != len(values):
        raise ValueError(f"got {len(names)} names for {len(values)} scores")
    order = sorted(
        range(len(names)),
        key=lambda i: ((-values[i]) if descending else values[i], str(names[i])),
    )
    return [(names[i], float(values[i])) for i in order[: max(0, int(k))]]
