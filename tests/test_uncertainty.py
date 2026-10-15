"""Step 5 tests: uncertainty maths, stable ranking, clustering, cluster tagging."""

from __future__ import annotations

import numpy as np
import pytest

from aerial_vit.eval.hardcases import (
    STAT_NAMES,
    cluster_embeddings,
    cluster_metric_table,
    scene_statistics,
    tag_clusters,
)
from aerial_vit.eval.uncertainty import (
    frame_uncertainty,
    knn_ood_scores,
    margin,
    normalized_entropy,
    predictive_entropy,
    topk_uncertainty,
)

N_CLASSES = 8


# --------------------------------------------------------------------------- #
# entropy / margin
# --------------------------------------------------------------------------- #


def test_entropy_is_maximal_on_a_uniform_distribution():
    p = np.full((4, 4), 0.25)
    h = predictive_entropy(p, dim=0)
    assert h.shape == (4,)
    assert np.allclose(h, np.log(4), atol=1e-9)


def test_entropy_is_near_zero_on_a_one_hot_distribution():
    p = np.zeros((3, 3))
    p[0] = 1.0  # one-hot along dim 0
    h = predictive_entropy(p, dim=0)
    assert np.allclose(h, 0.0, atol=1e-9)
    assert np.all(np.asarray(h) < 1e-6)


def test_entropy_is_between_the_two_extremes():
    uniform = np.full(4, 0.25)
    peaked = np.array([0.85, 0.05, 0.05, 0.05])
    assert float(predictive_entropy(peaked)) < float(predictive_entropy(uniform))
    assert float(predictive_entropy(peaked)) > 0.0


def test_normalized_entropy_is_bounded_by_one():
    uniform = np.full(8, 1 / 8)
    assert float(normalized_entropy(uniform)) == pytest.approx(1.0, abs=1e-9)
    onehot = np.zeros(8)
    onehot[0] = 1.0
    assert float(normalized_entropy(onehot)) == pytest.approx(0.0, abs=1e-9)


def test_margin_is_maximal_on_one_hot_and_zero_on_uniform():
    onehot = np.array([1.0, 0.0, 0.0])
    uniform = np.full(3, 1 / 3)
    assert float(margin(onehot)) == pytest.approx(1.0)
    assert float(margin(uniform)) == pytest.approx(0.0, abs=1e-12)


def test_margin_uses_the_two_highest_classes():
    p = np.array([0.1, 0.7, 0.2])
    assert float(margin(p)) == pytest.approx(0.5)


def test_frame_uncertainty_keys_and_monotonicity():
    confident = np.zeros((N_CLASSES, 8, 8))
    confident[0] = 0.99
    confident[1:] = 0.01 / (N_CLASSES - 1)
    unsure = np.full((N_CLASSES, 8, 8), 1 / N_CLASSES)

    a = frame_uncertainty(confident)
    b = frame_uncertainty(unsure)
    for k in ("entropy_mean", "entropy_p95", "entropy_normalized"):
        assert b[k] > a[k]
    for k in ("margin_mean", "confidence_mean"):
        assert b[k] < a[k]


def test_frame_uncertainty_accepts_torch():
    import torch

    p = torch.softmax(torch.randn(N_CLASSES, 16, 16), dim=0)
    out = frame_uncertainty(p)
    assert 0.0 <= out["entropy_normalized"] <= 1.0 + 1e-6
    assert 0.0 <= out["confidence_mean"] <= 1.0


# --------------------------------------------------------------------------- #
# ranking stability
# --------------------------------------------------------------------------- #


def test_topk_ranking_is_sorted_and_stable_under_seed():
    scores = {"a": 0.9, "b": 0.1, "c": 0.5, "d": 0.5, "e": 0.7}
    first = topk_uncertainty(scores, k=3)
    for _ in range(5):
        assert topk_uncertainty(scores, k=3) == first
    assert [n for n, _ in first] == ["a", "e", "c"]
    # ties broken by name, deterministically
    assert [n for n, _ in topk_uncertainty(scores, k=5)] == ["a", "e", "c", "d", "b"]


def test_topk_ranking_accepts_an_array_aligned_with_names():
    scores = np.array([0.2, 0.9, 0.4])
    ranked = topk_uncertainty(scores, ["x", "y", "z"], k=2)
    assert [n for n, _ in ranked] == ["y", "z"]
    assert ranked[0][1] == pytest.approx(0.9)


def test_topk_ranking_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        topk_uncertainty(np.array([0.1, 0.2]), ["only-one"], k=1)


def test_topk_ranking_handles_k_larger_than_frames():
    ranked = topk_uncertainty(np.array([0.1, 0.2]), ["a", "b"], k=10)
    assert len(ranked) == 2


def test_topk_ranking_ascending_order_option():
    ranked = topk_uncertainty(np.array([0.1, 0.9]), ["a", "b"], k=1, descending=False)
    assert ranked[0][0] == "a"


# --------------------------------------------------------------------------- #
# kNN OOD
# --------------------------------------------------------------------------- #


def test_knn_ood_is_zero_for_a_training_point_itself():
    train = np.eye(5)
    score = knn_ood_scores(train, train[:1], k=1)
    assert float(score[0]) == pytest.approx(0.0, abs=1e-9)


def test_knn_ood_grows_with_distance_from_the_training_manifold():
    train = np.eye(4)
    near = np.array([[1.0, 0.1, 0.0, 0.0]])
    far = np.array([[0.25, 0.25, 0.25, 0.25]])
    s_near = float(knn_ood_scores(train, near, k=1)[0])
    s_far = float(knn_ood_scores(train, far, k=1)[0])
    assert s_far > s_near


def test_knn_ood_clamps_k_to_the_training_set_size():
    train = np.eye(3)
    assert knn_ood_scores(train, np.eye(3), k=99).shape == (3,)


def test_knn_ood_empty_train_set_is_nan():
    out = knn_ood_scores(np.zeros((0, 4)), np.ones((2, 4)), k=1)
    assert out.shape == (2,) and np.all(np.isnan(out))


def test_knn_ood_euclidean_metric_matches_manual_computation():
    train = np.array([[0.0, 0.0]])
    query = np.array([[3.0, 4.0]])
    assert float(knn_ood_scores(train, query, k=1, metric="l2")[0]) == pytest.approx(5.0)


# --------------------------------------------------------------------------- #
# scene statistics
# --------------------------------------------------------------------------- #


def test_scene_statistics_detects_glare():
    dark = np.zeros((64, 64, 3), np.uint8)
    blown = np.full((64, 64, 3), 255, np.uint8)
    assert scene_statistics(blown)["glare"] == pytest.approx(1.0)
    assert scene_statistics(dark)["glare"] == pytest.approx(0.0)


def test_scene_statistics_contrast_separates_haze():
    crisp = np.zeros((64, 64, 3), np.uint8)
    crisp[::2] = 255
    hazy = np.full((64, 64, 3), 128, np.uint8)
    assert scene_statistics(crisp)["contrast"] > scene_statistics(hazy)["contrast"]
    assert scene_statistics(hazy)["contrast"] == pytest.approx(0.0, abs=1e-6)


def test_scene_statistics_measures_instances_and_occlusion():
    img = np.zeros((100, 100, 3), np.uint8)
    boxes = np.array([[0, 0, 20, 20], [10, 10, 30, 30], [60, 60, 80, 80]], np.float32)
    st = scene_statistics(img, boxes)
    assert st["n_instances"] == 3
    assert st["median_instance_area"] == pytest.approx(400.0)
    assert st["occlusion"] > 0.0, "overlapping boxes must register as occlusion"


def test_scene_statistics_no_boxes_is_safe():
    st = scene_statistics(np.zeros((32, 32, 3), np.uint8))
    assert st["n_instances"] == 0 and st["occlusion"] == 0.0 and st["median_instance_area"] == 0.0


def test_scene_statistics_boundary_ambiguity_is_higher_when_unsure():
    mask = np.zeros((16, 16), np.uint8)
    mask[:, 8:] = 1
    sure = np.zeros((2, 16, 16))
    sure[mask] = 0.99
    sure[1 - mask, 1] = 0.99
    sure[1 - mask, 0] = 0.01
    sure[mask, 1] = 0.01
    sure[mask, 0] = 0.99
    unsure = np.full((2, 16, 16), 0.5)
    assert (
        scene_statistics(np.zeros((16, 16, 3), np.uint8), None, mask, unsure)["boundary_ambiguity"]
        > scene_statistics(np.zeros((16, 16, 3), np.uint8), None, mask, sure)["boundary_ambiguity"]
    )


# --------------------------------------------------------------------------- #
# clustering
# --------------------------------------------------------------------------- #


def test_cluster_embeddings_recovers_three_separated_groups():
    rng = np.random.default_rng(0)
    x = np.concatenate(
        [rng.normal(c, 0.05, (25, 6)) for c in ([0, 0, 0, 0, 0, 0], [5] * 6, [-5] * 6)]
    )
    res = cluster_embeddings(x, k_range=(2, 5), seed=1)
    assert res["k"] in (3, 2)
    assert res["silhouette"] is not None and res["silhouette"] > 0.5
    assert len(set(res["labels"].tolist())) == res["k"]
    assert res["labels"].shape == (len(x),)


def test_cluster_embeddings_is_deterministic():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, (30, 5))
    a = cluster_embeddings(x, k_range=(2, 4), seed=7)
    b = cluster_embeddings(x, k_range=(2, 4), seed=7)
    assert a["k"] == b["k"]
    assert np.array_equal(a["labels"], b["labels"])


def test_cluster_embeddings_handles_tiny_inputs():
    assert cluster_embeddings(np.zeros((0, 4)))["k"] == 0
    assert cluster_embeddings(np.ones((1, 4)))["k"] == 1
    assert cluster_embeddings(np.ones((2, 4)))["k"] == 1


# --------------------------------------------------------------------------- #
# cluster tagging
# --------------------------------------------------------------------------- #


def _stat_row(**kw):
    row = dict.fromkeys(STAT_NAMES, 0.0)
    row["altitude_ratio"] = 1.0
    row.update(kw)
    return row


def test_tag_clusters_returns_exactly_one_tag_set_per_cluster():
    stats = [
        _stat_row(
            glare=0.9,
        ),
        _stat_row(glare=0.8),
        _stat_row(glare=0.02),
        _stat_row(glare=0.01),
    ]
    labels = np.array([0, 0, 1, 1])
    tags = tag_clusters(stats, labels)
    assert set(tags) == {0, 1}
    for info in tags.values():
        assert isinstance(info["tags"], list) and len(info["tags"]) >= 1
        assert info["tag"] == info["tags"][0]
        assert info["n_frames"] == 2
    assert tags[0]["tag"] == "glare"
    assert tags[1]["tag"] != "glare"


def test_tag_clusters_flags_low_contrast_as_haze():
    stats = [
        _stat_row(contrast=0.05),
        _stat_row(contrast=0.05),
        _stat_row(contrast=0.6),
        _stat_row(contrast=0.7),
    ]
    tags = tag_clusters(stats, np.array([0, 0, 1, 1]))
    assert tags[0]["tag"] == "low-contrast/haze"


def test_tag_clusters_flags_small_objects():
    stats = [
        _stat_row(median_instance_area=12.0),
        _stat_row(median_instance_area=10.0),
        _stat_row(median_instance_area=1200.0),
        _stat_row(median_instance_area=1400.0),
    ]
    tags = tag_clusters(stats, np.array([0, 0, 1, 1]))
    assert tags[0]["tag"] == "small-object"


def test_tag_clusters_flags_occlusion_and_density():
    stats = [
        _stat_row(occlusion=0.8, instance_density=9.0),
        _stat_row(occlusion=0.7, instance_density=8.0),
        _stat_row(occlusion=0.05, instance_density=1.0),
        _stat_row(occlusion=0.02, instance_density=1.2),
    ]
    tags = tag_clusters(stats, np.array([0, 0, 1, 1]))
    assert tags[0]["tag"] in {"occlusion", "dense-objects"}
    assert tags[1]["tag"] == "within-normal-range"


def test_tag_clusters_is_deterministic_and_complete():
    rng = np.random.default_rng(5)
    stats = [_stat_row(glare=float(g), contrast=float(c)) for g, c in rng.random((20, 2))]
    labels = np.repeat([0, 1, 2, 3], 5)
    a = tag_clusters(stats, labels)
    b = tag_clusters(stats, labels)
    assert a == b
    assert set(a) == {0, 1, 2, 3}


# --------------------------------------------------------------------------- #
# per-cluster metric table
# --------------------------------------------------------------------------- #


def test_cluster_metric_table_computes_deltas():
    cluster_info = {
        0: {"tag": "small-object", "tags": ["small-object"], "test_frames": ["a", "b"]},
        1: {"tag": "glare", "tags": ["glare"], "test_frames": ["c"]},
    }
    per_frame = {
        "a": {"miou": 0.1, "map_50": 0.0},
        "b": {"miou": 0.3, "map_50": 0.2},
        "c": {"miou": 0.6, "map_50": 0.5},
    }
    split = {"miou": 0.5, "map_50": 0.4}
    rows = cluster_metric_table(cluster_info, per_frame, split, metric_keys=("miou", "map_50"))
    by = {r["cluster"]: r for r in rows}
    assert by[0]["miou"] == pytest.approx(0.2)
    assert by[0]["delta_miou"] == pytest.approx(-0.3)
    assert by[1]["delta_miou"] == pytest.approx(0.1)
    assert by[0]["n_frames"] == 2
    assert by[0]["split_miou"] == pytest.approx(0.5)


def test_cluster_metric_table_skips_missing_metrics():
    cluster_info = {0: {"tag": "t", "tags": ["t"], "test_frames": ["a", "b"]}}
    rows = cluster_metric_table(
        cluster_info, {"a": {"miou": 0.2}}, {"miou": 0.4}, metric_keys=("miou",)
    )
    assert rows[0]["miou"] == pytest.approx(0.2)
    assert rows[0]["delta_miou"] == pytest.approx(-0.2)


def test_cluster_metric_table_prefers_test_frames_over_hard_frames():
    cluster_info = {0: {"tag": "t", "tags": ["t"], "frames": ["hard1"], "test_frames": ["t1"]}}
    rows = cluster_metric_table(
        cluster_info,
        {"hard1": {"miou": 0.0}, "t1": {"miou": 0.9}},
        {"miou": 0.5},
        metric_keys=("miou",),
    )
    assert rows[0]["miou"] == pytest.approx(0.9)
    assert rows[0]["n_frames"] == 1
