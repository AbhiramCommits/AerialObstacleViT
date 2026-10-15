"""Dataclass-based configuration, loaded from YAML in ``configs/``.

Every script takes ``--config`` and every runtime behaviour that matters for
reproducibility (seed, device, image size, label fraction) lives here, not in
code.  Configs compose: ``base.yaml`` holds shared defaults, ``smoke.yaml`` and
``full.yaml`` override them, and CLI flags override the file.

Usage
-----
>>> cfg = load_config("configs/smoke.yaml")
>>> cfg.data.image_size
224
>>> cfg = load_config("configs/smoke.yaml", overrides={"train.epochs": 1})
>>> cfg.train.epochs
1
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"
CONFIGS_DIR = CONFIG_DIR  # alias
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
REPORTS_DIR = REPO_ROOT / "reports"
RESULTS_DIR = ARTIFACTS_DIR / "results"
FIGURES_DIR = REPORTS_DIR / "figures"


@dataclass
class SyntheticCfg:
    """Knobs for the procedural aerial-scene generator (the offline fallback)."""

    n_scenes: int = 24
    base_size: int = 384
    min_objects: int = 12
    max_objects: int = 26
    # Degradations. Each is a probability in [0, 1]; 1.0 = always applied.
    glare_prob: float = 0.25
    blur_prob: float = 0.25
    haze_prob: float = 0.25
    # Per-scene altitude scaling factor; simulates ground-sample-distance change.
    altitude_scales: tuple[float, ...] = (0.7, 1.0, 1.4)
    altitude_shift_prob: float = 0.5
    # Texture strength.
    texture: float = 0.18
    seed: int = 1337

    def __post_init__(self) -> None:
        if isinstance(self.altitude_scales, list):
            self.altitude_scales = tuple(float(s) for s in self.altitude_scales)


@dataclass
class DataCfg:
    """Dataset + taxonomy configuration."""

    source: str = "auto"  # auto | semantic_drone | synthetic
    src: str | None = None  # URL or local path for the real dataset
    root: str | None = None  # resolved dataset root (else AERIALVIT_DATA_DIR)
    image_size: int = 224
    classes: list[str] = field(
        default_factory=lambda: [
            "background",
            "paved-area",
            "grass",
            "roof",
            "tree",
            "water",
            "person",
            "vehicle",
        ]
    )
    landable_classes: list[str] = field(default_factory=lambda: ["paved-area", "grass"])
    n_train: int = 64
    n_val: int = 16
    n_test: int = 16
    rare_pixel_threshold: float = 0.01
    rare_instance_threshold: int = 50
    augment: bool = True
    num_workers: int = 0
    synthetic: SyntheticCfg = field(default_factory=SyntheticCfg)
    # Native SDD class name -> canonical index (see configs/taxonomy.yaml).
    sdd_class_map: dict[str, int] = field(default_factory=dict)

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def class_index(self, name: str) -> int:
        return self.classes.index(name)

    @property
    def landable_indices(self) -> list[int]:
        return [self.class_index(c) for c in self.landable_classes]

    @property
    def landable_index_set(self) -> set[int]:
        return set(self.landable_indices)


@dataclass
class BackboneCfg:
    """Frozen (or LoRA-adapted) ViT backbone."""

    name: str = "facebook/dinov2-small"
    patch_size: int = 14
    freeze: bool = True
    # Which HF model class family to use; currently only "dinov2".
    family: str = "dinov2"


@dataclass
class ModelCfg:
    """Perception head configuration."""

    seg_head: str = "linear"  # linear | conv
    head_dim: int = 256
    conv_layers: int = 2
    box_head: bool = True
    box_head_dim: int = 128
    dropout: float = 0.1
    # FCOS-lite strides on the token grid (in token units).
    box_strides: tuple[int, ...] = (1,)


@dataclass
class TrainCfg:
    """Optimisation + adaptation mode."""

    mode: str = "linear"  # linear | lora | full
    epochs: int = 2
    batch_size: int = 4
    lr: float = 1e-3
    backbone_lr: float = 1e-5  # used only when mode == "full"
    weight_decay: float = 0.05
    warmup_frac: float = 0.05
    label_smoothing: float = 0.0
    amp: bool = True
    grad_clip: float = 1.0
    dice_weight: float = 1.0
    focal_weight: float = 1.0
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    bbox_weight: float = 1.0
    giou_weight: float = 0.5
    obj_weight: float = 1.0
    early_stop_patience: int = 10
    max_steps: int = 0  # 0 = unlimited (smoke runs use this to stay cheap)
    log_every: int = 10
    # Label scarcity
    label_fraction: float = 1.0
    # LoRA
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_targets: list[str] = field(default_factory=lambda: ["query", "key", "value", "dense"])


@dataclass
class EvalCfg:
    """Evaluation / hard-case-mining knobs."""

    batch_size: int = 4
    score_thresh: float = 0.05
    nms_iou: float = 0.5
    max_dets: int = 100
    top_k_hard: int = 24
    mc_dropout_samples: int = 8
    cluster_k_range: tuple[int, int] = (2, 6)


@dataclass
class AblationCfg:
    """Grid for the label-scarcity ablation sweep."""

    modes: list[str] = field(default_factory=lambda: ["linear", "lora_r4", "lora_r16", "full"])
    label_fractions: list[float] = field(default_factory=lambda: [0.01, 0.05, 0.1, 0.25, 1.0])
    seeds: list[int] = field(default_factory=lambda: [1337, 2024])
    # Cells at or below this label fraction get every seed; larger cells get one.
    multiseed_max_fraction: float = 0.25
    max_steps: int = 0  # 0 = full epochs


@dataclass
class RunCfg:
    """Run bookkeeping."""

    name: str = "smoke"
    out_dir: str = "artifacts"
    device: str = "auto"  # auto | cpu | mps | cuda
    seed: int = 1337
    deterministic: bool = True
    tags: list[str] = field(default_factory=list)


@dataclass
class Config:
    """Root config object."""

    run: RunCfg = field(default_factory=RunCfg)
    data: DataCfg = field(default_factory=DataCfg)
    backbone: BackboneCfg = field(default_factory=BackboneCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    ablation: AblationCfg = field(default_factory=AblationCfg)
    taxonomy_path: str = "configs/taxonomy.yaml"

    # -- serialisation -----------------------------------------------------#
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path | None = None) -> str:
        text = json.dumps(self.to_dict(), indent=2, default=str, sort_keys=False)
        if path is not None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text + "\n")
        return text

    # -- paths -------------------------------------------------------------#
    @property
    def run_dir(self) -> Path:
        return (REPO_ROOT / self.run.out_dir / "runs" / self.run.name).resolve()

    @property
    def results_dir(self) -> Path:
        return (REPO_ROOT / self.run.out_dir / "results").resolve()

    @property
    def data_root(self) -> Path:
        if self.data.root:
            return (REPO_ROOT / self.data.root).resolve()
        return (REPO_ROOT / self.run.out_dir / "data").resolve()


# --------------------------------------------------------------------------- #
# Loading / merging
# --------------------------------------------------------------------------- #


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _build(cls: type, data: dict | None):
    """Recursively build a dataclass from a plain dict, ignoring unknown keys."""
    if data is None:
        return cls()
    kwargs: dict[str, Any] = {}
    known = {f.name: f for f in fields(cls)}
    for key, value in data.items():
        if key not in known:
            continue
        ftype = known[key].type
        nested = _TYPE_MAP.get(ftype) if isinstance(ftype, str) else None
        if nested is not None and isinstance(value, dict):
            kwargs[key] = _build(nested, value)
        else:
            kwargs[key] = _coerce(value, ftype)
    return cls(**kwargs)


def _coerce(value: Any, ftype: Any):
    """Best-effort conversion of YAML values to declared field types."""
    if isinstance(ftype, str):
        if "tuple" in ftype and isinstance(value, list):
            return tuple(value)
        if ("list" in ftype) and isinstance(value, (tuple, str)):
            return list(value)
        return value
    return value


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Load a YAML config, applying ``overrides`` (dotted keys) on top."""
    path = Path(path)
    if not path.is_absolute():
        candidate = CONFIG_DIR / path
        if candidate.exists():
            path = candidate
        elif (REPO_ROOT / path).exists():
            path = REPO_ROOT / path
    raw = yaml.safe_load(Path(path).read_text()) or {}

    # Support a `defaults:` include chain (single level of inheritance).
    if "defaults" in raw:
        base_name = raw.pop("defaults")
        base_path = CONFIG_DIR / f"{base_name}.yaml"
        base_raw = yaml.safe_load(base_path.read_text()) or {}
        raw = _deep_merge(base_raw, raw)

    if overrides:
        raw = apply_dotted_overrides(raw, overrides)

    cfg = Config()
    for f in fields(Config):
        if f.name not in raw:
            continue
        ftype = f.type
        if isinstance(ftype, str) and ftype in _TYPE_MAP:
            setattr(cfg, f.name, _build(_TYPE_MAP[ftype], raw[f.name]))
        else:
            setattr(cfg, f.name, _coerce(raw[f.name], ftype))

    # Attach taxonomy from the shared taxonomy.yaml unless inlined in the config.
    tax_path = REPO_ROOT / cfg.taxonomy_path
    if tax_path.exists():
        tax = yaml.safe_load(tax_path.read_text())
        if not cfg.data.sdd_class_map:
            cfg.data.sdd_class_map = dict(tax.get("sdd_class_map", {}))
    return cfg


_TYPE_MAP: dict[str, type] = {
    "RunCfg": RunCfg,
    "DataCfg": DataCfg,
    "BackboneCfg": BackboneCfg,
    "ModelCfg": ModelCfg,
    "TrainCfg": TrainCfg,
    "EvalCfg": EvalCfg,
    "SyntheticCfg": SyntheticCfg,
    "AblationCfg": AblationCfg,
}


def apply_dotted_overrides(raw: dict, overrides: dict[str, Any]) -> dict:
    """Apply ``{"train.epochs": 3}`` style overrides onto a nested dict."""
    out = copy.deepcopy(raw)
    for dotted, value in overrides.items():
        if value is None:
            continue
        parts = dotted.split(".")
        node = out
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return out


# --------------------------------------------------------------------------- #
# CLI parsing helpers (used by every typer command)
# --------------------------------------------------------------------------- #


def parse_override_args(items: list[str] | None) -> dict[str, Any]:
    """Parse repeated ``key=value`` CLI overrides into a dict with YAML typing."""
    out: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        key, _, value = item.partition("=")
        try:
            out[key.strip()] = yaml.safe_load(value)
        except yaml.YAMLError:
            out[key.strip()] = value
    return out


def resolve_class_names(cfg: DataCfg) -> list[str]:
    return list(cfg.classes)


def is_dataclass_instance(obj: Any) -> bool:
    return is_dataclass(obj) and not isinstance(obj, type)
