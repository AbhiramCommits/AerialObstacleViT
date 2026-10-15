"""``fetch_data`` -- obtain the dataset, or generate the synthetic fallback.

Resolution order:

1. ``--src`` (URL or local path) or the ``AERIALVIT_DATA_DIR`` environment
   variable pointing at an already-prepared canonical dataset root.
2. A local copy of the Semantic Drone Dataset (``--src`` points at a directory
   or zip containing ``images/`` + ``label_images_semantic/`` + ``class_dict.csv``).
3. **Synthetic fallback** -- ``aerial_vit.data.synthetic`` generates a
   deterministic procedural aerial dataset.

Whichever route is taken is recorded honestly in ``artifacts/data_manifest.json``
under ``source`` / ``source_detail``.
"""

from __future__ import annotations

import os
import tempfile
import zipfile
from pathlib import Path

import typer

from aerial_vit.commands._common import ConfigOpt, SetOpt, load_cli_config
from aerial_vit.config import Config
from aerial_vit.data.semantic_drone import (
    build_manifest,
    default_manifest_path,
    make_splits,
    prepare_semantic_drone,
    read_splits,
    write_manifest,
    write_splits,
)
from aerial_vit.data.synthetic import generate_synthetic_dataset
from aerial_vit.utils.logging import get_logger

app = typer.Typer(add_completion=False, help="Fetch or generate the aerial dataset.")
log = get_logger("aerialvit.fetch")


def _is_prepared_root(p: Path) -> bool:
    return (p / "images").is_dir() and (p / "masks").is_dir() and (p / "splits.json").exists()


def _download(src: str, dest: Path) -> Path:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s -> %s", src, dest)
    urllib.request.urlretrieve(src, dest)  # noqa: S310 - explicit user-supplied URL
    return dest


def _maybe_unzip(path: Path, workdir: Path) -> Path:
    if path.suffix.lower() != ".zip":
        return path
    workdir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as zf:
        zf.extractall(workdir)
    return workdir


def run_fetch(cfg: Config, src: str | None, force: bool) -> dict:
    data_root = cfg.data_root
    manifest_p = default_manifest_path(cfg)
    if manifest_p.exists() and not force:
        log.info("manifest already exists at %s (use --force to regenerate)", manifest_p)

    env_src = os.environ.get("AERIALVIT_DATA_DIR")
    requested = src or cfg.data.src or env_src
    source_pref = cfg.data.source

    n_need = cfg.data.n_train + cfg.data.n_val + cfg.data.n_test
    source_detail = ""
    real_root: Path | None = None

    # --- route 1: already-prepared canonical root -------------------------- #
    if requested and not requested.startswith(("http://", "https://")):
        cand = Path(requested).expanduser()
        if cand.is_dir() and _is_prepared_root(cand):
            real_root = cand
            source_detail = f"prepared canonical root at {cand}"
            log.info("using prepared dataset root %s", cand)

    # --- route 2: real Semantic Drone Dataset ----------------------------- #
    if real_root is None and source_pref in ("auto", "semantic_drone") and requested:
        tmp = None
        try:
            if requested.startswith(("http://", "https://")):
                tmp = Path(tempfile.mkdtemp(prefix="aerialvit_dl_"))
                dl = _download(requested, tmp / Path(requested).name)
                extracted = _maybe_unzip(dl, tmp / "unzipped")
            else:
                extracted = Path(requested).expanduser()
                extracted = _maybe_unzip(extracted, extracted.parent / f"{extracted.stem}_unzipped")
            out_root = data_root / "semantic_drone"
            out_root.mkdir(parents=True, exist_ok=True)
            log.info("preparing Semantic Drone Dataset from %s", extracted)
            info = prepare_semantic_drone(
                extracted,
                out_root,
                cfg.data,
                cfg.data.n_train,
                cfg.data.n_val,
                cfg.data.n_test,
                seed=cfg.run.seed,
            )
            real_root = out_root
            source_detail = f"Semantic Drone Dataset prepared from {requested}"
            log.info("prepared %d scenes", info["n_scenes"])
        except Exception as exc:  # noqa: BLE001 - fall back honestly
            log.warning("Semantic Drone Dataset unavailable (%s: %s)", type(exc).__name__, exc)
            real_root = None
            source_detail = f"semantic_drone unavailable: {type(exc).__name__}: {exc}"

    if source_pref == "semantic_drone" and real_root is None:
        log.warning("source=semantic_drone requested but unavailable; falling back to synthetic")

    # --- route 3: synthetic fallback --------------------------------------- #
    if real_root is None:
        syn_root = data_root / "synthetic"
        n_scenes = max(cfg.data.synthetic.n_scenes, n_need)
        log.info("generating synthetic dataset (%d scenes) at %s", n_scenes, syn_root)
        generate_synthetic_dataset(syn_root, cfg.data, n_scenes=n_scenes)
        names = sorted(p.stem for p in (syn_root / "images").glob("*.png"))
        splits = make_splits(
            names, cfg.data.n_train, cfg.data.n_val, cfg.data.n_test, seed=cfg.run.seed
        )
        write_splits(syn_root, splits)
        real_root = syn_root
        source_detail = source_detail or "synthetic procedural fallback (deterministic)"

    # --- manifest ---------------------------------------------------------- #
    splits = read_splits(real_root)
    manifest = build_manifest(
        real_root,
        cfg.data,
        source="synthetic" if real_root.name == "synthetic" else "semantic_drone",
        splits=splits,
        image_size=cfg.data.image_size,
        extra={
            "source_detail": source_detail,
            "requested_src": requested,
            "config_source_pref": source_pref,
            "synthetic_cfg": cfg.data.synthetic.__dict__,
            "canonical_layout": "images/ masks/ boxes/ meta/ splits.json",
        },
    )
    write_manifest(manifest, manifest_p)
    log.info("wrote manifest -> %s", manifest_p)
    return manifest


@app.command()
def main(
    config: str = ConfigOpt,
    src: str = typer.Option(None, "--src", help="URL or local path to the (real) dataset."),
    force: bool = typer.Option(False, "--force", help="Regenerate even if a manifest exists."),
    set_: list[str] = SetOpt,
) -> None:
    """Fetch the Semantic Drone Dataset, else generate the synthetic fallback."""
    cfg = load_cli_config(config, set_)
    manifest = run_fetch(cfg, src, force)
    typer.echo(
        "\n".join(
            [
                f"source          : {manifest['source']}",
                f"detail          : {manifest.get('source_detail', '')}",
                f"root            : {manifest['root']}",
                f"splits          : train={manifest['n_train']} val={manifest['n_val']} test={manifest['n_test']}",
                f"image_size      : {manifest['image_size']}",
                f"classes         : {', '.join(manifest['classes'])}",
                f"rare classes    : {', '.join(manifest['rare_classes']) or '(none)'}",
                f"manifest        : {default_manifest_path(cfg)}",
            ]
        )
    )


if __name__ == "__main__":  # pragma: no cover
    app()
