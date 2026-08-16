"""YAML config loading, merge, validation, snapshot. See CLAUDE.md §7.

Layout assumed (fixed by CLAUDE.md §3, not itself configurable):

    configs/base.yaml
    configs/data/{name}.yaml
    configs/model/{name}.yaml
    configs/exp/{exp_id}.yaml

An exp config's top-level `data` / `model` keys are either a string naming a
file under configs/data/ or configs/model/ (resolved and merged in), or an
inline dict (merged in as-is) — both are valid, so a self-contained exp file
needs no separate data/model file. The fully-expanded result of merging all
layers is what §7.2's example config shows, and what `snapshot()` writes out.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REQUIRED_KEYS = (
    "exp_id",
    "seed",
    "data.path",
    "data.target_byte",
    "data.trace_len",
    "split.n_attacker",
    "split.n_val",
    "split.n_defender",
    "split.seed",
    "leakage.model",
    "leakage.n_classes",
    "model.name",
    "train.epochs",
    "attack.max_traces",
    "attack.n_runs",
    "attack.seed",
)
VALID_LEAKAGE_MODELS = ("ID", "ID_MASKED", "HW")


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with open(path) as f:
        loaded = yaml.safe_load(f)
    return loaded or {}


def _resolve_named(dir_path: Path, name: str) -> Path:
    return dir_path / name if name.endswith((".yaml", ".yml")) else dir_path / f"{name}.yaml"


def _resolve_layer(configs_root: Path, subdir: str, ref: Any) -> dict[str, Any]:
    """`ref` is either a string (load configs/{subdir}/{ref}.yaml) or an inline dict."""
    if isinstance(ref, str):
        return _load_yaml(_resolve_named(configs_root / subdir, ref))
    if isinstance(ref, dict):
        return {subdir: ref}
    raise TypeError(f"'{subdir}' must be a string reference or an inline dict, got {type(ref).__name__}")


def load_config(exp_path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Merge base.yaml -> data/*.yaml -> model/*.yaml -> exp/*.yaml -> CLI overrides.

    `overrides` are "dotted.key=value" strings (see CLAUDE.md §7.3).
    Later sources win. Referenced data/model configs are resolved from the
    `data`/`model` keys inside the exp config.
    """
    exp_path = Path(exp_path)
    configs_root = exp_path.parent.parent  # configs/exp/{exp_id}.yaml -> configs/

    base_cfg = _load_yaml(configs_root / "base.yaml")
    exp_cfg = _load_yaml(exp_path)

    data_ref = exp_cfg.pop("data", None)
    model_ref = exp_cfg.pop("model", None)

    layers = [base_cfg]
    if data_ref is not None:
        layers.append(_resolve_layer(configs_root, "data", data_ref))
    if model_ref is not None:
        layers.append(_resolve_layer(configs_root, "model", model_ref))
    layers.append(exp_cfg)

    cfg = merge(*layers)
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    validate(cfg)
    return cfg


def merge(*configs: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge configs left to right; later dicts override earlier ones."""
    result: dict[str, Any] = {}
    for cfg in configs:
        _deep_merge_into(result, cfg)
    return result


def _deep_merge_into(base: dict[str, Any], incoming: dict[str, Any]) -> None:
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge_into(base[key], value)
        else:
            base[key] = copy.deepcopy(value)


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply "dotted.key=value" CLI overrides on top of a merged config."""
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        key_path, sep, raw_value = item.partition("=")
        if not sep:
            raise ValueError(f"override must be in dotted.key=value form, got: {item!r}")
        value = yaml.safe_load(raw_value)
        keys = key_path.split(".")
        node = cfg
        for k in keys[:-1]:
            if not isinstance(node.get(k), dict):
                node[k] = {}
            node = node[k]
        node[keys[-1]] = value
    return cfg


def _get_path(cfg: dict[str, Any], dotted_key: str) -> Any:
    node: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def validate(cfg: dict[str, Any]) -> None:
    """Raise on missing/malformed required keys before any stage runs."""
    errors = [f"missing required key: {key}" for key in REQUIRED_KEYS if _get_path(cfg, key) is None]

    leakage_model = _get_path(cfg, "leakage.model")
    if leakage_model is not None and leakage_model not in VALID_LEAKAGE_MODELS:
        errors.append(f"leakage.model must be one of {VALID_LEAKAGE_MODELS}, got {leakage_model!r}")
    if leakage_model == "ID_MASKED" and _get_path(cfg, "leakage.mask_index") is None:
        errors.append("leakage.mask_index is required when leakage.model is ID_MASKED")

    for key in ("split.n_attacker", "split.n_val", "split.n_defender", "data.trace_len",
                "train.epochs", "attack.max_traces", "attack.n_runs"):
        value = _get_path(cfg, key)
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
            errors.append(f"{key} must be a positive integer, got {value!r}")

    if errors:
        raise ValueError("invalid config:\n  " + "\n  ".join(errors))


def snapshot(cfg: dict[str, Any], run_dir: str | Path) -> None:
    """Write the fully-expanded config to `{run_dir}/config_snapshot.yaml`."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config_snapshot.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
