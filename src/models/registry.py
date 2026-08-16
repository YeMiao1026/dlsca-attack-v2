"""name -> builder lookup so new architectures need only a new config + module.
See CLAUDE.md acceptance criterion: "新增一個攻擊模型只需新增一個 model config"."""

from __future__ import annotations

from typing import Any, Callable

import keras

Builder = Callable[..., keras.Model]

_REGISTRY: dict[str, Builder] = {}


def register(name: str) -> Callable[[Builder], Builder]:
    """Decorator: `@register("cnn_light")` on a `build(**kwargs) -> keras.Model` function."""
    def decorator(builder: Builder) -> Builder:
        _REGISTRY[name] = builder
        return builder
    return decorator


def get(name: str) -> Builder:
    if name not in _REGISTRY:
        raise KeyError(f"unknown model {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def build(name: str, **kwargs: Any) -> keras.Model:
    return get(name)(**kwargs)
