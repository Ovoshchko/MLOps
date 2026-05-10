from __future__ import annotations

import json
from typing import Any, Callable

import numpy as np
import pandas as pd


_TRANSFORMS: dict[str, Callable[..., pd.DataFrame]] = {}


def _register(name: str):
    def decorator(fn: Callable[..., pd.DataFrame]) -> Callable[..., pd.DataFrame]:
        _TRANSFORMS[name] = fn
        return fn
    return decorator


def normalize_column_transforms(raw: dict[str, Any] | None) -> dict[str, list[str]]:
    """Accept ``{col: str}`` or ``{col: [str, ...]}`` and return the latter."""
    out: dict[str, list[str]] = {}
    for k, v in (raw or {}).items():
        if isinstance(v, str):
            out[str(k)] = [v]
        elif isinstance(v, list):
            out[str(k)] = [str(x) for x in v]
        else:
            raise TypeError(f"column_transforms.{k}: expected str or list, got {type(v)}")
    return out


def select_features(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Return a frame with exactly ``feature_names`` columns; missing -> ``pd.NA``."""
    out = pd.DataFrame(index=df.index)
    for c in feature_names:
        out[c] = df[c] if c in df.columns else pd.NA
    return out


@_register("embed")
def _col_embed(
    df: pd.DataFrame,
    col: str,
    *,
    embedder: Any,
    embedding_batch_size: int,
    **_: Any,
) -> pd.DataFrame:
    if embedder is None:
        raise ValueError(f"transform 'embed' for column {col!r} requires an embedder")
    out = df.copy()
    texts = out[col].fillna("").astype(str).tolist()
    vectors = embedder.encode(texts, batch_size=embedding_batch_size, show_progress_bar=False)
    out[f"{col}_embedding"] = [row.tolist() for row in np.asarray(vectors)]
    return out


@_register("log1p")
def _col_log1p(df: pd.DataFrame, col: str, **_: Any) -> pd.DataFrame:
    out = df.copy()
    x = pd.to_numeric(out[col], errors="coerce").clip(lower=0)
    out[f"{col}_log1p"] = np.log1p(x)
    return out


@_register("coerce_bool")
def _col_coerce_bool(df: pd.DataFrame, col: str, **_: Any) -> pd.DataFrame:
    out = df.copy()
    s = out[col]

    def one(x: Any) -> bool | None:
        if pd.isna(x):
            return None
        if isinstance(x, bool):
            return x
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            if x == 1:
                return True
            if x == 0:
                return False
        if isinstance(x, str):
            t = x.strip().lower()
            if t in ("true", "1", "yes"):
                return True
            if t in ("false", "0", "no", ""):
                return False
        return None

    mapped = s.map(one)
    if mapped.notna().mean() > 0.5:
        out[col] = mapped.fillna(False).astype(bool)
    return out


@_register("json_to_string")
def _col_json_to_string(df: pd.DataFrame, col: str, **_: Any) -> pd.DataFrame:
    out = df.copy()
    s = out[col]

    def norm(v: Any) -> Any:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return v
        if isinstance(v, (dict, list)):
            return json.dumps(v, ensure_ascii=False, default=str)
        return v

    sample = s.dropna().head(64)
    if not sample.empty and any(isinstance(v, (dict, list)) for v in sample):
        out[col] = s.map(norm)
    return out


def apply_transforms(
    df: pd.DataFrame,
    feature_names: list[str],
    transforms: dict[str, list[str]],
    *,
    embedder: Any = None,
    embedding_batch_size: int = 32,
) -> pd.DataFrame:
    """Select ``feature_names`` and apply ``transforms`` per column.

    ``transforms`` is the normalized mapping from :func:`normalize_column_transforms`.
    ``embedder`` is required only if any column uses the ``"embed"`` transform.
    """
    out = select_features(df, feature_names)
    for col in feature_names:
        if col not in out.columns:
            continue
        for name in transforms.get(col, []):
            fn = _TRANSFORMS.get(name)
            if fn is None:
                allowed = ", ".join(sorted(_TRANSFORMS))
                raise ValueError(
                    f"unknown transform {name!r} for column {col!r}; allowed: {allowed}"
                )
            out = fn(out, col, embedder=embedder, embedding_batch_size=embedding_batch_size)
    return out
