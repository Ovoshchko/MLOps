from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

_COLUMN_TRANSFORM_REGISTRY: dict[str, str] = {
    "embed": "embed",
    "log1p": "log1p",
    "coerce_bool": "coerce_bool",
    "json_to_string": "json_to_string",
}


def _manifest_column_transforms(manifest: dict[str, Any]) -> dict[str, list[str]]:
    raw = manifest.get("column_transforms") or {}
    out: dict[str, list[str]] = {}
    for k, v in raw.items():
        if isinstance(v, str):
            out[str(k)] = [v]
        elif isinstance(v, list):
            out[str(k)] = [str(x) for x in v]
        else:
            raise TypeError(f"column_transforms.{k}: expected str or list, got {type(v)}")
    return out


def _select_features(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in feature_names:
        out[c] = df[c] if c in df.columns else pd.NA
    return out


def _col_embed(
    df: pd.DataFrame,
    col: str,
    embedder: Any,
    embedding_batch_size: int,
) -> pd.DataFrame:
    out = df.copy()
    texts = out[col].fillna("").astype(str).tolist()
    vectors = embedder.encode(texts, batch_size=embedding_batch_size, show_progress_bar=False)
    out[f"{col}_embedding"] = [row.tolist() for row in np.asarray(vectors)]
    return out


def _col_log1p(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    x = pd.to_numeric(out[col], errors="coerce").clip(lower=0)
    out[f"{col}_log1p"] = np.log1p(x)
    return out


def _col_coerce_bool(df: pd.DataFrame, col: str) -> pd.DataFrame:
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


def _col_json_to_string(df: pd.DataFrame, col: str) -> pd.DataFrame:
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


def _apply_column_transform(
    df: pd.DataFrame,
    col: str,
    transform_type: str,
    *,
    embedder: Any,
    embedding_batch_size: int,
) -> pd.DataFrame:
    key = transform_type.strip()
    method_suffix = _COLUMN_TRANSFORM_REGISTRY.get(key)
    if method_suffix is None:
        allowed = ", ".join(sorted(_COLUMN_TRANSFORM_REGISTRY))
        raise ValueError(f"unknown transform {transform_type!r} for column {col!r}; allowed: {allowed}")
    if method_suffix == "embed":
        return _col_embed(df, col, embedder, embedding_batch_size)
    if method_suffix == "log1p":
        return _col_log1p(df, col)
    if method_suffix == "coerce_bool":
        return _col_coerce_bool(df, col)
    if method_suffix == "json_to_string":
        return _col_json_to_string(df, col)
    raise AssertionError(method_suffix)


def transform_features(
    df: pd.DataFrame,
    manifest: dict[str, Any],
    embedder: Any,
    *,
    embedding_batch_size: int,
) -> pd.DataFrame:
    feature_names = list(manifest["features"])
    transforms = _manifest_column_transforms(manifest)
    out = _select_features(df, feature_names)
    for col in feature_names:
        if col not in out.columns:
            continue
        for t in transforms.get(col, []):
            out = _apply_column_transform(
                out,
                col,
                t,
                embedder=embedder,
                embedding_batch_size=embedding_batch_size,
            )
    return out
