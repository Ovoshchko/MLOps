from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import load_yaml_config
from .write_dispatcher import WriteDispatcher


# Имя типа → метод ``_col_<name>``
_COLUMN_TRANSFORM_REGISTRY: dict[str, str] = {
    "embed": "embed",
    "log1p": "log1p",
    "coerce_bool": "coerce_bool",
    "json_to_string": "json_to_string",
}


class DataTransformer:
    """
    1) Всегда оставляет только ``transform.features_to_select``.
    2) Для каждой колонки из этого списка, если в ``transform.column_transforms``
       заданы дополнительные шаги — применяет их по порядку; иначе колонка не меняется.
    """

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path).resolve()
        self._config: dict[str, Any] | None = None
        self._writer: WriteDispatcher | None = None

    @property
    def config(self) -> dict[str, Any]:
        if self._config is None:
            self._config = load_yaml_config(str(self.config_path))
        return self._config

    @property
    def writer(self) -> WriteDispatcher:
        if self._writer is None:
            self._writer = WriteDispatcher.from_config(self.config)
        return self._writer

    @property
    def transform_cfg(self) -> dict[str, Any]:
        cfg = self.config.get("transform")
        if not cfg:
            raise ValueError("config.yaml: missing top-level key 'transform'")
        return cfg

    def input_dir(self) -> str:
        base = self.config_path.parent
        raw = self.transform_cfg.get("input_dir")
        if raw:
            return WriteDispatcher.resolve_uri_or_path(str(raw), base)
        raw = self.config.get("data_loader", {}).get("output_dir")
        if not raw:
            raise ValueError("set data_loader.output_dir or transform.input_dir in config.yaml")
        return WriteDispatcher.resolve_uri_or_path(str(raw), base)

    def output_dir(self) -> str:
        raw = self.transform_cfg.get("output_dir")
        if not raw:
            raise ValueError("set transform.output_dir in config.yaml")
        return WriteDispatcher.resolve_uri_or_path(str(raw), self.config_path.parent)

    def partition_path(self, load_date: str, input: bool = True) -> str:
        root = self.input_dir() if input else self.output_dir()
        return WriteDispatcher.partition_path(root, load_date)

    def read_partition(self, load_date: str) -> pd.DataFrame:
        path = self.partition_path(load_date, input=True)
        return self.writer.read_parquet(path)

    def save_partition(self, df: pd.DataFrame, load_date: str) -> str:
        path = self.partition_path(load_date, input=False)
        return self.writer.save_parquet(df, path, index=False)

    def features_to_select(self) -> list[str]:
        return list(self.transform_cfg.get("features_to_select", []))

    def column_transforms(self) -> dict[str, list[str]]:
        """Имя колонки → упорядоченный список типов трансформаций (строки)."""
        raw = self.transform_cfg.get("column_transforms") or {}
        out: dict[str, list[str]] = {}
        for k, v in raw.items():
            if isinstance(v, str):
                out[str(k)] = [v]
            elif isinstance(v, list):
                out[str(k)] = [str(x) for x in v]
            else:
                raise TypeError(f"column_transforms.{k}: expected str or list, got {type(v)}")
        return out

    def text_vectorizer_name(self) -> str:
        return str(self.config.get("data_loader", {}).get("text_vectorizer", "intfloat/multilingual-e5-small"))

    def embedding_batch_size(self) -> int:
        return int(self.transform_cfg.get("embedding_batch_size", 32))

    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = self.features_to_select()
        out = pd.DataFrame(index=df.index)
        for c in cols:
            out[c] = df[c] if c in df.columns else pd.NA
        return out
    
    def _col_embed(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "transform 'embed' requires sentence-transformers (pip install sentence-transformers)."
            ) from e

        out = df.copy()
        texts = out[col].fillna("").astype(str).tolist()
        model = SentenceTransformer(self.text_vectorizer_name())
        vectors = model.encode(texts, batch_size=self.embedding_batch_size(), show_progress_bar=False)
        out[f"{col}_embedding"] = [row.tolist() for row in np.asarray(vectors)]
        return out

    def _col_log1p(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        out = df.copy()
        x = pd.to_numeric(out[col], errors="coerce").clip(lower=0)
        out[f"{col}_log1p"] = np.log1p(x)
        return out

    def _col_coerce_bool(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
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

    def _col_json_to_string(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
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

    def _apply_column_transform(self, df: pd.DataFrame, col: str, transform_type: str) -> pd.DataFrame:
        key = transform_type.strip()
        method_suffix = _COLUMN_TRANSFORM_REGISTRY.get(key)
        if method_suffix is None:
            allowed = ", ".join(sorted(_COLUMN_TRANSFORM_REGISTRY))
            raise ValueError(f"unknown transform {transform_type!r} for column {col!r}; allowed: {allowed}")
        method = getattr(self, f"_col_{method_suffix}")
        return method(df, col)

    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обязательный отбор фич, затем опциональные трансформации по конфигу (по колонкам)."""
        out = self._select_features(df)
        transforms = self.column_transforms()

        for col in self.features_to_select():
            if col not in out.columns:
                continue
            for t in transforms.get(col, []):
                out = self._apply_column_transform(out, col, t)
        return out

    def run(self, load_date: str) -> str:
        """read → transform_features → save."""
        df = self.read_partition(load_date)
        return self.save_partition(self.transform_features(df), load_date)
