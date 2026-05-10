from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from common.column_transforms import apply_transforms, normalize_column_transforms
from common.yaml import load_yaml_config

from .write_dispatcher import WriteDispatcher


class DataTransformer:
    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path).resolve()
        self._config: dict[str, Any] | None = None
        self._writer: WriteDispatcher | None = None
        self._embedder: Any = None

    @property
    def config(self) -> dict[str, Any]:
        if self._config is None:
            self._config = load_yaml_config(str(self.config_path))
        return self._config

    @property
    def writer(self) -> WriteDispatcher:
        if self._writer is None:
            self._writer = WriteDispatcher.from_env(env_path=self.config_path.parent / ".env")
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
        return normalize_column_transforms(self.transform_cfg.get("column_transforms"))

    def text_vectorizer_name(self) -> str:
        return str(
            self.config.get("data_loader", {}).get(
                "text_vectorizer", "intfloat/multilingual-e5-small"
            )
        )

    def embedding_batch_size(self) -> int:
        return int(self.transform_cfg.get("embedding_batch_size", 32))

    def _embed_model(self) -> Any:
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise RuntimeError(
                    "transform 'embed' requires sentence-transformers "
                    "(pip install sentence-transformers)."
                ) from e
            self._embedder = SentenceTransformer(self.text_vectorizer_name())
        return self._embedder

    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_names = self.features_to_select()
        transforms = self.column_transforms()
        needs_embedder = any("embed" in t for t in transforms.values())
        return apply_transforms(
            df,
            feature_names=feature_names,
            transforms=transforms,
            embedder=self._embed_model() if needs_embedder else None,
            embedding_batch_size=self.embedding_batch_size(),
        )

    def run(self, load_date: str) -> str:
        df = self.read_partition(load_date)
        return self.save_partition(self.transform_features(df), load_date)
