from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class WriteDispatcher:
    """
    Универсальный диспетчер записи parquet.
    Определяет тип пути (локальный/s3) и вызывает соответствующий метод.
    """

    def __init__(self, s3_storage_options: dict[str, Any] | None = None):
        self.s3_storage_options = s3_storage_options or {}

    @staticmethod
    def is_s3_path(save_path: str) -> bool:
        return save_path.startswith("s3://")

    @classmethod
    def s3_options_from_config(cls, config: dict[str, Any]) -> dict[str, Any]:
        s3 = config.get("s3", {}) or {}
        endpoint = s3.get("endpoint_url")
        key = s3.get("access_key")
        secret = s3.get("secret_key")
        secure = bool(s3.get("secure", True))
        if not (endpoint and key and secret):
            return {}
        return {
            "key": str(key),
            "secret": str(secret),
            "client_kwargs": {
                "endpoint_url": str(endpoint),
                "verify": secure,
            },
        }

    @classmethod
    def resolve_uri_or_path(cls, raw: str, base_dir: Path) -> str:
        if not raw:
            raise ValueError("path is empty")
        if cls.is_s3_path(str(raw)):
            return str(raw).rstrip("/")
        p = Path(raw)
        if not p.is_absolute():
            p = (base_dir / p).resolve()
        return str(p)

    def save_parquet(self, df: pd.DataFrame, save_path: str, index: bool = False) -> str:
        if self.is_s3_path(save_path):
            return self._save_s3(df, save_path, index=index)
        return self._save_local(df, save_path, index=index)

    def _save_local(self, df: pd.DataFrame, save_path: str, index: bool = False) -> str:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=index)
        return str(path)

    def _save_s3(self, df: pd.DataFrame, save_path: str, index: bool = False) -> str:
        df.to_parquet(save_path, index=index, storage_options=self.s3_storage_options)
        return save_path
