from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv


class WriteDispatcher:
    def __init__(self, s3_storage_options: dict[str, Any] | None = None):
        self.s3_storage_options = s3_storage_options or {}

    @classmethod
    def from_env(cls, env_path: Path | None = None) -> "WriteDispatcher":
        return cls(s3_storage_options=cls.s3_options_from_env(env_path=env_path))

    @staticmethod
    def is_s3_path(save_path: str) -> bool:
        return save_path.startswith("s3://")

    @classmethod
    def s3_options_from_env(cls, env_path: Path | None = None) -> dict[str, Any]:
        load_dotenv(dotenv_path=env_path, override=False)
        endpoint = os.getenv("S3_ENDPOINT_URL")
        key = os.getenv("S3_ACCESS_KEY")
        secret = os.getenv("S3_SECRET_KEY")
        secure_raw = os.getenv("S3_SECURE")
        secure = True if secure_raw is None else str(secure_raw).strip().lower() in ("1", "true", "yes", "on")

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

    @staticmethod
    def partition_path(root: str, load_date: str) -> str:
        return f"{root.rstrip('/')}/load_date={load_date}.parquet"

    def save_parquet(self, df: pd.DataFrame, save_path: str, index: bool = False) -> str:
        if self.is_s3_path(save_path):
            return self._save_s3(df, save_path, index=index)
        return self._save_local(df, save_path, index=index)

    def read_parquet(self, path: str) -> pd.DataFrame:
        if self.is_s3_path(path):
            return pd.read_parquet(path, storage_options=self.s3_storage_options)
        local_path = Path(path)
        if not local_path.is_file():
            raise FileNotFoundError(path)
        return pd.read_parquet(local_path)

    def _save_local(self, df: pd.DataFrame, save_path: str, index: bool = False) -> str:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=index)
        return str(path)

    def _save_s3(self, df: pd.DataFrame, save_path: str, index: bool = False) -> str:
        df.to_parquet(save_path, index=index, storage_options=self.s3_storage_options)
        return save_path
