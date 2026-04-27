from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import fsspec
import pandas as pd

from data_pipeline.src.write_dispatcher import WriteDispatcher


_LOAD_DATE_PATTERN = re.compile(r"load_date=(\d{4}-\d{2}-\d{2})\.parquet$")


def _read_text(path: str, storage_options: dict[str, Any]) -> str:
    if not path.startswith("s3://"):
        raise ValueError(f"expected S3 path, got {path}")
    with fsspec.open(path, mode="rt", encoding="utf-8", **storage_options) as file:
        return file.read()


def load_manifest(
    manifest_uri: str,
    *,
    env_path: str = "data_pipeline/.env",
) -> dict[str, Any]:
    writer = WriteDispatcher.from_env(Path(env_path))
    return json.loads(_read_text(manifest_uri, writer.s3_storage_options))


def load_registered_dataset(
    manifest_uri: str,
    *,
    env_path: str = "data_pipeline/.env",
) -> tuple[dict[str, Any], pd.DataFrame]:
    manifest = load_manifest(manifest_uri, env_path=env_path)
    writer = WriteDispatcher.from_env(Path(env_path))

    frames = []
    for path in manifest["registered_partition_paths"]:
        frame = writer.read_parquet(path)
        match = _LOAD_DATE_PATTERN.search(path)
        if match:
            frame["load_date"] = pd.to_datetime(match.group(1))
        frames.append(frame)
    if not frames:
        return manifest, pd.DataFrame()
    return manifest, pd.concat(frames, ignore_index=True)
