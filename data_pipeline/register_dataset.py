from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import fsspec
import pyarrow.parquet as pq

from data_pipeline.script_configs import DatasetRegistrationConfig
from data_pipeline.src.utils import load_yaml_config
from data_pipeline.src.write_dispatcher import WriteDispatcher


def _iter_dates(date_from: str, date_to: str) -> list[str]:
    start = date.fromisoformat(date_from)
    end = date.fromisoformat(date_to)
    if start > end:
        raise ValueError(f"date_from must be <= date_to, got {date_from} > {date_to}")

    days: list[str] = []
    current = start
    while current <= end:
        days.append(current.isoformat())
        current += timedelta(days=1)
    return days


def _bucket_root(uri: str) -> str:
    if not uri.startswith("s3://"):
        raise ValueError(f"expected s3:// URI, got {uri}")
    bucket = uri[5:].split("/", 1)[0]
    return f"s3://{bucket}"


def _build_dataset_id(subreddit_name: str, date_from: str | None, date_to: str | None, version: str) -> str:
    normalized = "".join(ch if ch.isalnum() else "_" for ch in subreddit_name.lower()).strip("_")
    version_token = version if version.startswith("v") else f"v{version}"
    if date_from and date_to:
        return f"reddit_{normalized}_{date_from}_{date_to}_{version_token}"
    return f"reddit_{normalized}_{version_token}"


@dataclass(frozen=True)
class RegisteredDataset:
    dataset_id: str
    dataset_uri: str
    manifest_uri: str
    source_root: str
    target_col: str
    subreddit_name: str
    version: str
    date_from: str | None
    date_to: str | None
    partition_paths: list[str]
    registered_partition_paths: list[str]
    features: list[str]
    column_transforms: dict[str, Any]
    row_count: int
    column_count: int
    partition_count: int
    schema: list[dict[str, str]]
    digest: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DatasetRegistrar:
    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path).resolve()
        self.pipeline_config = load_yaml_config(str(self.config_path))
        self.writer = WriteDispatcher.from_env(self.config_path.parent / ".env")
        self.storage_options = self.writer.s3_storage_options

    @property
    def data_loader_cfg(self) -> dict[str, Any]:
        return self.pipeline_config.get("data_loader", {})

    @property
    def transform_cfg(self) -> dict[str, Any]:
        cfg = self.pipeline_config.get("transform", {})
        if not cfg:
            raise ValueError("config.yaml: missing top-level key 'transform'")
        return cfg

    @property
    def processed_root(self) -> str:
        raw = str(self.transform_cfg["output_dir"])
        return WriteDispatcher.resolve_uri_or_path(raw, self.config_path.parent)

    @property
    def default_dataset_root(self) -> str:
        if self.processed_root.startswith("s3://"):
            return f"{_bucket_root(self.processed_root)}/datasets/processed"
        return str((self.config_path.parent / "datasets" / "processed").resolve())

    def resolve_partition_paths(self, registration: DatasetRegistrationConfig) -> list[str]:
        if registration.partition_paths:
            return registration.partition_paths

        if registration.date_from and registration.date_to:
            return [
                WriteDispatcher.partition_path(self.processed_root, load_date)
                for load_date in _iter_dates(registration.date_from, registration.date_to)
            ]

        raise ValueError("set dataset_registration.partition_paths or both date_from/date_to")

    def register(self, registration: DatasetRegistrationConfig) -> RegisteredDataset:
        partition_paths = self.resolve_partition_paths(registration)
        self._ensure_partitions_exist(partition_paths)

        dataset_id = registration.dataset_id or _build_dataset_id(
            subreddit_name=str(self.data_loader_cfg["subreddit_name"]),
            date_from=registration.date_from,
            date_to=registration.date_to,
            version=registration.version,
        )

        dataset_root = registration.dataset_root or self.default_dataset_root
        dataset_uri = f"{dataset_root.rstrip('/')}/{dataset_id}"
        manifest_uri = f"{dataset_uri}/manifest.json"

        row_count = 0
        column_count = 0
        schema: list[dict[str, str]] = []
        registered_paths: list[str] = []
        digest_source = hashlib.sha256()

        for src_path in partition_paths:
            metadata = self._read_metadata(src_path)
            row_count += metadata.num_rows
            if column_count == 0:
                column_count = metadata.num_columns
                schema = [
                    {"name": field.name, "type": str(field.type)}
                    for field in metadata.schema.to_arrow_schema()
                ]

            digest_source.update(src_path.encode("utf-8"))
            digest_source.update(str(metadata.num_rows).encode("utf-8"))

            dst_path = f"{dataset_uri}/partitions/{Path(src_path).name}"
            self._copy_file(src_path, dst_path)
            registered_paths.append(dst_path)

        dataset = RegisteredDataset(
            dataset_id=dataset_id,
            dataset_uri=dataset_uri,
            manifest_uri=manifest_uri,
            source_root=self.processed_root,
            target_col=str(self.data_loader_cfg["target_col"]),
            subreddit_name=str(self.data_loader_cfg["subreddit_name"]),
            version=registration.version if registration.version.startswith("v") else f"v{registration.version}",
            date_from=registration.date_from,
            date_to=registration.date_to,
            partition_paths=partition_paths,
            registered_partition_paths=registered_paths,
            features=list(self.transform_cfg.get("features_to_select", [])),
            column_transforms=dict(self.transform_cfg.get("column_transforms", {})),
            row_count=row_count,
            column_count=column_count,
            partition_count=len(partition_paths),
            schema=schema,
            digest=digest_source.hexdigest(),
        )

        self._write_json(manifest_uri, dataset.to_dict())
        return dataset

    def _ensure_partitions_exist(self, partition_paths: list[str]) -> None:
        missing = [path for path in partition_paths if not self._exists(path)]
        if not missing:
            return
        preview = ", ".join(missing[:3])
        suffix = "" if len(missing) <= 3 else f" and {len(missing) - 3} more"
        raise FileNotFoundError(f"missing processed partitions: {preview}{suffix}")

    def _exists(self, path: str) -> bool:
        if path.startswith("s3://"):
            fs = fsspec.filesystem("s3", **self.storage_options)
            return fs.exists(path)
        return Path(path).exists()

    def _read_metadata(self, path: str) -> pq.FileMetaData:
        if path.startswith("s3://"):
            with fsspec.open(path, mode="rb", **self.storage_options) as file:
                return pq.read_metadata(file)
        return pq.read_metadata(path)

    def _copy_file(self, src_path: str, dst_path: str) -> None:
        if dst_path.startswith("s3://"):
            fs, _, paths = fsspec.get_fs_token_paths(dst_path, storage_options=self.storage_options)
            fs.makedirs(str(Path(paths[0]).parent), exist_ok=True)
        else:
            Path(dst_path).parent.mkdir(parents=True, exist_ok=True)

        with self._open(src_path, mode="rb") as src:
            with self._open(dst_path, mode="wb") as dst:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)

    def _write_json(self, path: str, payload: dict[str, Any]) -> None:
        content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        if not path.startswith("s3://"):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self._open(path, mode="wt", encoding="utf-8") as file:
            file.write(content)

    def _open(self, path: str, mode: str, encoding: str | None = None):
        kwargs: dict[str, Any] = {}
        if encoding is not None:
            kwargs["encoding"] = encoding
        if path.startswith("s3://"):
            kwargs.update(self.storage_options)
        return fsspec.open(path, mode=mode, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a versioned processed dataset from selected partitions."
    )
    parser.add_argument("--config", default="data_pipeline/configs/register_dataset.yaml")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    registration = DatasetRegistrationConfig.from_yaml(args.config)
    registrar = DatasetRegistrar(registration.pipeline_config_path)
    dataset = registrar.register(registration)

    print(f"dataset_id={dataset.dataset_id}")
    print(f"dataset_uri={dataset.dataset_uri}")
    print(f"manifest_uri={dataset.manifest_uri}")
    print(f"partitions={dataset.partition_count}")
    print(f"rows={dataset.row_count}")
    print(f"digest={dataset.digest}")


if __name__ == "__main__":
    main()
