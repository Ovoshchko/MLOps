from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from data_pipeline.src.utils import load_yaml_config


@dataclass(frozen=True)
class BackfillConfig:
    pipeline_config_path: str
    date_from: str
    date_to: str
    stop_on_error: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BackfillConfig":
        resolved = Path(path).resolve()
        config = load_yaml_config(str(resolved)).get("backfill", {})
        return cls(
            pipeline_config_path=str(config.get("pipeline_config_path", "data_pipeline/config.yaml")),
            date_from=str(config["date_from"]),
            date_to=str(config["date_to"]),
            stop_on_error=bool(config.get("stop_on_error", True)),
        )


@dataclass(frozen=True)
class DatasetRegistrationConfig:
    pipeline_config_path: str
    dataset_id: str | None
    version: str
    dataset_root: str | None
    partition_paths: list[str]
    date_from: str | None
    date_to: str | None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DatasetRegistrationConfig":
        resolved = Path(path).resolve()
        config = load_yaml_config(str(resolved))
        registration_cfg = config.get("dataset_registration", {})
        raw_partition_paths = registration_cfg.get("partition_paths") or []

        return cls(
            pipeline_config_path=str(registration_cfg.get("pipeline_config_path", "data_pipeline/config.yaml")),
            dataset_id=str(registration_cfg["dataset_id"]) if registration_cfg.get("dataset_id") is not None else None,
            version=str(registration_cfg.get("version", "v1")),
            dataset_root=str(registration_cfg["dataset_root"]) if registration_cfg.get("dataset_root") is not None else None,
            partition_paths=[str(path) for path in raw_partition_paths],
            date_from=str(registration_cfg["date_from"]) if registration_cfg.get("date_from") is not None else None,
            date_to=str(registration_cfg["date_to"]) if registration_cfg.get("date_to") is not None else None,
        )
