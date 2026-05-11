from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import tritonclient.grpc as grpcclient
from pydantic import BaseModel

from common.yaml import load_yaml_config
from experiments.features.build import apply_feature_engineering, build_X, infer_feature_columns
from experiments.tracking.mlflow_tracking import configure_mlflow
from inference.utils.online_transform import transform_features
from inference.utils.predict_schema import build_input_row_model, manifest_defaults


def load_manifest(uri: str) -> dict[str, Any]:
    artifact = mlflow.artifacts.download_artifacts(uri)
    with open(artifact, "r") as f:
        return json.load(f)


class PredictionService:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.manifest = load_manifest(config["mlflow_schema_path"])
        self.field_aliases = dict(config.get("input_mapping") or {})
        self.input_row_model = build_input_row_model(self.manifest, self.field_aliases)
        self.defaults = manifest_defaults(self.manifest)
        self.feature_names = list(self.manifest["features"])
        self.embedder = None
        self.embedding_batch_size = 0
        self.feature_config: dict[str, Any] = {}
        self.triton: grpcclient.InferenceServerClient | None = None
        self.triton_model_name = ""

    @classmethod
    def from_config_path(cls, path: str | Path) -> "PredictionService":
        configure_mlflow()
        return cls(load_yaml_config(str(path)))

    def start(self) -> None:
        from sentence_transformers import SentenceTransformer

        embed_cfg = self.config["embedder"]
        triton_cfg = self.config.get("triton", {})
        self.embedder = SentenceTransformer(embed_cfg["model_name"])
        self.embedding_batch_size = int(embed_cfg["embedding_batch_size"])
        self.feature_config = dict(self.config.get("features", {}))
        self.triton = grpcclient.InferenceServerClient(
            url=str(triton_cfg.get("grpc_url", "localhost:8001")),
        )
        self.triton_model_name = str(triton_cfg.get("model_name", "regression"))

    def validate_rows(self, rows: list[dict[str, Any] | BaseModel]) -> list[BaseModel]:
        validated = []
        for row in rows:
            if isinstance(row, BaseModel):
                validated.append(row)
            else:
                validated.append(self.input_row_model.model_validate(row))
        return validated

    def predict_rows(self, rows: list[dict[str, Any] | BaseModel]) -> list[float]:
        if self.embedder is None or self.triton is None:
            raise RuntimeError("PredictionService is not started")

        df = self._build_dataframe(self.validate_rows(rows))
        df = transform_features(
            df,
            self.manifest,
            self.embedder,
            embedding_batch_size=self.embedding_batch_size,
        )
        df = apply_feature_engineering(df, self.feature_config)

        feature_columns = infer_feature_columns(df, target_col=self.manifest["target_col"])
        x = np.ascontiguousarray(build_X(df, feature_columns), dtype=np.float32)

        inp = grpcclient.InferInput("features", x.shape, "FP32")
        inp.set_data_from_numpy(x)
        out = self.triton.infer(
            self.triton_model_name,
            inputs=[inp],
            outputs=[grpcclient.InferRequestedOutput("predictions")],
        )
        return out.as_numpy("predictions").reshape(-1).tolist()

    def _build_dataframe(self, rows: list[BaseModel]) -> pd.DataFrame:
        filled = []
        for row in rows:
            data = row.model_dump()
            filled.append(
                {
                    col: data[col] if data[col] is not None else default
                    for col, default in self.defaults.items()
                }
            )
        return pd.DataFrame(filled, columns=self.feature_names)
