import json
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import tritonclient.grpc as grpcclient
from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict

from common.yaml import load_yaml_config
from experiments.features.build import apply_feature_engineering, build_X, infer_feature_columns
from experiments.tracking.mlflow_tracking import configure_mlflow
from inference.utils.online_transform import transform_features
from inference.utils.predict_schema import build_input_row_model, manifest_defaults


configure_mlflow()
config = load_yaml_config(str(Path(__file__).parent / "config.yml"))


def load_manifest(uri: str) -> dict:
    artifact = mlflow.artifacts.download_artifacts(uri)
    with open(artifact, "r") as f:
        return json.load(f)


_manifest = load_manifest(config["mlflow_schema_path"])
_field_aliases: dict[str, str] = dict(config.get("input_mapping") or {})
InputRow = build_input_row_model(_manifest, _field_aliases)
_defaults = manifest_defaults(_manifest)
_feature_names = list(_manifest["features"])


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    rows: list[InputRow]


class PredictResponse(BaseModel):
    predictions: list[float]


def _build_dataframe(rows: list[InputRow]) -> pd.DataFrame:
    """Materialize validated rows into a DataFrame with type-appropriate defaults."""
    filled = []
    for row in rows:
        d = row.model_dump()
        filled.append(
            {col: (d[col] if d[col] is not None else default)
             for col, default in _defaults.items()}
        )
    return pd.DataFrame(filled, columns=_feature_names)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from sentence_transformers import SentenceTransformer

    embed_cfg = config["embedder"]
    triton_cfg = config.get("triton", {})
    app.state.embedder = SentenceTransformer(embed_cfg["model_name"])
    app.state.embedding_batch_size = int(embed_cfg["embedding_batch_size"])
    app.state.feature_config = dict(config.get("features", {}))
    app.state.triton = grpcclient.InferenceServerClient(
        url=str(triton_cfg.get("grpc_url", "localhost:8001")),
    )
    app.state.triton_model_name = str(triton_cfg.get("model_name", "regression"))
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/predict", response_model=PredictResponse)
def predict(request: Request, body: PredictRequest) -> PredictResponse:
    df = _build_dataframe(body.rows)
    df = transform_features(
        df,
        _manifest,
        request.app.state.embedder,
        embedding_batch_size=request.app.state.embedding_batch_size,
    )
    df = apply_feature_engineering(df, request.app.state.feature_config)

    feature_columns = infer_feature_columns(df, target_col=_manifest["target_col"])
    x = np.ascontiguousarray(build_X(df, feature_columns), dtype=np.float32)

    inp = grpcclient.InferInput("features", x.shape, "FP32")
    inp.set_data_from_numpy(x)
    out = request.app.state.triton.infer(
        request.app.state.triton_model_name,
        inputs=[inp],
        outputs=[grpcclient.InferRequestedOutput("predictions")],
    )
    return PredictResponse(predictions=out.as_numpy("predictions").reshape(-1).tolist())
