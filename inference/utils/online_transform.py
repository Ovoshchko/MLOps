from __future__ import annotations

from typing import Any

import pandas as pd

from common.column_transforms import apply_transforms, normalize_column_transforms


def transform_features(
    df: pd.DataFrame,
    manifest: dict[str, Any],
    embedder: Any,
    *,
    embedding_batch_size: int,
) -> pd.DataFrame:
    return apply_transforms(
        df,
        feature_names=list(manifest["features"]),
        transforms=normalize_column_transforms(manifest.get("column_transforms")),
        embedder=embedder,
        embedding_batch_size=embedding_batch_size,
    )
