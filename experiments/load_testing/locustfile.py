import os
from typing import Any

import numpy as np
import pandas as pd
from locust import HttpUser, between, task

BATCH_SIZE = 1
PARQUET_PATH = "load_date=2025-11-25.parquet"
ENDPOINT = os.environ.get("LOCUST_HOST", "http://127.0.0.1:8000")


class PredictUser(HttpUser):
    host = ENDPOINT
    wait_time = between(0.05, 0.2)

    def on_start(self) -> None:
        self.df = pd.read_parquet(PARQUET_PATH).replace({np.nan: None})
        self.batch_size = min(BATCH_SIZE, len(self.df))

    @task
    def predict(self) -> None:
        idx = np.random.choice(len(self.df), size=self.batch_size, replace=False)
        rows = [self.df.iloc[i].to_dict() for i in idx]
        self.client.post("/predict", json={"rows": rows}, name="/predict")
