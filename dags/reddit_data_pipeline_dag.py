from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task
from airflow.operators.python import get_current_context


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.src.data_loader import DataLoader
from data_pipeline.src.data_transformer import DataTransformer


@dag(
    dag_id="reddit_data_pipeline",
    description="Daily Reddit load and transform pipeline",
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["reddit", "data_pipeline"],
)
def reddit_data_pipeline():
    config_path = str(PROJECT_ROOT / "data_pipeline" / "config.yaml")

    @task(task_id="load_data")
    def load_data() -> dict[str, str]:
        ctx = get_current_context()
        dag_run = ctx.get("dag_run")
        dag_conf = dag_run.conf if dag_run and dag_run.conf else {}
        load_date = str(dag_conf.get("load_date", ctx["ds"]))
        datetime.strptime(load_date, "%Y-%m-%d")
        loader = DataLoader(config_path=config_path)
        output_path = loader.run(load_date=load_date)
        return {"load_date": load_date, "raw_path": output_path}

    @task(task_id="transform_data")
    def transform_data(payload: dict[str, str]) -> dict[str, str]:
        transformer = DataTransformer(config_path=config_path)
        output_path = transformer.run(load_date=payload["load_date"])
        return {"load_date": payload["load_date"], "transformed_path": output_path}

    transform_data(load_data())


reddit_data_pipeline()
