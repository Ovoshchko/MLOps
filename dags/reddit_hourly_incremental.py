import pendulum
from airflow.decorators import dag, task
from airflow.operators.python import get_current_context

from reddit_pipeline.reddit_ingestion import run_collection_job


@dag(
    dag_id="reddit_hourly_incremental",
    schedule="0 * * * *",
    start_date=pendulum.datetime(2026, 3, 1, tz="UTC"),
    catchup=False,
    tags=["reddit", "hourly", "minio", "parquet"],
)
def reddit_hourly_incremental():
    @task
    def collect():
        context = get_current_context()
        window_start = context["data_interval_start"].in_timezone("UTC")
        window_end = context["data_interval_end"].in_timezone("UTC")
        return run_collection_job(
            mode="hourly",
            window_start=window_start,
            window_end=window_end,
        )

    collect()


reddit_hourly_incremental()
