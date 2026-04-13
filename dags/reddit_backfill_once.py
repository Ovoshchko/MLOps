import pendulum

from airflow.decorators import dag, task

from reddit_pipeline.reddit_ingestion import run_collection_job


@dag(
    dag_id="reddit_backfill_once",
    schedule="@once",
    start_date=pendulum.datetime(2026, 3, 1, tz="UTC"),
    catchup=False,
    tags=["reddit", "backfill", "minio", "parquet"],
)
def reddit_backfill_once():
    @task
    def collect():
        now = pendulum.now("UTC")
        window_start = pendulum.datetime(now.year, 3, 1, tz="UTC")
        window_end = now.start_of("day")
        return run_collection_job(
            mode="backfill",
            window_start=window_start,
            window_end=window_end,
        )

    collect()


reddit_backfill_once()
