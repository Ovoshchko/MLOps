import os
import time
from pathlib import Path

import boto3
import pandas as pd
import requests


BASE_URL = "https://arctic-shift.photon-reddit.com/api/posts/search"


def get_env_config():
    return {
        "subreddits": [item.strip() for item in os.getenv("REDDIT_SUBREDDITS", "technology").split(",") if item.strip()],
        "max_posts": int(os.getenv("REDDIT_MAX_POSTS_PER_WINDOW", "500000")),
        "sleep_seconds": float(os.getenv("REDDIT_SLEEP_SECONDS", "0.2")),
        "output_dir": Path(os.getenv("REDDIT_OUTPUT_DIR", "/opt/airflow/data")),
        "minio_endpoint_url": os.getenv("MINIO_ENDPOINT_URL", "http://minio:9000"),
        "minio_access_key": os.getenv("MINIO_ACCESS_KEY", "minio"),
        "minio_secret_key": os.getenv("MINIO_SECRET_KEY", "minio123"),
        "minio_bucket": os.getenv("MINIO_BUCKET", "reddit-processed"),
        "minio_region": os.getenv("MINIO_REGION", "us-east-1"),
    }


def fetch_subreddit_posts(subreddit, after, before, max_posts, sleep_seconds):
    all_posts = []
    seen_ids = set()
    params = {
        "subreddit": subreddit,
        "after": after,
        "before": before,
        "limit": 100,
        "sort": "asc",
    }

    while len(all_posts) < max_posts:
        response = requests.get(BASE_URL, params=params, timeout=60)
        response.raise_for_status()
        data = response.json().get("data", [])

        if not data:
            break

        added_this_page = 0

        for post in data:
            if len(all_posts) >= max_posts:
                break

            post_id = post.get("id")
            if not post_id or post_id in seen_ids:
                continue

            seen_ids.add(post_id)
            all_posts.append(post)
            added_this_page += 1

        if added_this_page == 0:
            break

        last_utc = data[-1].get("created_utc")
        if last_utc is None:
            break

        params["after"] = last_utc + 1
        time.sleep(sleep_seconds)

    return all_posts


def extract_window(subreddits, window_start, window_end, max_posts, sleep_seconds):
    records = []
    after = int(window_start.timestamp())
    before = int(window_end.timestamp())

    for subreddit in subreddits:
        records.extend(
            fetch_subreddit_posts(
                subreddit=subreddit,
                after=after,
                before=before,
                max_posts=max_posts,
                sleep_seconds=sleep_seconds,
            )
        )

    if not records:
        return pd.DataFrame()

    return pd.DataFrame.from_records(records).drop_duplicates(subset="id")


def preprocess_posts(df, mode, window_start, window_end, collected_at):
    if df.empty:
        return df

    working_df = df.copy()

    numeric_columns = [
        "created_utc",
        "retrieved_on",
        "score",
        "ups",
        "upvote_ratio",
        "num_comments",
        "subreddit_subscribers",
        "thumbnail_height",
        "thumbnail_width",
    ]
    for column in numeric_columns:
        if column in working_df.columns:
            working_df[column] = pd.to_numeric(working_df[column], errors="coerce")

    working_df["created_dt"] = pd.to_datetime(working_df["created_utc"], unit="s", utc=True, errors="coerce")
    working_df["retrieved_dt"] = pd.to_datetime(working_df["retrieved_on"], unit="s", utc=True, errors="coerce")
    working_df["age_seconds"] = (working_df["retrieved_dt"] - working_df["created_dt"]).dt.total_seconds()
    working_df["target_horizon_hours"] = working_df["age_seconds"] / 3600
    working_df["created_date"] = working_df["created_dt"].dt.strftime("%Y-%m-%d")
    working_df["created_hour"] = working_df["created_dt"].dt.hour
    working_df["created_day_of_week"] = working_df["created_dt"].dt.day_name()
    working_df["title"] = working_df["title"].fillna("").astype(str)
    working_df["selftext"] = working_df["selftext"].fillna("").astype(str)
    working_df["domain"] = working_df["domain"].fillna("unknown").astype(str)
    working_df["link_flair_text"] = working_df["link_flair_text"].fillna("unknown").astype(str)
    working_df["post_hint"] = working_df["post_hint"].fillna("unknown").astype(str)
    working_df["thumbnail"] = working_df["thumbnail"].fillna("missing").astype(str)
    working_df["removed_by_category"] = working_df["removed_by_category"].fillna("not_removed").astype(str)
    working_df["title_len"] = working_df["title"].str.len()
    working_df["selftext_len"] = working_df["selftext"].str.len()
    working_df["title_word_count"] = working_df["title"].str.split().str.len()
    working_df["has_selftext"] = working_df["selftext_len"] > 0
    working_df["has_thumbnail"] = ~working_df["thumbnail"].isin(["self", "default", "nsfw", "image", "spoiler", "missing", ""])
    working_df["is_removed"] = working_df["removed_by_category"].ne("not_removed")
    working_df["title_has_question"] = working_df["title"].str.contains(r"\?", regex=True)
    working_df["title_has_number"] = working_df["title"].str.contains(r"\d", regex=True)
    working_df["title_upper_share"] = working_df["title"].str.count(r"[A-Z]").div(working_df["title_len"].replace(0, pd.NA)).fillna(0.0)
    working_df["pipeline_mode"] = mode
    working_df["window_start"] = window_start
    working_df["window_end"] = window_end
    working_df["collected_at"] = collected_at.isoformat()
    working_df["ingestion_date"] = collected_at.strftime("%Y-%m-%d")
    working_df["ingestion_time"] = collected_at.strftime("%H-%M-%S")

    for column in ["is_self", "is_video", "over_18", "spoiler", "stickied", "locked"]:
        if column in working_df.columns:
            working_df[column] = working_df[column].fillna(False).astype(bool)

    processed_df = working_df.drop(columns=["id", "url", "score", "retrieved_on"], errors="ignore")
    processed_df = processed_df.sort_values(["created_utc", "subreddit", "title"]).reset_index(drop=True)

    return processed_df


def build_storage_paths(output_dir, mode, collected_at):
    date_part = collected_at.strftime("%Y-%m-%d")
    time_part = collected_at.strftime("%H-%M-%S")
    local_dir = output_dir / f"mode={mode}" / f"date={date_part}" / f"time={time_part}"
    local_file = local_dir / "reddit_posts.parquet"
    object_key = f"reddit/mode={mode}/date={date_part}/time={time_part}/reddit_posts.parquet"
    return local_dir, local_file, object_key


def get_s3_client(endpoint_url, access_key, secret_key, region):
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )


def ensure_bucket(s3_client, bucket_name):
    existing_buckets = {bucket["Name"] for bucket in s3_client.list_buckets().get("Buckets", [])}
    if bucket_name not in existing_buckets:
        s3_client.create_bucket(Bucket=bucket_name)


def write_and_upload(df, mode, output_dir, minio_endpoint_url, minio_access_key, minio_secret_key, minio_bucket, minio_region, collected_at):
    local_dir, local_file, object_key = build_storage_paths(output_dir=output_dir, mode=mode, collected_at=collected_at)
    local_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(local_file, index=False)

    s3_client = get_s3_client(
        endpoint_url=minio_endpoint_url,
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        region=minio_region,
    )
    ensure_bucket(s3_client, minio_bucket)
    s3_client.upload_file(str(local_file), minio_bucket, object_key)

    return local_file, object_key


def run_collection_job(mode, window_start, window_end):
    config = get_env_config()
    collected_at = pd.Timestamp.now(tz="UTC")
    raw_df = extract_window(
        subreddits=config["subreddits"],
        window_start=window_start,
        window_end=window_end,
        max_posts=config["max_posts"],
        sleep_seconds=config["sleep_seconds"],
    )
    processed_df = preprocess_posts(
        df=raw_df,
        mode=mode,
        window_start=window_start.isoformat(),
        window_end=window_end.isoformat(),
        collected_at=collected_at,
    )

    if processed_df.empty:
        return {
            "mode": mode,
            "rows": 0,
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "local_path": None,
            "minio_object": None,
        }

    local_file, object_key = write_and_upload(
        df=processed_df,
        mode=mode,
        output_dir=config["output_dir"],
        minio_endpoint_url=config["minio_endpoint_url"],
        minio_access_key=config["minio_access_key"],
        minio_secret_key=config["minio_secret_key"],
        minio_bucket=config["minio_bucket"],
        minio_region=config["minio_region"],
        collected_at=collected_at,
    )

    return {
        "mode": mode,
        "rows": int(len(processed_df)),
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "local_path": str(local_file),
        "minio_bucket": config["minio_bucket"],
        "minio_object": object_key,
    }
