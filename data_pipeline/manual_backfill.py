from __future__ import annotations

import argparse
from datetime import date, timedelta

from data_pipeline.script_configs import BackfillConfig
from data_pipeline.src.data_loader import DataLoader
from data_pipeline.src.data_transformer import DataTransformer


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load and transform daily partitions for a date range without Airflow."
    )
    parser.add_argument("--config", default="data_pipeline/configs/manual_backfill.yaml")
    parser.add_argument("--date-from", default=None, help="Inclusive start date in YYYY-MM-DD format.")
    parser.add_argument("--date-to", default=None, help="Inclusive end date in YYYY-MM-DD format.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = BackfillConfig.from_yaml(args.config)

    pipeline_config_path = config.pipeline_config_path
    date_from = args.date_from or config.date_from
    date_to = args.date_to or config.date_to

    loader = DataLoader(config_path=pipeline_config_path)
    transformer = DataTransformer(config_path=pipeline_config_path)

    for load_date in _iter_dates(date_from, date_to):
        try:
            raw_path = loader.run(load_date=load_date)
            transformed_path = transformer.run(load_date=load_date)
        except Exception:
            if config.stop_on_error:
                raise
            print(f"failed load_date={load_date}")
            continue

        print(f"load_date={load_date}")
        print(f"raw_path={raw_path}")
        print(f"transformed_path={transformed_path}")


if __name__ == "__main__":
    main()
