from __future__ import annotations

import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import requests

from .utils import load_yaml_config
from .write_dispatcher import WriteDispatcher


N_POSTS_PER_QUERY = 100


class DataLoader:
    """
    Загрузка сырых данных и сохранение в parquet без изменения значений ячеек.
    Оставляет только колонки из config.yaml → data_loader.features_to_select.
    Имя файла-партиции: load_date=YYYY-MM-DD.parquet
    """

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path).resolve()
        self._config: dict[str, Any] | None = None

    @property
    def config(self) -> dict[str, Any]:
        if self._config is None:
            self._config = load_yaml_config(str(self.config_path))
        return self._config

    def _dl(self) -> dict[str, Any]:
        return self.config.get("data_loader", {})

    def output_dir(self) -> str:
        raw = str(self._dl().get("output_dir", "artifacts/raw"))
        return WriteDispatcher.resolve_uri_or_path(raw, self.config_path.parent)

    def features_to_select(self) -> list[str]:
        return list(self._dl().get("features_to_select", []))

    @staticmethod
    def _materialize_column_for_parquet(s: pd.Series) -> pd.Series:
        """
        Столбцы на базе PyArrow (struct/map/list) не всегда имеют dtype object;
        пустой struct Arrow не пишет в Parquet. Разворачиваем в object + python list/dict.
        """
        dtype = s.dtype
        if isinstance(dtype, pd.ArrowDtype):
            ap = dtype.pyarrow_dtype
            if pa.types.is_struct(ap) or pa.types.is_map(ap) or pa.types.is_list(ap) or pa.types.is_large_list(ap):
                return pd.Series(s.tolist(), index=s.index, dtype=object)
        return s

    def select_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Только колонки из конфига, в заданном порядке; лишние (например gildings) отбрасываются."""
        cols = self.features_to_select()
        out = pd.DataFrame(index=df.index)
        for c in cols:
            if c not in df.columns:
                out[c] = pd.NA
            else:
                out[c] = self._materialize_column_for_parquet(df[c])
        return out

    @staticmethod
    def _check_columns_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
        """
        PyArrow не пишет пустые struct (частый кейс Reddit: пустой {{}} в dict-поле).
        Любые dict/list в ячейках сериализуем в JSON-строку.
        """
        out = df.copy()

        def _needs_json_normalize(ser: pd.Series) -> bool:
            for v in ser.dropna().head(256):
                if isinstance(v, (dict, list)):
                    return True
            return False

        def _norm(v: Any) -> Any:
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return v
            if isinstance(v, (dict, list)):
                return json.dumps(v, ensure_ascii=False, default=str)
            return v

        for c in out.columns:
            ser = out[c]
            if not _needs_json_normalize(ser):
                continue
            out[c] = ser.map(_norm)
        return out

    def _fetch_from_api(self, after: str, before: str) -> pd.DataFrame:
        cfg = self._dl()
        url = str(cfg["base_url"])
        subreddit = str(cfg["subreddit_name"])
        max_posts = int(cfg["max_posts_per_day"])
        sleep_s = float(cfg["request_sleep_seconds"])

        all_posts: list[dict] = []
        params: dict = {
            "subreddit": subreddit,
            "after": after,
            "before": before,
            "limit": cfg["n_posts_per_query"],
            "sort": "asc",
        }

        while len(all_posts) < max_posts:
            resp = requests.get(url, params=params, timeout=120)
            resp.raise_for_status()
            data = resp.json().get("data", [])

            if not data:
                break

            for post in data:
                meta = post.get("_meta") or {}
                post["retrieved_2nd_on"] = meta.get("retrieved_2nd_on")
                post.pop("_meta", None)
                all_posts.append(post)

            last_utc = data[-1]["created_utc"]
            params["after"] = last_utc
            time.sleep(sleep_s)

        return pd.DataFrame(all_posts)

    def _fetch_raw_dataframe(self, load_date: date | str) -> pd.DataFrame:
        raw_csv = self._dl().get("raw_csv_path")
        if raw_csv:
            path = Path(raw_csv)
            if not path.is_absolute():
                path = (self.config_path.parent / path).resolve()
            return pd.read_csv(path)

        d = load_date if isinstance(load_date, date) else date.fromisoformat(str(load_date))
        day_before = d - timedelta(days=1)
        after = day_before.isoformat()
        before = d.isoformat()
        return self._fetch_from_api(after=after, before=before)

    def load_raw(self, load_date: date | str) -> pd.DataFrame:
        """Источник по конфигу (CSV или API), полный набор колонок источника."""
        return self._fetch_raw_dataframe(load_date)

    def load_and_select(self, load_date: date | str) -> pd.DataFrame:
        """Сырой df → только features_to_select."""
        return self.select_columns(self.load_raw(load_date))

    def partition_path(self, load_date: date | str) -> str:
        d = load_date if isinstance(load_date, date) else date.fromisoformat(str(load_date))
        name = f"load_date={d.isoformat()}.parquet"
        return f"{self.output_dir()}/{name}"

    def save_partition(self, df: pd.DataFrame, load_date: date | str) -> str:
        """Сохранить партицию: всегда только features_to_select + безопасные типы для Parquet."""
        path = self.partition_path(load_date)
        narrowed = self.select_columns(df)
        safe = self._check_columns_for_parquet(narrowed)
        writer = WriteDispatcher(s3_storage_options=WriteDispatcher.s3_options_from_config(self.config))
        return writer.save_parquet(safe, path, index=False)

    def run(self, load_date: date | str) -> str:
        """Загрузить, отфильтровать колонки, сохранить партицию."""
        df = self.load_and_select(load_date)
        return self.save_partition(df, load_date)
