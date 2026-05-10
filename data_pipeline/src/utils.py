from __future__ import annotations

from datetime import date, timedelta


def iter_dates(date_from: str, date_to: str) -> list[str]:
    """Inclusive date range as a list of ``YYYY-MM-DD`` strings."""
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
