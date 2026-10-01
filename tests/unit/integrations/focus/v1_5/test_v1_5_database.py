from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone

import pydantic
import pytest

from litellm.integrations.focus.v1_5.database import SPEND_LOG_BUCKETS_SQL, FocusSpendLogsDatabase

START = datetime(2026, 5, 25, tzinfo=timezone.utc)
END = datetime(2026, 5, 26, tzinfo=timezone.utc)
BUCKET = {
    "charge_period_start": "2026-05-25T00:00:00Z",
    "charge_period_end": "2026-05-26T00:00:00Z",
    "principal_id": "hermes",
    "principal_name": None,
    "principal_email": None,
    "credential_id": "33b20aab1a63380e19e8",
    "credential_name": "hermes-primary",
    "model": "openai/gpt-5.4-mini",
    "custom_llm_provider": "openai",
    "team_id": None,
    "team_alias": None,
    "service_tier": "default",
    "request_tags": "[]",
    "meter": "Output Tokens",
    "quantity": 7,
    "list_cost": 0.0000315,
    "contracted_cost": 0.0000315,
    "billed_cost": 0.0000315,
}


class _RecordingDb:
    def __init__(self, rows: Sequence[Mapping[str, object]]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, tuple[object, ...]]] = []  # mutable-ok: records calls for assertions

    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]:
        self.calls.append((query, args))
        return self.rows


def _database(db: _RecordingDb, *, disabled: bool = False) -> FocusSpendLogsDatabase:
    return FocusSpendLogsDatabase(granularity="daily", resolve_db=lambda: db, spend_logs_disabled=lambda: disabled)


@pytest.mark.asyncio
async def test_passes_day_buckets_window_and_limit_to_the_query() -> None:
    db = _RecordingDb([BUCKET])

    frame = await _database(db).get_usage_data(limit=10, start_time_utc=START, end_time_utc=END)

    assert db.calls == [(SPEND_LOG_BUCKETS_SQL, ("day", START, END, 10))]
    assert frame.to_dicts() == [BUCKET]


@pytest.mark.asyncio
async def test_refuses_to_export_when_spend_logs_are_disabled() -> None:
    db = _RecordingDb([BUCKET])

    with pytest.raises(RuntimeError, match="disable_spend_logs"):
        await _database(db, disabled=True).get_usage_data()
    assert db.calls == []


@pytest.mark.asyncio
async def test_rejects_negative_limit_before_querying() -> None:
    db = _RecordingDb([])

    with pytest.raises(ValueError, match="non-negative"):
        await _database(db).get_usage_data(limit=-1)
    assert db.calls == []


@pytest.mark.asyncio
async def test_rejects_rows_that_do_not_match_the_bucket_shape() -> None:
    db = _RecordingDb([{**BUCKET, "meter": "Reasoning Tokens"}])

    with pytest.raises(pydantic.ValidationError):
        await _database(db).get_usage_data()
