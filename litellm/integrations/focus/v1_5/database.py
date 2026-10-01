"""LiteLLM_SpendLogs access for FOCUS 1.5 export."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Final, Literal, Protocol, TypeAlias

import polars as pl
from pydantic import TypeAdapter
from typing_extensions import ReadOnly, TypedDict

from ..settings import FocusDataGranularity

BucketUnit: TypeAlias = Literal["day"]

SPEND_LOG_BUCKETS_SQL: Final = """
WITH bucketed AS (
    SELECT
        date_trunc($1::text, sl."startTime") AS bucket_start,
        NULLIF(sl."user", '') AS principal_id,
        NULLIF(sl.api_key, '') AS credential_id,
        NULLIF(sl.model, '') AS model,
        NULLIF(sl.custom_llm_provider, '') AS custom_llm_provider,
        NULLIF(sl.team_id, '') AS team_id,
        sl.spend
    FROM "LiteLLM_SpendLogs" sl
    WHERE ($2::timestamptz IS NULL OR sl."startTime" >= ($2::timestamptz AT TIME ZONE 'UTC'))
      AND ($3::timestamptz IS NULL OR sl."startTime" < ($3::timestamptz AT TIME ZONE 'UTC'))
)
SELECT
    to_char(bucket_start, 'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS charge_period_start,
    to_char(bucket_start + ('1 ' || $1::text)::interval, 'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS charge_period_end,
    principal_id,
    credential_id,
    model,
    custom_llm_provider,
    team_id,
    SUM(spend) AS spend
FROM bucketed
GROUP BY bucket_start, principal_id, credential_id, model, custom_llm_provider, team_id
ORDER BY bucket_start, principal_id, credential_id, model, custom_llm_provider, team_id
LIMIT $4
"""

SPEND_LOG_BUCKET_SCHEMA: Final = pl.Schema(
    (
        ("charge_period_start", pl.String),
        ("charge_period_end", pl.String),
        ("principal_id", pl.String),
        ("credential_id", pl.String),
        ("model", pl.String),
        ("custom_llm_provider", pl.String),
        ("team_id", pl.String),
        ("spend", pl.Float64),
    )
)


class SpendLogBucket(TypedDict):
    charge_period_start: ReadOnly[str]
    charge_period_end: ReadOnly[str]
    principal_id: ReadOnly[str | None]
    credential_id: ReadOnly[str | None]
    model: ReadOnly[str | None]
    custom_llm_provider: ReadOnly[str | None]
    team_id: ReadOnly[str | None]
    spend: ReadOnly[float]


_BUCKETS_ADAPTER: Final = TypeAdapter(tuple[SpendLogBucket, ...])


class SupportsQueryRaw(Protocol):
    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]: ...


def _proxy_db() -> SupportsQueryRaw:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise RuntimeError(
            "Database not connected. Connect a database to your proxy - "
            "https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
        )
    return prisma_client.db


def _proxy_spend_logs_disabled() -> bool:
    from litellm.proxy import proxy_server

    return bool(proxy_server.disable_spend_logs)


def _bucket_unit(granularity: FocusDataGranularity) -> BucketUnit:
    match granularity:
        case "daily":
            return "day"


class FocusSpendLogsDatabase:
    """Sums LiteLLM_SpendLogs into per-period buckets keyed by principal, credential, model, provider and team."""

    def __init__(
        self,
        *,
        granularity: FocusDataGranularity,
        resolve_db: Callable[[], SupportsQueryRaw] = _proxy_db,
        spend_logs_disabled: Callable[[], bool] = _proxy_spend_logs_disabled,
    ) -> None:
        self._bucket_unit: Final = _bucket_unit(granularity)
        self._resolve_db: Final = resolve_db
        self._spend_logs_disabled: Final = spend_logs_disabled

    async def get_usage_data(
        self,
        *,
        limit: int | None = None,
        start_time_utc: datetime | None = None,
        end_time_utc: datetime | None = None,
    ) -> pl.DataFrame:
        """Return one row per bucket in [start_time_utc, end_time_utc); an unset bound is unbounded."""
        if self._spend_logs_disabled():
            raise RuntimeError("FOCUS 1.5 export reads LiteLLM_SpendLogs, but disable_spend_logs is enabled")
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")
        rows: Final = await self._resolve_db().query_raw(
            SPEND_LOG_BUCKETS_SQL, self._bucket_unit, start_time_utc, end_time_utc, limit
        )
        buckets: Final = _BUCKETS_ADAPTER.validate_python(rows)
        return pl.DataFrame(buckets, schema=SPEND_LOG_BUCKET_SCHEMA)


__all__ = ("SPEND_LOG_BUCKETS_SQL", "FocusSpendLogsDatabase", "SpendLogBucket", "SupportsQueryRaw")
