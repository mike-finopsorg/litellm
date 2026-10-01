"""LiteLLM_SpendLogs access for FOCUS 1.5 export."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Final, Literal, Protocol, TypeAlias

import polars as pl
from pydantic import TypeAdapter
from typing_extensions import ReadOnly, TypedDict

from ..settings import FocusDataGranularity

BucketUnit: TypeAlias = Literal["day", "hour"]
SkuMeter: TypeAlias = Literal[
    "Input Tokens", "Cached Input Tokens", "Cache Write Tokens", "Output Tokens", "Other Usage"
]

SPEND_LOG_BUCKETS_SQL: Final = """
WITH logs AS (
    SELECT
        date_trunc($1::text, sl."startTime") AS bucket_start,
        NULLIF(sl."user", '') AS principal_id,
        NULLIF(sl.api_key, '') AS credential_id,
        NULLIF(sl.metadata->>'user_api_key_alias', '') AS credential_name,
        NULLIF(sl.model, '') AS model,
        NULLIF(sl.model_id, '') AS model_id,
        NULLIF(sl.model_group, '') AS model_group,
        NULLIF(sl.custom_llm_provider, '') AS custom_llm_provider,
        NULLIF(sl.team_id, '') AS team_id,
        NULLIF(sl.metadata->>'user_api_key_team_alias', '') AS team_alias,
        NULLIF(sl.metadata->'cost_breakdown'->>'service_tier', '') AS service_tier,
        CASE WHEN jsonb_typeof(sl.request_tags) = 'array' THEN sl.request_tags ELSE '[]'::jsonb END AS request_tags,
        sl.spend,
        jsonb_typeof(sl.metadata->'cost_breakdown') = 'object' AS has_cost_breakdown,
        GREATEST(sl.prompt_tokens, 0)::bigint AS prompt_tokens,
        GREATEST(sl.completion_tokens, 0)::bigint AS completion_tokens,
        COALESCE((sl.metadata->'usage_object'->'prompt_tokens_details'->>'cached_tokens')::bigint, 0) AS cached_tokens,
        GREATEST(
            COALESCE((sl.metadata->'usage_object'->'prompt_tokens_details'->>'cache_creation_tokens')::bigint, 0),
            COALESCE((sl.metadata->'usage_object'->>'cache_creation_input_tokens')::bigint, 0)
        ) AS cache_write_tokens,
        COALESCE((sl.metadata->'cost_breakdown'->>'input_cost')::float8, 0) AS input_cost,
        COALESCE((sl.metadata->'cost_breakdown'->>'cache_read_cost')::float8, 0) AS cache_read_cost,
        COALESCE((sl.metadata->'cost_breakdown'->>'cache_creation_cost')::float8, 0) AS cache_creation_cost,
        COALESCE((sl.metadata->'cost_breakdown'->>'output_cost')::float8, 0) AS output_cost,
        COALESCE((sl.metadata->'cost_breakdown'->>'original_cost')::float8, 0) AS original_cost,
        COALESCE((sl.metadata->'cost_breakdown'->>'discount_percent')::float8, 0) AS discount_percent
    FROM "LiteLLM_SpendLogs" sl
    WHERE ($2::timestamptz IS NULL OR sl."startTime" >= ($2::timestamptz AT TIME ZONE 'UTC'))
      AND ($3::timestamptz IS NULL OR sl."startTime" < ($3::timestamptz AT TIME ZONE 'UTC'))
),
skus AS (
    SELECT
        logs.*,
        sku.meter,
        sku.quantity,
        sku.list_cost,
        CASE
            WHEN sku.meter = 'Other Usage' AND logs.original_cost = 0 THEN logs.spend
            WHEN sku.meter = 'Other Usage'
                THEN logs.spend - (logs.input_cost + logs.output_cost) * logs.spend / logs.original_cost
            WHEN logs.original_cost = 0 THEN 0
            ELSE sku.list_cost * logs.spend / logs.original_cost
        END AS billed_cost
    FROM logs
    CROSS JOIN LATERAL (
        VALUES
            (
                'Input Tokens',
                GREATEST(logs.prompt_tokens - logs.cached_tokens - logs.cache_write_tokens, 0),
                logs.input_cost - logs.cache_read_cost - logs.cache_creation_cost
            ),
            ('Cached Input Tokens', logs.cached_tokens, logs.cache_read_cost),
            ('Cache Write Tokens', logs.cache_write_tokens, logs.cache_creation_cost),
            ('Output Tokens', logs.completion_tokens, logs.output_cost),
            (
                'Other Usage',
                1::bigint,
                CASE
                    WHEN logs.has_cost_breakdown THEN logs.original_cost - logs.input_cost - logs.output_cost
                    ELSE logs.spend
                END
            )
    ) AS sku(meter, quantity, list_cost)
),
usage_rows AS (
    SELECT * FROM skus
    WHERE (meter <> 'Other Usage' AND quantity <> 0) OR abs(list_cost) > 1e-12 OR abs(billed_cost) > 1e-12
)
SELECT
    to_char(u.bucket_start, 'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS charge_period_start,
    to_char(u.bucket_start + ('1 ' || $1::text)::interval, 'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS charge_period_end,
    u.principal_id,
    ut.user_alias AS principal_name,
    ut.user_email AS principal_email,
    u.credential_id,
    u.credential_name,
    u.model,
    u.model_id,
    u.model_group,
    u.custom_llm_provider,
    u.team_id,
    u.team_alias,
    u.service_tier,
    u.request_tags::text AS request_tags,
    u.meter,
    SUM(u.quantity)::bigint AS quantity,
    SUM(u.list_cost) AS list_cost,
    SUM(u.list_cost * (1 - u.discount_percent)) AS contracted_cost,
    SUM(u.billed_cost) AS billed_cost
FROM usage_rows u
LEFT JOIN "LiteLLM_UserTable" ut ON ut.user_id = u.principal_id
GROUP BY
    u.bucket_start, u.principal_id, ut.user_alias, ut.user_email, u.credential_id, u.credential_name, u.model,
    u.model_id, u.model_group, u.custom_llm_provider, u.team_id, u.team_alias, u.service_tier, u.request_tags, u.meter
ORDER BY
    u.bucket_start, u.principal_id, u.credential_id, u.model, u.model_id, u.custom_llm_provider, u.team_id, u.meter
LIMIT $4
"""

SPEND_LOG_BUCKET_SCHEMA: Final = pl.Schema(
    (
        ("charge_period_start", pl.String),
        ("charge_period_end", pl.String),
        ("principal_id", pl.String),
        ("principal_name", pl.String),
        ("principal_email", pl.String),
        ("credential_id", pl.String),
        ("credential_name", pl.String),
        ("model", pl.String),
        ("model_id", pl.String),
        ("model_group", pl.String),
        ("custom_llm_provider", pl.String),
        ("team_id", pl.String),
        ("team_alias", pl.String),
        ("service_tier", pl.String),
        ("request_tags", pl.String),
        ("meter", pl.String),
        ("quantity", pl.Int64),
        ("list_cost", pl.Float64),
        ("contracted_cost", pl.Float64),
        ("billed_cost", pl.Float64),
    )
)


class SpendLogBucket(TypedDict):
    charge_period_start: ReadOnly[str]
    charge_period_end: ReadOnly[str]
    principal_id: ReadOnly[str | None]
    principal_name: ReadOnly[str | None]
    principal_email: ReadOnly[str | None]
    credential_id: ReadOnly[str | None]
    credential_name: ReadOnly[str | None]
    model: ReadOnly[str | None]
    model_id: ReadOnly[str | None]
    model_group: ReadOnly[str | None]
    custom_llm_provider: ReadOnly[str | None]
    team_id: ReadOnly[str | None]
    team_alias: ReadOnly[str | None]
    service_tier: ReadOnly[str | None]
    request_tags: ReadOnly[str]
    meter: ReadOnly[SkuMeter]
    quantity: ReadOnly[int]
    list_cost: ReadOnly[float]
    contracted_cost: ReadOnly[float]
    billed_cost: ReadOnly[float]


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
        case "hourly":
            return "hour"


class FocusSpendLogsDatabase:
    """Sums LiteLLM_SpendLogs into per-period, per-SKU buckets keyed by requester, model, provider, team and tags."""

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


__all__ = ("SPEND_LOG_BUCKETS_SQL", "FocusSpendLogsDatabase", "SkuMeter", "SpendLogBucket", "SupportsQueryRaw")
