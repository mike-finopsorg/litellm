"""
Behavior tests for the FOCUS 1.5 SpendLogs export against a real Postgres. Bucketing, the per-SKU split of
token counts and costs, the empty-string-to-null mapping, the user join and the half-open window live in SQL,
so these tests are the ones that exercise them
"""

import json
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Final

import polars as pl
import pytest

from litellm.integrations.focus.serializers import FocusCsvSerializer
from litellm.integrations.focus.settings import FocusBillingSettings, FocusExportSettings
from litellm.integrations.focus.v1_5.database import FocusSpendLogsDatabase
from litellm.integrations.focus.v1_5.transformer import Focus15Transformer
from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload
from litellm.proxy.utils import hash_token

pytestmark = pytest.mark.asyncio(loop_scope="session")

DAY: Final = datetime(2002, 6, 14, tzinfo=timezone.utc)
NEXT_DAY: Final = DAY + timedelta(days=1)
RUN: Final = uuid.uuid4().hex
TEAM: Final = f"team-{RUN}"
HERMES: Final = f"hermes-{RUN}"
HONCHO: Final = f"honcho-{RUN}"
CREDENTIAL_A: Final = hash_token(f"sk-hermes-a-{RUN}")
CREDENTIAL_B: Final = hash_token(f"sk-hermes-b-{RUN}")
HONCHO_CREDENTIAL: Final = hash_token(f"sk-honcho-{RUN}")
RAW_KEY: Final = f"sk-raw-{RUN}"
MODEL: Final = f"openai/gpt-5.4-mini-{RUN}"
CACHED_MODEL: Final = f"cached-model-{RUN}"
DISCOUNTED_MODEL: Final = f"discounted-model-{RUN}"
UNPRICED_MODEL: Final = f"unpriced-model-{RUN}"
TOOL_MODEL: Final = f"tool-model-{RUN}"
CACHE_WRITE_MODEL: Final = f"cache-write-model-{RUN}"
ANONYMOUS_MODEL: Final = f"anonymous-model-{RUN}"
FAILED_MODEL: Final = f"failed-model-{RUN}"
TAGGED_MODEL: Final = f"tagged-model-{RUN}"
RAW_KEY_MODEL: Final = f"raw-key-model-{RUN}"
BILLING: Final = FocusBillingSettings(
    include_spend=True,
    billing_account_id="billing-1",
    billing_account_name="Example AI Platform",
    sub_account_id="account-1",
    sub_account_name="Example Account",
)


def _metadata(
    *,
    input_cost: float,
    output_cost: float,
    cached_tokens: int = 0,
    cache_read_cost: float = 0.0,
    cache_creation_tokens: int = 0,
    cache_creation_cost: float = 0.0,
    tool_usage_cost: float = 0.0,
    discount_percent: float = 0.0,
    key_alias: str | None = None,
) -> dict[str, object]:
    return {
        "user_api_key_alias": key_alias,
        "usage_object": {
            "prompt_tokens_details": {"cached_tokens": cached_tokens},
            "cache_creation_input_tokens": cache_creation_tokens,
        },
        "cost_breakdown": {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "cache_read_cost": cache_read_cost,
            "cache_creation_cost": cache_creation_cost,
            "tool_usage_cost": tool_usage_cost,
            "original_cost": input_cost + output_cost + tool_usage_cost,
            "discount_percent": discount_percent,
            "service_tier": "default",
        },
    }


async def _spend_log(
    db,
    *,
    model: str,
    spend: float,
    prompt_tokens: int,
    completion_tokens: int,
    metadata: dict[str, object] | None,
    user: str = HERMES,
    api_key: str = CREDENTIAL_A,
    start: datetime = DAY,
    request_tags: tuple[str, ...] = (),
) -> None:
    await db.execute_raw(
        'INSERT INTO "LiteLLM_SpendLogs" ("request_id", "call_type", "api_key", "startTime", "endTime", "user", '
        '"model", "custom_llm_provider", "team_id", "spend", "prompt_tokens", "completion_tokens", "metadata", '
        '"request_tags") '
        "VALUES ($1, 'acompletion', $2, $3::timestamp, $3::timestamp, $4, $5, 'openai', $6, $7, $8, $9, "
        "$10::jsonb, $11::jsonb)",
        f"focus-{RUN}-{uuid.uuid4()}",
        api_key,
        start.replace(tzinfo=None),
        user,
        model,
        TEAM,
        spend,
        prompt_tokens,
        completion_tokens,
        json.dumps(metadata) if metadata is not None else "{}",
        json.dumps(list(request_tags)),
    )


def _payload_for_raw_key() -> tuple[str, str]:
    payload: Final = get_logging_payload(
        kwargs={
            "model": RAW_KEY_MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "call_type": "acompletion",
            "litellm_params": {"metadata": {"user_api_key": RAW_KEY, "user_api_key_user_id": HERMES}},
        },
        response_obj=None,
        start_time=DAY,
        end_time=DAY,
    )
    return payload["user"], payload["api_key"]


def _plain(spend: float, **metadata: object) -> dict[str, object]:
    return _metadata(input_cost=spend * 0.25, output_cost=spend * 0.75, **metadata)


@pytest.fixture(scope="module", autouse=True)
async def seeded(db):
    await db.execute_raw(
        'INSERT INTO "LiteLLM_UserTable" ("user_id", "user_alias", "user_email") VALUES ($1, $2, $3)',
        HERMES,
        "Hermes Agent",
        "hermes@example.test",
    )
    hermes_alias: Final = {"key_alias": "hermes-primary"}
    await _spend_log(
        db, model=MODEL, spend=0.4, prompt_tokens=10, completion_tokens=5, metadata=_plain(0.4, **hermes_alias)
    )
    await _spend_log(
        db,
        model=MODEL,
        spend=0.8,
        prompt_tokens=20,
        completion_tokens=10,
        metadata=_plain(0.8, **hermes_alias),
        start=DAY.replace(hour=23, minute=59),
    )
    await _spend_log(
        db,
        model=MODEL,
        spend=1.6,
        prompt_tokens=40,
        completion_tokens=20,
        metadata=_plain(1.6),
        api_key=CREDENTIAL_B,
        start=DAY.replace(hour=12),
    )
    await _spend_log(
        db,
        model=MODEL,
        spend=3.2,
        prompt_tokens=80,
        completion_tokens=40,
        metadata=_plain(3.2),
        user=HONCHO,
        api_key=HONCHO_CREDENTIAL,
    )
    await _spend_log(
        db, model=MODEL, spend=64.0, prompt_tokens=1, completion_tokens=1, metadata=_plain(64.0), start=NEXT_DAY
    )
    await _spend_log(
        db,
        model=MODEL,
        spend=128.0,
        prompt_tokens=1,
        completion_tokens=1,
        metadata=_plain(128.0),
        start=DAY - timedelta(microseconds=1000),
    )
    await _spend_log(
        db,
        model=CACHED_MODEL,
        spend=0.0038,
        prompt_tokens=4049,
        completion_tokens=174,
        metadata=_metadata(input_cost=0.000618, output_cost=0.000783, cached_tokens=3584, cache_read_cost=0.0002688),
    )
    await _spend_log(
        db,
        model=DISCOUNTED_MODEL,
        spend=0.0099,
        prompt_tokens=100,
        completion_tokens=50,
        metadata=_metadata(input_cost=0.006, output_cost=0.004, discount_percent=0.1),
    )
    await _spend_log(db, model=UNPRICED_MODEL, spend=0.5, prompt_tokens=30, completion_tokens=3, metadata=None)
    await _spend_log(
        db,
        model=TOOL_MODEL,
        spend=0.03,
        prompt_tokens=10,
        completion_tokens=10,
        metadata=_metadata(input_cost=0.005, output_cost=0.015, tool_usage_cost=0.01),
    )
    await _spend_log(
        db,
        model=CACHE_WRITE_MODEL,
        spend=0.012,
        prompt_tokens=1200,
        completion_tokens=0,
        metadata=_metadata(input_cost=0.012, output_cost=0.0, cache_creation_tokens=1000, cache_creation_cost=0.01),
    )
    await _spend_log(
        db,
        model=ANONYMOUS_MODEL,
        spend=4.0,
        prompt_tokens=1,
        completion_tokens=1,
        metadata=_plain(4.0),
        user="",
        api_key="",
    )
    await _spend_log(db, model=FAILED_MODEL, spend=0.0, prompt_tokens=0, completion_tokens=0, metadata=None)
    await _spend_log(
        db,
        model=TAGGED_MODEL,
        spend=0.1,
        prompt_tokens=1,
        completion_tokens=1,
        metadata=_plain(0.1),
        request_tags=("a",),
    )
    await _spend_log(
        db,
        model=TAGGED_MODEL,
        spend=0.2,
        prompt_tokens=1,
        completion_tokens=1,
        metadata=_plain(0.2),
        request_tags=("b",),
    )
    payload_user, payload_api_key = _payload_for_raw_key()
    await _spend_log(
        db,
        model=RAW_KEY_MODEL,
        spend=0.32,
        prompt_tokens=1,
        completion_tokens=1,
        metadata=_plain(0.32),
        user=payload_user,
        api_key=payload_api_key,
    )
    yield
    await db.execute_raw('DELETE FROM "LiteLLM_SpendLogs" WHERE "request_id" LIKE $1', f"focus-{RUN}-%")
    await db.execute_raw('DELETE FROM "LiteLLM_UserTable" WHERE "user_id" = $1', HERMES)


async def _export(db, *, granularity: str = "daily") -> pl.DataFrame:
    database: Final = FocusSpendLogsDatabase(
        granularity=granularity, resolve_db=lambda: db, spend_logs_disabled=lambda: False
    )
    frame: Final = await database.get_usage_data(start_time_utc=DAY, end_time_utc=NEXT_DAY)
    settings: Final = FocusExportSettings(version="1.5", billing=BILLING)
    return Focus15Transformer(settings).transform(frame).filter(pl.col("Tags").str.contains(TEAM))


def _by_meter(rows: pl.DataFrame, model: str) -> dict[str, tuple[Decimal, Decimal, Decimal, Decimal]]:
    return {
        row["SkuMeter"]: (row["PricingQuantity"], row["ListCost"], row["ContractedCost"], row["BilledCost"])
        for row in rows.filter(pl.col("ResourceId") == model).to_dicts()
    }


def _close(actual: tuple[Decimal, ...], expected: tuple[float, ...]) -> bool:
    return all(abs(float(a) - e) < 1e-9 for a, e in zip(actual, expected, strict=True))


async def test_each_principal_and_credential_gets_its_own_daily_rows(db):
    rows: Final = (await _export(db)).filter((pl.col("ResourceId") == MODEL) & (pl.col("SkuMeter") == "Output Tokens"))

    assert {
        (row["PrincipalId"], row["CredentialId"]): (int(row["PricingQuantity"]), float(row["BilledCost"]))
        for row in rows.to_dicts()
    } == {
        (HERMES, CREDENTIAL_A): (15, 0.9),
        (HERMES, CREDENTIAL_B): (20, 1.2),
        (HONCHO, HONCHO_CREDENTIAL): (40, 2.4),
    }


async def test_rows_carry_the_utc_day_and_month_as_charge_and_billing_periods(db):
    row: Final = (await _export(db)).filter(pl.col("CredentialId") == CREDENTIAL_A).row(0, named=True)

    assert (row["ChargePeriodStart"], row["ChargePeriodEnd"]) == (DAY, NEXT_DAY)
    assert (row["BillingPeriodStart"], row["BillingPeriodEnd"]) == (
        datetime(2002, 6, 1, tzinfo=timezone.utc),
        datetime(2002, 7, 1, tzinfo=timezone.utc),
    )


async def test_cached_input_is_split_from_uncached_input(db):
    meters: Final = _by_meter(await _export(db), CACHED_MODEL)

    assert set(meters) == {"Input Tokens", "Cached Input Tokens", "Output Tokens"}
    assert _close(meters["Input Tokens"], (465, 0.0003492, 0.0003492, 0.0003492 * 0.0038 / 0.001401))
    assert _close(meters["Cached Input Tokens"], (3584, 0.0002688, 0.0002688, 0.0002688 * 0.0038 / 0.001401))
    assert _close(meters["Output Tokens"], (174, 0.000783, 0.000783, 0.000783 * 0.0038 / 0.001401))


async def test_cache_writes_are_their_own_sku(db):
    meters: Final = _by_meter(await _export(db), CACHE_WRITE_MODEL)

    assert _close(meters["Cache Write Tokens"][:2], (1000, 0.01))
    assert _close(meters["Input Tokens"][:2], (200, 0.002))


async def test_discount_lowers_contracted_cost_and_spend_is_split_by_list_cost(db):
    meters: Final = _by_meter(await _export(db), DISCOUNTED_MODEL)

    assert _close(meters["Input Tokens"], (100, 0.006, 0.0054, 0.00594))
    assert _close(meters["Output Tokens"], (50, 0.004, 0.0036, 0.00396))


async def test_tool_usage_cost_lands_in_other_usage(db):
    meters: Final = _by_meter(await _export(db), TOOL_MODEL)

    assert _close(meters["Other Usage"], (1, 0.01, 0.01, 0.01))


async def test_spend_without_a_cost_breakdown_lands_in_other_usage(db):
    meters: Final = _by_meter(await _export(db), UNPRICED_MODEL)

    assert _close(meters["Other Usage"], (1, 0.5, 0.5, 0.5))
    assert _close(meters["Input Tokens"], (30, 0, 0, 0))


async def test_billed_cost_reconciles_to_spend_for_every_model(db):
    rows: Final = await _export(db)

    assert (
        abs(
            float(rows["BilledCost"].sum())
            - (0.4 + 0.8 + 1.6 + 3.2 + 0.0038 + 0.0099 + 0.5 + 0.03 + 0.012 + 4.0 + 0.3 + 0.32)
        )
        < 1e-9
    )


async def test_requests_without_usage_produce_no_rows(db):
    assert (await _export(db)).filter(pl.col("ResourceId") == FAILED_MODEL).height == 0


async def test_principal_details_come_from_the_user_table(db):
    row: Final = (
        (await _export(db))
        .filter((pl.col("CredentialId") == CREDENTIAL_A) & (pl.col("ResourceId") == MODEL))
        .row(0, named=True)
    )

    assert json.loads(row["RequesterDetails"]) == [
        {"key": "Principal", "value": {"Type": "User", "Name": "Hermes Agent", "Email": "hermes@example.test"}},
        {"key": "Credential", "value": {"Type": "API Key", "Name": "hermes-primary"}},
    ]


async def test_missing_principal_and_credential_export_as_null(db):
    rows: Final = (await _export(db)).filter(pl.col("ResourceId") == ANONYMOUS_MODEL)

    assert set(rows.select("PrincipalId", "CredentialId", "RequesterDetails").rows()) == {(None, None, None)}


async def test_different_request_tags_are_separate_rows(db):
    rows: Final = (await _export(db)).filter(
        (pl.col("ResourceId") == TAGGED_MODEL) & (pl.col("SkuMeter") == "Output Tokens")
    )

    assert sorted(
        (sorted(k for k in json.loads(row["Tags"]) if not k.startswith("litellm/")), float(row["BilledCost"]))
        for row in rows.to_dicts()
    ) == [(["a"], 0.075), (["b"], 0.15)]


async def test_export_holds_the_hashed_credential_never_the_raw_key(db):
    rows: Final = await _export(db)
    csv: Final = FocusCsvSerializer().serialize(rows).decode()
    raw_key_rows: Final = rows.filter(pl.col("ResourceId") == RAW_KEY_MODEL)

    assert RAW_KEY not in csv
    assert set(raw_key_rows.select("PrincipalId", "CredentialId").rows()) == {(HERMES, hash_token(RAW_KEY))}
