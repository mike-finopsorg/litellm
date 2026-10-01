"""
Behavior tests for the FOCUS 1.5 SpendLogs export against a real Postgres. Bucketing, the
empty-string-to-null mapping and the half-open window live in SQL, so these tests are the
ones that exercise them
"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Final

import polars as pl
import pytest

from litellm.integrations.focus.serializers import FocusCsvSerializer
from litellm.integrations.focus.v1_5.database import FocusSpendLogsDatabase
from litellm.integrations.focus.v1_5.transformer import Focus15Transformer
from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload
from litellm.proxy.utils import hash_token

pytestmark = pytest.mark.asyncio(loop_scope="session")

DAY: Final = datetime(2002, 6, 14, tzinfo=timezone.utc)
NEXT_DAY: Final = DAY + timedelta(days=1)
RUN: Final = uuid.uuid4().hex
HERMES: Final = f"hermes-{RUN}"
HONCHO: Final = f"honcho-{RUN}"
TEAM: Final = f"team-{RUN}"
CREDENTIAL_A: Final = hash_token(f"sk-hermes-a-{RUN}")
CREDENTIAL_B: Final = hash_token(f"sk-hermes-b-{RUN}")
HONCHO_CREDENTIAL: Final = hash_token(f"sk-honcho-{RUN}")
RAW_KEY: Final = f"sk-raw-{RUN}"
ANONYMOUS_MODEL: Final = f"anonymous-model-{RUN}"
RAW_KEY_MODEL: Final = f"raw-key-model-{RUN}"


async def _spend_log(db, *, user: str, api_key: str, start: datetime, spend: float, model: str = "gpt-5.4-mini") -> None:
    await db.execute_raw(
        'INSERT INTO "LiteLLM_SpendLogs" ("request_id", "call_type", "api_key", "startTime", "endTime", "user", '
        '"model", "custom_llm_provider", "team_id", "spend") '
        "VALUES ($1, 'acompletion', $2, $3::timestamp, $3::timestamp, $4, $5, 'openai', $6, $7)",
        f"focus-{RUN}-{uuid.uuid4()}",
        api_key,
        start.replace(tzinfo=None),
        user,
        model,
        TEAM,
        spend,
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


@pytest.fixture(scope="module", autouse=True)
async def seeded(db):
    await _spend_log(db, user=HERMES, api_key=CREDENTIAL_A, start=DAY, spend=0.25)
    await _spend_log(db, user=HERMES, api_key=CREDENTIAL_A, start=DAY.replace(hour=23, minute=59), spend=0.5)
    await _spend_log(db, user=HERMES, api_key=CREDENTIAL_B, start=DAY.replace(hour=12), spend=1.0)
    await _spend_log(db, user=HONCHO, api_key=HONCHO_CREDENTIAL, start=DAY.replace(hour=6), spend=2.0)
    await _spend_log(db, user="", api_key="", start=DAY.replace(hour=9), spend=4.0, model=ANONYMOUS_MODEL)
    await _spend_log(db, user=HERMES, api_key=CREDENTIAL_A, start=NEXT_DAY, spend=8.0)
    await _spend_log(db, user=HERMES, api_key=CREDENTIAL_A, start=DAY - timedelta(microseconds=1000), spend=16.0)
    payload_user, payload_api_key = _payload_for_raw_key()
    await _spend_log(db, user=payload_user, api_key=payload_api_key, start=DAY, spend=32.0, model=RAW_KEY_MODEL)
    yield
    await db.execute_raw('DELETE FROM "LiteLLM_SpendLogs" WHERE "request_id" LIKE $1', f"focus-{RUN}-%")


async def _export_day(db):
    database: Final = FocusSpendLogsDatabase(granularity="daily", resolve_db=lambda: db, spend_logs_disabled=lambda: False)
    frame: Final = await database.get_usage_data(start_time_utc=DAY, end_time_utc=NEXT_DAY)
    return Focus15Transformer().transform(frame).filter(pl.col("SubAccountId") == TEAM)


async def _rows_by_identity(db) -> dict[tuple[str | None, str | None, str | None], dict]:
    rows: Final = await _export_day(db)
    return {(row["PrincipalId"], row["CredentialId"], row["ResourceId"]): row for row in rows.to_dicts()}


async def test_each_principal_and_credential_gets_its_own_daily_row(db):
    rows: Final = await _rows_by_identity(db)

    assert {
        identity: float(row["BilledCost"]) for identity, row in rows.items() if identity[2] == "gpt-5.4-mini"
    } == {
        (HERMES, CREDENTIAL_A, "gpt-5.4-mini"): 0.75,
        (HERMES, CREDENTIAL_B, "gpt-5.4-mini"): 1.0,
        (HONCHO, HONCHO_CREDENTIAL, "gpt-5.4-mini"): 2.0,
    }


async def test_rows_carry_the_utc_day_as_the_charge_period(db):
    rows: Final = await _rows_by_identity(db)
    row: Final = rows[(HERMES, CREDENTIAL_A, "gpt-5.4-mini")]

    assert (row["ChargePeriodStart"], row["ChargePeriodEnd"]) == ("2002-06-14T00:00:00Z", "2002-06-15T00:00:00Z")
    assert (row["ServiceProviderName"], row["SubAccountId"], row["ChargeCategory"]) == ("openai", TEAM, "Usage")


async def test_missing_principal_and_credential_export_as_null(db):
    rows: Final = await _rows_by_identity(db)

    assert float(rows[(None, None, ANONYMOUS_MODEL)]["BilledCost"]) == 4.0


async def test_export_holds_the_hashed_credential_never_the_raw_key(db):
    rows: Final = await _export_day(db)
    csv: Final = FocusCsvSerializer().serialize(rows).decode()
    raw_key_row: Final = rows.filter(rows["ResourceId"] == RAW_KEY_MODEL).to_dicts()

    assert RAW_KEY not in csv
    assert [(row["PrincipalId"], row["CredentialId"]) for row in raw_key_row] == [(HERMES, hash_token(RAW_KEY))]
