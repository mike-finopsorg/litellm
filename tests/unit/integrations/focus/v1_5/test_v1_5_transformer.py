from __future__ import annotations

import io
from datetime import datetime, timezone
from decimal import Decimal

import polars as pl

from litellm.integrations.focus.serializers import FocusCsvSerializer, FocusParquetSerializer
from litellm.integrations.focus.v1_5.database import SPEND_LOG_BUCKET_SCHEMA
from litellm.integrations.focus.v1_5.transformer import FOCUS_1_5_SCHEMA, Focus15Transformer


def _bucket(**overrides: object) -> dict[str, object]:
    return {
        "charge_period_start": "2026-05-25T00:00:00Z",
        "charge_period_end": "2026-05-26T00:00:00Z",
        "principal_id": "hermes",
        "credential_id": "33b20aab1a63380e19e8",
        "model": "gpt-5.4-mini",
        "custom_llm_provider": "openai",
        "team_id": "team-1",
        "spend": 0.125,
    } | overrides


def _transform(*buckets: dict[str, object]) -> pl.DataFrame:
    return Focus15Transformer().transform(pl.DataFrame(list(buckets), schema=SPEND_LOG_BUCKET_SCHEMA))


def test_each_bucket_field_lands_in_its_focus_column() -> None:
    assert _transform(_bucket()).to_dicts() == [
        {
            "BilledCost": Decimal("0.125000"),
            "BillingCurrency": "USD",
            "ChargeCategory": "Usage",
            "ChargePeriodStart": datetime(2026, 5, 25, tzinfo=timezone.utc),
            "ChargePeriodEnd": datetime(2026, 5, 26, tzinfo=timezone.utc),
            "CredentialId": "33b20aab1a63380e19e8",
            "PrincipalId": "hermes",
            "ResourceId": "gpt-5.4-mini",
            "ServiceProviderName": "openai",
            "SubAccountId": "team-1",
        }
    ]


def test_missing_identity_stays_null() -> None:
    row = _transform(_bucket(principal_id=None, credential_id=None)).row(0, named=True)

    assert (row["PrincipalId"], row["CredentialId"]) == (None, None)


def test_rotated_credential_keeps_the_principal() -> None:
    frame = _transform(_bucket(credential_id="credential-A"), _bucket(credential_id="credential-B"))

    assert frame.select("PrincipalId", "CredentialId").rows() == [
        ("hermes", "credential-A"),
        ("hermes", "credential-B"),
    ]


def test_empty_input_still_carries_the_full_schema() -> None:
    assert _transform().schema == FOCUS_1_5_SCHEMA


def test_parquet_stores_charge_periods_as_utc_timestamps() -> None:
    payload = FocusParquetSerializer().serialize(_transform(_bucket()))

    assert pl.read_parquet(io.BytesIO(payload)).select("ChargePeriodStart", "ChargePeriodEnd").row(0) == (
        datetime(2026, 5, 25, tzinfo=timezone.utc),
        datetime(2026, 5, 26, tzinfo=timezone.utc),
    )


def test_csv_writes_charge_periods_as_iso_8601_utc() -> None:
    rows = pl.read_csv(io.BytesIO(FocusCsvSerializer().serialize(_transform(_bucket()))), infer_schema=False)

    assert rows.select("ChargePeriodStart", "ChargePeriodEnd").row(0) == (
        "2026-05-25T00:00:00Z",
        "2026-05-26T00:00:00Z",
    )


def test_sub_micro_dollar_costs_keep_ten_decimal_places() -> None:
    assert _transform(_bucket(spend=0.0000001234)).item(0, "BilledCost") == Decimal("0.0000001234")
