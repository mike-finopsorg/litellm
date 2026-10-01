from __future__ import annotations

from decimal import Decimal

import polars as pl

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
            "ChargePeriodStart": "2026-05-25T00:00:00Z",
            "ChargePeriodEnd": "2026-05-26T00:00:00Z",
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
