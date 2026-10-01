from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from decimal import Decimal

import polars as pl
import pytest

from litellm.integrations.focus.serializers import FocusCsvSerializer, FocusParquetSerializer
from litellm.integrations.focus.settings import FocusBillingSettings, FocusExportSettings
from litellm.integrations.focus.v1_5.database import SPEND_LOG_BUCKET_SCHEMA
from litellm.integrations.focus.v1_5.transformer import FOCUS_1_5_SCHEMA, Focus15Transformer

BILLING = FocusBillingSettings(
    include_spend=True,
    billing_account_id="billing-1",
    billing_account_name="Example AI Platform",
    sub_account_id="account-1",
    sub_account_name="Example Account",
)


def _bucket(**overrides: object) -> dict[str, object]:
    return {
        "charge_period_start": "2026-05-25T00:00:00Z",
        "charge_period_end": "2026-05-26T00:00:00Z",
        "principal_id": "hermes",
        "principal_name": "Hermes Agent",
        "principal_email": "hermes@example.test",
        "credential_id": "33b20aab1a63380e19e8",
        "credential_name": "hermes-primary",
        "model": "openai/gpt-5.4-mini",
        "custom_llm_provider": "openai",
        "team_id": "team-1",
        "team_alias": "Agents",
        "service_tier": "default",
        "request_tags": '["prod"]',
        "meter": "Input Tokens",
        "quantity": 4000,
        "list_cost": 0.003,
        "contracted_cost": 0.0027,
        "billed_cost": 0.00297,
    } | overrides


def _transform(*buckets: dict[str, object], billing: FocusBillingSettings = BILLING) -> pl.DataFrame:
    settings = FocusExportSettings(version="1.5", billing=billing)
    return Focus15Transformer(settings).transform(pl.DataFrame(list(buckets), schema=SPEND_LOG_BUCKET_SCHEMA))


def _row(**overrides: object) -> dict[str, object]:
    return _transform(_bucket(**overrides)).row(0, named=True)


def test_token_bucket_maps_onto_every_focus_column() -> None:
    row = _row()

    assert {**row, "RequesterDetails": json.loads(row["RequesterDetails"]), "Tags": json.loads(row["Tags"])} == {
        "BilledCost": Decimal("0.00297"),
        "BillingAccountId": "billing-1",
        "BillingAccountName": "Example AI Platform",
        "BillingAccountType": "LiteLLM Billing Account",
        "BillingCurrency": "USD",
        "BillingPeriodEnd": datetime(2026, 6, 1, tzinfo=timezone.utc),
        "BillingPeriodStart": datetime(2026, 5, 1, tzinfo=timezone.utc),
        "ChargeCategory": "Usage",
        "ChargeClass": None,
        "ChargeDescription": "openai/gpt-5.4-mini Input Tokens",
        "ChargeFrequency": "Usage-Based",
        "ChargePeriodEnd": datetime(2026, 5, 26, tzinfo=timezone.utc),
        "ChargePeriodStart": datetime(2026, 5, 25, tzinfo=timezone.utc),
        "ConsumedQuantity": Decimal(4000),
        "ConsumedUnit": "Tokens",
        "ContractedCost": Decimal("0.0027"),
        "ContractedUnitPrice": Decimal("0.000000675"),
        "CredentialId": "33b20aab1a63380e19e8",
        "EffectiveCost": Decimal("0.00297"),
        "HostProviderName": "OpenAI",
        "InvoiceIssuerName": "Example AI Platform",
        "ListCost": Decimal("0.003"),
        "ListUnitPrice": Decimal("0.00000075"),
        "PricingCategory": "Standard",
        "PricingCurrency": "USD",
        "PricingCurrencyContractedUnitPrice": Decimal("0.000000675"),
        "PricingCurrencyEffectiveCost": Decimal("0.00297"),
        "PricingCurrencyListUnitPrice": Decimal("0.00000075"),
        "PricingQuantity": Decimal(4000),
        "PricingUnit": "Tokens",
        "PrincipalId": "hermes",
        "RequesterDetails": [
            {"key": "Principal", "value": {"Type": "User", "Name": "Hermes Agent", "Email": "hermes@example.test"}},
            {"key": "Credential", "value": {"Type": "API Key", "Name": "hermes-primary"}},
        ],
        "ResourceId": "openai/gpt-5.4-mini",
        "ResourceType": "Model",
        "ServiceCategory": "AI and Machine Learning",
        "ServiceName": "OpenAI API",
        "ServiceProviderName": "OpenAI",
        "ServiceSubcategory": "Generative AI",
        "SkuId": "openai/gpt-5.4-mini/input-tokens",
        "SkuMeter": "Input Tokens",
        "SkuPriceId": "openai/gpt-5.4-mini/input-tokens/default",
        "SubAccountId": "account-1",
        "SubAccountName": "Example Account",
        "SubAccountType": "LiteLLM Account",
        "Tags": {"litellm/team_alias": "Agents", "litellm/team_id": "team-1", "prod": True},
    }


def test_without_spend_the_gateway_bills_nothing_and_the_vendor_issues_the_invoice() -> None:
    billing = FocusBillingSettings(
        include_spend=False,
        billing_account_id="billing-1",
        billing_account_name="Example AI Platform",
        sub_account_id="account-1",
        sub_account_name="Example Account",
    )
    row = _transform(_bucket(), billing=billing).row(0, named=True)

    assert (row["BilledCost"], row["EffectiveCost"], row["PricingCurrencyEffectiveCost"]) == (0, 0, 0)
    assert (row["ListCost"], row["ContractedCost"]) == (Decimal("0.003"), Decimal("0.0027"))
    assert row["InvoiceIssuerName"] == "OpenAI"


@pytest.mark.parametrize(
    ("overrides", "expected"),
    (
        (
            {"principal_name": None, "principal_email": None, "credential_name": None},
            [{"key": "Principal", "value": {"Type": "User"}}, {"key": "Credential", "value": {"Type": "API Key"}}],
        ),
        (
            {"principal_id": None, "credential_id": "litellm_proxy_master_key", "credential_name": None},
            [{"key": "Credential", "value": {"Type": "Master Key"}}],
        ),
        (
            {"credential_id": None},
            [{"key": "Principal", "value": {"Type": "User", "Name": "Hermes Agent", "Email": "hermes@example.test"}}],
        ),
    ),
)
def test_requester_details_only_carry_entries_for_known_identities(
    overrides: dict[str, object], expected: list[dict[str, object]]
) -> None:
    assert json.loads(_row(**overrides)["RequesterDetails"]) == expected


def test_requester_details_are_null_without_principal_or_credential() -> None:
    row = _row(principal_id=None, credential_id=None)

    assert (row["PrincipalId"], row["CredentialId"], row["RequesterDetails"]) == (None, None, None)


def test_health_check_key_is_a_service_account_key() -> None:
    details = json.loads(_row(principal_id=None, credential_id="litellm-internal-health-check")["RequesterDetails"])

    assert details == [{"key": "Credential", "value": {"Type": "Service Account Key", "Name": "hermes-primary"}}]


def test_tags_are_null_without_request_tags_or_team() -> None:
    assert _row(request_tags="[]", team_id=None, team_alias=None)["Tags"] is None


def test_other_usage_is_priced_per_request() -> None:
    row = _row(meter="Other Usage", quantity=2, list_cost=0.02, contracted_cost=0.02, billed_cost=0.02)

    assert (row["PricingUnit"], row["ConsumedUnit"], row["ListUnitPrice"], row["SkuId"]) == (
        "Requests",
        "Requests",
        Decimal("0.01"),
        "openai/gpt-5.4-mini/other-usage",
    )


def test_unit_prices_are_null_when_no_quantity_was_recorded() -> None:
    row = _row(quantity=0)

    assert (row["ListUnitPrice"], row["ContractedUnitPrice"]) == (None, None)


def test_self_hosted_models_are_provided_and_hosted_by_the_operator() -> None:
    row = _row(custom_llm_provider="hosted_vllm", model="hosted_vllm/spark")

    assert (row["ServiceProviderName"], row["HostProviderName"]) == ("Example AI Platform", "Example AI Platform")


def test_unknown_providers_fall_back_to_their_slug() -> None:
    row = _row(custom_llm_provider="some_new_provider")

    assert (row["ServiceProviderName"], row["HostProviderName"], row["ServiceName"]) == (
        "some_new_provider",
        "some_new_provider",
        "some_new_provider",
    )


def test_billing_period_rolls_over_the_year_end() -> None:
    row = _row(charge_period_start="2026-12-31T23:00:00Z", charge_period_end="2027-01-01T00:00:00Z")

    assert (row["BillingPeriodStart"], row["BillingPeriodEnd"]) == (
        datetime(2026, 12, 1, tzinfo=timezone.utc),
        datetime(2027, 1, 1, tzinfo=timezone.utc),
    )


def test_rotated_credential_keeps_the_principal() -> None:
    frame = _transform(_bucket(credential_id="credential-A"), _bucket(credential_id="credential-B"))

    assert frame.select("PrincipalId", "CredentialId").rows() == [
        ("hermes", "credential-A"),
        ("hermes", "credential-B"),
    ]


def test_empty_input_still_carries_the_full_schema() -> None:
    assert _transform().schema == FOCUS_1_5_SCHEMA


def test_sub_micro_dollar_costs_keep_ten_decimal_places() -> None:
    assert _row(billed_cost=0.0000001234)["BilledCost"] == Decimal("0.0000001234")


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
