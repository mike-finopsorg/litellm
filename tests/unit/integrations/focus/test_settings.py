from __future__ import annotations

import pydantic
import pytest

from litellm.integrations.focus.settings import (
    FocusBillingSettings,
    FocusDataGranularity,
    FocusExportSettings,
    parse_focus_export_settings,
    resolve_export_frequency,
)

ACCOUNT = {"billing_account_id": "billing-1", "billing_account_name": "Example AI Platform"}


@pytest.mark.parametrize(("version", "data_granularity"), ((None, None), ("", ""), ("  ", "  ")))
def test_unset_values_fall_back_to_defaults(version: str | None, data_granularity: str | None) -> None:
    settings = parse_focus_export_settings(version=version, data_granularity=data_granularity)

    assert settings == FocusExportSettings(version="1.2", data_granularity="daily", billing=None)


def test_values_are_normalized_before_validation() -> None:
    settings = parse_focus_export_settings(version=" 1.2 ", data_granularity=" DAILY ")

    assert settings == FocusExportSettings(version="1.2", data_granularity="daily")


@pytest.mark.parametrize("version", ("1.0", "1.20", "v1.2", "latest"))
def test_unsupported_version_is_rejected(version: str) -> None:
    with pytest.raises(ValueError, match="FOCUS_VERSION"):
        parse_focus_export_settings(version=version, data_granularity=None)


@pytest.mark.parametrize("data_granularity", ("weekly", "minute", "day"))
def test_unsupported_data_granularity_is_rejected(data_granularity: str) -> None:
    with pytest.raises(ValueError, match="FOCUS_DATA_GRANULARITY"):
        parse_focus_export_settings(version=None, data_granularity=data_granularity)


def test_1_2_rejects_hourly_granularity() -> None:
    with pytest.raises(ValueError, match=r"FOCUS_VERSION=1\.2 only supports FOCUS_DATA_GRANULARITY=daily"):
        parse_focus_export_settings(version="1.2", data_granularity="hourly")


def test_1_5_needs_the_billing_account() -> None:
    with pytest.raises(ValueError, match="billing_account_id and billing_account_name"):
        parse_focus_export_settings(version="1.5", data_granularity=None)


def test_1_5_billing_defaults_to_the_operator_and_one_account() -> None:
    settings = parse_focus_export_settings(version="1.5", data_granularity=None, config=ACCOUNT)

    assert settings == FocusExportSettings(
        version="1.5",
        data_granularity="daily",
        billing=FocusBillingSettings(
            billed_by="operator",
            billing_account_id="billing-1",
            billing_account_name="Example AI Platform",
            sub_account_id="billing-1",
            sub_account_name="Example AI Platform",
        ),
    )


def test_config_block_sets_every_billing_column() -> None:
    config = {
        **ACCOUNT,
        "billed_by": "vendor",
        "sub_account_id": "account-1",
        "sub_account_name": "Example Account",
        "region_id": "us-east",
        "region_name": "US East",
        "invoice_issuer_name": "Reseller Inc",
    }

    assert parse_focus_export_settings(version="1.5", data_granularity=None, config=config).billing == (
        FocusBillingSettings(
            billed_by="vendor",
            billing_account_id="billing-1",
            billing_account_name="Example AI Platform",
            sub_account_id="account-1",
            sub_account_name="Example Account",
            region_id="us-east",
            region_name="US East",
            invoice_issuer_name="Reseller Inc",
        )
    )


def test_config_block_wins_over_the_environment() -> None:
    settings = parse_focus_export_settings(
        version="1.2", data_granularity="daily", config={**ACCOUNT, "version": "1.5", "data_granularity": "hourly"}
    )

    assert (settings.version, settings.data_granularity) == ("1.5", "hourly")


def test_config_block_accepts_a_yaml_number_as_the_version() -> None:
    settings = parse_focus_export_settings(version=None, data_granularity=None, config={**ACCOUNT, "version": 1.5})

    assert settings.version == "1.5"


@pytest.mark.parametrize("config", ({"region_id": "us-east"}, {"region_name": "US East"}))
def test_region_id_and_name_must_be_set_together(config: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="region_id and region_name must be set together"):
        parse_focus_export_settings(version="1.5", data_granularity=None, config={**ACCOUNT, **config})


@pytest.mark.parametrize("config", ({"billed_by": "sometimes"}, {"include_spend": True}, {"billing_id": "x"}))
def test_unknown_or_invalid_config_keys_are_rejected(config: dict[str, object]) -> None:
    with pytest.raises(pydantic.ValidationError):
        parse_focus_export_settings(version="1.5", data_granularity=None, config={**ACCOUNT, **config})


def test_1_2_ignores_the_billing_block() -> None:
    settings = parse_focus_export_settings(version="1.2", data_granularity=None, config={"billed_by": "vendor"})

    assert settings == FocusExportSettings(version="1.2", data_granularity="daily", billing=None)


@pytest.mark.parametrize(
    ("configured", "frequency"), ((None, "hourly"), ("", "hourly"), ("daily", "daily"), ("interval", "interval"))
)
def test_1_2_frequency_comes_from_the_environment_and_defaults_to_hourly(
    configured: str | None, frequency: str
) -> None:
    assert resolve_export_frequency(FocusExportSettings(version="1.2"), configured) == frequency


@pytest.mark.parametrize(
    ("granularity", "configured"), (("daily", None), ("daily", "daily"), ("hourly", None), ("hourly", "hourly"))
)
def test_1_5_writes_one_file_per_bucket(granularity: FocusDataGranularity, configured: str | None) -> None:
    settings = FocusExportSettings(version="1.5", data_granularity=granularity)

    assert resolve_export_frequency(settings, configured) == granularity


@pytest.mark.parametrize(
    ("granularity", "configured"), (("daily", "hourly"), ("hourly", "daily"), ("daily", "interval"))
)
def test_1_5_rejects_a_frequency_that_differs_from_the_granularity(
    granularity: FocusDataGranularity, configured: str
) -> None:
    settings = FocusExportSettings(version="1.5", data_granularity=granularity)

    with pytest.raises(ValueError, match="FOCUS_FREQUENCY"):
        resolve_export_frequency(settings, configured)
