from __future__ import annotations

import pytest

from litellm.integrations.focus.settings import (
    FocusBillingSettings,
    FocusExportSettings,
    parse_focus_billing_settings,
    parse_focus_export_settings,
    validate_export_frequency,
)


@pytest.mark.parametrize(
    ("version", "data_granularity"),
    ((None, None), ("", ""), ("  ", "  ")),
)
def test_unset_values_fall_back_to_defaults(version: str | None, data_granularity: str | None) -> None:
    settings = parse_focus_export_settings(version=version, data_granularity=data_granularity)

    assert settings == FocusExportSettings(version="1.2", data_granularity="daily")


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


def test_version_1_5_is_accepted() -> None:
    assert parse_focus_export_settings(version="1.5", data_granularity=None).version == "1.5"


@pytest.mark.parametrize("frequency", ("hourly", "interval"))
def test_1_5_daily_rejects_frequencies_that_split_a_day(frequency: str) -> None:
    settings = FocusExportSettings(version="1.5", data_granularity="daily")

    with pytest.raises(ValueError, match="FOCUS_FREQUENCY=daily"):
        validate_export_frequency(settings, frequency)


def test_1_5_daily_accepts_daily_frequency() -> None:
    validate_export_frequency(FocusExportSettings(version="1.5", data_granularity="daily"), "daily")


@pytest.mark.parametrize("frequency", ("hourly", "daily", "interval"))
def test_1_2_keeps_accepting_every_frequency(frequency: str) -> None:
    validate_export_frequency(FocusExportSettings(version="1.2", data_granularity="daily"), frequency)


def test_billing_defaults_to_spend_and_a_hostname_label() -> None:
    assert parse_focus_billing_settings({}, hostname="spark") == FocusBillingSettings(
        include_spend=True,
        billing_account_id="spark-litellm",
        billing_account_name="spark-litellm",
        sub_account_id="spark-litellm",
        sub_account_name="spark-litellm",
    )


def test_billing_values_come_from_focus_env() -> None:
    env = {
        "FOCUS_INCLUDE_SPEND": " False ",
        "FOCUS_BILLING_ID": "billing-1",
        "FOCUS_BILLING_NAME": "Example AI Platform",
        "FOCUS_ACCOUNT_ID": "account-1",
        "FOCUS_ACCOUNT_NAME": "Example Account",
    }

    assert parse_focus_billing_settings(env, hostname="spark") == FocusBillingSettings(
        include_spend=False,
        billing_account_id="billing-1",
        billing_account_name="Example AI Platform",
        sub_account_id="account-1",
        sub_account_name="Example Account",
    )


def test_blank_billing_values_fall_back_to_the_hostname_label() -> None:
    settings = parse_focus_billing_settings({"FOCUS_BILLING_NAME": "  ", "FOCUS_INCLUDE_SPEND": ""}, hostname="spark")

    assert (settings.billing_account_name, settings.include_spend) == ("spark-litellm", True)


@pytest.mark.parametrize("raw", ("maybe", "2", "enabled"))
def test_unsupported_include_spend_is_rejected(raw: str) -> None:
    with pytest.raises(ValueError, match="FOCUS_INCLUDE_SPEND"):
        parse_focus_billing_settings({"FOCUS_INCLUDE_SPEND": raw}, hostname="spark")


def test_region_defaults_to_null() -> None:
    settings = parse_focus_billing_settings({}, hostname="spark")

    assert (settings.region_id, settings.region_name) == (None, None)


def test_region_comes_from_focus_env() -> None:
    settings = parse_focus_billing_settings(
        {"FOCUS_REGION_ID": " us-east ", "FOCUS_REGION_NAME": "US East"}, hostname="spark"
    )

    assert (settings.region_id, settings.region_name) == ("us-east", "US East")


@pytest.mark.parametrize("env", ({"FOCUS_REGION_ID": "us-east"}, {"FOCUS_REGION_NAME": "US East"}))
def test_region_id_and_name_must_be_set_together(env: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="FOCUS_REGION_ID and FOCUS_REGION_NAME must be set together"):
        parse_focus_billing_settings(env, hostname="spark")
