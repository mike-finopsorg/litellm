from __future__ import annotations

import pytest

from litellm.integrations.focus.settings import (
    FocusExportSettings,
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
