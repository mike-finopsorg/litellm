from __future__ import annotations

import pytest

from litellm.integrations.focus.settings import FocusExportSettings, parse_focus_export_settings


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
