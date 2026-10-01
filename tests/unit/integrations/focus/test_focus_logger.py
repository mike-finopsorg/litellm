from __future__ import annotations

import pytest

from litellm.integrations.focus.focus_logger import FocusLogger
from litellm.integrations.focus.settings import FocusExportSettings


def test_settings_default_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FOCUS_VERSION", raising=False)
    monkeypatch.delenv("FOCUS_DATA_GRANULARITY", raising=False)

    assert FocusLogger().settings == FocusExportSettings(version="1.2", data_granularity="daily")


def test_settings_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOCUS_VERSION", "1.2")
    monkeypatch.setenv("FOCUS_DATA_GRANULARITY", "DAILY")

    assert FocusLogger().settings == FocusExportSettings(version="1.2", data_granularity="daily")


def test_constructor_args_take_precedence_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOCUS_VERSION", "bogus")
    monkeypatch.setenv("FOCUS_DATA_GRANULARITY", "bogus")

    logger = FocusLogger(focus_version="1.2", data_granularity="daily")

    assert logger.settings == FocusExportSettings(version="1.2", data_granularity="daily")


@pytest.mark.parametrize("env_var", ("FOCUS_VERSION", "FOCUS_DATA_GRANULARITY"))
def test_invalid_env_value_fails_at_construction(monkeypatch: pytest.MonkeyPatch, env_var: str) -> None:
    monkeypatch.setenv(env_var, "bogus")

    with pytest.raises(ValueError, match=env_var):
        FocusLogger()

