from __future__ import annotations

import pytest

from litellm.integrations.focus.settings import FocusExportSettings
from litellm.integrations.vantage.vantage_logger import VantageLogger


def test_ignores_focus_env_and_stays_on_1_2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOCUS_VERSION", "bogus")
    monkeypatch.setenv("FOCUS_DATA_GRANULARITY", "bogus")

    assert VantageLogger().settings == FocusExportSettings(version="1.2", data_granularity="daily")
