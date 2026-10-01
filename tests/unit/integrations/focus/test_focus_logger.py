from __future__ import annotations

from decimal import Decimal

import polars as pl
import pytest

from litellm.integrations.focus.database import FocusLiteLLMDatabase
from litellm.integrations.focus.focus_logger import FocusLogger
from litellm.integrations.focus.schema import FOCUS_NORMALIZED_SCHEMA
from litellm.integrations.focus.settings import FocusExportSettings
from litellm.integrations.focus.v1_5.database import FocusSpendLogsDatabase
from litellm.integrations.focus.v1_5.transformer import FOCUS_1_5_SCHEMA


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


def test_1_5_with_the_default_hourly_frequency_fails_at_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FOCUS_FREQUENCY", raising=False)

    with pytest.raises(ValueError, match="FOCUS_FREQUENCY=daily"):
        FocusLogger(focus_version="1.5")


def test_1_5_engine_reads_spend_logs_and_emits_the_1_5_schema() -> None:
    engine = FocusLogger(
        focus_version="1.5", frequency="daily", destination_config={"bucket_name": "focus-test"}
    )._ensure_engine()

    assert isinstance(engine._database, FocusSpendLogsDatabase)
    assert engine._transformer.schema == FOCUS_1_5_SCHEMA


def test_default_engine_keeps_the_1_2_pipeline() -> None:
    engine = FocusLogger(destination_config={"bucket_name": "focus-test"})._ensure_engine()

    assert isinstance(engine._database, FocusLiteLLMDatabase)
    assert engine._transformer.schema == FOCUS_NORMALIZED_SCHEMA


def test_1_5_csv_export_writes_tiny_costs_without_scientific_notation() -> None:
    engine = FocusLogger(
        focus_version="1.5", frequency="daily", export_format="csv", destination_config={"bucket_name": "focus-test"}
    )._ensure_engine()
    frame = pl.DataFrame({"BilledCost": [Decimal("0.0000001234")]}, schema={"BilledCost": pl.Decimal(38, 10)})

    assert engine._serializer.serialize(frame).decode().splitlines() == ["BilledCost", "0.0000001234"]
