from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import polars as pl
import pytest

from litellm.integrations.focus.database import FocusLiteLLMDatabase
from litellm.integrations.focus.destinations.base import FocusTimeWindow
from litellm.integrations.focus.focus_logger import FocusLogger, aligned_windows
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


def test_1_5_hourly_from_env_runs_on_the_default_hourly_frequency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOCUS_VERSION", "1.5")
    monkeypatch.setenv("FOCUS_DATA_GRANULARITY", "hourly")
    monkeypatch.delenv("FOCUS_FREQUENCY", raising=False)

    logger = FocusLogger()

    assert (logger.settings, logger.frequency) == (
        FocusExportSettings(version="1.5", data_granularity="hourly"),
        "hourly",
    )


def _at(day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(2026, 9, day, hour, minute, tzinfo=timezone.utc)


def _bounds(windows: tuple[FocusTimeWindow, ...]) -> list[tuple[datetime, datetime]]:
    return [(window.start_time, window.end_time) for window in windows]


def test_aligned_windows_widen_a_partial_range_to_whole_hours() -> None:
    windows = aligned_windows(
        start_time_utc=_at(10, 10, 30), end_time_utc=_at(10, 12, 15), frequency="hourly", now=_at(20)
    )

    assert _bounds(windows) == [(_at(10, 10), _at(10, 11)), (_at(10, 11), _at(10, 12)), (_at(10, 12), _at(10, 13))]
    assert {window.frequency for window in windows} == {"hourly"}


def test_aligned_windows_do_not_widen_an_end_already_on_a_boundary() -> None:
    windows = aligned_windows(start_time_utc=_at(10), end_time_utc=_at(12), frequency="daily", now=_at(20))

    assert _bounds(windows) == [(_at(10), _at(11)), (_at(11), _at(12))]


def test_aligned_windows_stop_before_the_bucket_still_in_progress() -> None:
    windows = aligned_windows(
        start_time_utc=_at(10, 9), end_time_utc=_at(10, 18), frequency="hourly", now=_at(10, 11, 20)
    )

    assert _bounds(windows) == [(_at(10, 9), _at(10, 10)), (_at(10, 10), _at(10, 11))]


def test_aligned_windows_treat_naive_times_as_utc_and_convert_offsets() -> None:
    windows = aligned_windows(
        start_time_utc=datetime(2026, 9, 10, 2, 0),
        end_time_utc=datetime(2026, 9, 10, 5, 0, tzinfo=timezone(timedelta(hours=2))),
        frequency="hourly",
        now=_at(20),
    )

    assert _bounds(windows) == [(_at(10, 2), _at(10, 3))]


def test_aligned_windows_are_empty_when_the_range_is_still_in_progress() -> None:
    assert (
        aligned_windows(start_time_utc=_at(10, 9), end_time_utc=_at(10, 10), frequency="hourly", now=_at(10, 9, 30))
        == ()
    )


def test_aligned_windows_reject_interval_frequency() -> None:
    with pytest.raises(ValueError, match="FOCUS_FREQUENCY=hourly or daily"):
        aligned_windows(start_time_utc=_at(10), end_time_utc=_at(11), frequency="interval", now=_at(20))


def test_aligned_windows_cap_the_number_of_windows() -> None:
    with pytest.raises(ValueError, match="limit is 744"):
        aligned_windows(
            start_time_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
            end_time_utc=datetime(2026, 3, 1, tzinfo=timezone.utc),
            frequency="hourly",
            now=_at(20),
        )


class _RecordingFocusLogger(FocusLogger):
    def __init__(self) -> None:
        super().__init__(focus_version="1.5", frequency="daily", destination_config={"bucket_name": "focus-test"})
        self.exported: list[tuple[FocusTimeWindow, int | None]] = []  # mutable-ok: records exports for assertions

    async def _export_window(self, *, window: FocusTimeWindow, limit: int | None) -> None:
        self.exported.append((window, limit))


@pytest.mark.asyncio
async def test_export_range_exports_each_aligned_window_in_order() -> None:
    logger = _RecordingFocusLogger()

    windows = await logger.export_range(start_time_utc=_at(10, 6), end_time_utc=_at(12, 6), limit=7, now=_at(20))

    assert logger.exported == [(window, 7) for window in windows]
    assert _bounds(windows) == [(_at(10), _at(11)), (_at(11), _at(12)), (_at(12), _at(13))]
