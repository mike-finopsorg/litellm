from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import polars as pl
import pytest

import litellm
from litellm import Router
from litellm.integrations.focus.database import FocusLiteLLMDatabase
from litellm.integrations.focus.destinations.base import FocusTimeWindow
from litellm.integrations.focus.focus_logger import FocusLogger, aligned_windows
from litellm.integrations.focus.schema import FOCUS_NORMALIZED_SCHEMA
from litellm.integrations.focus.settings import FocusExportSettings
from litellm.integrations.focus.v1_5.database import FocusSpendLogsDatabase
from litellm.integrations.focus.v1_5.transformer import FOCUS_1_5_SCHEMA
from litellm.proxy import proxy_server

FOCUS_LOCK = "cronjob_lock:focus_export_usage_data"
FOCUS_SETTINGS = {"billing_account_id": "billing-1", "billing_account_name": "Example AI Platform"}
VANTAGE_LOCK = "cronjob_lock:vantage_export_usage_data"


class _InMemoryRedis:
    def __init__(self, locks: dict[str, str]) -> None:
        self.locks = locks
        self.granted: list[tuple[str, str]] = []

    async def async_set_cache(self, key: str, value: str, nx: bool, ttl: int) -> bool:
        if nx and key in self.locks:
            return False
        self.locks[key] = value
        self.granted.append((key, value))
        return True

    async def async_get_cache(self, key: str) -> str | None:
        return self.locks.get(key)

    async def async_delete_cache(self, key: str) -> int:
        return 1 if self.locks.pop(key, None) is not None else 0


@pytest.mark.asyncio
@pytest.mark.parametrize("proxy_logging_registered", [True, False])
async def test_scheduled_focus_export_without_a_redis_lock_runs_until_the_database_boundary(
    monkeypatch: pytest.MonkeyPatch, proxy_logging_registered: bool
) -> None:
    monkeypatch.setattr(proxy_server.proxy_logging_obj.db_spend_update_writer.pod_lock_manager, "redis_cache", None)
    if not proxy_logging_registered:
        monkeypatch.setattr(proxy_server, "proxy_logging_obj", None)
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    logger = FocusLogger(provider="s3", destination_config={"bucket_name": "focus-exports"})

    with pytest.raises(RuntimeError, match="Database not connected"):
        await logger.initialize_focus_export_job()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("locks_before", "locks_after", "granted"),
    [
        pytest.param({}, {}, [(FOCUS_LOCK, "this-pod")], id="lock-free"),
        pytest.param({FOCUS_LOCK: "this-pod"}, {}, [], id="lock-already-held-by-this-pod"),
        pytest.param(
            {VANTAGE_LOCK: "another-pod"},
            {VANTAGE_LOCK: "another-pod"},
            [(FOCUS_LOCK, "this-pod")],
            id="vantage-lock-held-by-another-pod",
        ),
    ],
)
async def test_scheduled_focus_export_runs_under_the_focus_pod_lock_and_releases_it_when_the_export_fails(
    monkeypatch: pytest.MonkeyPatch,
    locks_before: dict[str, str],
    locks_after: dict[str, str],
    granted: list[tuple[str, str]],
) -> None:
    lock_manager = proxy_server.proxy_logging_obj.db_spend_update_writer.pod_lock_manager
    redis = _InMemoryRedis(dict(locks_before))
    monkeypatch.setattr(lock_manager, "redis_cache", redis)
    monkeypatch.setattr(lock_manager, "pod_id", "this-pod")
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    logger = FocusLogger(provider="s3", destination_config={"bucket_name": "focus-exports"})

    with pytest.raises(RuntimeError, match="Database not connected"):
        await logger.initialize_focus_export_job()

    assert redis.locks == locks_after
    assert redis.granted == granted


@pytest.mark.asyncio
async def test_scheduled_focus_export_is_skipped_while_another_pod_holds_the_focus_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_manager = proxy_server.proxy_logging_obj.db_spend_update_writer.pod_lock_manager
    redis = _InMemoryRedis({FOCUS_LOCK: "another-pod"})
    monkeypatch.setattr(lock_manager, "redis_cache", redis)
    monkeypatch.setattr(lock_manager, "pod_id", "this-pod")
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    logger = FocusLogger(provider="s3", destination_config={"bucket_name": "focus-exports"})

    await logger.initialize_focus_export_job()

    assert redis.locks == {FOCUS_LOCK: "another-pod"}
    assert redis.granted == []


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


@pytest.mark.parametrize(("granularity", "frequency"), (("daily", "daily"), ("hourly", "hourly")))
def test_1_5_defaults_the_frequency_to_one_bucket_per_file(
    monkeypatch: pytest.MonkeyPatch, granularity: str, frequency: str
) -> None:
    monkeypatch.delenv("FOCUS_FREQUENCY", raising=False)

    logger = FocusLogger(focus_version="1.5", data_granularity=granularity, focus_settings=FOCUS_SETTINGS)

    assert logger.frequency == frequency


def test_1_5_rejects_a_frequency_that_differs_from_the_granularity() -> None:
    with pytest.raises(ValueError, match="FOCUS_FREQUENCY"):
        FocusLogger(focus_version="1.5", frequency="hourly", focus_settings=FOCUS_SETTINGS)


def test_1_5_needs_the_billing_account() -> None:
    with pytest.raises(ValueError, match="billing_account_id and billing_account_name"):
        FocusLogger(focus_version="1.5")


def test_callback_settings_block_configures_the_generic_logger(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FOCUS_VERSION", raising=False)
    monkeypatch.setattr(
        litellm, "callback_settings", {"focus": {**FOCUS_SETTINGS, "version": "1.5", "billed_by": "vendor"}}
    )

    logger = FocusLogger(destination_config={"bucket_name": "focus-test"})

    assert logger.settings.version == "1.5"
    assert logger.settings.billing is not None
    assert logger.settings.billing.billed_by == "vendor"


def test_a_pinned_version_ignores_the_callback_settings_block(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "callback_settings", {"focus": {**FOCUS_SETTINGS, "version": "1.5"}})

    assert FocusLogger(focus_version="1.2").settings == FocusExportSettings(version="1.2", data_granularity="daily")


def test_1_5_engine_reads_spend_logs_and_emits_the_1_5_schema() -> None:
    engine = FocusLogger(
        focus_version="1.5", focus_settings=FOCUS_SETTINGS, destination_config={"bucket_name": "focus-test"}
    )._ensure_engine()

    assert isinstance(engine._database, FocusSpendLogsDatabase)
    assert engine._transformer.schema == FOCUS_1_5_SCHEMA


def test_default_engine_keeps_the_1_2_pipeline() -> None:
    engine = FocusLogger(destination_config={"bucket_name": "focus-test"})._ensure_engine()

    assert isinstance(engine._database, FocusLiteLLMDatabase)
    assert engine._transformer.schema == FOCUS_NORMALIZED_SCHEMA


def test_1_5_csv_export_writes_tiny_costs_without_scientific_notation() -> None:
    engine = FocusLogger(
        focus_version="1.5",
        focus_settings=FOCUS_SETTINGS,
        export_format="csv",
        destination_config={"bucket_name": "focus-test"},
    )._ensure_engine()
    frame = pl.DataFrame({"BilledCost": [Decimal("0.0000001234")]}, schema={"BilledCost": pl.Decimal(38, 10)})

    assert engine._serializer.serialize(frame).decode().splitlines() == ["BilledCost", "0.0000001234"]


def test_1_5_hourly_from_env_runs_on_the_default_hourly_frequency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOCUS_VERSION", "1.5")
    monkeypatch.setenv("FOCUS_DATA_GRANULARITY", "hourly")
    monkeypatch.delenv("FOCUS_FREQUENCY", raising=False)
    monkeypatch.setattr(litellm, "callback_settings", {"focus": FOCUS_SETTINGS})

    logger = FocusLogger()

    assert (logger.settings.version, logger.settings.data_granularity, logger.frequency) == ("1.5", "hourly", "hourly")


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


@pytest.mark.parametrize("frequency", ("hourly", "daily"))
def test_aligned_windows_cap_the_span_at_31_days(frequency: str) -> None:
    with pytest.raises(ValueError, match="limit is 31 days"):
        aligned_windows(
            start_time_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
            end_time_utc=datetime(2026, 3, 1, tzinfo=timezone.utc),
            frequency=frequency,
            now=_at(20),
        )


def test_aligned_windows_allow_a_full_31_day_span() -> None:
    windows = aligned_windows(
        start_time_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
        end_time_utc=datetime(2026, 2, 1, tzinfo=timezone.utc),
        frequency="daily",
        now=_at(20),
    )

    assert len(windows) == 31


class _RecordingFocusLogger(FocusLogger):
    def __init__(self) -> None:
        super().__init__(
            focus_version="1.5", focus_settings=FOCUS_SETTINGS, destination_config={"bucket_name": "focus-test"}
        )
        self.exported: list[tuple[FocusTimeWindow, int | None]] = []  # mutable-ok: records exports for assertions

    async def _export_window(self, *, window: FocusTimeWindow, limit: int | None) -> None:
        self.exported.append((window, limit))


@pytest.mark.asyncio
async def test_export_range_exports_each_aligned_window_in_order() -> None:
    logger = _RecordingFocusLogger()

    windows = await logger.export_range(start_time_utc=_at(10, 6), end_time_utc=_at(12, 6), limit=7, now=_at(20))

    assert logger.exported == [(window, 7) for window in windows]
    assert _bounds(windows) == [(_at(10), _at(11)), (_at(11), _at(12)), (_at(12), _at(13))]


def _router_with_a_broken_focus_block() -> Router:
    return Router(
        model_list=[
            {
                "model_name": "spark",
                "litellm_params": {"model": "hosted_vllm/spark", "api_base": "http://localhost:1"},
                "model_info": {"id": "spark-1", "focus": {"region_id": "on-prem"}},
            }
        ]
    )


def test_1_5_logger_rejects_a_broken_focus_block_when_the_job_is_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(proxy_server, "llm_router", _router_with_a_broken_focus_block())

    with pytest.raises(ValueError, match="deployment spark-1"):
        FocusLogger(focus_version="1.5", focus_settings=FOCUS_SETTINGS).validate_deployment_overrides()


def test_1_2_logger_ignores_focus_blocks(  # test-quality-ok: passes only if validation does not raise, which is the behavior under test
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(proxy_server, "llm_router", _router_with_a_broken_focus_block())

    FocusLogger().validate_deployment_overrides()


@pytest.mark.parametrize(
    ("frequency", "start", "end"),
    (("hourly", _at(10, 9), _at(10, 10)), ("daily", _at(9), _at(10))),
)
def test_last_closed_window_is_the_bucket_before_now(frequency: str, start: datetime, end: datetime) -> None:
    from litellm.integrations.focus.focus_logger import last_closed_window

    window = last_closed_window(frequency=frequency, now=_at(10, 10, 20))

    assert (window.start_time, window.end_time, window.frequency) == (start, end, frequency)
