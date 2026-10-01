from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal

import polars as pl
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm.integrations.focus.destinations.base import FocusTimeWindow
from litellm.integrations.focus.settings import FocusExportSettings
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.spend_tracking.focus_endpoints import get_focus_export_target, router

DAY = datetime(2026, 9, 30, tzinfo=timezone.utc)
NEXT_DAY = datetime(2026, 10, 1, tzinfo=timezone.utc)
FRAME = pl.DataFrame(
    {
        "BilledCost": [Decimal("0.0000315"), Decimal("0.0000105")],
        "ChargePeriodStart": [DAY, DAY],
        "PrincipalId": ["hermes", None],
    },
    schema={
        "BilledCost": pl.Decimal(38, 10),
        "ChargePeriodStart": pl.Datetime(time_unit="us", time_zone="UTC"),
        "PrincipalId": pl.String,
    },
)


@dataclass
class _Target:
    settings: FocusExportSettings = field(
        default_factory=lambda: FocusExportSettings(version="1.5", data_granularity="daily")
    )
    frequency: str = "daily"
    previews: list[tuple[int | None, datetime | None, datetime | None]] = field(
        default_factory=list
    )  # mutable-ok: records calls
    exports: list[tuple[datetime, datetime]] = field(default_factory=list)  # mutable-ok: records calls
    error: ValueError | None = None

    async def preview(
        self, *, limit: int | None, start_time_utc: datetime | None, end_time_utc: datetime | None
    ) -> pl.DataFrame:
        self.previews.append((limit, start_time_utc, end_time_utc))
        return FRAME

    async def export_range(self, *, start_time_utc: datetime, end_time_utc: datetime) -> tuple[FocusTimeWindow, ...]:
        if self.error is not None:
            raise self.error
        self.exports.append((start_time_utc, end_time_utc))
        return (FocusTimeWindow(start_time=DAY, end_time=NEXT_DAY, frequency="daily"),)


def _client(target: _Target | None, role: LitellmUserRoles = LitellmUserRoles.PROXY_ADMIN) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_role=role, user_id="u1")
    if target is not None:
        app.dependency_overrides[get_focus_export_target] = lambda: target
    return TestClient(app)


@pytest.fixture
def target() -> Iterator[_Target]:
    yield _Target()


def test_dry_run_returns_json_safe_rows_for_the_widened_window(target: _Target) -> None:
    response = _client(target).post(
        "/focus/dry-run",
        json={"start_time_utc": "2026-09-30T06:00:00Z", "end_time_utc": "2026-09-30T18:00:00Z", "limit": 10},
    )

    assert response.status_code == 200
    assert target.previews == [(10, DAY, NEXT_DAY)]
    assert response.json() == {
        "focus_version": "1.5",
        "data_granularity": "daily",
        "window": {"start_time_utc": "2026-09-30T00:00:00Z", "end_time_utc": "2026-10-01T00:00:00Z"},
        "total_rows": 2,
        "total_billed_cost": pytest.approx(0.000042),
        "rows": [
            {"BilledCost": 0.0000315, "ChargePeriodStart": "2026-09-30T00:00:00Z", "PrincipalId": "hermes"},
            {"BilledCost": 0.0000105, "ChargePeriodStart": "2026-09-30T00:00:00Z", "PrincipalId": None},
        ],
    }


def test_dry_run_without_a_window_previews_everything(target: _Target) -> None:
    response = _client(target).post("/focus/dry-run", json={})

    assert (response.status_code, response.json()["window"], target.previews) == (200, None, [(500, None, None)])


def test_export_returns_the_windows_it_uploaded(target: _Target) -> None:
    response = _client(target).post(
        "/focus/export", json={"start_time_utc": "2026-09-30T00:00:00Z", "end_time_utc": "2026-10-01T00:00:00Z"}
    )

    assert response.status_code == 200
    assert target.exports == [(DAY, NEXT_DAY)]
    assert response.json()["windows"] == [
        {"start_time_utc": "2026-09-30T00:00:00Z", "end_time_utc": "2026-10-01T00:00:00Z"}
    ]


def test_export_reports_invalid_ranges_as_bad_requests() -> None:
    target = _Target(error=ValueError("Manual FOCUS export spans 900 windows; the limit is 744"))
    response = _client(target).post(
        "/focus/export", json={"start_time_utc": "2026-01-01T00:00:00Z", "end_time_utc": "2026-03-01T00:00:00Z"}
    )

    assert (response.status_code, response.json()["detail"]["error"]) == (
        400,
        "Manual FOCUS export spans 900 windows; the limit is 744",
    )


@pytest.mark.parametrize("path", ("/focus/dry-run", "/focus/export"))
def test_end_must_be_after_start(target: _Target, path: str) -> None:
    response = _client(target).post(
        path, json={"start_time_utc": "2026-09-30T00:00:00Z", "end_time_utc": "2026-09-30T00:00:00Z"}
    )

    assert response.status_code == 422
    assert (target.previews, target.exports) == ([], [])


@pytest.mark.parametrize("path", ("/focus/dry-run", "/focus/export"))
def test_only_proxy_admins_can_preview_or_export(target: _Target, path: str) -> None:
    response = _client(target, role=LitellmUserRoles.INTERNAL_USER).post(
        path, json={"start_time_utc": "2026-09-30T00:00:00Z", "end_time_utc": "2026-10-01T00:00:00Z"}
    )

    assert response.status_code == 403
    assert (target.previews, target.exports) == ([], [])


def test_endpoints_explain_how_to_enable_focus_when_the_callback_is_missing() -> None:
    response = _client(None).post("/focus/dry-run", json={})

    assert response.status_code == 404
    assert "litellm_settings.callbacks" in response.json()["detail"]["error"]
