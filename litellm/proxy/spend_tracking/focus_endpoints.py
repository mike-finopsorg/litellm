"""Admin endpoints to preview and backfill the FOCUS export configured through the ``focus`` callback."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Annotated, Final, Protocol

from fastapi import APIRouter, Depends, HTTPException
from pydantic import TypeAdapter
from typing_extensions import ReadOnly, TypedDict

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.integrations.focus.destinations.base import FocusTimeWindow
from litellm.integrations.focus.focus_logger import FocusLogger, aligned_windows
from litellm.integrations.focus.settings import FocusExportSettings
from litellm.proxy._types import CommonProxyErrors, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.proxy.focus_endpoints import (
    FocusCell,
    FocusDryRunRequest,
    FocusDryRunResponse,
    FocusExportRequest,
    FocusExportResponse,
    FocusExportWindow,
)

if TYPE_CHECKING:
    import polars as pl

router: Final = APIRouter()

_ROWS_ADAPTER: Final = TypeAdapter(tuple[Mapping[str, FocusCell], ...])


class _ErrorDetail(TypedDict):
    error: ReadOnly[str]


def _http_error(status_code: int, message: str) -> HTTPException:
    detail: Final[_ErrorDetail] = {"error": message}
    return HTTPException(status_code=status_code, detail=detail)


class FocusExportTarget(Protocol):
    @property
    def settings(self) -> FocusExportSettings: ...

    @property
    def frequency(self) -> str: ...

    async def preview(
        self, *, limit: int | None, start_time_utc: datetime | None, end_time_utc: datetime | None
    ) -> pl.DataFrame: ...

    async def export_range(
        self, *, start_time_utc: datetime, end_time_utc: datetime
    ) -> tuple[FocusTimeWindow, ...]: ...


def get_focus_export_target() -> FocusExportTarget:
    """The FocusLogger registered through ``litellm_settings.callbacks: ["focus"]``."""
    registered: Final = tuple(
        callback
        for callback in litellm.logging_callback_manager.get_custom_loggers_for_type(callback_type=FocusLogger)
        if type(callback) is FocusLogger
    )
    if not registered:
        raise _http_error(404, 'FOCUS export is not enabled. Add "focus" to litellm_settings.callbacks.')
    return registered[0]


def _require_admin(user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)]) -> None:
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise _http_error(403, CommonProxyErrors.not_allowed_access.value)


def _window(windows: tuple[FocusTimeWindow, ...]) -> FocusExportWindow | None:
    if not windows:
        return None
    return FocusExportWindow(start_time_utc=windows[0].start_time, end_time_utc=windows[-1].end_time)


def _column_total(frame: pl.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return 0.0
    return float(frame.get_column(column).sum() or 0)


def _json_rows(frame: pl.DataFrame) -> tuple[Mapping[str, FocusCell], ...]:
    import polars as pl

    readable: Final = frame.with_columns(
        pl.col(pl.Decimal).cast(pl.Float64),  # cast-ok: polars dtype conversion for JSON
        pl.col(pl.Datetime).dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    return _ROWS_ADAPTER.validate_python(readable.to_dicts())


@router.post(
    "/focus/dry-run",
    tags=["FOCUS"],
    dependencies=[Depends(_require_admin)],
    response_model=FocusDryRunResponse,
)
async def focus_dry_run(
    request: FocusDryRunRequest,
    target: Annotated[FocusExportTarget, Depends(get_focus_export_target)],
) -> FocusDryRunResponse:
    """Return the FOCUS rows an export of [start_time_utc, end_time_utc) would upload, without uploading them. Totals
    cover every row in the window and ``limit`` caps the rows returned. The window is widened to whole buckets the
    same way /focus/export widens it. Only proxy admins can call it."""
    try:
        windows: Final = aligned_windows(
            start_time_utc=request.start_time_utc,
            end_time_utc=request.end_time_utc,
            frequency=target.frequency,
            now=datetime.now(timezone.utc),
        )
        window: Final = _window(windows)
        frame: Final = await target.preview(
            limit=None,
            start_time_utc=window.start_time_utc if window is not None else request.start_time_utc,
            end_time_utc=window.end_time_utc if window is not None else request.end_time_utc,
        )
    except ValueError as exc:
        raise _http_error(400, str(exc)) from exc
    returned: Final = frame.head(request.limit)
    return FocusDryRunResponse(
        focus_version=target.settings.version,
        data_granularity=target.settings.data_granularity,
        window=window,
        total_rows=frame.height,
        total_billed_cost=_column_total(frame, "BilledCost"),
        total_effective_cost=_column_total(frame, "EffectiveCost"),
        total_list_cost=_column_total(frame, "ListCost"),
        returned_rows=returned.height,
        rows=_json_rows(returned),
    )


@router.post(
    "/focus/export",
    tags=["FOCUS"],
    dependencies=[Depends(_require_admin)],
    response_model=FocusExportResponse,
)
async def focus_export(
    request: FocusExportRequest,
    target: Annotated[FocusExportTarget, Depends(get_focus_export_target)],
) -> FocusExportResponse:
    """Export [start_time_utc, end_time_utc) to the configured destination, one file per scheduled window, e.g. to
    backfill history. The range is widened to whole buckets and stops before the bucket still in progress, so
    re-running a range overwrites the same files. With FOCUS_VERSION=1.2 the range selects daily rollups by their
    last update time rather than by usage date. Only proxy admins can call it."""
    try:
        windows: Final = await target.export_range(
            start_time_utc=request.start_time_utc, end_time_utc=request.end_time_utc
        )
    except ValueError as exc:
        raise _http_error(400, str(exc)) from exc
    verbose_proxy_logger.info("FOCUS export uploaded %d windows", len(windows))
    return FocusExportResponse(
        focus_version=target.settings.version,
        data_granularity=target.settings.data_granularity,
        windows=tuple(
            FocusExportWindow(start_time_utc=window.start_time, end_time_utc=window.end_time) for window in windows
        ),
    )
