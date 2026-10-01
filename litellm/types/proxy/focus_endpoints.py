"""FOCUS export endpoint types for LiteLLM Proxy."""

from __future__ import annotations

from datetime import datetime
from typing import TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

FocusCell: TypeAlias = str | float | int | bool | None


def _require_end_after_start(start: datetime | None, end: datetime | None) -> None:
    if start is not None and end is not None and end <= start:
        raise ValueError("end_time_utc must be after start_time_utc")


class FocusDryRunRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    start_time_utc: datetime | None = Field(None, description="Inclusive start of the preview window (UTC)")
    end_time_utc: datetime | None = Field(None, description="Exclusive end of the preview window (UTC)")
    limit: int = Field(500, ge=1, le=5000, description="Maximum number of FOCUS rows to return; totals cover all rows")

    @model_validator(mode="after")
    def _end_after_start(self) -> Self:
        _require_end_after_start(self.start_time_utc, self.end_time_utc)
        return self


class FocusExportRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    start_time_utc: datetime = Field(..., description="Inclusive start of the export range (UTC)")
    end_time_utc: datetime = Field(..., description="Exclusive end of the export range (UTC)")

    @model_validator(mode="after")
    def _end_after_start(self) -> Self:
        _require_end_after_start(self.start_time_utc, self.end_time_utc)
        return self


class FocusExportWindow(BaseModel):
    model_config = ConfigDict(frozen=True)

    start_time_utc: datetime
    end_time_utc: datetime


class FocusDryRunResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    focus_version: str
    data_granularity: str
    window: FocusExportWindow | None
    total_rows: int
    total_billed_cost: float
    total_effective_cost: float
    total_list_cost: float
    returned_rows: int
    rows: tuple[dict[str, FocusCell], ...]


class FocusExportResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    focus_version: str
    data_granularity: str
    windows: tuple[FocusExportWindow, ...]
