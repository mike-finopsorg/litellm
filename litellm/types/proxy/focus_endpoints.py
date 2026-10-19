"""FOCUS export endpoint types for LiteLLM Proxy."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

FocusCell: TypeAlias = str | float | int | bool | None


def _as_utc(moment: datetime) -> datetime:
    return moment.replace(tzinfo=timezone.utc) if moment.tzinfo is None else moment.astimezone(timezone.utc)


class FocusDryRunRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    start_time_utc: datetime | None = Field(
        None,
        description="Inclusive start of the preview window in UTC; a value without an offset is read as UTC. "
        "Leave both bounds out to preview the last closed scheduler window",
    )
    end_time_utc: datetime | None = Field(
        None, description="Exclusive end of the preview window in UTC; a value without an offset is read as UTC"
    )
    limit: int = Field(500, ge=1, le=5000, description="Maximum number of FOCUS rows to return; totals cover all rows")

    @field_validator("start_time_utc", "end_time_utc")
    @classmethod
    def _utc(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _as_utc(value)

    @model_validator(mode="after")
    def _both_bounds_in_order(self) -> Self:
        if (self.start_time_utc is None) != (self.end_time_utc is None):
            raise ValueError("start_time_utc and end_time_utc must be given together")
        if self.start_time_utc is not None and self.end_time_utc is not None:
            if self.end_time_utc <= self.start_time_utc:
                raise ValueError("end_time_utc must be after start_time_utc")
        return self


class FocusExportRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    start_time_utc: datetime = Field(
        ..., description="Inclusive start of the export range in UTC; a value without an offset is read as UTC"
    )
    end_time_utc: datetime = Field(
        ..., description="Exclusive end of the export range in UTC; a value without an offset is read as UTC"
    )

    @field_validator("start_time_utc", "end_time_utc")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        return _as_utc(value)

    @model_validator(mode="after")
    def _end_after_start(self) -> Self:
        if self.end_time_utc <= self.start_time_utc:
            raise ValueError("end_time_utc must be after start_time_utc")
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
    rows: tuple[Mapping[str, FocusCell], ...]


class FocusExportResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    focus_version: str
    data_granularity: str
    windows: tuple[FocusExportWindow, ...]
