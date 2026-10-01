"""Version and data granularity settings for Focus export."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

FocusVersion: TypeAlias = Literal["1.2"]
FocusDataGranularity: TypeAlias = Literal["daily"]

DEFAULT_FOCUS_VERSION: Final[FocusVersion] = "1.2"
DEFAULT_FOCUS_DATA_GRANULARITY: Final[FocusDataGranularity] = "daily"


@dataclass(frozen=True, slots=True)
class FocusExportSettings:
    version: FocusVersion = DEFAULT_FOCUS_VERSION
    data_granularity: FocusDataGranularity = DEFAULT_FOCUS_DATA_GRANULARITY


def _parse_version(raw: str) -> FocusVersion:
    match raw:
        case "1.2":
            return raw
        case _:
            raise ValueError(f"Unsupported FOCUS_VERSION '{raw}'. Supported: 1.2")


def _parse_data_granularity(raw: str) -> FocusDataGranularity:
    match raw:
        case "daily":
            return raw
        case _:
            raise ValueError(f"Unsupported FOCUS_DATA_GRANULARITY '{raw}'. Supported: daily")


def parse_focus_export_settings(*, version: str | None, data_granularity: str | None) -> FocusExportSettings:
    """Validate raw version/granularity values, treating empty values as unset."""
    stripped_version: Final = (version or "").strip()
    stripped_granularity: Final = (data_granularity or "").strip().lower()
    return FocusExportSettings(
        version=_parse_version(stripped_version) if stripped_version else DEFAULT_FOCUS_VERSION,
        data_granularity=(
            _parse_data_granularity(stripped_granularity) if stripped_granularity else DEFAULT_FOCUS_DATA_GRANULARITY
        ),
    )


__all__ = (
    "DEFAULT_FOCUS_DATA_GRANULARITY",
    "DEFAULT_FOCUS_VERSION",
    "FocusDataGranularity",
    "FocusExportSettings",
    "FocusVersion",
    "parse_focus_export_settings",
)
