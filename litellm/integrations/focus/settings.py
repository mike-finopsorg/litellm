"""Version and data granularity settings for Focus export."""

from __future__ import annotations

import socket
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

FocusVersion: TypeAlias = Literal["1.2", "1.5"]
FocusDataGranularity: TypeAlias = Literal["daily", "hourly"]

DEFAULT_FOCUS_VERSION: Final[FocusVersion] = "1.2"
DEFAULT_FOCUS_DATA_GRANULARITY: Final[FocusDataGranularity] = "daily"


_TRUE_VALUES: Final = frozenset(("true", "1", "yes", "on"))
_FALSE_VALUES: Final = frozenset(("false", "0", "no", "off"))


@dataclass(frozen=True, slots=True)
class FocusBillingSettings:
    """Values for the FOCUS 1.5 billing and account columns, which LiteLLM data does not carry."""

    include_spend: bool
    billing_account_id: str
    billing_account_name: str
    sub_account_id: str
    sub_account_name: str
    region_id: str | None = None
    region_name: str | None = None


def default_account_label(hostname: str | None = None) -> str:
    return f"{hostname or socket.gethostname()}-litellm"


def _default_billing_settings() -> FocusBillingSettings:
    label: Final = default_account_label()
    return FocusBillingSettings(
        include_spend=True,
        billing_account_id=label,
        billing_account_name=label,
        sub_account_id=label,
        sub_account_name=label,
    )


@dataclass(frozen=True, slots=True)
class FocusExportSettings:
    version: FocusVersion = DEFAULT_FOCUS_VERSION
    data_granularity: FocusDataGranularity = DEFAULT_FOCUS_DATA_GRANULARITY
    billing: FocusBillingSettings = field(default_factory=_default_billing_settings)


def _parse_version(raw: str) -> FocusVersion:
    match raw:
        case "1.2" | "1.5":
            return raw
        case _:
            raise ValueError(f"Unsupported FOCUS_VERSION '{raw}'. Supported: 1.2, 1.5")


def _parse_data_granularity(raw: str) -> FocusDataGranularity:
    match raw:
        case "daily" | "hourly":
            return raw
        case _:
            raise ValueError(f"Unsupported FOCUS_DATA_GRANULARITY '{raw}'. Supported: daily, hourly")


def _parse_include_spend(raw: str) -> bool:
    if raw in _TRUE_VALUES:
        return True
    if raw in _FALSE_VALUES:
        return False
    raise ValueError(f"Unsupported FOCUS_INCLUDE_SPEND '{raw}'. Supported: true, false")


def parse_focus_billing_settings(env: Mapping[str, str], *, hostname: str | None = None) -> FocusBillingSettings:
    """Read FOCUS_INCLUDE_SPEND and the FOCUS_BILLING_* / FOCUS_ACCOUNT_* / FOCUS_REGION_* values, treating empty
    values as unset."""
    label: Final = default_account_label(hostname)

    def _value(name: str) -> str:
        return env.get(name, "").strip()

    include_spend: Final = _value("FOCUS_INCLUDE_SPEND").lower()
    region_id: Final = _value("FOCUS_REGION_ID") or None
    region_name: Final = _value("FOCUS_REGION_NAME") or None
    if (region_id is None) != (region_name is None):
        raise ValueError("FOCUS_REGION_ID and FOCUS_REGION_NAME must be set together")
    return FocusBillingSettings(
        include_spend=_parse_include_spend(include_spend) if include_spend else True,
        billing_account_id=_value("FOCUS_BILLING_ID") or label,
        billing_account_name=_value("FOCUS_BILLING_NAME") or label,
        sub_account_id=_value("FOCUS_ACCOUNT_ID") or label,
        sub_account_name=_value("FOCUS_ACCOUNT_NAME") or label,
        region_id=region_id,
        region_name=region_name,
    )


def parse_focus_export_settings(
    *,
    version: str | None,
    data_granularity: str | None,
    env: Mapping[str, str] = MappingProxyType({}),
    hostname: str | None = None,
) -> FocusExportSettings:
    """Validate raw version/granularity values and FOCUS_* env values, treating empty values as unset."""
    stripped_version: Final = (version or "").strip()
    stripped_granularity: Final = (data_granularity or "").strip().lower()
    settings: Final = FocusExportSettings(
        version=_parse_version(stripped_version) if stripped_version else DEFAULT_FOCUS_VERSION,
        data_granularity=(
            _parse_data_granularity(stripped_granularity) if stripped_granularity else DEFAULT_FOCUS_DATA_GRANULARITY
        ),
        billing=parse_focus_billing_settings(env, hostname=hostname),
    )
    if settings.version == "1.2" and settings.data_granularity != "daily":
        raise ValueError("FOCUS_VERSION=1.2 only supports FOCUS_DATA_GRANULARITY=daily")
    return settings


def _frequencies_covering_whole_buckets(granularity: FocusDataGranularity) -> tuple[str, ...]:
    match granularity:
        case "daily":
            return ("daily",)
        case "hourly":
            return ("hourly", "daily")


def validate_export_frequency(settings: FocusExportSettings, frequency: str) -> None:
    """Require 1.5 exports to cover whole charge periods so no bucket is split across files."""
    if settings.version != "1.5":
        return
    allowed: Final = _frequencies_covering_whole_buckets(settings.data_granularity)
    if frequency not in allowed:
        raise ValueError(
            f"FOCUS_VERSION=1.5 with FOCUS_DATA_GRANULARITY={settings.data_granularity} "
            f"requires FOCUS_FREQUENCY={' or '.join(allowed)}, got '{frequency}'"
        )


__all__ = (
    "DEFAULT_FOCUS_DATA_GRANULARITY",
    "DEFAULT_FOCUS_VERSION",
    "FocusBillingSettings",
    "FocusDataGranularity",
    "FocusExportSettings",
    "FocusVersion",
    "default_account_label",
    "parse_focus_billing_settings",
    "parse_focus_export_settings",
    "validate_export_frequency",
)
