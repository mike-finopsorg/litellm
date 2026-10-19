"""Version, granularity and billing settings for FOCUS export."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from typing_extensions import Self

FocusVersion: TypeAlias = Literal["1.2", "1.5"]
FocusDataGranularity: TypeAlias = Literal["daily", "hourly"]
BilledBy: TypeAlias = Literal["operator", "vendor"]

DEFAULT_FOCUS_VERSION: Final[FocusVersion] = "1.2"
DEFAULT_FOCUS_DATA_GRANULARITY: Final[FocusDataGranularity] = "daily"


@dataclass(frozen=True, slots=True)
class FocusBillingSettings:
    """Values for the FOCUS 1.5 billing, account and region columns, which LiteLLM data does not carry."""

    billed_by: BilledBy
    billing_account_id: str
    billing_account_name: str
    sub_account_id: str
    sub_account_name: str
    region_id: str | None = None
    region_name: str | None = None
    invoice_issuer_name: str | None = None


@dataclass(frozen=True, slots=True)
class FocusExportSettings:
    version: FocusVersion = DEFAULT_FOCUS_VERSION
    data_granularity: FocusDataGranularity = DEFAULT_FOCUS_DATA_GRANULARITY
    billing: FocusBillingSettings | None = None


class FocusCallbackSettings(BaseModel):
    """The ``callback_settings.focus`` block of the proxy config."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: FocusVersion | None = None
    data_granularity: FocusDataGranularity | None = None
    billed_by: BilledBy | None = None
    billing_account_id: str | None = None
    billing_account_name: str | None = None
    sub_account_id: str | None = None
    sub_account_name: str | None = None
    region_id: str | None = None
    region_name: str | None = None
    invoice_issuer_name: str | None = None

    @field_validator("version", mode="before")
    @classmethod
    def _version_as_text(cls, value: object) -> object:
        return str(value) if isinstance(value, (int, float)) else value

    @model_validator(mode="after")
    def _region_is_set_as_a_pair(self) -> Self:
        if (self.region_id is None) != (self.region_name is None):
            raise ValueError("callback_settings.focus region_id and region_name must be set together")
        return self


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


def _billing_settings(block: FocusCallbackSettings) -> FocusBillingSettings:
    if not block.billing_account_id or not block.billing_account_name:
        raise ValueError(
            "FOCUS 1.5 needs callback_settings.focus.billing_account_id and billing_account_name, "
            "the account the export is produced for"
        )
    return FocusBillingSettings(
        billed_by=block.billed_by or "operator",
        billing_account_id=block.billing_account_id,
        billing_account_name=block.billing_account_name,
        sub_account_id=block.sub_account_id or block.billing_account_id,
        sub_account_name=block.sub_account_name or block.billing_account_name,
        region_id=block.region_id,
        region_name=block.region_name,
        invoice_issuer_name=block.invoice_issuer_name,
    )


def parse_focus_export_settings(
    *,
    version: str | None,
    data_granularity: str | None,
    config: Mapping[str, object] | None = None,
) -> FocusExportSettings:
    """Validate the version and granularity from the environment and the ``callback_settings.focus`` block, where
    the block wins, and read the 1.5 billing settings from the block."""
    block: Final = FocusCallbackSettings.model_validate(config) if config is not None else FocusCallbackSettings()
    stripped_version: Final = (version or "").strip()
    stripped_granularity: Final = (data_granularity or "").strip().lower()
    parsed_version: Final = block.version or (
        _parse_version(stripped_version) if stripped_version else DEFAULT_FOCUS_VERSION
    )
    parsed_granularity: Final = block.data_granularity or (
        _parse_data_granularity(stripped_granularity) if stripped_granularity else DEFAULT_FOCUS_DATA_GRANULARITY
    )
    match parsed_version:
        case "1.2":
            if parsed_granularity != "daily":
                raise ValueError("FOCUS_VERSION=1.2 only supports FOCUS_DATA_GRANULARITY=daily")
            return FocusExportSettings(version="1.2", data_granularity="daily")
        case "1.5":
            return FocusExportSettings(
                version="1.5", data_granularity=parsed_granularity, billing=_billing_settings(block)
            )


def resolve_export_frequency(settings: FocusExportSettings, configured: str | None) -> str:
    """The scheduler frequency: FOCUS_FREQUENCY (default hourly) on 1.2; one file per charge period on 1.5, where
    FOCUS_FREQUENCY may only restate the granularity."""
    requested: Final = (configured or "").strip().lower()
    match settings.version:
        case "1.2":
            return requested or "hourly"
        case "1.5":
            if requested and requested != settings.data_granularity:
                raise ValueError(
                    f"FOCUS_FREQUENCY='{requested}' is not used with FOCUS_VERSION=1.5; one file is written per "
                    f"FOCUS_DATA_GRANULARITY={settings.data_granularity} bucket, so unset it or set it to "
                    f"'{settings.data_granularity}'"
                )
            return settings.data_granularity


__all__ = (
    "DEFAULT_FOCUS_DATA_GRANULARITY",
    "DEFAULT_FOCUS_VERSION",
    "BilledBy",
    "FocusBillingSettings",
    "FocusCallbackSettings",
    "FocusDataGranularity",
    "FocusExportSettings",
    "FocusVersion",
    "parse_focus_export_settings",
    "resolve_export_frequency",
)
