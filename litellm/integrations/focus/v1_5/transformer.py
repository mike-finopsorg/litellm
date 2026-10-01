"""FOCUS 1.5 transformer for SpendLogs SKU buckets."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import astuple, fields
from types import MappingProxyType
from typing import Final, Literal

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from litellm.constants import (
    LITELLM_PROXY_MASTER_KEY_ALIAS,
    LITTELM_CLI_SERVICE_ACCOUNT_NAME,
    LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME,
)

from ..settings import FocusExportSettings
from .overrides import ResolvedColumns, ResolveDeploymentOverrides, proxy_deployment_overrides, resolve_columns

_UTC_TIMESTAMP: Final = pl.Datetime(time_unit="us", time_zone="UTC")
_COST: Final = pl.Decimal(38, 10)
_UNIT_PRICE: Final = pl.Decimal(38, 18)

# see: https://focus.finops.org/focus-specification/v1-5/
FOCUS_1_5_SCHEMA: Final = pl.Schema(
    (
        ("BilledCost", _COST),
        ("BillingAccountId", pl.String),
        ("BillingAccountName", pl.String),
        ("BillingAccountType", pl.String),
        ("BillingCurrency", pl.String),
        ("BillingPeriodEnd", _UTC_TIMESTAMP),
        ("BillingPeriodStart", _UTC_TIMESTAMP),
        ("ChargeCategory", pl.String),
        ("ChargeClass", pl.String),
        ("ChargeDescription", pl.String),
        ("ChargeFrequency", pl.String),
        ("ChargePeriodEnd", _UTC_TIMESTAMP),
        ("ChargePeriodStart", _UTC_TIMESTAMP),
        ("ConsumedQuantity", _COST),
        ("ConsumedUnit", pl.String),
        ("ContractedCost", _COST),
        ("ContractedUnitPrice", _UNIT_PRICE),
        ("CredentialId", pl.String),
        ("EffectiveCost", _COST),
        ("HostProviderName", pl.String),
        ("InvoiceIssuerName", pl.String),
        ("ListCost", _COST),
        ("ListUnitPrice", _UNIT_PRICE),
        ("PricingCategory", pl.String),
        ("PricingCurrency", pl.String),
        ("PricingCurrencyContractedUnitPrice", _UNIT_PRICE),
        ("PricingCurrencyEffectiveCost", _COST),
        ("PricingCurrencyListUnitPrice", _UNIT_PRICE),
        ("PricingQuantity", _COST),
        ("PricingUnit", pl.String),
        ("PrincipalId", pl.String),
        ("RegionId", pl.String),
        ("RegionName", pl.String),
        ("RequesterDetails", pl.String),
        ("ResourceId", pl.String),
        ("ResourceName", pl.String),
        ("ResourceType", pl.String),
        ("ServiceCategory", pl.String),
        ("ServiceName", pl.String),
        ("ServiceProviderName", pl.String),
        ("ServiceSubcategory", pl.String),
        ("SkuId", pl.String),
        ("SkuMeter", pl.String),
        ("SkuPriceId", pl.String),
        ("SubAccountId", pl.String),
        ("SubAccountName", pl.String),
        ("SubAccountType", pl.String),
        ("Tags", pl.String),
    )
)

_METER_SLUGS: Final = MappingProxyType(
    {
        "Input Tokens": "input-tokens",
        "Cached Input Tokens": "cached-input-tokens",
        "Cache Write Tokens": "cache-write-tokens",
        "Output Tokens": "output-tokens",
        "Other Usage": "other-usage",
    }
)

_CREDENTIAL_TYPES: Final = MappingProxyType(
    {
        LITELLM_PROXY_MASTER_KEY_ALIAS: "Master Key",
        LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME: "Service Account Key",
        LITTELM_CLI_SERVICE_ACCOUNT_NAME: "Service Account Key",
    }
)


class _RequesterValue(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: str = Field(serialization_alias="Type")
    name: str | None = Field(default=None, serialization_alias="Name")
    email: str | None = Field(default=None, serialization_alias="Email")


class _RequesterEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: Literal["Principal", "Credential"]
    value: _RequesterValue


_REQUESTER_DETAILS_ADAPTER: Final = TypeAdapter(tuple[_RequesterEntry, ...])
_REQUEST_TAGS_ADAPTER: Final = TypeAdapter(tuple[JsonValue, ...])
_TAGS_ADAPTER: Final = TypeAdapter(dict[str, str | bool])


def _requester_details(row: Mapping[str, str | None]) -> str | None:
    principal_id: Final = row["principal_id"]
    credential_id: Final = row["credential_id"]
    principal: Final = (
        (
            _RequesterEntry(
                key="Principal",
                value=_RequesterValue(kind="User", name=row["principal_name"], email=row["principal_email"]),
            ),
        )
        if principal_id is not None
        else ()
    )
    credential: Final = (
        (
            _RequesterEntry(
                key="Credential",
                value=_RequesterValue(
                    kind=_CREDENTIAL_TYPES.get(credential_id, "API Key"), name=row["credential_name"]
                ),
            ),
        )
        if credential_id is not None
        else ()
    )
    entries: Final = principal + credential
    if not entries:
        return None
    return _REQUESTER_DETAILS_ADAPTER.dump_json(entries, by_alias=True, exclude_none=True).decode()


def _tags(row: Mapping[str, str | None]) -> str | None:
    user_tags: Final = tuple(
        (str(tag), True) for tag in _REQUEST_TAGS_ADAPTER.validate_json(row["request_tags"] or "[]") if tag is not None
    )
    litellm_tags: Final = tuple(
        (f"litellm/{name}", value)
        for name, value in (("team_id", row["team_id"]), ("team_alias", row["team_alias"]))
        if value is not None
    )
    tags: Final = MappingProxyType({key: value for key, value in sorted(user_tags + litellm_tags)})
    return _TAGS_ADAPTER.dump_json(_TAGS_ADAPTER.validate_python(tags)).decode() if tags else None


def _json_column(
    columns: tuple[str, ...], build: Callable[[Mapping[str, str | None]], str | None], alias: str
) -> pl.Expr:
    return pl.struct(columns).map_elements(build, return_dtype=pl.String, skip_nulls=False).alias(alias)


def _utc(column: pl.Expr) -> pl.Expr:
    return column.str.to_datetime("%Y-%m-%dT%H:%M:%SZ", time_unit="us", time_zone="UTC")


def _cost(column: pl.Expr) -> pl.Expr:
    return column.cast(_COST)  # cast-ok: polars dtype conversion


def _unit_price(cost: str) -> pl.Expr:
    price: Final = pl.when(pl.col("quantity") > 0).then(pl.col(cost) / pl.col("quantity"))
    return price.cast(_UNIT_PRICE)  # cast-ok: polars dtype conversion


_RESOLVED_FIELDS: Final = tuple(field.name for field in fields(ResolvedColumns))
_LOOKUP_SCHEMA: Final = pl.Schema(
    (
        ("model_id", pl.String),
        ("custom_llm_provider", pl.String),
        *((name, pl.Boolean if name == "include_spend" else pl.String) for name in _RESOLVED_FIELDS),
    )
)


class Focus15Transformer:
    """Maps FocusSpendLogsDatabase SKU buckets onto the FOCUS 1.5 Cost and Usage columns LiteLLM can populate."""

    schema = FOCUS_1_5_SCHEMA

    def __init__(
        self,
        settings: FocusExportSettings | None = None,
        resolve_overrides: ResolveDeploymentOverrides = proxy_deployment_overrides,
    ) -> None:
        self._billing: Final = (settings or FocusExportSettings()).billing
        self._resolve_overrides: Final = resolve_overrides

    def _lookup(self, frame: pl.DataFrame) -> pl.DataFrame:
        keys: Final = frame.select("model_id", "custom_llm_provider").unique().rows()
        rows: Final = tuple(
            (
                model_id,
                provider,
                *astuple(
                    resolve_columns(
                        self._billing,
                        self._resolve_overrides(model_id) if model_id is not None else None,
                        provider,
                    )
                ),
            )
            for model_id, provider in keys
        )
        return pl.DataFrame(rows, schema=_LOOKUP_SCHEMA, orient="row")

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        joined: Final = frame.join(
            self._lookup(frame),
            on=("model_id", "custom_llm_provider"),
            how="left",
            nulls_equal=True,
            maintain_order="left",
        )
        is_token_meter: Final = pl.col("meter") != "Other Usage"
        unit: Final = pl.when(is_token_meter).then(pl.lit("Tokens")).otherwise(pl.lit("Requests"))
        billed: Final = _cost(pl.when(pl.col("include_spend")).then(pl.col("billed_cost")).otherwise(0))
        has_deployment: Final = pl.col("model_id").is_not_null()
        sku_id: Final = pl.concat_str(
            pl.col("model").fill_null("unknown-model"),
            pl.col("meter").replace_strict(_METER_SLUGS, return_dtype=pl.String),
            separator="/",
        )
        charge_period_start: Final = _utc(pl.col("charge_period_start"))
        selected: Final = joined.select(
            billed.alias("BilledCost"),
            pl.col("billing_account_id").alias("BillingAccountId"),
            pl.col("billing_account_name").alias("BillingAccountName"),
            pl.lit("LiteLLM Billing Account").alias("BillingAccountType"),
            pl.lit("USD").alias("BillingCurrency"),
            charge_period_start.dt.truncate("1mo").dt.offset_by("1mo").alias("BillingPeriodEnd"),
            charge_period_start.dt.truncate("1mo").alias("BillingPeriodStart"),
            pl.lit("Usage").alias("ChargeCategory"),
            pl.lit(None, dtype=pl.String).alias("ChargeClass"),
            pl.concat_str(pl.col("model").fill_null("unknown-model"), pl.col("meter"), separator=" ").alias(
                "ChargeDescription"
            ),
            pl.lit("Usage-Based").alias("ChargeFrequency"),
            _utc(pl.col("charge_period_end")).alias("ChargePeriodEnd"),
            charge_period_start.alias("ChargePeriodStart"),
            _cost(pl.col("quantity")).alias("ConsumedQuantity"),
            unit.alias("ConsumedUnit"),
            _cost(pl.col("contracted_cost")).alias("ContractedCost"),
            _unit_price("contracted_cost").alias("ContractedUnitPrice"),
            pl.col("credential_id").alias("CredentialId"),
            billed.alias("EffectiveCost"),
            pl.col("host_provider_name").alias("HostProviderName"),
            pl.col("invoice_issuer_name").alias("InvoiceIssuerName"),
            _cost(pl.col("list_cost")).alias("ListCost"),
            _unit_price("list_cost").alias("ListUnitPrice"),
            pl.lit("Standard").alias("PricingCategory"),
            pl.lit("USD").alias("PricingCurrency"),
            _unit_price("contracted_cost").alias("PricingCurrencyContractedUnitPrice"),
            billed.alias("PricingCurrencyEffectiveCost"),
            _unit_price("list_cost").alias("PricingCurrencyListUnitPrice"),
            _cost(pl.col("quantity")).alias("PricingQuantity"),
            unit.alias("PricingUnit"),
            pl.col("principal_id").alias("PrincipalId"),
            pl.col("region_id").alias("RegionId"),
            pl.col("region_name").alias("RegionName"),
            _json_column(
                ("principal_id", "principal_name", "principal_email", "credential_id", "credential_name"),
                _requester_details,
                "RequesterDetails",
            ),
            pl.col("model_id").alias("ResourceId"),
            pl.when(has_deployment).then(pl.col("model_group")).alias("ResourceName"),
            pl.when(has_deployment).then(pl.lit("LiteLLM Deployment")).alias("ResourceType"),
            pl.lit("AI and Machine Learning").alias("ServiceCategory"),
            pl.col("service_name").alias("ServiceName"),
            pl.col("service_provider_name").alias("ServiceProviderName"),
            pl.lit("Generative AI").alias("ServiceSubcategory"),
            sku_id.alias("SkuId"),
            pl.col("meter").alias("SkuMeter"),
            pl.concat_str(sku_id, pl.col("service_tier").fill_null("default"), separator="/").alias("SkuPriceId"),
            pl.col("sub_account_id").alias("SubAccountId"),
            pl.col("sub_account_name").alias("SubAccountName"),
            pl.lit("LiteLLM Account").alias("SubAccountType"),
            _json_column(("request_tags", "team_id", "team_alias"), _tags, "Tags"),
        )
        return selected.cast(self.schema)  # cast-ok: polars dtype conversion


__all__ = ("FOCUS_1_5_SCHEMA", "Focus15Transformer")
