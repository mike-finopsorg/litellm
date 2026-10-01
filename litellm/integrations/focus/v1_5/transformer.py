"""FOCUS 1.5 transformer for SpendLogs buckets."""

from __future__ import annotations

from typing import Final

import polars as pl

# see: https://focus.finops.org/focus-specification/v1-5/
FOCUS_1_5_SCHEMA: Final = pl.Schema(
    (
        ("BilledCost", pl.Decimal(18, 6)),
        ("BillingCurrency", pl.String),
        ("ChargeCategory", pl.String),
        ("ChargePeriodStart", pl.Datetime(time_unit="us", time_zone="UTC")),
        ("ChargePeriodEnd", pl.Datetime(time_unit="us", time_zone="UTC")),
        ("CredentialId", pl.String),
        ("PrincipalId", pl.String),
        ("ResourceId", pl.String),
        ("ServiceProviderName", pl.String),
        ("SubAccountId", pl.String),
    )
)


def _utc(column: pl.Expr) -> pl.Expr:
    return column.str.to_datetime("%Y-%m-%dT%H:%M:%SZ", time_unit="us", time_zone="UTC")


class Focus15Transformer:
    """Maps FocusSpendLogsDatabase buckets onto the FOCUS 1.5 columns LiteLLM populates."""

    schema = FOCUS_1_5_SCHEMA

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame.select(
            pl.col("spend").cast(pl.Decimal(18, 6)).alias("BilledCost"),  # cast-ok: polars dtype conversion
            pl.lit("USD").alias("BillingCurrency"),
            pl.lit("Usage").alias("ChargeCategory"),
            _utc(pl.col("charge_period_start")).alias("ChargePeriodStart"),
            _utc(pl.col("charge_period_end")).alias("ChargePeriodEnd"),
            pl.col("credential_id").alias("CredentialId"),
            pl.col("principal_id").alias("PrincipalId"),
            pl.col("model").alias("ResourceId"),
            pl.col("custom_llm_provider").alias("ServiceProviderName"),
            pl.col("team_id").alias("SubAccountId"),
        )


__all__ = ("FOCUS_1_5_SCHEMA", "Focus15Transformer")
