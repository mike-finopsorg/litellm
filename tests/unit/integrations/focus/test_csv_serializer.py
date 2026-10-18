"""Tests for FocusCsvSerializer."""

from __future__ import annotations

import io
from datetime import datetime, timezone
from decimal import Decimal

import polars as pl

from litellm.integrations.focus.serializers.csv import FocusCsvSerializer


def test_should_serialize_dataframe_to_csv():
    frame = pl.DataFrame({"BilledCost": [1.5, 2.0], "ServiceName": ["openai", "anthropic"]})
    serializer = FocusCsvSerializer()
    result = serializer.serialize(frame)

    assert isinstance(result, bytes)
    lines = result.decode("utf-8").strip().split("\n")
    assert lines[0] == "BilledCost,ServiceName"
    assert len(lines) == 3  # header + 2 data rows


def test_should_return_header_only_for_empty_frame():
    frame = pl.DataFrame(schema={"BilledCost": pl.Float64, "ServiceName": pl.Utf8})
    serializer = FocusCsvSerializer()
    result = serializer.serialize(frame)

    lines = result.decode("utf-8").strip().split("\n")
    assert lines[0] == "BilledCost,ServiceName"
    assert len(lines) == 1  # header only


def test_should_cast_decimal_columns_to_float():
    frame = pl.DataFrame({"BilledCost": [1, 2], "ServiceName": ["openai", "anthropic"]}).cast(
        {"BilledCost": pl.Decimal(18, 6)}
    )
    serializer = FocusCsvSerializer()
    result = serializer.serialize(frame)

    lines = result.decode("utf-8").strip().split("\n")
    # Should use floating-point notation (e.g. "1.0") not fixed-point ("1.000000")
    assert "1.000000" not in lines[1]
    assert lines[1].startswith("1.0,")


def test_extension_should_be_csv():
    assert FocusCsvSerializer.extension == "csv"


def _tiny_cost_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {"BilledCost": [Decimal("0.0000001234")]},
        schema={"BilledCost": pl.Decimal(38, 10)},
    )


def test_1_5_floats_never_use_scientific_notation():
    result = FocusCsvSerializer(version="1.5").serialize(_tiny_cost_frame())

    assert result.decode().splitlines() == ["BilledCost", "0.0000001234"]


def _timestamp_and_float_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ChargePeriodStart": [datetime(2026, 1, 1, 5, tzinfo=timezone.utc)],
            "BilledCost": [0.0000001234],
        }
    )


def test_1_2_output_matches_plain_polars_csv():
    frame = _timestamp_and_float_frame()
    buffer = io.BytesIO()
    frame.write_csv(buffer)

    assert FocusCsvSerializer(version="1.2").serialize(frame) == buffer.getvalue()


def test_1_5_writes_utc_second_precision_timestamps():
    result = FocusCsvSerializer(version="1.5").serialize(_timestamp_and_float_frame())

    assert result.decode().splitlines()[1].startswith("2026-01-01T05:00:00Z,")
