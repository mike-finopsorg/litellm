"""CSV serializer for Focus export."""

from __future__ import annotations

import io
from typing import Final

import polars as pl

from ..settings import DEFAULT_FOCUS_VERSION, FocusVersion
from .base import FocusSerializer


class FocusCsvSerializer(FocusSerializer):
    """Serialize normalized Focus frames to CSV bytes."""

    extension = "csv"

    def __init__(self, *, version: FocusVersion = DEFAULT_FOCUS_VERSION) -> None:
        self._version: Final = version

    def serialize(self, frame: pl.DataFrame) -> bytes:
        """Encode the provided frame as a CSV payload."""
        # Cast Decimal columns to Float64 so CSV output uses standard
        # floating-point notation (e.g. "1.5") instead of fixed-point
        # strings (e.g. "1.500000") that some parsers may reject.
        decimal_cols: Final = [col for col, dtype in zip(frame.columns, frame.dtypes) if isinstance(dtype, pl.Decimal)]
        if decimal_cols:
            frame = frame.with_columns([pl.col(c).cast(pl.Float64) for c in decimal_cols])
        buffer: Final = io.BytesIO()
        match self._version:
            case "1.2":
                frame.write_csv(buffer)
            case "1.5":
                frame.write_csv(buffer, datetime_format="%Y-%m-%dT%H:%M:%SZ", float_scientific=False)
        return buffer.getvalue()
