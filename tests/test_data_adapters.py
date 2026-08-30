from __future__ import annotations

import pandas as pd
import pytest

from src.fetch_eia import process_region
from src.fetch_entso_e import _parse_generation_xml


def test_eia_fuel_codes_are_not_collapsed_to_other() -> None:
    raw = pd.DataFrame(
        [
            {"period": "2025-01-01T00", "type-name": "COL", "value": "60"},
            {"period": "2025-01-01T00", "type-name": "NG", "value": "40"},
        ]
    )
    result = process_region(raw, "PJM")
    assert result.loc[0, "coal_share"] == pytest.approx(0.6)
    assert result.loc[0, "natural_gas_share"] == pytest.approx(0.4)
    assert result.loc[0, "other_share"] == pytest.approx(0.0)


def test_entsoe_quarter_hour_solar_is_aggregated_to_mwh() -> None:
    points = "".join(
        f"<Point><position>{index}</position><quantity>100</quantity></Point>"
        for index in range(1, 5)
    )
    xml = f"""
    <GL_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0">
      <TimeSeries>
        <MktPSRType><psrType>B17</psrType></MktPSRType>
        <Period>
          <timeInterval><start>2025-01-01T00:00Z</start></timeInterval>
          <resolution>PT15M</resolution>
          {points}
        </Period>
      </TimeSeries>
    </GL_MarketDocument>
    """
    rows = _parse_generation_xml(xml, "TEST")
    assert len(rows) == 1
    assert rows[0]["fuel_type"] == "solar"
    assert rows[0]["generation_mwh"] == pytest.approx(100.0)
