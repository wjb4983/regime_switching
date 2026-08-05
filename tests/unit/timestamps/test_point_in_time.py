import pandas as pd

from regime.validation import point_in_time_snapshot


def test_publication_timestamp_is_respected() -> None:
    releases = pd.DataFrame(
        {
            "indicator": ["cpi", "payrolls"],
            "publication_ts": ["2024-01-05 08:30", "2024-01-05 10:00"],
            "value": [3.1, 200.0],
        }
    )

    visible = point_in_time_snapshot(releases, "2024-01-05 09:00", key_columns=["indicator"])

    assert visible["indicator"].tolist() == ["cpi"]


def test_vendor_received_timestamp_is_respected_when_available() -> None:
    releases = pd.DataFrame(
        {
            "indicator": ["cpi", "payrolls"],
            "publication_ts": ["2024-01-05 08:30"] * 2,
            "vendor_received_ts": [None, "2024-01-05 09:05"],
        }
    )

    visible = point_in_time_snapshot(releases, "2024-01-05 09:00", key_columns=["indicator"])

    assert visible["indicator"].tolist() == ["cpi"]


def test_macro_revision_does_not_replace_vintage_before_its_publication() -> None:
    releases = pd.DataFrame(
        {
            "indicator": ["gdp", "gdp"],
            "period": ["2023-Q4", "2023-Q4"],
            "publication_ts": ["2024-01-25", "2024-02-28"],
            "revision_id": ["advance", "second"],
            "actual_value": [3.3, 3.2],
        }
    )

    january = point_in_time_snapshot(releases, "2024-01-31", key_columns=["indicator", "period"])
    march = point_in_time_snapshot(releases, "2024-03-01", key_columns=["indicator", "period"])

    assert january.loc[0, "revision_id"] == "advance"
    assert march.loc[0, "revision_id"] == "second"
