from pathlib import Path

import pandas as pd
import pytest

from zentables import zentables

file_path = Path(__file__).parent / "fixtures" / "superstore.csv"
df = pd.read_csv(file_path)


@pytest.fixture()
def pivot():
    return df.pivot_table(
        index=["Segment", "Region"],
        columns=["Category", "Sub-Category"],
        values="Sales",
        aggfunc=["count", "mean", "std"],
        margins=True,
    )


####################################
# Test helper functions
####################################


def test__get_font_style():

    global_font_size = zentables._options.font_size
    global_font_family = zentables._options.font_family

    assert zentables._get_font_style() == zentables._get_font_style(
        global_font_size, global_font_family
    )

    font_size_processed, font_family_processed = zentables._get_font_style()

    assert zentables._get_font_style(font_family="Arial") == [
        font_size_processed,
        "font-family: Arial",
    ]
    assert zentables._get_font_style(font_size=11) == [
        "font-size: 11pt",
        font_family_processed,
    ]

    assert zentables._get_font_style(font_size=11, font_family="Arial") == [
        "font-size: 11pt",
        "font-family: Arial",
    ]

    assert zentables._get_font_style(font_size="11em", font_family="Arial") == [
        "font-size: 11em",
        "font-family: Arial",
    ]


def test__convert_names():
    l = ["str1", "str2"]

    assert zentables._convert_names("str") == ["str"]
    assert zentables._convert_names(l) == l
    assert zentables._convert_names(pd.Series(l)) == l

    with pytest.raises(ValueError):
        zentables._convert_names(l, 1)

    with pytest.raises(ValueError):
        zentables._convert_names(l, 1, "Error!")


def test__swap_column_levels(pivot):

    pivot_swapped = zentables._swap_column_levels(pivot)
    col_levels = pivot_swapped.columns.nlevels

    assert col_levels == pivot.columns.nlevels

    last_level_values = pivot_swapped.columns.get_level_values(
        col_levels - 1
    ).to_series()

    for i in range(0, len(last_level_values), 3):
        assert list(last_level_values.iloc[i : i + 3]) == ["count", "mean", "std"]


####################################
# Test global functions
####################################


def test_global_set_options(monkeypatch):
    # Monkey patch in a new OptionsWrapper object just for this test
    monkeypatch.setattr(zentables, "_options", zentables.OptionsWrapper())

    zentables.set_options(font_size="11pt")
    assert zentables._options.font_size == "11pt"

    with pytest.raises(KeyError):
        zentables.set_options(foobar=12)
