from pathlib import Path

import pandas as pd
import pytest

import zentables.options
from zentables.accessor import _convert_names, _swap_column_levels
from zentables.options import OptionsWrapper, _options, set_options
from zentables.pretty_styler import _get_font_style

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
    global_font_size = _options.font_size
    global_font_family = _options.font_family

    assert _get_font_style() == _get_font_style(global_font_size, global_font_family)

    font_size_processed, font_family_processed = _get_font_style()

    assert _get_font_style(font_family="Arial") == [
        font_size_processed,
        "font-family: Arial",
    ]
    assert _get_font_style(font_size=11) == [
        "font-size: 11pt",
        font_family_processed,
    ]

    assert _get_font_style(font_size=11, font_family="Arial") == [
        "font-size: 11pt",
        "font-family: Arial",
    ]

    assert _get_font_style(font_size="11em", font_family="Arial") == [
        "font-size: 11em",
        "font-family: Arial",
    ]


def test__convert_names():
    test_list = ["str1", "str2"]

    assert _convert_names("str") == ["str"]
    assert _convert_names(test_list) == test_list
    assert _convert_names(pd.Series(test_list)) == test_list

    with pytest.raises(ValueError):
        _convert_names(test_list, 1)

    with pytest.raises(ValueError):
        _convert_names(test_list, 1, "Error!")


def test__swap_column_levels(pivot):
    pivot_swapped = _swap_column_levels(pivot)
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
    monkeypatch.setattr(zentables.options, "_options", OptionsWrapper())

    set_options(font_size="11pt")
    assert _options.font_size == "11pt"

    with pytest.raises(KeyError):
        set_options(foobar=12)
