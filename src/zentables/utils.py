#################################################
# Helper functions
#################################################

from __future__ import annotations

from typing import Any, Dict, Iterable, List, cast

import numpy as np
import pandas as pd
from numpy.random import Generator

from .zentables import _options


def _get_font_style(
    font_size: int | str | None = None, font_family: str | None = None
) -> List[str]:

    font_size = font_size or _options.font_size
    if isinstance(font_size, int):
        font_size = f"{font_size}pt"

    font_family = font_family or _options.font_family

    return [f"font-size: {font_size}", rf"font-family: {font_family}"]


def _convert_names(
    names, max_levels: int | None = None, err_msg: str | None = None
) -> List[str]:
    """Helper function that converts arguments of index, columns, values to list.

    Also performs check on number of levels. If it exceeds `max_levels`,
    raise ValueError with `err_msg`.
    """
    result = None
    if isinstance(names, str):
        result = [names]
    elif isinstance(names, list):
        result = names
    else:
        result = list(names)

    if max_levels and len(result) > max_levels:
        raise ValueError(err_msg)

    return result


def _sort_all_levels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function that sorts each level of the DataFrame to while maintaining
    the original order.
    """
    nlevels = df.columns.nlevels

    for level in range(nlevels):
        original_order = df.columns.get_level_values(level).unique()
        df = df.reindex(original_order, axis=1, level=level, copy=False)

    return df


def _swap_column_levels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function that moves the names of the calculation to the last level
    of columns from the first.
    """
    nlevels = df.columns.nlevels
    df = df.reorder_levels(list(range(1, nlevels)) + [0], axis=1)
    df = _sort_all_levels(df)
    return df


def _combine_n_pct(
    df_n: pd.DataFrame,
    df_pct: pd.DataFrame,
    digits: int = 1,
    suppress_symbol: str | None = None,
) -> pd.DataFrame:
    """
    Helper function that formats and combines n and pct values
    """
    df_n_str = df_n.astype(str)
    df_pct_str = df_pct.applymap(lambda x: f" ({x:.{digits}%})", na_action="ignore")

    if suppress_symbol is not None:
        return df_n_str.add(df_pct_str).fillna(f"{suppress_symbol}")
    return df_n_str.add(df_pct_str)


def _seed_to_rng(seed: int | Generator | None = None) -> Generator:
    if seed is None:
        # When there is no seed set, we start a RNG based on system entropy
        return np.random.default_rng()

    if isinstance(seed, (int, np.integer)):
        # When a seed is specified, begin RNG based on seed
        return np.random.default_rng(seed=seed)

    else:
        # When the input is already a generator, simply return the generator
        # N.B. mypy gets confused at this point because `np.integer` is the type
        #      asserted above. So we simply cast to the Generator type (which is
        #      effectively a noop).
        return cast(Generator, seed)


def _local_suppression(
    mini_df_n: pd.DataFrame,
    low: int = 1,
    high: int = 10,
    seed: int | Generator | None = None,
):
    """
    Helper function that applies cell suppression to mini_df_n.
    mini_df_n is assumed to be a smaller portion of a larger
    """

    df = mini_df_n.copy()
    colnames = mini_df_n.columns
    rownames = mini_df_n.index
    df.columns = np.arange(len(colnames))
    df.index = np.arange(len(rownames))

    rng = _seed_to_rng(seed)
    mask = np.logical_and(df >= low, df < high)
    not_sparse = df.abs().max().max() * 2

    while True:
        roi = np.where(mask.sum(axis=1) == 1)[0]

        if len(roi) > 0:
            tie_breaker_df = (
                df
                + rng.uniform(0, 1e-2, size=df.shape)  # break ties with random number
                + (mask * not_sparse).astype(
                    "float"
                )  # ensure numbers that are not sparse are made large
            )
            min_cols = tie_breaker_df.idxmin(axis=1)
            # Suppress the identified minimums per row.
            mask.values[roi, min_cols.values[roi]] = True

        coi = np.where(mask.sum(axis=0) == 1)[0]
        if len(coi) > 0:
            tie_breaker_df = (
                df
                + rng.uniform(0, 1e-2, size=df.shape)
                + (mask * not_sparse).astype("float")
            )
            min_rows = tie_breaker_df.idxmin(axis=0)
            mask.values[min_rows.values[coi], coi] = True

        if (len(coi) != 0 or len(roi) != 0) and 1 in mask.shape:
            # Corner Case: if there is one row or column, and one is masked, the rest needs to be masked
            return pd.DataFrame(
                np.ones(mask.shape, dtype=bool), columns=colnames, index=rownames
            )
        if len(coi) == 0 and len(roi) == 0:
            break

    # reassign column names and row names
    mask.index = rownames
    mask.columns = colnames
    assert (mask.columns == mini_df_n.columns).all()
    assert (mask.index == mini_df_n.index).all()
    return mask


def _do_suppression(
    df_n: pd.DataFrame,
    low: int = 1,
    high: int = 10,
    seed: int = 2021,
    fillna: bool = True,
) -> pd.DataFrame:
    """
    Helper function that applies cell suppression to input dataframe.

    Raises:
        ValueError: when the input DataFrame includes NaN values

    Returns:
        pd.DataFrame where entries are True if df_n needs to be suppress and False if not.
    """
    # See if there are any NaNs, and raise error
    if fillna:
        df_n.fillna(0, inplace=True)
    if df_n.isnull().values.any():
        raise ValueError("DataFrame contains NaN values that cannot be suppressed")

    df_n = df_n.abs()

    # see if there are sub-margins that we need to suppress
    if df_n.index.nlevels > 1:
        mask_list = []
        unique_index = df_n.index.get_level_values(0).unique()
        for idx in unique_index:
            mask_list.append(
                _local_suppression(
                    df_n.iloc[df_n.index.get_level_values(0) == idx],
                    low=low,
                    high=high,
                    seed=seed,
                )
            )
        mask = pd.concat(mask_list)
    else:
        mask = _local_suppression(df_n, low=low, high=high, seed=seed)
    return mask.astype(bool)


def _combine_mean_std(
    df_mean: pd.DataFrame, df_std: pd.DataFrame, digits: int = 1, na_rep: str = "N/A"
) -> pd.DataFrame:
    """
    Helper function that formats and combines mean and standard deviation values
    """
    df_mean_str = df_mean.applymap(lambda x: f"{x:.{digits}f}", na_action="ignore")
    df_std_str = df_std.applymap(lambda x: f" ({x:.{digits}f})", na_action="ignore")
    result = df_mean_str.add(df_std_str)
    is_sample_size_1 = ~df_mean_str.isna() & df_std_str.isna()
    is_missing = df_mean_str.isna() & df_std_str.isna()
    result[is_sample_size_1] = df_mean_str[is_sample_size_1].add(f" ({na_rep})")
    result[is_missing] = na_rep
    return result


def _add_style_in_element(ele: Dict[str, Any], style: str | Iterable[str]) -> None:
    """
    Helper function that sets the `style` field in a dict to `style` if it doesn't
    already exist or updates the style field. Maintain the value as a list.
    """

    if "styles" in ele:
        if isinstance(style, str):
            ele["styles"].append(style)
        elif isinstance(style, list):
            ele["styles"] += style
        else:
            ele["styles"] += list(style)

    else:
        if isinstance(style, str):
            ele["styles"] = [style]
        elif isinstance(style, list):
            ele["styles"] = style
        else:
            ele["styles"] = list(style)
