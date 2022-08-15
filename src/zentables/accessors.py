from __future__ import annotations

import warnings

import pandas as pd

from .style import PrettyStyler
from .utils import (
    _combine_mean_std,
    _combine_n_pct,
    _convert_names,
    _do_suppression,
    _swap_column_levels,
)


@pd.api.extensions.register_dataframe_accessor("zen")
class ZenTablesDataFrameAccessor:
    """An accessor class registered to the Pandas API

    This accessor class provides all DataFrame operations in the package.

    Attributes:
        _pandas_obj: the pandas DataFrame passed to the class.
    """

    def __init__(self, pandas_obj: pd.DataFrame):
        """Constructor for the accessor class.

        Args:
            pandas_obj: the pandas object that this class wraps around.
        """
        self._pandas_obj = pandas_obj

    def pretty(self, **kwargs) -> PrettyStyler:
        """Formats a DataFrame for it to look pretty.

        When `fast` is `True`, returns the native pandas `Styler` in combination with
        CSS styles, which works with all DataFrames. When `False`, uses the
        `PrettyStyler` class with custom Jinja templates.

        It is not as fast with large DataFrames, because it styles each `th` and `td`
        elements of the table, but the formatting can be preserved when the table is
        copied into Google Docs or Word.

        Args:
            font_size: Sets the font size of the entire table. Can be an `int`, which
            is the font size in points, or a `str` which is passed to the `font-size`
            property of the table. When not specified, uses global font_size setting.
            font_family: Sets the font-family property of the table. When not specified,
                uses global font_family setting.
            fast: toggles `fast` and `slow` mode. Slow mode preserves most formatting
                when copied to Google Docs or Word.

        Returns:
            The styler object
        """

        return PrettyStyler(self._pandas_obj, **kwargs)

    def freq_table(
        self,
        index,
        columns,
        values: str,
        totals: bool = True,
        totals_name: str = "Total",
        subtotals: bool = False,
        subtotals_name: str | None = "Subtotal",
        props: str | None = None,
        digits: int = 1,
        suppress: bool = False,
        suppress_symbol: str = "*",
        low: int = 1,
        high: int = 10,
        **kwargs,
    ) -> pd.DataFrame:
        """Produces a frequency table.

        Args:
            index (str or list-like): The categories on the index. Only at
                most two levels are supported currently. Passed to the same
                argument in the pandas `pivot_table` function.
            columns (str or list-like): The categories on the columns. Only
                one level is supported currently. Passed to the same argument
                in the pandas `pivot_table` function.
            values (str or list-like): The column to be summarized. Only one
                level is supported currently.
            totals: Whether to add a totals row and column. Defaults to True.
            totals_name: The labels of the totals row and column. Defaults to "Total".
            props: Controls whether and how percentages of counts are added. When None,
                no proportions will be added. When an str is passed, it determines
                whether proportions will be calculated along the rows ('index'), columns
                ('columns'), or by the (sub)totals of the category ('all').
            digits: Controls the number of digits for the percentages.
            subtotals: Controls whether subtotals will be calculated.
            subtotals_name: The labels of the subtotal rows.
            suppress: Controls whether to apply cell suppression.
            ceiling: Maximum value for suppression. Only used if suppress=True

        Raises:
            ValueError: When any of index, columns, or values is None.
            ValueError: When more than two levels are used on the index.
            ValueError: When more than one level is used on the columns.
            ValueError: When passing an invalid parameter to `props`.

        Returns:
            A frequency table
        """

        if index is None or columns is None or values is None:
            raise ValueError("`index`, `columns` and `values` cannot be None")

        index = _convert_names(
            index, 2, "Currently at most two levels are supported for index."
        )
        columns = _convert_names(
            columns, 1, "Currently only 1 level is supported for columns."
        )

        if props not in [None, "index", "columns", "all"]:
            raise ValueError(
                "`props` arguments only accepts the following values: "
                "[None, 'index', 'columns', 'all']"
            )

        if isinstance(values, str):
            return self._internal_freq_table(
                index=index,
                columns=columns,
                values=values,
                totals=totals,
                totals_name=totals_name,
                subtotals=subtotals,
                subtotals_name=subtotals_name,
                props=props,
                digits=digits,
                suppress=suppress,
                suppress_symbol=suppress_symbol,
                low=low,
                high=high,
                **kwargs,
            )

        values = _convert_names(values)

        freq_tables = [
            self._internal_freq_table(
                index=index,
                columns=columns,
                values=[value],  # convert to list to preserve header
                totals=totals,
                totals_name=totals_name,
                subtotals=subtotals,
                subtotals_name=subtotals_name,
                props=props,
                digits=digits,
                suppress=suppress,
                suppress_symbol=suppress_symbol,
                low=low,
                high=high,
                **kwargs,
            )
            for value in values
        ]

        return pd.concat(freq_tables, axis=1)

    def _internal_freq_table(
        self,
        index,
        columns,
        values: str,
        totals: bool = False,
        totals_name: str = "Total",
        subtotals: bool = False,
        subtotals_name: str | None = "Subtotal",
        props: str | None = None,
        digits: int = 1,
        suppress: bool = False,
        suppress_symbol: str = "*",
        low: int = 1,
        high: int = 10,
        **kwargs,
    ) -> pd.DataFrame:

        if props is None:

            pivot = self._internal_pivot_table(
                index=index,
                columns=columns,
                values=values,
                aggfunc="count",
                margins=totals,
                margins_name=totals_name,
                submargins=subtotals,
                submargins_name=subtotals_name,
                **kwargs,
            ).astype("Int64")

            if suppress:
                mask = _do_suppression(pivot, low, high)
                return pivot.where(~mask).astype(str).replace("<NA>", suppress_symbol)

            return pivot

        pivot = self._internal_pivot_table(
            index=index,
            columns=columns,
            values=values,
            aggfunc="count",
            margins=True,
            margins_name=totals_name,
            submargins=True,
            submargins_name=subtotals_name,
            **kwargs,
        ).astype("Int64")

        index_levels = len(index)

        # If there is only one level on the index, calculate percentages and call
        # it a day.
        if index_levels == 1:

            if props == "index":
                pivot_props = pivot.div(pivot.iloc[:, -1], axis=0)
            elif props == "columns":
                pivot_props = pivot.div(pivot.iloc[-1, :], axis=1)
            elif props == "all":
                pivot_props = pivot.div(pivot.iloc[-1, -1].squeeze())

            if not totals:

                if suppress:
                    mask = _do_suppression(pivot, low, high)
                    return (
                        _combine_n_pct(
                            pivot.where(~mask),
                            pivot_props.where(~mask),
                            digits,
                            suppress_symbol,
                        )
                        .iloc[:-1, :-1]
                        .copy()
                    )

                return _combine_n_pct(pivot, pivot_props, digits).iloc[:-1, :-1].copy()

            if suppress:
                mask = _do_suppression(pivot, low, high)
                return _combine_n_pct(
                    pivot.where(~mask),
                    pivot_props.where(~mask),
                    digits,
                    suppress_symbol,
                )
            return _combine_n_pct(pivot, pivot_props, digits)

        # If there is more than one level on the index,
        # calculate percentages for each label on level0
        level0_values = pivot.index.get_level_values(0).unique().to_list()

        sub_freqs = []
        sub_props = []

        for level in level0_values if totals else level0_values[:-1]:
            sub_freq = pivot.xs(level, drop_level=False)

            if props == "index":
                sub_prop = sub_freq.div(sub_freq.iloc[:, -1], axis=0)
            elif props == "columns":
                sub_prop = sub_freq.div(sub_freq.iloc[-1, :], axis=1)
            else:
                sub_prop = sub_freq.div(sub_freq.iloc[-1, -1].squeeze())

            if not subtotals:
                sub_freq = sub_freq.drop(index=[subtotals_name], level=1)
                sub_prop = sub_prop.drop(index=[subtotals_name], level=1)

            sub_freqs.append(sub_freq)
            sub_props.append(sub_prop)

        if totals:
            freqs_frame = pd.concat(sub_freqs)
            props_frame = pd.concat(sub_props)
        else:
            freqs_frame = pd.concat(sub_freqs).iloc[:, :-1].copy()
            props_frame = pd.concat(sub_props).iloc[:, :-1].copy()

        if suppress:
            mask = _do_suppression(freqs_frame, low, high)
            return _combine_n_pct(
                freqs_frame.where(~mask),
                props_frame.where(~mask),
                digits,
                suppress_symbol,
            )
        return _combine_n_pct(freqs_frame, props_frame, digits)

    def _internal_pivot_table(
        self,
        index=None,
        columns=None,
        values=None,
        aggfunc=None,
        margins: bool = False,
        margins_name: str | None = "All",
        submargins: bool = False,
        submargins_name: str | None = "All",
        **kwargs,
    ) -> pd.DataFrame:
        """
        Helper function that calls the pandas pivot_table and
        adds submargins if `submargins` is set to True.
        """
        pivot = self._pandas_obj.pivot_table(
            index=index,
            columns=columns,
            values=values,
            aggfunc=aggfunc,
            # When submargins is True, ignore margins settings
            margins=True if submargins else margins,
            margins_name=margins_name,
            **kwargs,
        )

        index = _convert_names(index)

        if not submargins or len(index) <= 1:
            return pivot

        if submargins and not margins:
            warnings.warn("`submargins` is set to True. Overriding `margins` settings")

        submargins_frames = [pivot]
        nlevels_index = len(index)

        for level in range(1, nlevels_index):

            submargin = self._pandas_obj.pivot_table(
                index=index[:level],
                columns=columns,
                values=values,
                aggfunc=aggfunc,
                margins=margins,
                margins_name=margins_name,
                **kwargs,
            ).reset_index()

            if margins:
                submargin = submargin.iloc[:-1, :]

            for i, col in enumerate(index[level:]):
                if i == 0:
                    submargin[col] = submargins_name
                else:
                    submargin[col] = ""

            submargin = submargin.set_index(index)
            submargins_frames.append(submargin)

        pt_with_submargins = pd.concat(submargins_frames)

        for level in index:
            original_order = pivot.index.get_level_values(level).unique().to_list()
            new_order = pd.Series(original_order + [submargins_name, ""]).unique()
            pt_with_submargins = pt_with_submargins.reindex(
                new_order, level=level, axis=0, copy=False
            )

        return pt_with_submargins

    def pivot_table(
        self,
        index=None,
        columns=None,
        values=None,
        aggfunc=None,
        margins: bool = False,
        margins_name: str = "All",
        submargins: bool = False,
        submargins_name="All",
        **kwargs,
    ) -> pd.DataFrame:
        """Extends the pandas pivot_table function.

        Provides two additional functionalities:
        1. you can use `submargins` to control whether a summary will be
           generated at each level of `index`.
        2. the pivot table generated will sort the columns by ["values",
           "columns`, "aggfunc"], whereas the pandas default is ["aggfunc",
           "columns, values"]

        Args:
            submargins: Controls whether a summary will
                be produced at each level. Defaults to False.
            submargins_name: Controls the name of the summary generated at
                each level. When submargins is true,

        Returns:
            pd.DataFrame: the pivot table.
        """
        pivot = self._internal_pivot_table(
            index=index,
            columns=columns,
            values=values,
            aggfunc=aggfunc,
            margins=margins,
            margins_name=margins_name,
            submargins=submargins,
            submargins_name=submargins_name,
            **kwargs,
        )

        return _swap_column_levels(pivot)

    def mean_sd_table(
        self,
        index,
        columns,
        values,
        digits: int = 1,
        margins: bool = True,
        margins_name: str = "All",
        submargins: bool = False,
        submargins_name: str | None = "All",
        na_rep: str = "N/A",
        suppress: bool = False,
        low: int = 1,
        high: int = 10,
        **kwargs,
    ) -> pd.DataFrame:
        """Produces a table with n, mean, and standard deviation.

        Args:
            index: The categories on the index. Passed to the same argument in the
                pandas `pivot_table`function.
            columns: The categories on the columns. Passed to the same argument in the
                pandas `pivot_table` function.
            values: The columns to be summarized. Defaults to None, which means all
                columns not included in `index` and `column` (can be slow).
            digits: Controls the number of digits for the means and standard deviations.
            margins: Whether to add a row and column for all data in the table.
            margins_name: The labels of the margin rows and columns.
            submargins: Controls whether aggregations at intermediate levels are
                calculated.
            submargins_name: The labels of the aggregations at intermediate levels.
            na_rep: The labels for missing data.

        Raises:
            ValueError: When more than three layers are used on the index.

        Returns:
            pd.DataFrame.
        """

        pivot = self._internal_pivot_table(
            index=index,
            columns=columns,
            values=values,
            aggfunc=["count", "mean", "std"],
            margins=margins,
            margins_name=margins_name,
            submargins=submargins,
            submargins_name=submargins_name,
            **kwargs,
        )

        count = pivot.xs("count", axis=1).astype("Int64").fillna(0)
        mean = pivot.xs("mean", axis=1)
        std = pivot.xs("std", axis=1)
        # issues can arise when the shape is inconsistent because some groupings are entirely empty
        assert (
            count.shape == mean.shape == std.shape
        ), "Ensure that all categories have at least some entry"

        mean_std = _combine_mean_std(mean, std, digits=digits, na_rep=na_rep)
        if suppress:
            mask = _do_suppression(count, low=low, high=high)
            result = pd.concat(
                {
                    "n": count.astype(int).where(~mask).fillna("*"),
                    "Mean (SD)": mean_std.where(~mask).fillna("*"),
                },
                axis=1,
            )
        else:
            result = pd.concat({"n": count, "Mean (SD)": mean_std}, axis=1)

        return _swap_column_levels(result)


@pd.api.extensions.register_series_accessor("zen")
class ZenTablesSeriesAccessor:
    """An accessor class registered to the Pandas API

    This accessor class provides all DataFrame operations in the package.

    Attributes:
        _pandas_obj: the pandas Series passed to the class.
    """

    def __init__(self, pandas_obj: pd.Series):
        """Constructor for the accessor class.

        Args:
            pandas_obj: the pandas object that this class wraps around.
        """
        self._pandas_obj = pandas_obj

    def value_counts(self, dropna: bool = True, digits: int = 1) -> pd.Series:
        """
        Replacement for the pandas Series.value_counts() method to include percentages along
        with the counts.

        Args:
            dropna (bool, optional): whether to count NAs. Defaults to False.
            digits (int, optional): how many digits after 0 to include in percentages. Defaults to 1.

        Returns:
            A pandas Series with values as the index and counts as the values.
        """

        val_counts = self._pandas_obj.value_counts(dropna=dropna, normalize=False)
        val_counts_pct = self._pandas_obj.value_counts(dropna=dropna, normalize=True)

        val_counts_combined = val_counts.astype(str) + val_counts_pct.apply(
            lambda pct: f" ({pct:.{digits:n}%})"
        )

        return val_counts_combined
