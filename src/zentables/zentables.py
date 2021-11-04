# -*- coding: utf-8 -*-
"""
Main functionalities for `ZenTables` package.

Provides a wrapper class around a `dict` for global options for the package.
Also provides an Accessor class registered with the `pandas` api to provide
access to package functions.

Examples:
    import zentables as zen
    df.zen.pretty()
"""
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Union, cast

import numpy as np
import pandas as pd
import pandas.core.common as com
from jinja2 import ChoiceLoader, Environment, PackageLoader
from numpy.random import Generator
from pandas.io.formats.style import FilePathOrBuffer, Styler, save_to_buffer
from pandas.io.formats.style_render import CSSStyles


@dataclass
class OptionsWrapper:
    """A wrapper class around a dict to provide global options functionalities."""

    font_size: str = "Arial, Helvetica, sans-serif"
    font_family: str = "11pt"
    show_index_names: bool = False
    show_column_names: bool = False
    show_copy_button: bool = True


_options = OptionsWrapper()


def set_options(**kwargs):
    """Utility function to set package-wide options.

    Args:
        kwargs: pass into the function the option name and value to be set.

    Raises:
        KeyError: if the option passed is not a valid option.

    Examples:
        import zentables as zen
        zen.set_options(option1=value1, option2=value2)
    """
    for opt, val in kwargs.items():
        if hasattr(_options, opt):
            setattr(_options, opt, val)
        else:
            raise KeyError(f"Invalid option: {opt}")


#########################################################
# Constants for creating css-based tables (faster option)
#########################################################


class PrettyStyler(Styler):
    """Custom subclass for pandas.io.format.Styler.

    It uses the two custom templates defined in
    the directory and is used by the pandas accessor class
    to create a custom Styler object
    """

    # Load the Jinja2 templates. Note that the "prettystyle.tpl" extends the
    # original template so we have to use the original styler as well.

    def __init__(
        self,
        data: Union[pd.DataFrame, pd.Series],
        precision: Optional[int] = None,
        table_styles: Optional[CSSStyles] = None,
        uuid: Optional[str] = None,
        caption: Union[tuple, str, None] = None,
        table_attributes: Optional[str] = None,
        cell_ids: bool = True,
        na_rep: Optional[str] = None,
        uuid_len: int = 5,
        decimal: str = ".",
        thousands: Optional[str] = None,
        escape: Optional[str] = None,
        font_family: Optional[str] = None,
        font_size: Union[str, int] = None,
        show_index_names: Optional[bool] = None,
        show_column_names: Optional[bool] = None,
        show_copy_button: Optional[bool] = None,
        row_borders: Optional[List[int]] = None,
    ):
        Styler.__init__(
            self,
            data=data,
            precision=precision,
            table_styles=table_styles,
            uuid=uuid,
            caption=caption,
            table_attributes=table_attributes,
            cell_ids=cell_ids,
            na_rep=na_rep,
            uuid_len=uuid_len,
            decimal=decimal,
            thousands=thousands,
            escape=escape,
        )

        self._table_local_styles = _get_font_style(font_size, font_family)
        self._index_names = (
            show_index_names
            if show_index_names is not None
            else _options.show_index_names
        )
        self._column_names = (
            show_column_names
            if show_column_names is not None
            else _options.show_column_names
        )
        self._copy_button = (
            show_copy_button
            if show_copy_button is not None
            else _options.show_copy_button
        )

        if row_borders is not None:
            for row_number in row_borders:
                if row_number >= len(data):
                    raise ValueError(
                        f"Row number {row_number} is out of range for the data."
                    )

        self.row_borders = row_borders

    env = Environment(
        loader=ChoiceLoader(
            [
                PackageLoader("zentables", "templates"),
                Styler.loader,  # the default templates
            ]
        )
    )

    template_html_table = env.get_template("prettyhtml.tpl")

    def render(
        self,
        sparse_index: Optional[bool] = None,
        sparse_columns: Optional[bool] = None,
        **kwargs,
    ) -> str:
        """
        Overrides the `render` method for the Styler class.
        """

        if sparse_index is None:
            sparse_index = pd.get_option("styler.sparse.index")
        if sparse_columns is None:
            sparse_columns = pd.get_option("styler.sparse.columns")
        return self._render_html(
            sparse_index,
            sparse_columns,
            table_local_styles=self._table_local_styles,
            show_copy_button=self._copy_button,
            **kwargs,
        )

    def show_index_names(self):
        """
        Shows the names of the index
        """
        self._index_names = True
        return self

    def show_column_names(self):
        """
        Shows the names of the columns
        """
        self._column_names = True
        return self

    def hide_copy_button(self):
        """
        Shows a "Copy Table" button below the rendered table.
        """
        self._copy_button = False
        return self

    def _translate(
        self, sparse_index: bool, sparse_cols: bool, blank: str = "&nbsp;"
    ) -> Dict[str, Any]:
        """
        Overrides the pandas method to add options to
        remove row/column names and add styles.

        Some code used directly from
        https://github.com/pandas-dev/pandas/blob/master/pandas/io/formats/style.py
        """

        result = Styler._translate(
            self, sparse_index=sparse_index, sparse_cols=sparse_cols, blank=blank
        )

        ### Wrangle the header

        head = result["head"]

        if (
            self.data.index.names
            and com.any_not_none(*self.data.index.names)
            and not self.hide_index_
            and not self.hide_columns_
            # The previous 4 conditions ensure there is a row with index names
            # If _index_names is false,
            # Then we need to pop the last row of head
            and not self._index_names
        ):
            head.pop()

        for row in head:
            for cell in row:
                if cell.get("type") == "th":
                    _add_style_in_element(cell, "text-align: center")

        # Add borders to the first and last line of the header
        for cell in head[0]:
            _add_style_in_element(cell, "border-top: 1.5pt solid black")

        for cell in head[-1]:
            _add_style_in_element(cell, "border-bottom: 1.5pt solid black")

        ### Wrangle the body

        body = result["body"]

        # Updates body to apply cell-wise style attribute
        # so that the style copies over to Word and Google Docs.

        if sparse_index and len(body) > 1:

            sep_rows = []
            max_th_count = 0
            for row_number, row in enumerate(body):
                th_count = 0

                for cell in row:
                    if cell.get("type") == "th" and cell.get("is_visible"):
                        _add_style_in_element(
                            cell, ["vertical-align: middle", "text-align: left"]
                        )
                        th_count += 1
                    if cell.get("type") == "td":
                        _add_style_in_element(
                            cell, ["vertical-align: middle", "text-align: center"]
                        )

                if th_count >= 2:
                    sep_rows.append(row_number)

                if row_number == 0:
                    max_th_count = th_count

            for row_number in sep_rows:
                for cell in body[row_number]:
                    _add_style_in_element(cell, "border-top: 1pt solid black")

            # Vertically walk through row headers to add a bottom border for the table.
            for i in range(max_th_count):

                for row in body:
                    if row[i].get("is_visible"):
                        last_th_for_level = row[i]

                if last_th_for_level and "styles" in last_th_for_level:
                    _add_style_in_element(
                        last_th_for_level, "border-bottom: 1.5pt solid black"
                    )
            # Add bottom border to all body rows
            for cell in body[-1]:
                if cell.get("type") == "td":
                    _add_style_in_element(cell, "border-bottom: 1.5pt solid black")

        elif len(body) > 1:

            for cell in body[-1]:
                _add_style_in_element(cell, "border-bottom: 1.5pt solid black")

        if self.row_borders is not None:
            for row_number in self.row_borders:
                for cell in body[row_number]:
                    _add_style_in_element(cell, "border-bottom: 1pt solid black")

        # If _column_names is false, remove column names
        if not self._column_names and body:

            max_th_count = 0
            for cell in body[0]:
                if cell.get("type") == "th":
                    max_th_count += 1

            for row in head:
                for col_number, cell in enumerate(row):
                    if col_number < max_th_count:
                        if "value" in cell:
                            cell["value"] = blank
                    else:
                        break

        return result

    def to_html(
        self,
        buf: Optional[FilePathOrBuffer[str]] = None,
        *,
        table_uuid: Optional[str] = None,
        table_attributes: Optional[str] = None,
        encoding: Optional[str] = None,
        doctype_html: bool = False,
        exclude_styles: bool = False,
    ):
        """Overrides Styler class's to_html methods for compatibility.

        Please see the pandas documentation for more defaults.

        Used source code from
        https://github.com/pandas-dev/pandas/blob/master/pandas/io/formats/style.py
        """
        if table_uuid:
            self.set_uuid(table_uuid)

        if table_attributes:
            self.set_table_attributes(table_attributes)

        # Build HTML string..
        html = self.render(
            exclude_styles=exclude_styles,
            encoding=encoding if encoding else "utf-8",
            doctype_html=doctype_html,
            show_copy_button=False,  # this is the only difference
        )

        return save_to_buffer(
            html, buf=buf, encoding=(encoding if buf is not None else None)
        )


@pd.api.extensions.register_dataframe_accessor("zen")
class ZenTablesAccessor:
    """An accessor class registered to the Pandas API

    This accessor class provides all DataFrame operations in the package.

    Attributes:
        _pandas_obj: the pandas DataFrame passed to the class.
    """

    def __init__(self, pandas_obj: Union[pd.Series, pd.DataFrame]):
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
        subtotals_name: Optional[str] = "Subtotal",
        props: Optional[str] = None,
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
                most two levels are supportedcurrently. Passed to the same
                argument in the pandas `pivot_table`function.
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
        subtotals_name: Optional[str] = "Subtotal",
        props: Optional[str] = None,
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
                pivot_props = pivot.div(pivot.iloc[-1, 0], axis=1)
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
        margins_name: Optional[str] = "All",
        submargins: bool = False,
        submargins_name: Optional[str] = "All",
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
        submargins_name: Optional[str] = "All",
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


#################################################
# Helper functions
#################################################


def _get_font_style(
    font_size: Optional[Union[int, str]] = None, font_family: Optional[str] = None
) -> List[str]:

    font_size = font_size or _options.font_size
    if isinstance(font_size, int):
        font_size = f"{font_size}pt"

    font_family = font_family or _options.font_family

    return [f"font-size: {font_size}", rf"font-family: {font_family}"]


def _convert_names(
    names, max_levels: Optional[int] = None, err_msg: Optional[str] = None
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
    suppress_symbol: Optional[str] = None,
) -> pd.DataFrame:
    """
    Helper function that formats and combines n and pct values
    """
    df_n_str = df_n.astype(str)
    df_pct_str = df_pct.applymap(lambda x: f" ({x:.{digits}%})", na_action="ignore")

    if suppress_symbol is not None:
        return df_n_str.add(df_pct_str).fillna(f"{suppress_symbol}")
    return df_n_str.add(df_pct_str)


def _seed_to_rng(seed: Optional[Union[int, Generator]] = None) -> Generator:
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
    seed: Optional[Union[int, Generator]] = None,
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


def _add_style_in_element(
    ele: Dict[str, Any], style: Union[str, Iterable[str]]
) -> None:
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
