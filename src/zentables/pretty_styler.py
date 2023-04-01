"""
The PrettyStyler class is inherited from the Styler class in pandas that extends
the functionalities of the Styler class for more styling options
"""


from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable, List

import pandas as pd
import pandas.core.common as com
from jinja2 import ChoiceLoader, Environment, PackageLoader
from pandas.io.formats.style import Styler, save_to_buffer

from .options import _options

if TYPE_CHECKING:
    from pandas._typing import FilePath, WriteBuffer
    from pandas.io.formats.style_render import CSSStyles


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
        data: pd.DataFrame | pd.Series,
        precision: int | None = None,
        table_styles: CSSStyles | None = None,
        uuid: str | None = None,
        caption: tuple | str | None = None,
        table_attributes: str | None = None,
        cell_ids: bool = True,
        na_rep: str | None = None,
        uuid_len: int = 5,
        decimal: str = ".",
        thousands: str | None = None,
        escape: str | None = None,
        font_family: str | None = None,
        font_size: str | int | None = None,
        show_index_names: bool | None = None,
        show_column_names: bool | None = None,
        show_copy_button: bool | None = None,
        row_borders: List[int] | None = None,
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
        sparse_index: bool | None = None,
        sparse_columns: bool | None = None,
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
        self, sparse_index: bool, sparse_cols: bool, blank: str = "&nbsp;", **kwargs
    ) -> Dict[str, Any]:
        """
        Overrides the pandas method to add options to
        remove row/column names and add styles.

        Some code used directly from
        https://github.com/pandas-dev/pandas/blob/master/pandas/io/formats/style.py
        """

        result = Styler._translate(
            self,
            sparse_index=sparse_index,
            sparse_cols=sparse_cols,
            blank=blank,
            **kwargs,
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
        buf: FilePath | WriteBuffer[str],
        *,
        table_uuid: str | None = None,
        table_attributes: str | None = None,
        encoding: str | None = None,
        doctype_html: bool = False,
        exclude_styles: bool = False,
    ) -> None:
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


#################################################
# Helper functions
#################################################


def _get_font_style(
    font_size: int | str | None = None, font_family: str | None = None
) -> List[str]:
    font_size = font_size or _options.font_size
    if isinstance(font_size, int):
        font_size = f"{font_size}pt"

    font_family = font_family or _options.font_family

    return [f"font-size: {font_size}", rf"font-family: {font_family}"]


def _add_style_in_element(ele: Dict[str, Any], style: str | Iterable[str]):
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
