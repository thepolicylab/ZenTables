from __future__ import annotations

import copy
from typing import Any, Dict, List

import pandas as pd
from jinja2 import ChoiceLoader, Environment, PackageLoader
from pandas._config import get_option
from pandas._typing import FilePath, WriteBuffer
from pandas.io.formats.format import save_to_buffer
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import CSSStyles

from .utils import _add_style_in_element, _get_font_style, _options


class PrettyStyler(Styler):
    """Custom subclass for pandas.io.format.Styler.

    It uses the two custom templates defined in
    the directory and is used by the pandas accessor class
    to create a custom Styler object
    """

    # Load the Jinja2 templates. Note that the "prettystyle.tpl" extends the
    # original template so we have to use the original styler as well.

    env = Environment(
        loader=ChoiceLoader(
            [
                PackageLoader("zentables", "templates"),
                Styler.loader,  # the default templates
            ]
        )
    )

    template_html_table = env.get_template("prettyhtml.tpl")

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
        hide_index_names: bool | None = None,
        hide_column_names: bool | None = None,
        hide_copy_button: bool | None = None,
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
        self.hide_index_names = (
            hide_index_names
            if hide_index_names is not None
            else _options.hide_index_names
        )
        self.hide_column_names = (
            hide_column_names
            if hide_column_names is not None
            else _options.hide_column_names
        )
        self.hide_copy_button = (
            hide_copy_button
            if hide_copy_button is not None
            else _options.hide_copy_button
        )

        if row_borders is not None:
            for row_number in row_borders:
                if row_number >= len(data):
                    raise ValueError(
                        f"Row number {row_number} is out of range for the data."
                    )

        self.row_borders = row_borders

    def render(
        self,
        sparse_index: bool | None = None,
        sparse_columns: bool | None = None,
        max_rows: int | None = None,
        max_cols: int | None = None,
        **kwargs,
    ) -> str:
        """
        Overrides the `render` method for the Styler class.
        """

        if "hide_copy_button" in kwargs:
            self.hide_copy_button = kwargs["hide_copy_button"]
            del kwargs["hide_copy_button"]

        if sparse_index is None:
            sparse_index = pd.get_option("styler.sparse.index")
        if sparse_columns is None:
            sparse_columns = pd.get_option("styler.sparse.columns")
        return self._render_html(
            sparse_index,
            sparse_columns,
            max_rows=max_rows,
            max_cols=max_cols,
            table_local_styles=self._table_local_styles,
            hide_copy_button=self.hide_copy_button,
            **kwargs,
        )

    def hide_button(self):
        """
        Shows a "Copy Table" button below the rendered table.
        """
        self.hide_copy_button = True
        return self

    def _translate(
        self,
        sparse_index: bool,
        sparse_cols: bool,
        max_rows: int | None = None,
        max_cols: int | None = None,
        blank: str = "&nbsp;",
    ) -> Dict[str, Any]:
        """
        Overrides the pandas method to add options to
        remove row/column names and add styles.

        Some code used directly from
        https://github.com/pandas-dev/pandas/blob/master/pandas/io/formats/style.py
        """

        result = super(Styler, self)._translate(
            sparse_index=sparse_index,
            sparse_cols=sparse_cols,
            max_rows=max_rows,
            max_cols=max_cols,
            blank=blank,
        )

        ### Wrangle the header

        head = result["head"]

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

        return result

    def _copy(self, deepcopy: bool = False) -> PrettyStyler:
        """
        Overrides the method with the same name in the Styler
        class, providing copying of more attributes

        Copies a Styler, allowing for deepcopy or shallow copy
        Copying a Styler aims to recreate a new Styler object which contains the same
        data and styles as the original.
        Data dependent attributes [copied and NOT exported]:
          - formatting (._display_funcs)
          - hidden index values or column values (.hidden_rows, .hidden_columns)
          - tooltips
          - cell_context (cell css classes)
          - ctx (cell css styles)
          - caption
        Non-data dependent attributes [copied and exported]:
          - css
          - hidden index state and hidden columns state (.hide_index_, .hide_columns_)
          - table_attributes
          - table_styles
          - applied styles (_todo)
        """
        # GH 40675
        styler = Styler(
            self.data  # populates attributes 'data', 'columns', 'index' as shallow
        )
        shallow = [  # simple string or boolean immutables
            "hide_index_",
            "hide_columns_",
            "hide_column_names",
            "hide_index_names",
            "table_attributes",
            "cell_ids",
            "caption",
            "uuid",
            "uuid_len",
            "template_latex",  # also copy templates if these have been customised
            "template_html_style",
            "template_html_table",
            "template_html",
            # Below are attributes unique to PrettyStyler
            "font_family",
            "font_size",
            "hide_copy_button",
        ]
        deep = [  # nested lists or dicts
            "css",
            "_display_funcs",
            "_display_funcs_index",
            "_display_funcs_columns",
            "hidden_rows",
            "hidden_columns",
            "ctx",
            "ctx_index",
            "ctx_columns",
            "cell_context",
            "_todo",
            "table_styles",
            "tooltips",
            # Below are attributes unique to PrettyStyler
            "row_borders",
        ]

        for attr in shallow:
            setattr(styler, attr, getattr(self, attr))

        for attr in deep:
            val = getattr(self, attr)
            setattr(styler, attr, copy.deepcopy(val) if deepcopy else val)

        return styler

    def to_html(
        self,
        buf: FilePath | WriteBuffer[str] | None = None,
        *,
        table_uuid: str | None = None,
        table_attributes: str | None = None,
        sparse_index: bool | None = None,
        sparse_columns: bool | None = None,
        bold_headers: bool = False,
        caption: str | None = None,
        max_rows: int | None = None,
        max_columns: int | None = None,
        encoding: str | None = None,
        doctype_html: bool = False,
        exclude_styles: bool = False,
        **kwargs,
    ):
        """Overrides Styler class's to_html methods for compatibility.

        Please see the pandas documentation for more defaults.

        Used source code from
        https://github.com/pandas-dev/pandas/blob/master/pandas/io/formats/style.py
        """
        obj = self._copy(deepcopy=True)

        if table_uuid:
            obj.set_uuid(table_uuid)

        if table_attributes:
            obj.set_table_attributes(table_attributes)

        if sparse_index is None:
            sparse_index = get_option("styler.sparse.index")
        if sparse_columns is None:
            sparse_columns = get_option("styler.sparse.columns")

        if bold_headers:
            obj.set_table_styles(
                [{"selector": "th", "props": "font-weight: bold;"}], overwrite=False
            )

        if caption is not None:
            obj.set_caption(caption)

        # Build HTML string..
        html = obj._render_html(
            sparse_index=sparse_index,
            sparse_columns=sparse_columns,
            max_rows=max_rows,
            max_cols=max_columns,
            exclude_styles=exclude_styles,
            encoding=encoding or get_option("styler.render.encoding"),
            doctype_html=doctype_html,
            **kwargs,
        )

        return save_to_buffer(
            html, buf=buf, encoding=(encoding if buf is not None else None)
        )
