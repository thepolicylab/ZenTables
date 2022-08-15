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
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OptionsWrapper:
    """A wrapper class around a dict to provide global options functionalities."""

    font_size: str = "Arial, Helvetica, sans-serif"
    font_family: str = "11pt"
    hide_index_names: bool = True
    hide_column_names: bool = True
    hide_copy_button: bool = False


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
