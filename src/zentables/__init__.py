# -*- coding: utf-8 -*-
"""
ZenTables: Stress-Free Descriptive Tables in Python

`ZenTables` bridges `pandas` and the academic world by making it easy to create
beautiful, publication-ready tables in Jupyter Notebooks and Jupyter Lab. Built on top
of the `pandas` styling system and `pivot_table`, it can reformat any pandas `DataFrame`
with one line of code to make it ready for publications.

Typical usage examples::

    import zentables as zen
    df.zen.pretty()
"""

from .zentables import ZenTablesAccessor, set_options

__all__ = ["ZenTablesAccessor", "set_options"]
