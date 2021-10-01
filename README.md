# ZenTables - Stress-Free Descriptive Tables in Python

`ZenTables` transforms your `pandas` DataFrames into beautiful, publishable tables in one line of code, which you can then transfer into Google Docs and other word processors with one click. Supercharge your workflow when you are writing papers and reports.

```python
import zentables as zen

df.zen.pretty()
```

![Formatting tables in one line](https://raw.githubusercontent.com/thepolicylab/ZenTables/main/docs/images/image1.png)

## Features

* Beautiful tables in one line
* Google Docs/Word ready in one click
* Descriptive statistics at varying levels of aggregation
* Control table aesthetics globally
* and many more to come....

## Installation

Via `pip` from PyPI:

```sh
pip install zentables
```

Via `pip` from GitHub directly

```sh
pip install -U git+https://github.com/thepolicylab/ZenTables
```

## How to use `ZenTables`

### 1. How to format any `DataFrame`

First, import the package alongside `pandas`:

```python
import pandas as pd
import zentables as zen
```

Then, to format any `DataFrame`, simply use:

```python
df.zen.pretty()
```

And this is the result:

![zen.pretty() result](https://raw.githubusercontent.com/thepolicylab/ZenTables/main/docs/images/image2.png)

Click on the `Copy Table` button to transfer the table to Google Docs and Word. Formatting will be preserved.

Results in Google Docs (Tested on Chrome, Firefox, and Safari):

![Results in Google Docs](https://raw.githubusercontent.com/thepolicylab/ZenTables/main/docs/images/image3.png)

Results in Microsoft Word:

![Results in Word](https://raw.githubusercontent.com/thepolicylab/ZenTables/main/docs/images/image4.png)

### 2. How to control my tables' appearance?

`ZenTables` provides two ways to control the aesthetics of the tables. You can use global settings to control the font and font size of the tables via:

```python
zen.set_options(font_family="Times New Roman, serif", font_size=12)
```

**Note:** When `font_size` is specified as an `int`, it will be interpreted as points (`pt`). All other CSS units are accepted as a `str`.

Or you can override any global options by specifying `font_family` and `font_size` in `zen.pretty()` method:

```python
df.zen.pretty(font_family="Times New Roman, serif", font_size="12pt")
```

Both will result in a table that looks like this

![Result change fonts](https://raw.githubusercontent.com/thepolicylab/ZenTables/main/docs/images/image5.png)

We are working on adding more customization options in a future release.

### 3. How to create common descriptive tables using `ZenTables`?

#### 3.1. Frequency Tables

Use `df.zen.freq_tables()` to create simple frequency tables:

```python
freq_table = df.zen.freq_table(
    index=["Segment", "Region"],
    columns=["Category"],
    values="Order ID",
    props="index",
    totals=True,
    subtotals=True,
    totals_names="Total"
    subtotals_names="Subtotal",
)
freq_table.zen.pretty() # You can also chain the methods
```

Use `props` to control whether to add percentages of counts. When `props` is not set (the default), no percentages will be added. You can also specify `props` to calculate percentages over `"index"` (rows), `"columns"`, or `"all"` (over the totals of the immediate top level).

Use `totals` and `subtotals` parameters to specify whether totals and subtotals will be added. Note that when `props` is not `None`, both `totals` and `subtotals` will be `True`, and when `subtotals` is set to `True`, this will also override `totals` settings to `True`.

Additionally, you can control the names of the total and subtotal categories using `totals_names` and `subtotals_names` parameters.

![Result freq_table()](https://raw.githubusercontent.com/thepolicylab/ZenTables/main/docs/images/image6.png)

#### 3.2. Mean and standard deviation tables

Use `df.zen.mean_sd_table()` to create descriptives with n, mean, and standard deviation:

```python
mean_sd_table = df.zen.mean_sd_table(
    index=["Segment", "Region"],
    columns=["Category"],
    values="Sales",
    margins=True,
    margins_name="All",
    submargins=True,
    submargins_name="All Regions",
)
mean_sd_table.zen.pretty() # You can also chain the methods
```

Similar to `freq_tables`, you can use `margins` and `submargins` parameters to specify whether aggregations at the top and intermediate levels will be added. Additionally, you can control the names of the total and subtotal categories using `margins_names` and `submargins_names` parameters.

![Result mean_sd_table()](https://raw.githubusercontent.com/thepolicylab/ZenTables/main/docs/images/image7.png)

#### 3.3 Other descriptive statistics tables

For all other types of tables, `ZenTables` provides its own `df.zen.pivot_table()` method:

```python
mean_median_table = df.zen.pivot_table(
    index=["Segment", "Region"],
    columns=["Category"],
    values="Sales",
    aggfunc=["count", "mean", "median"],
    margins=True,
    margins_name="All",
    submargins=True,
    submargins_name="All Regions",
).rename( # rename columns
    columns={
        "count": "n",
        "mean": "Mean",
        "median": "Median",
    }
)
mean_median_table.zen.pretty().format(precision=1) # Specify level of precision
```

There are two differences between this `pivot_table()` method and the `pandas` `pivot_table` method. First, like `mean_sd_table()`, it provides `submargins` and `submargins_names` for creating intermediate-level aggregations. Second, results are grouped by `values`, `columns`, and `aggfuncs`, instead of `aggfuncs`, `values`, and `columns`. This provides more readability than what the `pandas` version provides.

![Result pivot_table()](https://raw.githubusercontent.com/thepolicylab/ZenTables/main/docs/images/image8.png)

### 4. Tips and tricks

1. `df.zen.pretty()` returns a subclass of `pandas` `Styler`, which means you can chain all other methods after `df.style`. `format()` in the previous section is an example. For more formatting options, please see [this page in `pandas` documentation](https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html)

2. All other methods in `ZenTables` returns a regular `DataFrame` that can be modified further.

3. The names of the index and columns are by default hidden. You can get them back by doing this:

```python
df.zen.pretty().show_index_names().show_column_names()
```

4. You can also disable the `Copy Table` button like this:

```python
df.zen.pretty().hide_copy_button()
```

## TODO

- [ ] More tests on compatibility with `Styler` in `pandas`
- [ ] More options for customization
- [ ] A theming system
- [ ] More to come...

## Contributing

Contributions are welcome, and they are greatly appreciated! If you have a new idea for a simple table that we should add, please submit an issue.

## Contributors

Principally written by Paul Xu at [The Policy Lab](https://thepolicylab.brown.edu). Other contributors:
  * Kevin H. Wilson
  * Edward Huh

## Special thanks

* All the members of [The Policy Lab](https://thepolicylab.brown.edu) at Brown University for their feedback
* The [`sidetable` package](https://github.com/chris1610/sidetable) for ideas and inspiration.
