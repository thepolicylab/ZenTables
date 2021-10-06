import numpy as np
import pandas as pd
import pytest

from zentables.zentables import _do_suppression

@pytest.fixture(scope="function")
def random() -> np.random.Generator:
    return np.random.default_rng(123456)


def test_negative_numbers():
    """
    Suppression should work on the _absolute value_ of the numbers, not the
    signed value
    """
    # In this case, -5 and 8 will get suppressed because there's only one value
    # in their column suppressed
    input_array = np.array([[1, 4, 5], [-5, 8, 9]])

    expected_array = np.array([[True, True, False], [True, True, False]])

    df = pd.DataFrame(input_array)

    mask = _do_suppression(df, low=1, high=5)
    assert (mask.values == expected_array).all()


def test_multiple_rows():
    """
    If there are several choices, then we should suppress the second lowest value
    """
    # In this case, -5 and 8 will get suppressed because there's only one value
    # in their column suppressed and they are the smallest numbers
    input_array = np.array([[1, 4, 5], [20, 40, 9], [-5, 8, 9]])

    expected_array = np.array(
        [[True, True, False], [False, False, False], [True, True, False]]
    )

    df = pd.DataFrame(input_array)

    mask = _do_suppression(df, low=1, high=5)
    assert (mask.values == expected_array).all()

    # But what if the -5 and 8 are in diferent rows?
    input_array = np.array([[1, 4, 5], [-5, 40, 9], [20, 8, 10]])

    expected_array = np.array(
        [[True, True, False], [True, False, True], [False, True, True]]
    )

    df = pd.DataFrame(input_array)

    mask = _do_suppression(df, low=1, high=5)
    assert (mask.values == expected_array).all()


def test_random_seed_tie_breaker():
    """
    If there are several choices, then suppression should be random (i.e change depending on seed)
    """
    input_array = [
        [3, 3, 3, 3, 3, 3],
        [30, 30, 30, 30, 30, 30],
        [30, 30, 30, 30, 30, 30],
        [30, 30, 30, 30, 30, 30],
    ]
    df = pd.DataFrame(input_array)
    first_seed, second_seed = np.random.randint(1e3, size=2)

    mask_with_first_seed = _do_suppression(df, low=1, high=5, seed=first_seed)
    mask_with_second_seed = _do_suppression(df, low=1, high=5, seed=second_seed)

    # Verify that they are not all the same
    assert (mask_with_first_seed.values != mask_with_second_seed.values).any()


def test_nan_values():
    """
    If there are NaN values in the DataFrame, effective suppression cannot happen.
    The user should be aware of the dimensions of input array, or apply appropriate `fillna()` commands.
    """
    # When the input array has mismatching lengths, pd.DataFrame automatically assumes NaN values
    input_array = [[1, 2, 3], [100, 200]]
    df = pd.DataFrame(input_array)
    with pytest.raises(ValueError):
        _do_suppression(df, fillna=False)

    expected_array = [[True, True, True], [True, True, True]]
    # Otherwise, can still return result
    mask = _do_suppression(df, low=1, high=5)
    assert (mask.values == expected_array).all()


def test_col_and_row_names():
    """
    Verify that the code runs with column names and row names
    """
    input_array = np.array([[1, 4, 5], [20, 40, 9], [-5, 8, 9]])

    expected_array = np.array(
        [[True, True, False], [False, False, False], [True, True, False]]
    )

    df = pd.DataFrame(input_array, columns=["A", "B", "C"], index=["A", "B", "C"])

    mask = _do_suppression(df, low=1, high=5)
    assert (mask.values == expected_array).all()


def test_no_suppression():
    input_array = np.array([[100, 400, 50], [20, 40, 90], [-500, 800, 900]])

    expected_array = np.array(
        [[False, False, False], [False, False, False], [False, False, False]]
    )

    df = pd.DataFrame(input_array)

    mask = _do_suppression(df, low=1, high=5)
    assert (mask.values == expected_array).all()


def test_single_row():
    input_array = np.array([4, 0, 10, 30])
    expected_array = np.array([True, True, True, True])
    df = pd.DataFrame(input_array)

    mask = _do_suppression(df, low=1, high=5)
    assert (mask.values == expected_array).all()

    input_array = np.array([[4], [0], [10], [30]])
    expected_array = np.array([[True], [True], [True], [True]])
    df = pd.DataFrame(input_array)

    mask = _do_suppression(df, low=1, high=5)
    assert (mask.values == expected_array).all()


def test_nan_in_mean_sd_table():
    total_length = 20
    cuisine = ["Chinese", "Korean", "Italian"]
    city = ["Boston", "Providence"]
    who = ["Paul", "Ed", "Kevin"]

    d = {
        "city": np.random.choice(city, size=total_length),
        "cuisine": np.random.choice(cuisine, size=total_length),
        "price": np.random.randint(low=10, high=40, size=total_length),
        "who": np.random.choice(who, size=total_length),
    }
    df = pd.DataFrame(d)

    number_of_empty_groupings = (
        df.zen.freq_table(
            index=["cuisine", "who"],
            columns="city",
            values="price",
            subtotals=False,
            totals=False,
        )
        .isna()
        .sum()
        .sum()
    )

    outcome_df = df.zen.mean_sd_table(
        index=["cuisine", "who"], columns="city", values="price", suppress=False, high=2
    )

    # Ensure that those that have '0' values have 'NA' values in sd
    is_null = outcome_df.xs("n", level=1, axis=1) == 0.0
    is_nan = outcome_df.xs("Mean (SD)", level=1, axis=1) == "N/A"

    pd.testing.assert_frame_equal(left=is_null, right=is_nan, check_dtype=False)

    # verify that the number of empty
    assert is_nan.sum().sum() == is_null.sum().sum() == number_of_empty_groupings

    ## Realizing that further development is necessary to ensure that 0 is not suppressed
    ## and if this should even always be true.
    suppressed_outcome_df = df.zen.mean_sd_table(
        index=["cuisine", "who"], columns="city", values="price", suppress=True, high=2
    )


