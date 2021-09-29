import numpy as np
import pandas as pd
import pytest

import zentables as zen


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

    mask = zen._do_suppression(df, low=1, high=5)
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

    mask = zen._do_suppression(df, low=1, high=5)
    assert (mask.values == expected_array).all()

    # But what if the -5 and 8 are in diferent rows?
    input_array = np.array([[1, 4, 5], [-5, 40, 9], [20, 8, 10]])

    expected_array = np.array(
        [[True, True, False], [True, False, True], [False, True, True]]
    )

    df = pd.DataFrame(input_array)

    mask = zen._do_suppression(df, low=1, high=5)
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

    mask_with_first_seed = zen._do_suppression(df, low=1, high=5, seed=first_seed)
    mask_with_second_seed = zen._do_suppression(df, low=1, high=5, seed=second_seed)

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
        zen._do_suppression(df, fillna=False)

    expected_array = [[True, True, True], [True, True, True]]
    # Otherwise, can still return result
    mask = zen._do_suppression(df, low=1, high=5)
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

    mask = zen._do_suppression(df, low=1, high=5)
    assert (mask.values == expected_array).all()


def test_no_suppression():
    input_array = np.array([[100, 400, 50], [20, 40, 90], [-500, 800, 900]])

    expected_array = np.array(
        [[False, False, False], [False, False, False], [False, False, False]]
    )

    df = pd.DataFrame(input_array)

    mask = zen._do_suppression(df, low=1, high=5)
    assert (mask.values == expected_array).all()


def test_single_row():
    input_array = np.array([4, 0, 10, 30])
    expected_array = np.array([True, True, True, True])
    df = pd.DataFrame(input_array)

    mask = zen._do_suppression(df, low=1, high=5)
    assert (mask.values == expected_array).all()

    input_array = np.array([[4], [0], [10], [30]])
    expected_array = np.array([[True], [True], [True], [True]])
    df = pd.DataFrame(input_array)

    mask = zen._do_suppression(df, low=1, high=5)
    assert (mask.values == expected_array).all()
