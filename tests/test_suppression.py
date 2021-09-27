import numpy as np
import pandas as pd

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
