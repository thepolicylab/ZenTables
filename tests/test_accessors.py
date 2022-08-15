import pandas as pd

import zentables


def test_count_values():

    s = pd.Series(["Apple"] * 2 + ["Orange"] * 3 + [pd.NA])
    result = s.zen.value_counts(dropna=True, digits=0).sort_index()
    result_na = s.zen.value_counts(dropna=False, digits=1).sort_index()

    assert len(result) == 2
    assert len(result_na) == 3

    print(result)

    assert result[0] == "2 (40%)"
    assert result_na[0] == "2 (33.3%)"
