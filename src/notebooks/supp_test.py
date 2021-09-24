import pandas as pd

import zentables as zen
from zentables import _do_suppression

A = [[1, 4, 5], [-5, 8, 9]]

_do_suppression(pd.DataFrame(A), low=1, high=5)
