import pandas as pd

import zentables as zen

A = [[1, 4, 5], [-5, 8, 9]]

zen._do_suppression(pd.DataFrame(A), low=1, high=5)
