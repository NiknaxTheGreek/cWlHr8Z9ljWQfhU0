import numpy as np
import pandas as pd
import pytest
from src.validation import require, require_columns, require_finite, require_range, require_unique


def test_validation_primitives():
    df=pd.DataFrame({"id":[1,2],"x":[0.0,1.0]}); require_columns(df,["id","x"],"C"); require_unique(df,"id","U"); require_finite(np.array([1.0,2.0]),"F"); require_range(df["x"],0,1,"R")


def test_require_raises():
    with pytest.raises(AssertionError): require(False,"X","broken")
