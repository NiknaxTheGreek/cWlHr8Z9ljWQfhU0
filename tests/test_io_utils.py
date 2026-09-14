import json
from pathlib import Path

import pandas as pd

from src.io_utils import atomic_write_csv, atomic_write_json, canonical_json_hash, sha256_file


def test_atomic_json_and_hash(tmp_path: Path):
    p=tmp_path/"x.json"; digest=atomic_write_json(p,{"b":2,"a":1})
    assert digest==sha256_file(p)
    assert json.loads(p.read_text())=={"a":1,"b":2}
    assert canonical_json_hash({"a":1,"b":2})==canonical_json_hash({"b":2,"a":1})


def test_atomic_csv_structure(tmp_path: Path):
    p=tmp_path/"x.csv"; df=pd.DataFrame({"a":[1,2],"b":["x","y"]}); atomic_write_csv(p,df); got=pd.read_csv(p)
    assert list(got.columns)==["a","b"] and len(got)==2
