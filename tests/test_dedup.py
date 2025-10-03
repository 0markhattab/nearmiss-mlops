import pandas as pd
from pandas.testing import assert_frame_equal

def dedup_pandas(df, keys, ts_col):
    return (df.sort_values(ts_col, ascending=False)
              .drop_duplicates(subset=keys, keep="first")
              .sort_values(keys)
              .reset_index(drop=True))

def test_dedup_simple():
    df = pd.DataFrame({
        "store_id":[1,1,1,2],
        "event_id":[10,10,11,10],
        "ts":[100,120,80,50],
        "x":[1,2,3,4]
    })
    out = dedup_pandas(df, ["store_id","event_id"], "ts")
    exp = pd.DataFrame({
        "store_id":[1,1,2],
        "event_id":[10,11,10],
        "ts":[120,80,50],
        "x":[2,3,4]
    }).reset_index(drop=True)
    assert_frame_equal(out, exp)
