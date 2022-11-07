"""Feature utility functions for data processing."""
import pandas as pd


def stack_dict_to_df(_dict: dict) -> pd.DataFrame:
    """Stack dict of dicts into dataframe."""
    stacked = []
    for k, v in _dict.items():
        df = pd.DataFrame.from_dict(v)
        df["index"] = k
        index_name = "level_0" if df.index.name is None else df.index.name
        df = df.reset_index().set_index(["index", index_name])
        df.index.names = ("", "")
        stacked.append(df)
    return pd.concat(stacked).T
