import json
from collections import defaultdict
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd

from . import DATA_PATH


def _min_max_scale(
    X: np.array, feature_range: tuple["int", "int"] = (0, 1), flipped: bool = False
):
    X = np.array(X)
    min_, max_ = feature_range
    X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    X_scaled = X_std * (max_ - min_) + min_
    
    if flipped:
        return 1 - X_scaled

    return X_scaled


def _read_lemma_table() -> pd.DataFrame:
    """Read the lemma table

    Returns:
        pd.DataFrame: The lemma table
    """
    lemmas = pd.read_csv(
        DATA_PATH / "lemma-30k-2017.txt",
        sep="\t",
        header=None,
        names=["pos", "lemma", "freq"],
        encoding="utf-8",
    )

    return lemmas


def _create_lemma_dict(
    min_max_scaled: Optional[tuple["int", "int"]] = None
) -> dict[any, any]:
    """Load data

    Returns:
        dict[any, any]: Each lemma is a key and the value is a dictionary
    with the part of speech as key and the frequency as value.
    """
    lemmas = _read_lemma_table()
    if min_max_scaled:
        lemmas.freq = _min_max_scale(lemmas.freq, feature_range=min_max_scaled)

    data = lemmas.set_index("lemma").to_dict("split")

    lemma_dict = defaultdict(dict)

    for i, lemma in enumerate(data["index"]):
        lemma_dict[lemma] |= {data["data"][i][0]: data["data"][i][1]}

    return lemma_dict


def _save_lemma_dict(
    min_max_scaled: Optional[tuple["int", "int"]] = None, filename="lemma_dict_raw"
):
    lemma_dict = _create_lemma_dict(min_max_scaled)

    with open((DATA_PATH / filename).with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(lemma_dict, f)


def _get_lemma_dict(freq: Literal["raw", "scaled"]) -> dict[any, any]:
    """Load data

    Returns:
        dict[any, any]: Each lemma is a key and the value is a dictionary
    with the part of speech as key and the frequency as value.
    """
    with open(DATA_PATH / f"lemma_dict_{freq}.json", "r") as f:
        lemma_dict = json.load(f)

    return lemma_dict


def get_lemma_freq(
    lemma: str,
    pos: Optional[str] = None,
    freq: Literal["raw", "scaled"] = "scaled",
    agg: Literal["max", "min", "sum", "mean"] = "max",
    out_of_vocab: Union[int, None] = None,
) -> int:
    """_summary_

    Args:
        lemma (str): The lemma to look up.
        pos (Optional[str], optional): Part of speech tag. Defaults to None.
        freq (Literal["raw", "scaled"], optional): The frequence type to use.
            Defaults to "scaled".
        agg (Literal["max", "sum", "mean"], optional): Aggregation method.
            Used if POS is not provided. Defaults to "max".
        out_of_vocab (Union[int, None], optional): Value to return if the lemma
            is not in the vocabulary. Defaults to 0.

    Returns:
        int: _description_
    """
    if pos:
        return LEMMA_DICTS[freq][lemma][pos]
    else:
        agg_func = getattr(np, agg)
        freqs = LEMMA_DICTS[freq].get(lemma)
        if freqs:
            return agg_func(list(freqs.values()))
        else:
            return out_of_vocab


LEMMA_DICTS = {
    "raw": _get_lemma_dict("raw"),
    "scaled": _get_lemma_dict("scaled"),
}


if __name__ == "__main__":
    for freq in ("raw", "scaled"):
        min_max_scaled = None if freq == "raw" else (0, 1)
        _save_lemma_dict(
            min_max_scaled=min_max_scaled,
            filename=f"lemma_dict_{freq}",
        )
