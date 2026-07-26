"""Reading the sorting-bin count table.

The expected layout is a table with one row per sequence and a 3-level column
header ``(cell_line, replicate, bin)``:

    seq,   A549,A549,A549,A549,A549,...    <- cell line
           rep1,rep1,rep1,rep1,rep2,...    <- replicate
           bin1,bin2,bin3,bin4,bin1,...    <- expression bin

Each cell line is loaded as an independent (N, S, B) tensor of counts.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class GroupData:
    name: str
    X: np.ndarray          # (N, S, B) counts
    mask: np.ndarray       # (N, S, B) bool, True = observed
    index: pd.Index
    s_names: list
    b_names: list


def read_counts(path, sep=None):
    """Read the count table; ``sep`` defaults to tab for .tsv, comma otherwise."""
    if sep is None:
        sep = "\t" if str(path).endswith((".tsv", ".txt")) else ","
    return pd.read_csv(path, sep=sep, index_col=0, header=[0, 1, 2])


def load_groups(path_or_df, groups=None, verbose=False):
    """Split the count table into per-cell-line (N, S, B) tensors.

    Empty fields are read as counts of ZERO, not as missing cells.  Some
    export formats write every zero as a blank; treating those as unobserved
    deletes exactly the informative zeros of the low-abundance sequences (an
    object recorded as (0,0,0,14) would reach the likelihood as (-,-,-,14),
    which carries no information about its bin profile at all).  Only rows that
    are zero everywhere are meant to drop out, and the likelihood handles those.

    Rows keep file order, so the returned index aligns with the fit output.
    """
    df = path_or_df if isinstance(path_or_df, pd.DataFrame) \
        else read_counts(path_or_df)
    n_blank = int(df.isna().to_numpy().sum())
    if n_blank and verbose:
        print(f"[data] {n_blank} empty fields read as zero counts")
    df = df.fillna(0.0)

    out = {}
    all_groups = list(dict.fromkeys(df.columns.get_level_values(0)))
    unknown = [g for g in (groups or ()) if g not in all_groups]
    if unknown:
        raise ValueError(f"unknown cell line(s) {unknown}; "
                         f"the table has {all_groups}")
    for g in (groups or all_groups):
        sub = df[g]
        s_names = list(dict.fromkeys(sub.columns.get_level_values(0)))
        b_names = list(dict.fromkeys(sub.columns.get_level_values(1)))
        cols = pd.MultiIndex.from_product([s_names, b_names])
        missing = [c for c in cols if c not in sub.columns]
        if missing:
            raise ValueError(f"group {g}: non-rectangular columns {missing}")
        sub = sub[cols]
        X = sub.to_numpy(dtype=np.float64).reshape(
            len(sub), len(s_names), len(b_names))
        out[g] = GroupData(name=g, X=X, mask=~np.isnan(X), index=sub.index,
                           s_names=s_names, b_names=b_names)
    return out


def bin_order(b_names):
    """Column order that sorts bins by the number in their name (bin1 < bin2)."""
    key = lambda b: int("".join(c for c in str(b) if c.isdigit()) or 0)
    return sorted(range(len(b_names)), key=lambda i: key(b_names[i]))


def prep(gd):
    """(X, mask, measured) with bins in ascending order.

    ``measured`` marks the objects with at least one read; the all-zero rows
    carry no activity signal (they are either silent or unsampled).
    """
    bo = bin_order(gd.b_names)
    X, mask = gd.X[:, :, bo], gd.mask[:, :, bo]
    measured = mask.any((1, 2)) & (np.where(mask, np.nan_to_num(X), 0)
                                   .sum((1, 2)) > 0)
    return X, mask, measured
