'`gemo.utils.py`'

import copy
import collections

import numpy as np


def validate_3d_vector(vector):

    if not isinstance(vector, np.ndarray):
        vector = np.array(vector)

    if len(vector.shape) > 1:
        vector = np.squeeze(vector)

    if vector.shape != (3, ):
        msg = ('Vector must be of size 3, not of size {}.')
        raise ValueError(msg.format(vector.size))

    return vector


def update_dict(base, upd):
    """Update an arbitrarily-nested dict."""

    for key, val in upd.items():
        if isinstance(base, collections.Mapping):
            if isinstance(val, collections.Mapping):
                r = update_dict(base.get(key, {}), val)
                base[key] = r
            else:
                base[key] = upd[key]
        else:
            base = {key: upd[key]}

    return base


def set_in_dict(base, address, value):

    val_dict_sub = base
    for idx, sub_dict in enumerate(address):
        if idx < len(address) - 1:
            val_dict_sub = val_dict_sub[sub_dict]
        else:
            val_dict_sub[sub_dict] = value


def nest(*lists, return_index=False):
    """Nest elements of multiple lists.

    Parameters
    ----------
    lists : sequence of lists

    Returns
    -------
    nested_list : list
        List whose elements are lists containing one 
        element for each input list.
    return_index : bool, optional
        If True, an index list is also retuned which records the
        indices used from each list to generate each output list element.

    Example
    -------
    >>> nest([1, 2], [3, 4, 5])
    [[1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5]]

    """

    N = len(lists)

    products = np.array([1] * (N + 1))
    for i in range(len(lists) - 1, -1, -1):
        products[:i + 1] *= len(lists[i])

    nested_list = [[None for x in range(N)] for y in range(products[0])]

    idx = []
    for row_idx, row in enumerate(nested_list):

        sub_idx = []
        for col_idx, _ in enumerate(row):

            num_repeats = products[col_idx + 1]
            sub_list_idx = int(row_idx / num_repeats) % len(lists[col_idx])
            nested_list[row_idx][col_idx] = copy.deepcopy(
                lists[col_idx][sub_list_idx])

            sub_idx.append(sub_list_idx)
        idx.append(sub_idx)

    if return_index:
        return (nested_list, idx)
    else:
        return nested_list
