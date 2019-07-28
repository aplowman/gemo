'`gemo.utils.py`'

import copy
import collections

import numpy as np


def intersperse(lst, item):
    'Add a given item between each existing item in a list.'
    result = [item] * (len(lst) * 2 - 1)
    result[0::2] = lst
    return result


def validate_3d_vector(vector):

    if vector is None:
        vector = [0, 0, 0]

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


def to_4d_array(arr):

    out = np.vstack([
        np.hstack([arr, np.zeros((3, 1))]),
        [0, 0, 0, 1]
    ])

    return out


def point_on_line(start, end, parameter):
    'Return the point on a parametrically defined line.'
    point = (end - start) * parameter + start
    return point


def get_lines_trace(lines):
    'Get a trace suitable for plotting lines.'

    if not lines.size:
        out = np.zeros((4, 0))
    else:
        nannys = np.ones((len(lines), lines.shape[1], 1)) * np.nan
        out = np.hstack(np.concatenate([lines, nannys], axis=2))

    return out


def get_box_edges(corners):
    'Get line segments representing box edges'

    edges = [
        corners[:, [0, 1]],
        corners[:, [1, 4]],
        corners[:, [4, 2]],
        corners[:, [2, 0]],

        corners[:, [3, 5]],
        corners[:, [5, 7]],
        corners[:, [7, 6]],
        corners[:, [6, 3]],

        #corners[:, [0, 1]],
        corners[:, [1, 5]],
        #corners[:, [5, 3]],
        corners[:, [3, 0]],

        #corners[:, [2, 4]],
        corners[:, [4, 7]],
        #corners[:, [7, 6]],
        corners[:, [6, 2]],

        #corners[:, [0, 2]],
        #corners[:, [2, 6]],
        #corners[:, [6, 3]],
        #corners[:, [3, 0]],

        #corners[:, [1, 4]],
        #corners[:, [4, 7]],
        #corners[:, [7, 5]],
        #corners[:, [5, 1]],
    ]
    return np.array(edges)  # [10:11]
