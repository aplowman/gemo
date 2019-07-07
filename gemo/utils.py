'`gemo.utils.py`'

import copy

import numpy as np


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
