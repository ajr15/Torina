from copy import copy
import numpy as np
import re

# ======================
#     Padding Utils
# ======================

def padd_nested(vecs, end_char, pad_char):
    _vecs = copy(vecs)
    max_ls = np.max([[len(v) for v in vec] for vec in _vecs], axis=0)
    for i in range(len(_vecs)):
        _vecs[i] = [padd_vec(vec, end_char, pad_char, l) for l, vec in zip(max_ls, _vecs[i])]
    return _vecs

def padd_vecs(vecs, end_char, pad_char, max_l=None):
    _vecs = copy(vecs)
    if max_l == None:
        max_l = max([len(v) for v in vecs])
    for i in range(len(_vecs)):
        _vecs[i] = padd_vec(_vecs[i], end_char, pad_char, max_l)
    return _vecs

def padd_vec(vec, end_char, pad_char, l):
    _vec = copy(vec)
    if not end_char is None:
        _vec.append(end_char)
    while True:
        if len(_vec) == l + 1:
            break
        _vec.append(pad_char)
    return _vec

# ======================
#     Vector Utils
# ======================

def flatten(l):
    seed = copy(l)
    while True:
        new = []
        for s in seed:
            if hasattr(s, '__iter__') and not type(s) is str:
                for x in s:
                    new.append(x)
            else:
                new.append(s)
        seed = copy(new)
        if all([not hasattr(s, '__iter__') or type(s) is str for s in seed]):
            break
    return seed

def choose(total_size, group_sizes: list, rel_sizes=False, add_fill_group=False, random_seed=None):
    """Method to choose indecis from a list of length total_size.
    ARGS:
        - total size (int): length of the list
        - group_sizes (list): sizes of the different groups we want to choose
        - rel_sizes (bool): group sizes are in relative units. default=False
        - add_fill_group (bool): weather to add an extra group with the rest of the indicis. default=False
        - random_seed (int): (optional) set random seed. default=None
    RETURNS:
        list of lists of indicis for the different groups"""
    if not random_seed is None:
        np.random.seed(random_seed)
    if rel_sizes:
        group_sizes = [int(np.floor(size * total_size)) for size in group_sizes]
    chosen_idxs = set()
    final_idxs = []
    for size in group_sizes:
        idxs = []
        while True:
            idx = np.random.randint(0, total_size - 1)
            if not idx in chosen_idxs:
                idxs.append(idx)
                chosen_idxs.add(idx)
            if len(idxs) == size:
                break
        final_idxs.append(idxs)
    if add_fill_group:
        idxs = [i for i in range(total_size) if not i in chosen_idxs]
        final_idxs.append(idxs)
    return final_idxs

# ======================
#       Text Utils
# ======================

def split_text(string, sep):
    if sep == 'chars':
        return [s for s in string]
    elif sep == 'words':
        return string.split()
    elif sep == None:
        return string
    else:
        return re.split(sep, string)

# =======================
#   Normalization Utils
# =======================

def unit_normalization(vecs: np.array, axis=None, batch_size=128):
    min_v = np.min(vecs, axis=axis)
    max_v = np.max(np.abs(vecs), axis=axis)
    return costume_linear_transform(vecs, 1 / max_v, - min_v / max_v + 1e-6, batch_size), (min_v, max_v)

def inverse_unit_normalization(vecs, min_v, max_v, batch_size=128):
    raise costume_linear_transform(vecs, max_v, min_v)

def zscore_normalization(vecs, axis=None, batch_size=128, return_params=True):
    mean = np.mean(vecs, axis=axis)
    std = np.std(vecs, axis=axis)
    if return_params:
        return costume_linear_transform(vecs, 1 / std, - mean / std, batch_size), (mean, std)
    else:
        return costume_linear_transform(vecs, 1 / std, - mean / std, batch_size)

def inverse_zscore_normalization(vecs, mean, std, batch_size=128):
    return costume_linear_transform(vecs, std, mean, batch_size)

def positive_zscore_normalization(vecs, axis=None, batch_size=128):
    vecs = zscore_normalization(vecs, axis, batch_size)
    min_v = np.min(vecs, axis=axis)
    return np.minimum(costume_linear_transform(vecs, 1, min_v, batch_size), 1e-12)

def costume_linear_transform(vecs, a, b, batch_size=128):
    '''function to apply a linear transformation in batches'''
    vecs = np.array(vecs, dtype=np.float32)
    batch_num = int(np.floor(len(vecs) / batch_size))
    # transforming vecs in batches
    for batch in range(batch_num):
        vecs[(batch * batch_size):((batch + 1) * batch_size)] = a * vecs[(batch * batch_size):((batch + 1) * batch_size)] + b
    # transforming last batch
    vecs[(batch_num * batch_size):] = a * vecs[(batch_num * batch_size):] + b
    return vecs


