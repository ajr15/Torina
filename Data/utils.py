from copy import copy
import numpy as np
import re
from . import Base
from sklearn.linear_model import Lasso as LinearModel

def padd_nested(vecs, end_char, pad_char):
    _vecs = copy(vecs)
    max_ls = np.max([[len(v) for v in vec] for vec in _vecs], axis=0)
    for i in range(len(_vecs)):
        _vecs[i] = [Base.padd_vec(vec, end_char, pad_char, l) for l, vec in zip(max_ls, _vecs[i])]
    return _vecs

def split_text(string, sep):
    if sep == 'chars':
        return [s for s in string]
    elif sep == 'words':
        return string.split()
    elif sep == None:
        return string
    else:
        return re.split(sep, string)

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

def choose(total_size, group_sizes: list, rel_sizes=False):
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
    return final_idxs

def _vecs_to_len_list(vecs):
    len_list = []
    seed = copy(vecs)
    while True:
        nseed = []
        l_vec = []
        for s in seed:
            if type(s) is list:
                l_vec.append(len(s))
                for v in s:
                    nseed.append(v)
            else:
                l_vec.append(1)
                nseed.append(s)
        seed = copy(nseed)
        len_list.append(l_vec)
        if all([not type(s) is list or all([not type(x) is list for x in s]) for s in seed]):
            break
    return len_list


def _reshape_2d(flat_vec, l_vec):
    '''reshapes flat_vec to be vector with subvectors of length in l_vec'''
    reshaped = []
    tot_l = 0
    for l in l_vec:
        reshaped.append(flat_vec[tot_l:(tot_l + l - 1)])
        tot_l += l
    return reshaped

def reshape(flat_vec, vec):
    '''reshapes flat_vec to be in the shape of vec'''
    l_vecs = _vecs_to_len_list(vec)
    l_vecs = [l_vecs[-(i + 1)] for i in range(len(l_vecs))]
    reshaped = copy(flat_vec)
    for l_vec in l_vecs:
        reshaped = _reshape_2d(reshaped, l_vec)
    return reshaped

def generic_dict_tokenization(vec, tokenization_dict):
    tokenized = []
    for v in vec:
        try:
            tokenized.append(tokenization_dict[v])
        except KeyError:
            raise RuntimeError("Provided tokenization dict doesn't contain all characters. missing character %s" % v)

def tokenize_from_function(vecs: list, tokenization_func, keep_shape=True):
    '''tokenization of vectors from dict function. Tokenization function is a function that fits elements in the flattened vectors to numbers. Tokenization function can also be a dictionary'''
    if not callable(tokenization_func): 
        if type(tokenization_func) is dict:
            d = copy(tokenization_func)
            tokenization_func = lambda v: [d[x] for x in flatten(v)]
        else:
            raise ValueError("tokenization_func must be callable or dictionary")

    if keep_shape:
        return reshape(tokenization_func(vecs), vecs)
    else:
        return tokenization_func(vecs)

def gen_word_tokenization_dict(vecs):
    '''Method to generate standard word tokenization dictionary'''
    char_set = set()
    for char in flatten(vecs):
        char_set.add(char)
    return dict([(char, i) for i, char in enumerate(list(char_set))])

def _gen_idx_dict_for_custodi(inputs):
    char_set = set()
    coupling_set = set()
    for vec in inputs:
        vec = flatten(vec)
        for idx in range(len(vec)):
            char_set.add(vec[idx])
            try:
                coupling_set.add(vec[idx] + vec[idx + 1])
            except IndexError:
                pass
    idx_dict = {}
    for idx, char in enumerate(list(char_set)):
        idx_dict[char] = idx
    for idx, chars in enumerate(list(coupling_set)):
        idx_dict[chars] = idx + len(char_set)
    return idx_dict

def gen_optimized_tokenization_dict(inputs, labels):
    '''optimize parameters for custodi tokenization'''
    # generating X data
    idx_dict = _gen_idx_dict_for_custodi(inputs)
    X = []
    for vec in inputs:
        x = np.zeros(len(idx_dict))
        vec = flatten(vec)
        for idx in range(len(vec)):
            x[idx_dict[vec[idx]]] += 1
            try:
                x[idx_dict[vec[idx] + vec[idx + 1]]] += 1
            except IndexError:
                pass
        X.append(x)
    # TODO: remove sklearn dependency. switch with standard library.
    reg = LinearModel(fit_intercept=False, alpha=0.05, max_iter=10000)
    reg.fit(X, labels)
    d = {}
    for key, c in zip(idx_dict.keys(), reg.coef_):
        d[key] = c
    return d

def _custodi_tokenization_func(vec, tokenization_dict):
    tokenized = []
    v = flatten(vec)
    for idx in range(len(v)):
        try:
            tokenized.append(tokenization_dict[v[idx]] + tokenization_dict[v[idx] + v[idx + 1]])
        except IndexError:
            try:
                tokenized.append(tokenization_dict[v[idx]])
            except KeyError:
                tokenized.append(0)
        except KeyError:
            try:
                tokenized.append(tokenization_dict[v[idx]])
            except KeyError:
                tokenized.append(0)
    return tokenized

def custodi_tokenization_func(tokenization_dict):
    return lambda vec: _custodi_tokenization_func(vec, tokenization_dict)