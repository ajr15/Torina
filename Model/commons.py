import numpy as np
from copy import copy

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


# ===============================
#   Estimator Calculation Utils
# ===============================

def _safe_calc_sum_of_binary_func(pred, true, func):
    """Method to calculate sum of binary function values on two vectors in a memory-safe way"""
    s = 0
    for p, t in zip(pred, true):
        val = func(p, t)
        if not val == [np.inf] and not val == np.inf:
            s = s + val
    return s

def calc_rmse(pred, true):
    f = lambda p, t: np.square(p - t)
    return np.sqrt(_safe_calc_sum_of_binary_func(pred, true, f) / len(pred))

def calc_mae(pred, true):
    f = lambda p, t: np.abs(p - t)
    return _safe_calc_sum_of_binary_func(pred, true, f) / len(pred)

def calc_mare(pred, true):
    f = lambda p, t: np.abs((p - t) / t) if not t == 0 else 0
    return _safe_calc_sum_of_binary_func(pred, true, f)/ len(pred)

def calc_r_squared(pred, true):
    avg_t = _safe_calc_sum_of_binary_func(pred, true, lambda p, t: t) / len(true)
    avg_p = _safe_calc_sum_of_binary_func(pred, true, lambda p, t: p) / len(pred)
    var_t = _safe_calc_sum_of_binary_func(pred, true, lambda p, t: np.square(t - avg_t)) / len(true)
    var_p = _safe_calc_sum_of_binary_func(pred, true, lambda p, t: np.square(p - avg_p)) / len(pred)
    cov = _safe_calc_sum_of_binary_func(pred, true, lambda p, t: (t - avg_t) * (p - avg_p)) / len(true)
    return cov**2 / (var_p * var_t)

# =================================================
#   Make cartesian products of dictionaries utils
# =================================================

def cartesian_prod(vecs1, vecs2):
    prod = []
    for vec1 in vecs1:
        for vec2 in vecs2:
            prod.append(vec1 + [vec2])
    return prod

def kw_cartesian_prod(kwargs_dict):
    '''generates a list of dict for all combinations of keywords in kwargs_dict'''
    vec_cartesian = [[]]
    for vals in kwargs_dict.values():
        # correcting non-list kw values
        if not type(vals) is list:
            vals = [vals]
        vec_cartesian = cartesian_prod(vec_cartesian, vals)
    dict_list = []
    for vec in vec_cartesian:
        d = dict([(k, v) for k, v in zip(kwargs_dict.keys(), vec)])
        dict_list.append(d)
    return dict_list

# ==============
#   Plot Utils
# ==============

def _plot_fit(ytrue, ypred, show=True, add_line=True, **plot_kwargs):
    plt.figure()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.scatter(ypred, ytrue, **plot_kwargs)
    if add_line:
        plt.plot(ypred, ypred, 'r-')
    if show:
        plt.show()