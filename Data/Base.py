import pandas as pd
from multiprocessing import Pool
import os
import numpy as np
from abc import ABC, abstractclassmethod
from copy import copy
import sys; sys.path.append('../')
import CompProtocol

class Specie:

    name = "None"

    def to_str(self):
        return NotImplemented

    def from_str(self, string):
        raise NotImplementedError('the method \'from_str\' is not implemented and can\'t be used')

    def from_file(self, filename):
        return NotImplemented

class Data:
    '''General Input object, to handle general input formats'''

    inputs = None
    labels = None
    vectorized_inputs = None
    vectorized_labels = None

    @property
    def parent_specie(self):
        return self._parent_specie

    @parent_specie.setter
    def parent_specie(self, specie):
        if not issubclass(specie, Specie):
            raise ValueError("Parent specie must be a Specie")
        self._parent_specie = specie

    @property
    def reps(self):
        return self._reps

    @reps.setter
    def reps(self, vecs):
        if not len(vecs) == len(self.inputs):
            raise Warning("Number of reps is different then number of inputs. This might cause problems.")
        self._reps = vecs

    def vectorize_inputs(self):
        '''Method to vectorize the inputs'''
        return self.inputs

    def vectorize_labels(self):
        '''Method to vectorize the labels'''
        return self.labels

    def to_specie(self, x):
        '''Method to convert a single input entry to a \'Mol\' like object. For example converting smiles input to Mol'''
        raise NotImplementedError("to_specie method is not implemented in this subclass")

    @staticmethod
    def from_specie(specie):
        '''Method to convert a specie to input.'''
        raise NotImplementedError("load_inputs method is not implemented in this subclass")

    def noramlize_vectors(self, normalize='all', method='unit_scale', axis=None, batch_size=128):
        if not method in normalization_methods.keys():
            raise ValueError("Unknown method. Recognized methods are only %s" % (normalization_methods.keys()))
        norm_func = normalization_methods[method]
        if normalize == 'inputs' or normalize == 'all':
            self.vectorized_inputs = norm_func(self.vectorized_inputs, axis, batch_size)
        if normalize == 'labels' or normalize == 'all':
            self.vectorized_labels = norm_func(self.vectorized_labels, axis, batch_size)

    def load_from_dataframe(self, df, input_columns=None, label_columns=None, reps_columns=None):
        # TODO: implement this methods, to load data from dataframe to Data instance
        pass

    def export_to_dataframe(self, include='all'):
        # TODO: implement this method, to export Data instance to df.
        pass

    def _padd_attr(self, attr, pad_char, end_char):
        setattr(self, attr, padd_vecs(getattr(self, attr), end_char, pad_char))

    def pad_data(self, pad_char=0, end_char=None, pad='all'):
        if pad == 'all' or pad == 'inputs':
            self._padd_attr('inputs', pad_char, end_char)
        if pad == 'all' or pad == 'labels':
            self._padd_attr('labels', pad_char, end_char)
        if pad == 'all_vecs' or pad == 'vectorized_inputs':
            self._padd_attr('vectorized_inputs', pad_char, end_char)
        if pad == 'all_vecs' or pad == 'vectorized_labels':
            self._padd_attr('vectorized_labels', pad_char, end_char)

def model_data_to_dataframe(inputs, labels=None, input_names: list=None, label_names: list=None):
    # checks validity of input
    if not input_names is None and not len(input_names) == len(inputs[0]):
        raise ValueError("input_names and input length must be the same.")
    if not labels is None and not len(labels) == len(inputs):
        raise ValueError("inputs and labels lengths must be the same.")
    if not label_names is None and not len(label_names) == len(labels[0]):
        raise ValueError("label_names and label length must be the same.")
    # generates a dict list for dataframe
    # TODO: try to find a better algorithm to do that!
    d = []
    for Input in inputs:
        if label_names is None:
            d.append({'inputs': Input})
        else:
            d.append(dict([(input_name, i) for input_name, i in zip(input_names, Input)]))
    if not labels is None:
        for idx, label in enumerate(labels):
            if label_names is None:
                d[idx]['labels'] = label
            else:
                for name, label in zip(label_names, labels):
                    d[idx][name] = label
    return pd.DataFrame(d)

def convert_inputs(data, inputs=None, input_idxs=None, verbose=0, nprocs=os.cpu_count()):
    '''Recommended function for conversion on multiple inputs/input_idxs to species. Supports parallel computing'''
    # TODO: add verbosity mode support
    if not inputs is None:
        if nprocs > 1:
            with Pool(processes=nprocs) as pool:
                species = pool.map(data.to_specie, inputs)
        else:
            species = []
            for Input in inputs:
                species.append(data.to_specie(Input))
    elif not input_idxs is None:
        if nprocs > 1:
            with Pool(processes=nprocs) as pool:
                species = pool.map(data.to_specie, [data.inputs[i] for i in input_idxs])
        else:
            species = []
            for Input in [data.inputs[i] for i in input_idxs]:
                species.append(data.to_specie(Input))
    return species

def generate_data_using_comp_protocol(data, comp_protocol, input_idxs=None, inputs=None, verbose=0, nprocs=os.cpu_count(), timeout=None):
    if not isinstance(comp_protocol, CompProtocol.Base.CompProtocol):
        raise ValueError("computation protocol must be an instance of a CompProtocol object")
    if not isinstance(data, Data):
        raise ValueError("data must be an instance of a Data object")

    species = convert_inputs(data, inputs, input_idxs, verbose, os.cpu_count())
    aux_df, res_df = CompProtocol.Base.run_protocol(comp_protocol, species, verbose, timeout, nprocs)
    labels = [vec[:-1] for vec in res_df.to_numpy()]
    new_data = copy(data)
    new_data.inputs = inputs if not inputs == None else [data.inputs[i] for i in input_idxs]
    new_data.labels = labels
    return new_data

def unit_normalization(vecs: np.array, axis=None, batch_size=128):
    min_v = np.min(vecs, axis=axis)
    max_v = np.max(vecs, axis=axis)
    return costume_linear_transform(vecs, 1 / max_v, - min_v / max_v, batch_size)

def zscore_normalization(vecs, axis=None, batch_size=128):
    mean = np.mean(vecs, axis=axis)
    std = np.std(vecs, axis=axis)
    return costume_linear_transform(vecs, 1 / std, - mean / std, batch_size)

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

normalization_methods = {
    'unit_scale': unit_normalization,
    'z_score': zscore_normalization
}

def padd_vecs(vecs, end_char, pad_char, max_l=None):
    if max_l == None:
        max_l = max([len(v) for v in vecs])
    for i in range(len(vecs)):
        vecs[i] = padd_vec(vecs[i], end_char, pad_char, max_l)
    return vecs

def padd_vec(vec, end_char, pad_char, l):
    if not end_char is None:
        vec.append(end_char)
    while True:
        if len(vec) == l + 1:
            break
        vec.append(pad_char)
    return vec

def flatten(l):
    seed = copy(l)
    while True:
        new = []
        for s in seed:
            if type(s) is list:
                for x in s:
                    new.append(x)
            else:
                new.append(s)
        seed = copy(new)
        if all([not type(s) is list for s in seed]):
            break
    return seed