import sys
from . import Base
from .utils import *
import re
from ..Molecule.Base import BaseMol
from ..Molecule.Objects import SubsMol
import numpy as np
from copy import copy

class Text (Base.Data):
    '''Module to handle textual data'''
    tokenization_func = None

    def load_inputs_from_text(self, text, sep=None, split_lines=True):
        if split_lines:
            self.inputs = []
            text = text.splitlines()
            for line in text:
                self.inputs.append(split_text(line, sep))
        else:
            self.inputs = split_text(text, sep)

    def load_inputs_from_file(self, path, sep=None, split_lines=True):
        with open(path, 'r') as f:
            if split_lines:
                text = f.readlines()
            else:
                text = f.read()
        self.load_inputs_from_text(text, sep, split_lines)

    def set_word_tokenization_func(self):
        self.tokenization_dict = gen_word_tokenization_dict(self.vectorized_inputs)
        self.tokenization_func = self.tokenization_dict

    def set_custodi_tokenization_func(self, use_idxs='all'):
        if use_idxs == 'all':
            idxs = range(len(self.vectorized_inputs))
        else:
            idxs = use_idxs
        self.tokenization_dict = gen_optimized_tokenization_dict([self.vectorized_inputs[i] for i in idxs], [self.vectorized_labels[i] for i in idxs])
        self.tokenization_func = custodi_tokenization_func(self.tokenization_dict)

    def tokenize(self, target, keep_shape=True):
        if self.tokenization_func is None:
            self.set_word_tokenization_func()
        try:
            vecs = getattr(self, target)
        except AttributeError:
            if type(target) is list:
                vecs = target
            else:
                raise ValueError("tokenization target must be a vector of vectors (nested list) or attribute name")
        return [tokenize_from_function(vec, self.tokenization_func, keep_shape) for vec in vecs]

    # def tokenize_inputs(self, sep='char'):
    #     self.char_dict = dict()
    #     vectorized = []
    #     counter = 0
    #     for In in self.inputs:
    #         v_in = []
    #         for c in split_text(In, sep):
    #             if not c in self.char_dict.keys():
    #                 counter += 1
    #                 self.char_dict[c] = counter + 1
    #             v_in.append(char_dict[c])
    #         vectorized.append(v_in)
    #     return vectorized
    
    def vectorize_inputs(self, pad_char=0, end_char=None, method='unit_scale', axis=None, batch_size=128):
        self.vectorize_inputs = self.tokenize_inputs()
        self.pad_data(pad_char=0, end_char=None, pad='vectorized_inputs')
        self.noramlize_vectors(normalize='inputs', method='unit_scale', axis=None, batch_size=128)
    
    def vectorize_labels(self):
        pass

class SMILES (Text):

    def __init__(self, parent_specie=BaseMol):
        self.parent_specie = BaseMol

    def to_specie(self, x):
        return self.parent_specie.from_str(''.join(x))

class TextVec (Base.Data):
    '''Module to hold and handel text vector inputs'''

    def __init__(self, vec_sep=" "):
        self.vec_sep = vec_sep

    def load_inputs_from_text(self, text, sep='char', split_lines=True):
        if split_lines:
            self.inputs = []
            text = text.splitlines()
            for line in text:
                self.inputs.append([split_text(s, sep) for s in split_text(line, self.vec_sep)])
        else:
            self.inputs = [split_text(s, self.vec_sep) for s in split_text(line, sep)]

    def load_inputs_from_file(self, path, sep='chars', split_lines=True):
        with open(path, 'r') as f:
            text = f.read()
        self.load_inputs_from_text(text, sep, split_lines)

    def set_word_tokenization_func(self):
        self.tokenization_dict = gen_word_tokenization_dict(self.vectorized_inputs)
        self.tokenization_func = self.tokenization_dict

    def set_custodi_tokenization_func(self, use_idxs='all'):
        if use_idxs == 'all':
            idxs = range(len(self.vectorized_inputs))
        else:
            idxs = use_idxs
        self.tokenization_dict = gen_optimized_tokenization_dict([self.vectorized_inputs[i] for i in idxs], [self.vectorized_labels[i] for i in idxs])
        self.tokenization_func = custodi_tokenization_func(self.tokenization_dict)

    def tokenize(self, target, keep_shape=True):
        if self.tokenization_func is None:
            self.set_word_tokenization_func()
        try:
            vecs = getattr(self, target)
        except AttributeError:
            if type(target) is list:
                vecs = target
            else:
                raise ValueError("tokenization target must be a vector of vectors (nested list) or attribute name")
        return [tokenize_from_function(vec, self.tokenization_func, keep_shape) for vec in vecs]

    # def tokenize(self, start_from=0, costume_dict=None, attr='inputs'):
    #     # TODO: change all the 'tokenize' methods to this format
    #     if costume_dict == None:
    #         self.char_dict = dict()
    #         call_dict = False
    #     else:
    #         self.char_dict = costume_dict
    #         call_dict = all([callable(val) for val in costume_dict.values()])
    #     vectorized = []
    #     counter = start_from
    #     for In in getattr(self, attr):
    #         vectorized_in = []
    #         for vec in In:
    #             v_in = []
    #             for c in vec:
    #                 if not c in self.char_dict.keys():
    #                     if not costume_dict == None:
    #                         raise ValueError("Supplied dictionary doesn't contain all the characters. The character %s is not in dict" % c)
    #                     counter += 1
    #                     self.char_dict[c] = counter + 1
    #                 if call_dict:
    #                     v_in.append(self.char_dict[c]())
    #                 else:
    #                     v_in.append(self.char_dict[c])
    #             vectorized_in.append(v_in)
    #         vectorized.append(vectorized_in)
    #     return vectorized

    def vectorize_inputs(self, pad_char=0, end_char=None, method='unit_scale', axis=None, batch_size=128):
        if end_char == None:
            self.vectorized_inputs = self.tokenize_inputs()
        else:
            self.vectorized_inputs = self.tokenize_inputs(start_from=end_char)
        self.pad_data(end_char, pad_char, pad='vectorized_inputs')
        self.vectorized_inputs = [Base.flatten(v) for v in self.vectorized_inputs]
        self.noramlize_vectors('inputs', method, axis, batch_size)

    def _padd_attr(self, attr, end_char, pad_char):
        if attr == 'inputs' or attr == 'vectorized_inputs':
            return padd_nested(getattr(self, attr), end_char, pad_char)
        else:
            super()._padd_attr(attr, padd_char, end_char)


class SmilesVec (TextVec):

    def __init__(self, parent_specie, vec_sep=" "):
        self.parent_specie = parent_specie
        self.vec_sep = vec_sep

    def to_specie(self, x):
        mol = self.parent_specie()
        mol.from_vec([''.join(v) for v in x])
        return mol

