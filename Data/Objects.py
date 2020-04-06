import sys
sys.path.append('../')
from Data import Base
import re
from Molecule.Base import BaseMol
from Molecule.Objects import SubsMol
import numpy as np

def split_text(string, sep):
    if sep == 'chars':
        return [s for s in string]
    elif sep == 'words':
        return string.split()
    elif sep == None:
        return string
    else:
        return re.split(sep, string)

class Text (Base.Data):
    '''Module to handle textual data'''

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

    def tokenize_inputs(self, sep='char'):
        self.char_dict = dict()
        vectorized = []
        counter = 0
        for In in self.inputs:
            v_in = []
            for c in split_text(In, sep):
                if not c in self.char_dict.keys():
                    counter += 1
                    self.char_dict[c] = counter + 1
                v_in.append(char_dict[c])
            vectorized.append(v_in)
        return vectorized
    
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

    def tokenize_inputs(self):
        self.char_dict = dict()
        vectorized = []
        counter = 0
        for In in self.inputs:
            vectorized_in = []
            for vec in In:
                v_in = []
                for c in vec:
                    if not c in self.char_dict.keys():
                        counter += 1
                        self.char_dict[c] = counter + 1
                    v_in.append(self.char_dict[c])
                vectorized_in.append(v_in)
            vectorized.append(vectorized_in)
        return vectorized
    
    def vectorize_inputs(self, pad_char=0, end_char=None, method='unit_scale', axis=None, batch_size=128):
        self.vectorized_inputs = self.tokenize_inputs()
        self.pad_data(end_char, pad_char, pad='vectorized_inputs')
        self.vectorized_inputs = [Base.flatten(v) for v in self.vectorized_inputs]
        self.noramlize_vectors('inputs', method, axis, batch_size)

    def _padd_attr(self, attr, end_char, pad_char):
        if attr == 'inputs' or attr == 'vectorized_inputs':
            setattr(self, attr, padd_nested(getattr(self, attr), end_char, pad_char))
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

def padd_nested(vecs, end_char, pad_char):
    max_ls = np.max([[len(v) for v in vec] for vec in vecs], axis=0)
    for i in range(len(vecs)):
        vecs[i] = [Base.padd_vec(vec, end_char, pad_char, l) for l, vec in zip(max_ls, vecs[i])]
    return vecs