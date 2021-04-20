from .Tokenizer import Tokenizer
from ..commons import flatten
import numpy as np

class OneHot (Tokenizer):
    """Abstract class to handel general input tokenizations"""

    def __init__(self):
        self.word_idxs_dict = {}

    def set_word_tokenization_dict(self, vecs):
        '''Method to generate standard word tokenization dictionary'''
        d = {}
        for char in flatten(vecs):
            if char not in d.keys():
                d[char] = len(d.keys())
        self.word_idxs_dict = d

    def tokenize_vector(self, vec):
        """Method to tokenize a string"""
        tok = []
        for s in vec:
            v = [0 for _ in range(len(self.word_idxs_dict))]
            v[self.word_idxs_dict[s]] = 1
            tok.append(v)
        return tok

    def translate_vector(self, vec):
        """Method to translate a tokenized vector back to a string"""
        return NotImplementedError