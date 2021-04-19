from .Tokenizer import Tokenizer
from ..commons import flatten
import numpy as np

class OneHot (Tokenizer):
    """Abstract class to handel general input tokenizations"""

    def __init__(self):
        self.word_idxs_dict = {}

    def set_word_tokenization_dict(self, vecs):
        '''Method to generate standard word tokenization dictionary'''
        char_set = set()
        for char in flatten(vecs):
            char_set.add(char)
        self.word_idxs_dict = dict([(char, i) for i, char in enumerate(list(char_set))])

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