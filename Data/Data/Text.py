import numpy as np
from .Data import Data
from ..commons import split_text

class Text (Data):
    '''Module to handle textual data'''
    
    def __init__(self):
        super().__init__()
        self.tokenizer = None

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

    def set_tokenizer(self, tokenizer):
        """Set a tokenizer (Tokenizer object) for database."""
        self.tokenizer = tokenizer

    def tokenize(self, target):
        """Tokenize text input to vectors
        ARGS:
            - target (str or nested list): name of attribute or nested list to tokenize
        RETURNS:
            tokenized vectors"""

        if self.tokenizer is None:
            raise RuntimeError("tokenizer is not definded, use set_tokenizer method to set a tokenizer for this text")
        try:
            vecs = getattr(self, target)
        except TypeError:
            if type(target) is list:
                vecs = target
            else:
                raise ValueError("tokenization target must be a vector of vectors (nested list) or attribute name")
        return np.array([np.array(self.tokenizer.tokenize_vector(vec)) for vec in vecs])

    def vectorize_inputs(self):
        raise NotImplementedError
    
    def vectorize_labels(self):
        raise NotImplementedError