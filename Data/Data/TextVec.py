from .Data import Data

class TextVec (Data):
    '''Module to hold and handel text vector inputs'''

    def __init__(self, vec_sep=" "):
        super().__init__()
        self.vec_sep = vec_sep
        self.tokenizer = None

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

    def set_tokenizer(self, tokenizer):
        """Set a tokenizer (Tokenizer object) for database."""
        self.tokenizer = tokenizer

    def tokenize(self, target):
        """Tokenize text input to vectors
        ARGS:
            - target (str or nested list): name of attribute or nested list to tokenize"""

        if self.tokenization_func is None:
            self.set_word_tokenization_func()
        try:
            vecs = getattr(self, target)
        except AttributeError:
            if type(target) is list:
                vecs = target
            else:
                raise ValueError("tokenization target must be a vector of vectors (nested list) or attribute name")
        return [self.tokenizer.tokenize_vector(vec) for vec in vecs]

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