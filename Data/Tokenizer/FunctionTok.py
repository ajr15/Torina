from .Tokenizer import Tokenizer

class FunctionTok (Tokenizer):
    """General tokenizer from tokenization function
    ARGS:
        - func (callable): tokenization function for sinle vector
        - kwrags: function key words"""

    def __init__(self, func, **kwargs):
        self.func = func
        self.kwargs = kwargs

    def tokenize_vector(self, vec):
        try:
            return self.func(vec, **self.kwargs)
        except Exception:
            print("Failed tokenizing vector, returning empty array")
            return []

    def translate_vector(self, vec):
        raise RuntimeError("Function tokenization does not support translation")