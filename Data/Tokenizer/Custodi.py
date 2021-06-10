from sklearn.linear_model import Ridge as LinearModel
from .Tokenizer import Tokenizer
from ..commons import flatten

class Custodi (Tokenizer):

    def __init__(self, degree=2, alpha=0.05, max_iter=10000):
        self.dictionary = {}
        self.degree = degree
        self.alpha = alpha
        self.max_iter = max_iter
        self.trained = False

    @staticmethod
    def _gen_idx_dict_for_custodi(inputs, degree):
        char_sets = [set() for _ in range(degree)]
        for vec in inputs:
            vec = flatten(vec)
            for idx in range(len(vec)):
                for i, s in enumerate(char_sets):
                    try:
                        # a = ''.join(vec[idx:(idx + i + 1)])
                        string = ''.join(vec)
                        a = string[idx:(idx + i + 1)]
                        if len(a) == i + 1:
                            s.add(a)
                    except IndexError:
                        pass
        idx_dict = {}
        for i, s in enumerate(char_sets):
            for j, char in enumerate(list(s)):
                if i == 0:
                    idx_dict[char] = j
                else:
                    idx_dict[char] = j + len(char_sets[i - 1])
        return idx_dict

    def tokenize_vector(self, vec):
        if not self.trained:
            raise RuntimeError("CUSTODI tokenizer is not trained! train tokenizer and the tokenize.")
        tokenized = []
        v = flatten(vec)
        for idx in range(len(v)):
            t = 0
            for i in range(self.degree):
                try:
                    if not len(v[idx:(idx + i + 1)]) == i + 1:
                        break
                    t += self.dictionary[''.join(v[idx:(idx + i + 1)])]
                except KeyError:
                    pass
            tokenized.append(t)
        return tokenized

    def translate_vector(self, vec):
        chars = [s for s in self.dictionary.keys() if len(s) == 1]
        recovered_string = []
        for x in reversed(vec):
            char = ''
            min_dif = 1e6
            reverse_recovered_string = [t for t in reversed(recovered_string)]
            for c in chars:
                trail_s = [c] + reverse_recovered_string[:(self.degree + 1)]
                tok_val = self.tokenize_vector(self.dictionary, self.degree, trail_s)[0]
                diff = abs(tok_val - x)
                if diff < min_dif:
                    char = c
                    min_dif = diff
            recovered_string.append(char)
        return "".join([s for s in reversed(recovered_string)])


    def train(self, inputs, labels, **train_kwrags):
        """Train a CUSTODI model on inputs and labels to get the tokenization dictionary.
        ARGS:
            - inputs: list of input vectors
            - lables: list of label vectors
            - train_kwargs: key words for training function (sklearn.LinearModel)
        RETURNS:
            optimal dictionary"""

        idx_dict = self._gen_idx_dict_for_custodi(inputs, self.degree)
        X = []
        for vec in inputs:
            x = [0 for _ in range(len(idx_dict))]
            vec = flatten(vec)
            for idx in range(len(vec)):
                for i in range(self.degree):
                    try:
                        x[idx_dict[''.join(vec[idx:(idx + i + 1)])]] += 1
                    except IndexError:
                        pass
            X.append(x)
        # TODO: remove sklearn dependency. switch with standard library.
        reg = LinearModel(fit_intercept=True, alpha=self.alpha, max_iter=self.max_iter)
        reg.fit(X, labels)
        d = {}
        for key, c in zip(idx_dict.keys(), reg.coef_):
            d[key] = c
        self.dictionary = d
        self.intercept = reg.intercept_
        self.trained = True
        return d