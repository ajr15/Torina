from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge as LinearModel
import numpy as np

from .commons import flatten
from ..Data.Tokenizer.Custodi import Custodi as custodi_tokenizer
from .Model import Model


class Custodi (Model):

    def __init__(self, degree=2, alpha=0.05, max_iter=10000):
        self.degree = degree
        self.alpha = alpha
        self.max_iter = max_iter
        self.dictionary = {}
        self.intercept = 0
        self.trained = False

    def get_optimal_train_idxs(inputs):
        char_set = set()
        idxs = [0]
        for i, vec in enumerate(inputs):
            vec = flatten(vec)
            for idx in range(len(vec)):
                try:
                    c = ''.join(vec[idx:(idx + i + 1)])
                except IndexError:
                    pass
                finally:
                    if not c in char_set:
                        char_set.add(c)
                        if not idxs[-1] == idx:
                            idxs.append(idx)
        return idxs

    @staticmethod
    def _gen_idx_dict_for_custodi(inputs, degree):
        char_sets = [set() for _ in range(degree)]
        for vec in inputs:
            vec = flatten(vec)
            for idx in range(len(vec)):
                for i, s in enumerate(char_sets):
                    try:
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
        
    def train(self, inputs, labels):
        idx_dict = self._gen_idx_dict_for_custodi(inputs, self.degree)
        X = []
        for vec in inputs:
            x = np.zeros(len(idx_dict))
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
        X = np.array(X)
        reg.fit(X, labels)
        d = {}
        for key, c in zip(idx_dict.keys(), reg.coef_):
            d[key] = c
        self.dictionary = d
        self.intercept = reg.intercept_
        self.trained = True
        return d

    def encode(self, inputs):
        # use Custodi tokenizer to encode
        # no need to load more than degree and dictionary
        tokenizer = custodi_tokenizer(degree=self.degree)
        tokenizer.dictionary = self.dictionary
        tokenizer.trained = True
        # tokenize
        tokenized = []       
        for v in inputs:
            tokenized.append(tokenizer.tokenize_vector(v))
        return tokenized

    def decode(self, encoded_inputs):
        # TODO: implement a decoding method for custodi!
        raise NotImplementedError

    def predict(self, inputs):
        encoded = self.encode(inputs)
        pred = [sum(v) + self.intercept for v in encoded]
        return pred

    def plot_encoding(self, x, y=None, show=True, **plot_kwargs):
        xencoded = self.encode(x)
        pca = PCA(n_components=2)
        xencoded = pca.fit_transform(xencoded)
        plt.figure()
        plt.plot(xencoded[0], xencoded[1], c=y, **plot_kwargs)
        plt.xlabel("PCA1")
        plt.ylabel("PCA2")
        plt.title("Costodi Encoding")
        if show:
            plt.show()