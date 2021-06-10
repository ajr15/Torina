from abc import abstractclassmethod, ABC
from copy import deepcopy, copy
import pandas as pd
from ..Data.Data.Data import Data
from .commons import *


class Model (ABC):
    '''Abstract Model object'''
    
    @abstractclassmethod
    def train(self, inputs, labels, **kwargs):
        pass

    @abstractclassmethod
    def predict(self, inputs):
        pass

    def estimate_fit(self, inputs: list, labels: list, estimators=[], estimators_names=[], prefix=''):
        """Method to estimate fits between predictions on inputs and true labels.
        ARGS:
            - inputs (list): input list for prediction
            - labels (list): true labels
            - estimators: list of strings (with names of estimators) or callables to calculate estimators. default=[]
            - estimators_names: (optional) list with custom names for estimators. default=[]
            - prefix (str): a prefix to add to the keys of the dict. default=''
        RETURNS:
            dictionary with estimator names and values"""
        # setting up default estimators
        estimators_dict = {
            'r_squared': calc_r_squared,
            'rmse': calc_rmse,
            'mae': calc_mae,
            'mare': calc_mare
        }
        # setting up estimators and estimators names
        # building estimator names (in case it is not given)
        if not len(estimators) == len(estimators_names):
            counter = 0
            names = []
            for i, estimator in enumerate(estimators):
                if type(estimator) is str:
                    names.append(estimator)
                else:
                    try:
                        names.append(estimators_names[counter])
                        counter += 1
                    except IndexError:
                        names.append("estimator{}".format(i))
            estimators_names = names
        # making estimators a list of callables
        cestimators = [None for _ in range(len(estimators))]
        for i, estimator in enumerate(estimators):
            if estimator in estimators_dict.keys():
                cestimators[i] = estimators_dict[estimator]
            elif not callable(estimator):
                raise ValueError("Ilegal estimator entered. can be a callable or {}".format(', '.join(list(estimators_dict.keys()))))
        # calculating estimators
        d = {}
        pred = self.predict(inputs)
        for f, name in zip(cestimators, estimators_names):
            d[prefix + name] = f(pred, labels)
        return d

    def estimate_fit(self, data: Data, estimators=[], estimators_names=[], prefix=''):
        """Method to estimate fits on given data object.
        ARGS:
            - data (Data): data to estimate fit on
            - estimators: list of strings (with names of estimators) or callables to calculate estimators. default=[]
            - estimators_names: (optional) list with custom names for estimators. default=[]
            - prefix (str): a prefix to add to the keys of the dict. default=''
        RETURNS:
            dictionary with estimator names and values"""
        # setting up default estimators
        estimators_dict = {
            'r_squared': calc_r_squared,
            'rmse': calc_rmse,
            'mae': calc_mae,
            'mare': calc_mare
        }
        # setting up estimators and estimators names
        # building estimator names (in case it is not given)
        if not len(estimators) == len(estimators_names):
            counter = 0
            names = []
            for i, estimator in enumerate(estimators):
                if type(estimator) is str:
                    names.append(estimator)
                else:
                    try:
                        names.append(estimators_names[counter])
                        counter += 1
                    except IndexError:
                        names.append("estimator{}".format(i))
            estimators_names = names
        # making estimators a list of callables
        cestimators = [None for _ in range(len(estimators))]
        for i, estimator in enumerate(estimators):
            if estimator in estimators_dict.keys():
                cestimators[i] = estimators_dict[estimator]
            elif not callable(estimator):
                raise ValueError("Ilegal estimator entered. can be a callable or {}".format(', '.join(list(estimators_dict.keys()))))
        # calculating estimators
        d = {}
        pred = self.predict(data.vectorized_inputs)
        labels = copy(data.vectorized_labels)
        if data._label_norm_params is not None:
            pred = data.unnormalize_vectors(pred, params='labels')
            labels = data.unnormalize_vectors(labels, params='labels')
        for f, name in zip(cestimators, estimators_names):
            d[prefix + name] = f(pred, labels)
        return d

    def plot_fit(self, inputs, labels, show=True, add_line=True, **plot_kwargs):
        ypred = self.predict(inputs)
        _plot_fit(labels, ypred, show, add_line, **plot_kwargs)

    def grid_train(self, inputs, labels, **train_kwargs):
        """Method to train models on combinations of training kwargs.
        ARGS:
            - inputs: training inputs
            - labels: training labels
            - train_kwargs: kwargs for training. each argument that will be given as a list will be included in combinations
        RETURNS:
            list of trained models"""
        kw_list = kw_cartesian_prod(train_kwargs)
        res = []
        for kwds in kw_list:
            m = deepcopy(self)
            m.train(inputs, labels, **kwds)
            res.append(m)
        return m

