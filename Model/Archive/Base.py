from abc import abstractclassmethod, ABC
from copy import copy
from ..Data.Base import Data
import progressbar


# ********************************
#     Astract Model Object
# ********************************

class Model (ABC):
    '''Abstract fitting model object.'''

    def __init__(self, *args, **kwargs):
        pass
    
    @abstractclassmethod
    def train(self, *args):
        pass

    @abstractclassmethod
    def test(self, *args):
        pass

    @abstractclassmethod
    def predict(self, *args):
        pass

    def gen_inputs(self, data: Data, *args):
        '''Method to generate input data for the specific model from args. Defaults to return the args. Use to generalize data generation for different models.'''
        return data.vectorized_inputs

    def gen_labels(self, data: Data, *args):
        '''Method to generate label data for the specific model from args. Defaults to return the args. Use to generalize data generation for different models.'''
        return data.vectorized_labels

    def plot_fit(self, *args):
        '''Method to plot fit results'''
        return NotImplemented

    def plot_results(self, *args):
        '''Method to plot the results of a calculation'''
        return NotImplemented

