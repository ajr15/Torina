from sklearn.kernel_ridge import KernelRidge as sk_kernel_ridge
from .Model import Model

class KernalRidge (Model):

    def __init__(self, **kwargs):
        '''Kernal Ridge Regressor as implemented in scikit-learn. Arguments
        are same as sklearn.kernal_ridge.KernalRidge parameters.'''
        self.model = sk_kernel_ridge(**kwargs)
    
    def train(self, inputs, labels):
        self.model = self.model.fit(inputs, labels)
        
    def predict(self, inputs):
        return self.model.predict(inputs)