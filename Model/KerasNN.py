import tensorflow as tf

from .Model import Model
from .commons import *

class KerasNN (Model):
    '''Basic wrapper around keras.Model. Supports variational input (input from variational auto-encoder)
    ARGS:
        - model (keras.Model): a keras model
        - optimizer: a keras optimizer for model compilation
        - loss: a keras (or custom) loss function for model compilation
        - trained (bool): weather model is trained
        - variational_input (bool): weather input is variational
        '''

    def __init__(self, model, optimizer, loss, trained=False, variational_input=False):
        if variational_input:
            latent_dim = model.input_shape[-1]
            Input1 = tf.keras.layers.Input(latent_dim)
            Input2 = tf.keras.layers.Input(latent_dim)
            Lambda = tf.keras.layers.Lambda(self._variational_sampler) ([Input1, Input2])
            Output = model.layers[1] (Lambda)
            if len(model.layers) > 2:
                for layer in model.layers[2:]:
                    Output = layer (Output)
            self.model = tf.keras.Model([Input1, Input2], Output)
        else:
            self.model = model
        self.model.compile(optimizer, loss=loss)
        self.trained = trained

    @staticmethod
    def _variational_sampler(args):
        z_mean, z_log_sigma = args
        epsilon = tf.keras.backend.random_normal(shape = (latent_dim, ), stddev = 1)
        return z_mean + tf.keras.backend.exp(z_log_sigma) * epsilon

    def train(self, x, y, **training_kwags):
        self.model.fit(x, y, **training_kwags)
        self.trained = True

    def predict(self, x, **kwargs):
        return self.model.predict(x, **kwargs)