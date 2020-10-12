import tensorflow as tf
from copy import copy
from matplotlib import pyplot as plt

class KerasArchitectures:
    '''Class for auto-generation of architectures for ML models'''

    @staticmethod
    def StandardDNN(hidden_layers_sizes = [], input_shape = None, target_input_shape = None, activations = [], drop_rate = 0):
        model = tf.keras.Sequential()
        if not target_input_shape == None:
            model.add(tf.keras.layers.Reshape(target_input_shape, input_shape=input_shape))

        for size, activation in zip(hidden_layers_sizes, activations):
            if activation == 'flatten':
                model.add(tf.keras.layers.Flatten())
                continue

            model.add(tf.keras.layers.Dense(size, activation = activation))
            model.add(tf.keras.layers.Dropout(drop_rate))

        return model
    
    @staticmethod
    def EmbedAndDNN(embedding_dim = None, input_dim = None, max_length = None, hidden_layers_sizes = [], activations = [], drop_rate = 0):     
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Embedding(input_dim, embedding_dim, input_length = max_length))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dropout(0.5))

        for size, activation in zip(hidden_layers_sizes, activations):
            model.add(tf.keras.layers.Dense(size, activation = activation))
            model.add(tf.keras.layers.Dropout(drop_rate))

        return model

    @staticmethod
    def GeneralANN(input_shape = None, layers = [], return_model = True, Input = None):
        if Input == None:
            Input = tf.keras.layers.Input(shape = input_shape)

        Output = layers[0] (Input)
        
        for layer in layers[1:]:
            Output = layer (Output)

        if return_model == True:
            return tf.keras.Model(Input, Output)
        else:
            return Output

    class AutoEncoders:

        def __init__(self, encoder = None, decoder = None, variational = False):
            self.encoder = encoder
            self.decoder = decoder
            self.variational = variational

        def set_encoder(self, input_shape = None, layers = []):
            '''
                method to set encoding layers. params:
                    input_shape: shape of input.
                    layers: layers for model.

                it is possible for one to split net for multiple outputs (for VAEs). for that one write a list of layers for each model.
                for example:
                    set_encoder(layers = [Dense, Flatten, [Dense, Dense]])
                    will set an encoder with Dense layer, Flatten layer and two Dense layers feeding from the output of the Flatten layer.
                
                ** it is not possible to split more then once and to more than 2 branches**
                ** after splitting the models can't be merged agian **
            '''

            is_splitted = False
            for index, layer in enumerate(layers):
                if type(layer) == list:
                    is_splitted = True
                    split_idx = index
                    if len(layer) > 2:
                        raise ValueError("Not possible to split for more then two branches.")
                    break

            if is_splitted == False:
                self.encoder = KerasArchitectures.GeneralANN(input_shape = input_shape, layers = layers, return_model = True)
            else:
                self.encoder = KerasArchitectures.GeneralANN(input_shape = input_shape, layers = layers[:(split_idx)], return_model = True)
                self.encoder = tf.keras.Model(
                    self.encoder.input, 
                    outputs = [KerasArchitectures.GeneralANN(input_shape = input_shape, layers = [layer[i] for layer in layers[split_idx:]], return_model = False, Input = self.encoder.output) for i in range(2)]
                )

        def set_decoder(self, layers = [], epsilon_std = 1):
            if self.encoder == None:
                raise Exception("Before setting a decoder you must define the encoder.")
            

            if self.variational == True:
                latent_dim = self.encoder.output_shape[-1]

                def sampler(args):
                    z_mean, z_log_sigma = args
                    epsilon = tf.keras.backend.random_normal(shape = (latent_dim[-1], ), stddev = epsilon_std)
                    return z_mean + tf.keras.backend.exp(z_log_sigma) * epsilon

                Input1 = tf.keras.layers.Input(latent_dim)
                Input2 = tf.keras.layers.Input(latent_dim)
                Lambda = tf.keras.layers.Lambda(sampler) ([Input1, Input2])
                
                Output = layers[0] (Lambda)
                for layer in layers[1:]:
                    Output = layer (Output)

                self.decoder = tf.keras.Model([Input1, Input2], Output)

            else:
                input_shape = self.encoder.output_shape
                self.decoder = KerasArchitectures.GeneralANN(layers = layers, input_shape = input_shape, return_model = True)

        def get_model(self):
            self.decoder.layers.pop(0)
            self.decoder.inbound_nodes.pop()
            new_out = self.decoder(self.encoder.output)
            model = tf.keras.Model(inputs = self.encoder.input, outputs = new_out)
            return model
        
        def __copy__(self):
            return KerasArchitectures.AutoEncoders(tf.keras.models.clone_model(self.encoder), tf.keras.models.clone_model(self.decoder), copy(self.variational))


def _plot_fit(ytrue, ypred, show=True, **plot_kwargs):
    plt.figure()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.scatter(ypred, ytrue, **plot_kwargs)
    plt.plot(ypred, ypred, 'r-')
    if show:
        plt.show()

def variational_sampler(args):
    z_mean, z_log_sigma = args
    epsilon = tf.keras.backend.random_normal(shape = (latent_dim, ), stddev = 1)
    return z_mean + tf.keras.backend.exp(z_log_sigma) * epsilon

# ****** costodi utils ******

def flatten(l):
    seed = copy(l)
    while True:
        new = []
        for s in seed:
            if type(s) is list:
                for x in s:
                    new.append(x)
            else:
                new.append(s)
        seed = copy(new)
        if all([not type(s) is list for s in seed]):
            break
    return seed

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

def _custodi_encode_vec(vec, degree, tokenization_dict):
    tokenized = []
    v = flatten(vec)
    for idx in range(len(v)):
        t = 0
        for i in range(degree):
            try:
                t += tokenization_dict[''.join(v[idx:(idx + i + 1)])]
            except KeyError:
                pass
        tokenized.append(t)
    return tokenized