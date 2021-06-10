<<<<<<< HEAD
import tensorflow as tf

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
=======
import pandas as pd
from time import time
from .commons import kw_cartesian_prod
from ..Data.Data.Data import Data

def _get_changing_kwds(kw_dict):
    l = []
    for k, v in kw_dict.items():
        if type(v) is list and len(v) > 1:
            l.append(k)
    return l

def grid_estimation(model_type, 
                    train_inputs: list, 
                    train_labels: list,
                    estimation_sets: list,
                    estimators=[], 
                    estimators_names=[],
                    additional_descriptors={},
                    write_to=None,
                    train_kwargs={},
                    init_kwargs={},
                    verbose=1):
    """Method to calculate estimators on combinations of training kwargs.
    ARGS:
        - model_type (Model): model type to train
        - train_inputs: training inputs
        - train_labels: training labels
        - estimation_sets: list of 3-tuples with (set name, inputs, labels)
        - estimators: list of estimators (for estimate_fit method)
        - estimators_names: list of estimator names (for estimate_fit method)
        - additional_descriptors (dict): dictionary with additional information to be added to results
        - write_to (str): (optional) csv file to write online results to
        - train_kwargs (dict): kwargs dict for training. each argument that will be given as a list will be included in combinations
        - init_kwargs (dict): kwargs dict for instancing the model. each argument that will be given as a list will be included in combinations
    RETURNS:
        Dataframe with all estimator values and results"""
    train_kw_list = kw_cartesian_prod(train_kwargs)
    init_kw_list = kw_cartesian_prod(init_kwargs)
    res = pd.DataFrame()
    if verbose == 1:
        train_changing_kwargs = _get_changing_kwds(train_kwargs) 
        init_changing_kwargs = _get_changing_kwds(init_kw_list)

    for init_kwds in init_kw_list:
        # creating model instance
        m = model_type(**init_kwds)
        for train_kwds in train_kw_list:
            if verbose == 1:
                print("Running with {}".format(', '.join([str(train_kwds[w]) for w in train_changing_kwds] + [str(init_kwds[w]) for w in init_changing_kwds])))
            # measuring execution time
            ti = time()
            # training model
            m.train(train_inputs, train_labels, **train_kwds)
            tf = time()
            # getting results dictionary
            d = {'computation_time': round(tf - ti, 2)}
            d.update(init_kwds) # add model parameters
            d.update(train_kwds) # add train parameters
            d.update(additional_descriptors) # additional value
            d.update(m.estimate_fit(train_inputs, train_labels, estimators, estimators_names, 'train_')) # train estimators
            for name, inputs, labels in estimation_sets:
                d.update(m.estimate_fit(inputs, labels, estimators, estimators_names, name + '_')) # other sets estimators
            res = res.append(d, ignore_index=True) # add dictionary to results
            if write_to is not None:
                res.to_csv(write_to)
    return res

def grid_estimation(model_type, 
                    train_data: Data,
                    estimation_datas: list,
                    estimators=[], 
                    estimators_names=[],
                    additional_descriptors={},
                    write_to=None,
                    train_kwargs={},
                    init_kwargs={},
                    verbose=1):
    """Method to calculate estimators on combinations of training kwargs.
    ARGS:
        - model_type (Model): model type to train
        - train_data (Data): data to train on
        - estimation_datas: list of 2-tuples with (set name, data)
        - estimators: list of estimators (for estimate_fit method)
        - estimators_names: list of estimator names (for estimate_fit method)
        - additional_descriptors (dict): dictionary with additional information to be added to results
        - write_to (str): (optional) csv file to write online results to
        - train_kwargs (dict): kwargs dict for training. each argument that will be given as a list will be included in combinations
        - init_kwargs (dict): kwargs dict for instancing the model. each argument that will be given as a list will be included in combinations
    RETURNS:
        Dataframe with all estimator values and results"""
    train_kw_list = kw_cartesian_prod(train_kwargs)
    init_kw_list = kw_cartesian_prod(init_kwargs)
    res = pd.DataFrame()
    if verbose == 1:
        train_changing_kwds = _get_changing_kwds(train_kwargs) 
        init_changing_kwds = _get_changing_kwds(init_kwargs)
    for init_kwds in init_kw_list:
        # creating model instance
        m = model_type(**init_kwds)
        for train_kwds in train_kw_list:
            if verbose == 1:
                print("Running with {}".format(', '.join([w + "=" + str(train_kwds[w]) for w in train_changing_kwds] + [w + "=" + str(init_kwds[w]) for w in init_changing_kwds])))
            # measuring execution time
            ti = time()
            # training model
            m.train(train_data.vectorized_inputs, train_data.vectorized_labels, **train_kwds)
            tf = time()
            # getting results dictionary
            d = {'computation_time': round(tf - ti, 2)}
            d.update(init_kwds) # add model parameters
            d.update(train_kwds) # add train parameters
            d.update(additional_descriptors) # additional value
            d.update(m.estimate_fit(train_data, estimators, estimators_names, 'train_')) # train estimators
            for name, data in estimation_datas:
                d.update(m.estimate_fit(data, estimators, estimators_names, name + '_')) # other sets estimators
            res = res.append(d, ignore_index=True) # add dictionary to results
            if write_to is not None:
                res.to_csv(write_to)
    return res
>>>>>>> dev
