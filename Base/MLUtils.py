import os
import tensorflow as tf
from tensorflow import python
import warnings
warnings.simplefilter(action = 'ignore', category=FutureWarning)
python.util.deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn import cluster
from abc import abstractclassmethod, ABC
from copy import copy
import progressbar

class Architectures:
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
                self.encoder = Architectures.GeneralANN(input_shape = input_shape, layers = layers, return_model = True)
            else:
                self.encoder = Architectures.GeneralANN(input_shape = input_shape, layers = layers[:(split_idx)], return_model = True)
                self.encoder = tf.keras.Model(
                    self.encoder.input, 
                    outputs = [Architectures.GeneralANN(input_shape = input_shape, layers = [layer[i] for layer in layers[split_idx:]], return_model = False, Input = self.encoder.output) for i in range(2)]
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
                self.decoder = Architectures.GeneralANN(layers = layers, input_shape = input_shape, return_model = True)

        def get_model(self):
            self.decoder.layers.pop(0)
            self.decoder.inbound_nodes.pop()
            new_out = self.decoder(self.encoder.output)
            model = tf.keras.Model(inputs = self.encoder.input, outputs = new_out)
            return model
        
        def __copy__(self):
            return Architectures.AutoEncoders(tf.keras.models.clone_model(self.encoder), tf.keras.models.clone_model(self.decoder), copy(self.variational))


class Model:
    '''Wrapper around keras.model to allow some more utilities (mainly for autoencoders)'''

    def __init__(self, model = None, optimizer: tf.keras.optimizers = None, loss: tf.keras.losses = None, variational_input=False, trained=False, training_epochs=None, add_geo_loss=False): 
        if not model is None:
            self.loss = loss
            if isinstance(model, Architectures.AutoEncoders):
                if model.variational is True:
                    self._autoencoder = model
                    self.model = self._autoencoder.get_model()

                    def loss(x, x_decoded = self.model.output, x_loss_func = loss):
                        x_loss = x_loss_func(x, x_decoded)
                        z_mean, z_log_sigma = self._autoencoder.encoder.output
                        kl_loss = -0.5 * tf.keras.backend.mean(1 + z_log_sigma - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_log_sigma), axis = -1)
                        return x_loss + kl_loss

                else:
                    self._autoencoder = model
                    self.model = self._autoencoder.get_model()
                if add_geo_loss:
                    def loss(x, x_decoded = self.model.output, pred_loss_func=loss):
                        pred_loss = pred_loss_func(x, x_decoded)
                        if self._autoencoder.variational:
                            x_encoded = self._autoencoder.encoder.output[0]
                        else:
                            x_encoded = self._autoencoder.encoder.output
                        geo_loss = tf.keras.backend.dot(x_encoded, tf.keras.backend.transpose(x_encoded))
                        return pred_loss + geo_loss
                
            else:
                self._autoencoder = None
                if variational_input:
                    latent_dim = model.input_shape[-1]
                    def sampler(args):
                        z_mean, z_log_sigma = args
                        epsilon = tf.keras.backend.random_normal(shape = (latent_dim, ), stddev = 1)
                        return z_mean + tf.keras.backend.exp(z_log_sigma) * epsilon

                    Input1 = tf.keras.layers.Input(latent_dim)
                    Input2 = tf.keras.layers.Input(latent_dim)
                    Lambda = tf.keras.layers.Lambda(sampler) ([Input1, Input2])
                    Output = model.layers[1] (Lambda)
                    if len(model.layers) > 2:
                        for layer in model.layers[2:]:
                            Output = layer (Output)
                    self.model = tf.keras.Model([Input1, Input2], Output)
                else:
                    self.model = model
            self.model.compile(optimizer, loss = loss)
            self.variational_input = variational_input
            self.optimizer = optimizer
            self._trained = trained
            self.training_epochs = training_epochs
        else:
            self.model = model
            self.loss = loss
            self.optimizer = optimizer
            self._trained = trained
            self.variational_input = variational_input
            self.training_epochs = training_epochs

    def train(self, xtrain, ytrain, EPOCHS=None, verbose=1, batch_size=None):
        if EPOCHS is None and self.training_epochs is None:
            EPOCHS = 10
        elif EPOCHS is None:
            EPOCHS = self.training_epochs
        self.model.fit(xtrain, ytrain, epochs=EPOCHS, verbose=verbose, batch_size=batch_size)
        self._trained = True

    def test(self, xtest, ytest, batch_size=None, verbose=1):
        return self.model.evaluate(xtest, ytest, batch_size=batch_size, verbose=verbose)

    def encode(self, xdata):
        if self._autoencoder is None:
            raise ValueError("Model must be an autoencoder to encode")
        encoded = self._autoencoder.encoder.predict(xdata)
        return np.array(encoded)

    def decode(self, xencoded):
        if self._autoencoder is None:
            raise ValueError("Model must be an autoencoder to encode")
        return self._autoencoder.decoder.predict(xencoded)
    
    def predict(self, xdata):
        return self.model.predict(xdata)
    
    def save(self, path):
        if not os.path.isdir(path):
            raise ValueError("path must be a directory")
        def save_model(model, name):
            h5_filename = os.path.join(path, name + '_weights.h5')
            json_filename = os.path.join(path, name + '.json')
            json_model = model.to_json()
            with open(json_filename, "w") as f:
                f.write(json_model)
            model.save_weights(h5_filename)

        if self._autoencoder is None:
            save_model(self.model, 'model')
        else:
            save_model(self._autoencoder.encoder, 'encoder')
            save_model(self._autoencoder.decoder, 'decoder')

    def load(self, path):
        if not os.path.isdir(path):
            raise ValueError("path must be a directory")
        def load_model(name):
            h5_filename = os.path.join(path, name + '_weights.h5')
            json_filename = os.path.join(path, name + '.json')
            with open(json_filename, "r") as f:
                model = tf.keras.models.model_from_json(f.read())
            
            model.load_weights(h5_filename)
            return model
        
        if 'encoder.json' in os.listdir(path):
            autoencoder = Architectures.AutoEncoders()
            autoencoder.encoder = load_model('encoder')
            autoencoder.decoder = load_model('decoder')
            self._autoencoder = autoencoder
            self.model = self._autoencoder.get_model()
        else:
            self.model = load_model('model')
        self._trained = True

    def plot_2d(self, xdata, ydata, use_coord=None, show=True):
        xpred = self.predict(xdata)
        ydata = np.array(ydata)
        if len(xpred.shape) == 1 and len(ydata.shape) == 1:
            plt.scatter(xpred, ydata)
        elif not use_coord is None:
            plt.scatter(xpred[use_coord], ydata[use_coord])
        else:
            pca = PCA(n_components=1)
            if len(xpred.shape) > 1:
                xpred = pca.fit_transform(xpred)
            if len(ydata.shape) > 1:
                ydata = pca.fit_transform(ydata)
            plt.scatter(xpred, ydata)
        if show:
            plt.show()

    def plot_encoding(self, xdata, use_coord=None, prop=None, alpha=1, encode=True, show=True, new_fig=True, return_x=False):
        if new_fig:
            plt.figure()
        if encode:
            if not self._autoencoder is None and self._autoencoder.variational:
                xencoded = np.array(self.encode(xdata))[0]
            else:
                xencoded = np.array(self.encode(xdata))
        else:
            xencoded = xdata
        if len(xencoded.shape) == 1:
            plt.hist(xdata)
        elif use_coord is None:
            if xencoded.shape[-1] > 2:
                pca = PCA(n_components=2)
                xencoded = pca.fit_transform(xencoded)
            xencoded = np.transpose(xencoded)
            plt.scatter(xencoded[0], xencoded[1], c=prop, alpha=alpha)
        else:
            xencoded = np.transpose(xencoded)
            plt.scatter(xencoded[use_coord[0]], xencoded[use_coord[1]], alpha=alpha, c=prop)
        if return_x:
            return np.transpose(xencoded)
        if show:
            plt.show()

    def reinitialize(self):
        session = tf.compat.v1.keras.backend.get_session()
        for layer in self.model.layers: 
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel.initializer.run(session=session)
        self._trained = False

    def __copy__(self):
        if self._autoencoder is None:
            return Model(model=tf.keras.models.clone_model(self.model), optimizer=copy(self.optimizer), loss=copy(self.loss), variational_input=self.variational_input, trained=self._trained, training_epochs=self.training_epochs)
        else:
            return Model(model=self._autoencoder.__copy__(), optimizer=copy(self.optimizer), loss=copy(self.loss), variational_input=self.variational_input, trained=self._trained, training_epochs=self.training_epochs)

class FilterStrategies:

    class ABSStrategy (ABC):
        
        def __init__(self):
            pass
        
        @abstractclassmethod
        def strategy(self, reps):
            raise NotImplementedError()

    class random_filtering (ABSStrategy):
        def __init__(self, num_samples=1000):
            self.num_samples = num_samples

        def strategy(self, reps):
            return [np.random.randint(0, len(reps) - 1) for _ in range(self.num_samples)]
    
    class cluster_filtering (ABSStrategy):
        def __init__(self, cluster_model: cluster, metric='default', dist_th=1, frac=0.001):
            self.cluster_model = cluster_model
            if metric is 'default':
                def metric(x, y):
                    x = np.array(x)
                    y = np.array(y)
                    return np.linalg.norm(x - y)
            self.metric = metric
            self.dist_th = dist_th
            self.frac = frac

        def strategy(self, reps):
            reps = np.array(reps)
            if len(reps.shape) > 2:
                raise ValueError("Dimentions of the represantation for clustering must be lower than 2")
            labels = self.cluster_model.fit_predict(reps)
            clusters = dict()
            for label in list(set(labels)):
                clusters[label] = []
            for idx, label in enumerate(labels):
                clusters[label].append(idx)
            filtered_mols = [np.random.randint(0, len(reps) - 1)]
            for label in list(clusters.keys()):
                max_n = int(max([np.floor(len(clusters[label]) * self.frac), 1]))
                for _ in range(max_n):
                    idx = clusters[label][0 if len(clusters[label]) == 1 else np.random.randint(0, len(clusters[label]) - 1)]
                    rep = reps[idx]
                    Add = True
                    for i in filtered_mols:
                        if self.metric(reps[i], rep) < self.dist_th:
                            Add = False
                            break
                    if Add is True:
                        filtered_mols.append(idx)
            return filtered_mols

    class grid_filtering (ABSStrategy):
        
        def __init__(self, dist_th='auto', metric='default', frac=0.001, verbose=1):
            '''add option for autocalc dist_th'''
            self.dist_th = dist_th
            if metric is 'default':
                def metric(x, y):
                    x = np.array(x)
                    y = np.array(y)
                    return np.linalg.norm(x - y)
            self.metric = metric
            self.verbose = verbose
            self.frac = frac

        def strategy(self, reps):
            reps = np.array(reps)
            rep_size = np.prod(reps.shape[1:])
            reduction_dim = np.log2(len(reps) * self.frac)
            if reduction_dim < rep_size:
                pca = PCA(n_components=int(np.floor(reduction_dim)))
                reps = pca.fit_transform(reps)
            if self.dist_th == 'auto':
                # self.dist_th = 0.5 * np.mean(np.max(reps, axis=0) - np.min(reps, axis=0))
                self.dist_th = 0.5 * (np.max(reps, axis=0) - np.min(reps, axis=0))
            min_vec = np.min(reps, axis=0)
            clusters = dict()
            rep_vecs = []
            if self.verbose ==1:
                bar = progressbar.ProgressBar(maxval=len(reps) * 2, widgets=[progressbar.Bar('=', '[', ']', '.'), ' ', progressbar.Percentage()], term_width=50)
                prog_count = 0
                bar.start()
            for idx, rep in enumerate(reps):
                if self.verbose == 1:
                    prog_count += 1
                    bar.update(prog_count)
                vec = np.array(np.floor(rep / self.dist_th))
                # for coord in rep:
                #     vec.append(int(np.floor(coord / (self.dist_th))))
                try:
                    clusters[str(vec)].append(idx)
                    rep_vec = min_vec + self.dist_th * vec
                    rep_vecs.append(rep_vec)
                except KeyError:
                    clusters[str(vec)] = [idx]
            filtered_mols = []
            for idx_vec, rep_vec in zip(list(clusters.values()), rep_vecs):
                # avg_rep = np.zeros(reps.shape[-1])
                # for idx in idx_vec:
                #     avg_rep = avg_rep + reps[idx]
                # avg_rep = avg_rep / len(idx_vec)
                filtered_idx = None
                min_dist = np.inf
                for idx in idx_vec:
                    if self.verbose == 1:
                        prog_count += 1
                        bar.update(prog_count)
                    # dist = self.metric(reps[idx], avg_rep)
                    dist = self.metric(reps[idx], rep_vec)
                    if dist < min_dist:
                        filtered_idx = idx
                filtered_mols.append(filtered_idx)
            if self.verbose == 1:
                bar.finish()
            return filtered_mols

                
