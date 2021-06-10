import os
import tensorflow as tf
from tensorflow import python
import warnings; warnings.simplefilter(action = 'ignore', category=FutureWarning)
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn import cluster
from sklearn.kernel_ridge import KernelRidge as sk_kernel_ridge
from sklearn.linear_model import Lasso as LinearModel

from .Base import Model
from . import utils

class KerasNN (Model):
    '''Basic wrapper around keras.Model'''

    def __init__(self, model, optimizer, loss, trained=False, batch_size=32, variational_input=False):
        if variational_input:
            latent_dim = model.input_shape[-1]
            Input1 = tf.keras.layers.Input(latent_dim)
            Input2 = tf.keras.layers.Input(latent_dim)
            Lambda = tf.keras.layers.Lambda(variational_sampler) ([Input1, Input2])
            Output = model.layers[1] (Lambda)
            if len(model.layers) > 2:
                for layer in model.layers[2:]:
                    Output = layer (Output)
            self.model = tf.keras.Model([Input1, Input2], Output)
        else:
            self.model = model
        self.model.compile(optimizer, loss=loss)
        self.batch_size = batch_size
        self.trained = trained

    def train(self, x, y, **training_kwags):
        if self.trained:
            warnings.warn("Training a trained model, overrides existing model.")
        self.model.fit(x, y, **training_kwags)
        self.trained = True

    def test(self, x, y, **kwargs):
        return self.model.evaluate(x, y, **kwargs)

    def predict(self, x, **kwargs):
        return self.model.predict(x, **kwargs)

    def plot_fit(self, x, y, show=True, add_line=True, **plot_kwargs):
        ypred = self.predict(x)
        utils._plot_fit(y, ypred, show, **plot_kwargs)

def vae_loss(x, output, x_decoded, x_loss_func):
    x_loss = x_loss_func(x, x_decoded)
    z_mean, z_log_sigma = output
    kl_loss = -0.5 * tf.keras.backend.mean(1 + z_log_sigma - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_log_sigma), axis = -1)
    return x_loss + kl_loss

class KerasAE (Model):

    def __init__(self, ae, optimizer, loss, variational=False, trained=False):
        self._autoencoder = ae
        self.model = self._autoencoder.get_model()
        if variational:
            loss = lambda x: vae_loss(x, self._autoencoder.encoder.output, self.model.output, loss)
        self.model.compile(optimizer, loss)
        self.variational = variational
        self.trained = trained

    def train(self, x, **training_kwags):
        if self.trained:
            warnings.warn("Training a trained model, overrides existing model.")
        self.model.fit(x, x, **training_kwags)
        self.trained = True

    def test(self, x, **test_kwargs):
        return self.model.evaluate(x, **test_kwargs)

    def predict(self, x, **predict_kwargs):
        return self.model.predict(x, **predict_kwargs)
    
    def encode(self, x, **kwargs):
        return self._autoencoder.encoder.predict(x, **kwargs)
    
    def decode(self, x, **kwargs):
        return self._autoencoder.decoder.predict(x)

    def plot_results(self, x, use_coords=None, props=None, **plot_kwargs):
        plt.figure()
        xencoded = self.encode(x)
        if self.variational:
            xencoded = xencoded[0]
        xencoded = np.array(xencoded)
        if use_coords is None:
            if xencoded.shape[-1] > 2:
                pca = PCA(n_components=2)
                xencoded = pca.fit_transform(xencoded)
            plt.plot(xencoded[0], xencoded[1], c=props, **plot_kwargs)
        else:
            xencoded = np.transpose(xencoded)
            plt.plot(xencoded[use_coords[0]], xencoded[use_coords[1]], c=props, **plot_kwargs)

    def gen_labels(self, data):
        return data.vectorized_inputs

class KernalRidge (Model):

    def __init__(self, *args, **kwargs):
        '''Kernal Ridge Regressor as implemented in scikit-learn. Arguments
        are same as sklearn.kernal_ridge.KernalRidge parameters.'''
        self.model = sk_kernel_ridge(*args, **kwargs)
    
    def train(self, x, y):
        self.model = self.model.fit(x, y)
    
    def test(self, x, y):
        '''returns R^2 of the fit'''
        return self.model.score(x, y)
    
    def predict(self, x):
        return self.model.predict(x)

    def plot_fit(self, x, y, show=True, **plot_kwargs):
        ypred = self.predict(x)
        utils._plot_fit(y, ypred, show, **plot_kwargs)

class CuSToDi:

    def __init__(self, degree=2, alpha=0.05, max_iter=10000):
        self.degree = degree
        self.alpha = alpha
        self.max_iter = max_iter
        self.dictionary = {}
        self.intercept = 0

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
        

    def train(self, inputs, labels):
        idx_dict = utils._gen_idx_dict_for_custodi(inputs, self.degree)
        X = []
        for vec in inputs:
            x = np.zeros(len(idx_dict))
            vec = utils.flatten(vec)
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
        return d

    def encode(self, inputs):
        tokenized = []
        for v in inputs:
            tokenized.append(utils._custodi_encode_vec(v, self.degree, self.dictionary))
        return tokenized

    def decode(self, encoded_inputs):
        # TODO: implement a decoding method for custodi!
        pass

    def predict(self, inputs):
        encoded = self.encode(inputs)
        pred = [np.sum(v) + self.intercept for v in encoded]
        return pred

    def plot_fit(self, x, y, show=True, **plot_kwargs):
        ypred = self.predict(x)
        utils._plot_fit(y, ypred, show, **plot_kwargs)

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

# class OldModel:
#     '''Wrapper around keras.model to allow some more utilities (mainly for autoencoders)'''

#     def __init__(self, model = None, optimizer: tf.keras.optimizers = None, loss: tf.keras.losses = None, variational_input=False, trained=False, training_epochs=None, add_geo_loss=False): 
#         if not model is None:
#             self.loss = loss
#             if isinstance(model, Architectures.AutoEncoders):
#                 if model.variational is True:
#                     self._autoencoder = model
#                     self.model = self._autoencoder.get_model()

#                     def loss(x, x_decoded = self.model.output, x_loss_func = loss):
#                         x_loss = x_loss_func(x, x_decoded)
#                         z_mean, z_log_sigma = self._autoencoder.encoder.output
#                         kl_loss = -0.5 * tf.keras.backend.mean(1 + z_log_sigma - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_log_sigma), axis = -1)
#                         return x_loss + kl_loss

#                 else:
#                     self._autoencoder = model
#                     self.model = self._autoencoder.get_model()
#                 if add_geo_loss:
#                     def loss(x, x_decoded = self.model.output, pred_loss_func=loss):
#                         pred_loss = pred_loss_func(x, x_decoded)
#                         if self._autoencoder.variational:
#                             x_encoded = self._autoencoder.encoder.output[0]
#                         else:
#                             x_encoded = self._autoencoder.encoder.output
#                         geo_loss = tf.keras.backend.dot(x_encoded, tf.keras.backend.transpose(x_encoded))
#                         return pred_loss + geo_loss
                
#             else:
#                 self._autoencoder = None
#                 if variational_input:
#                     latent_dim = model.input_shape[-1]
#                     def sampler(args):
#                         z_mean, z_log_sigma = args
#                         epsilon = tf.keras.backend.random_normal(shape = (latent_dim, ), stddev = 1)
#                         return z_mean + tf.keras.backend.exp(z_log_sigma) * epsilon

#                     Input1 = tf.keras.layers.Input(latent_dim)
#                     Input2 = tf.keras.layers.Input(latent_dim)
#                     Lambda = tf.keras.layers.Lambda(sampler) ([Input1, Input2])
#                     Output = model.layers[1] (Lambda)
#                     if len(model.layers) > 2:
#                         for layer in model.layers[2:]:
#                             Output = layer (Output)
#                     self.model = tf.keras.Model([Input1, Input2], Output)
#                 else:
#                     self.model = model
#             self.model.compile(optimizer, loss = loss)
#             self.variational_input = variational_input
#             self.optimizer = optimizer
#             self._trained = trained
#             self.training_epochs = training_epochs
#         else:
#             self.model = model
#             self.loss = loss
#             self.optimizer = optimizer
#             self._trained = trained
#             self.variational_input = variational_input
#             self.training_epochs = training_epochs

#     def train(self, xtrain, ytrain, EPOCHS=None, verbose=1, batch_size=None):
#         if EPOCHS is None and self.training_epochs is None:
#             EPOCHS = 10
#         elif EPOCHS is None:
#             EPOCHS = self.training_epochs
#         self.model.fit(xtrain, ytrain, epochs=EPOCHS, verbose=verbose, batch_size=batch_size)
#         self._trained = True

#     def test(self, xtest, ytest, batch_size=None, verbose=1):
#         return self.model.evaluate(xtest, ytest, batch_size=batch_size, verbose=verbose)

#     def encode(self, xdata):
#         if self._autoencoder is None:
#             raise ValueError("Model must be an autoencoder to encode")
#         encoded = self._autoencoder.encoder.predict(xdata)
#         return np.array(encoded)

#     def decode(self, xencoded):
#         if self._autoencoder is None:
#             raise ValueError("Model must be an autoencoder to encode")
#         return self._autoencoder.decoder.predict(xencoded)
    
#     def predict(self, xdata):
#         return self.model.predict(xdata)
    
#     def save(self, path):
#         if not os.path.isdir(path):
#             raise ValueError("path must be a directory")
#         def save_model(model, name):
#             h5_filename = os.path.join(path, name + '_weights.h5')
#             json_filename = os.path.join(path, name + '.json')
#             json_model = model.to_json()
#             with open(json_filename, "w") as f:
#                 f.write(json_model)
#             model.save_weights(h5_filename)

#         if self._autoencoder is None:
#             save_model(self.model, 'model')
#         else:
#             save_model(self._autoencoder.encoder, 'encoder')
#             save_model(self._autoencoder.decoder, 'decoder')

#     def load(self, path):
#         if not os.path.isdir(path):
#             raise ValueError("path must be a directory")
#         def load_model(name):
#             h5_filename = os.path.join(path, name + '_weights.h5')
#             json_filename = os.path.join(path, name + '.json')
#             with open(json_filename, "r") as f:
#                 model = tf.keras.models.model_from_json(f.read())
            
#             model.load_weights(h5_filename)
#             return model
        
#         if 'encoder.json' in os.listdir(path):
#             autoencoder = Architectures.AutoEncoders()
#             autoencoder.encoder = load_model('encoder')
#             autoencoder.decoder = load_model('decoder')
#             self._autoencoder = autoencoder
#             self.model = self._autoencoder.get_model()
#         else:
#             self.model = load_model('model')
#         self._trained = True

#     def plot_2d(self, xdata, ydata, use_coord=None, show=True):
#         xpred = self.predict(xdata)
#         ydata = np.array(ydata)
#         if len(xpred.shape) == 1 and len(ydata.shape) == 1:
#             plt.scatter(xpred, ydata)
#         elif not use_coord is None:
#             plt.scatter(xpred[use_coord], ydata[use_coord])
#         else:
#             pca = PCA(n_components=1)
#             if len(xpred.shape) > 1:
#                 xpred = pca.fit_transform(xpred)
#             if len(ydata.shape) > 1:
#                 ydata = pca.fit_transform(ydata)
#             plt.scatter(xpred, ydata)
#         if show:
#             plt.show()

#     def plot_encoding(self, xdata, use_coord=None, prop=None, alpha=1, encode=True, show=True, new_fig=True, return_x=False):
#         if new_fig:
#             plt.figure()
#         if encode:
#             if not self._autoencoder is None and self._autoencoder.variational:
#                 xencoded = np.array(self.encode(xdata))[0]
#             else:
#                 xencoded = np.array(self.encode(xdata))
#         else:
#             xencoded = xdata
#         if len(xencoded.shape) == 1:
#             plt.hist(xdata)
#         elif use_coord is None:
#             if xencoded.shape[-1] > 2:
#                 pca = PCA(n_components=2)
#                 xencoded = pca.fit_transform(xencoded)
#             xencoded = np.transpose(xencoded)
#             plt.scatter(xencoded[0], xencoded[1], c=prop, alpha=alpha)
#         else:
#             xencoded = np.transpose(xencoded)
#             plt.scatter(xencoded[use_coord[0]], xencoded[use_coord[1]], alpha=alpha, c=prop)
#         if return_x:
#             return np.transpose(xencoded)
#         if show:
#             plt.show()

#     def reinitialize(self):
#         session = tf.compat.v1.keras.backend.get_session()
#         for layer in self.model.layers: 
#             if hasattr(layer, 'kernel_initializer'):
#                 layer.kernel.initializer.run(session=session)
#         self._trained = False

#     def __copy__(self):
#         if self._autoencoder is None:
#             return Model(model=tf.keras.models.clone_model(self.model), optimizer=copy(self.optimizer), loss=copy(self.loss), variational_input=self.variational_input, trained=self._trained, training_epochs=self.training_epochs)
#         else:
#             return Model(model=self._autoencoder.__copy__(), optimizer=copy(self.optimizer), loss=copy(self.loss), variational_input=self.variational_input, trained=self._trained, training_epochs=self.training_epochs)

# class FilterStrategies:

#     class ABSStrategy (ABC):
        
#         def __init__(self):
#             pass
        
#         @abstractclassmethod
#         def strategy(self, reps):
#             raise NotImplementedError()

#     class random_filtering (ABSStrategy):
#         def __init__(self, num_samples=1000):
#             self.num_samples = num_samples

#         def strategy(self, reps):
#             return [np.random.randint(0, len(reps) - 1) for _ in range(self.num_samples)]
    
#     class cluster_filtering (ABSStrategy):
#         def __init__(self, cluster_model: cluster, metric='default', dist_th=1, frac=0.001):
#             self.cluster_model = cluster_model
#             if metric is 'default':
#                 def metric(x, y):
#                     x = np.array(x)
#                     y = np.array(y)
#                     return np.linalg.norm(x - y)
#             self.metric = metric
#             self.dist_th = dist_th
#             self.frac = frac

#         def strategy(self, reps):
#             reps = np.array(reps)
#             if len(reps.shape) > 2:
#                 raise ValueError("Dimentions of the represantation for clustering must be lower than 2")
#             labels = self.cluster_model.fit_predict(reps)
#             clusters = dict()
#             for label in list(set(labels)):
#                 clusters[label] = []
#             for idx, label in enumerate(labels):
#                 clusters[label].append(idx)
#             filtered_mols = [np.random.randint(0, len(reps) - 1)]
#             for label in list(clusters.keys()):
#                 max_n = int(max([np.floor(len(clusters[label]) * self.frac), 1]))
#                 for _ in range(max_n):
#                     idx = clusters[label][0 if len(clusters[label]) == 1 else np.random.randint(0, len(clusters[label]) - 1)]
#                     rep = reps[idx]
#                     Add = True
#                     for i in filtered_mols:
#                         if self.metric(reps[i], rep) < self.dist_th:
#                             Add = False
#                             break
#                     if Add is True:
#                         filtered_mols.append(idx)
#             return filtered_mols

#     class grid_filtering (ABSStrategy):
        
#         def __init__(self, dist_th='auto', metric='default', frac=0.001, verbose=1):
#             '''add option for autocalc dist_th'''
#             self.dist_th = dist_th
#             if metric is 'default':
#                 def metric(x, y):
#                     x = np.array(x)
#                     y = np.array(y)
#                     return np.linalg.norm(x - y)
#             self.metric = metric
#             self.verbose = verbose
#             self.frac = frac

#         def strategy(self, reps):
#             reps = np.array(reps)
#             rep_size = np.prod(reps.shape[1:])
#             reduction_dim = np.log2(len(reps) * self.frac)
#             if reduction_dim < rep_size:
#                 pca = PCA(n_components=int(np.floor(reduction_dim)))
#                 reps = pca.fit_transform(reps)
#             if self.dist_th == 'auto':
#                 # self.dist_th = 0.5 * np.mean(np.max(reps, axis=0) - np.min(reps, axis=0))
#                 self.dist_th = 0.5 * (np.max(reps, axis=0) - np.min(reps, axis=0))
#             min_vec = np.min(reps, axis=0)
#             clusters = dict()
#             rep_vecs = []
#             if self.verbose ==1:
#                 bar = progressbar.ProgressBar(maxval=len(reps) * 2, widgets=[progressbar.Bar('=', '[', ']', '.'), ' ', progressbar.Percentage()], term_width=50)
#                 prog_count = 0
#                 bar.start()
#             for idx, rep in enumerate(reps):
#                 if self.verbose == 1:
#                     prog_count += 1
#                     bar.update(prog_count)
#                 vec = np.array(np.floor(rep / self.dist_th))
#                 # for coord in rep:
#                 #     vec.append(int(np.floor(coord / (self.dist_th))))
#                 try:
#                     clusters[str(vec)].append(idx)
#                     rep_vec = min_vec + self.dist_th * vec
#                     rep_vecs.append(rep_vec)
#                 except KeyError:
#                     clusters[str(vec)] = [idx]
#             filtered_mols = []
#             for idx_vec, rep_vec in zip(list(clusters.values()), rep_vecs):
#                 # avg_rep = np.zeros(reps.shape[-1])
#                 # for idx in idx_vec:
#                 #     avg_rep = avg_rep + reps[idx]
#                 # avg_rep = avg_rep / len(idx_vec)
#                 filtered_idx = None
#                 min_dist = np.inf
#                 for idx in idx_vec:
#                     if self.verbose == 1:
#                         prog_count += 1
#                         bar.update(prog_count)
#                     # dist = self.metric(reps[idx], avg_rep)
#                     dist = self.metric(reps[idx], rep_vec)
#                     if dist < min_dist:
#                         filtered_idx = idx
#                 filtered_mols.append(filtered_idx)
#             if self.verbose == 1:
#                 bar.finish()
#             return filtered_mols