import os
import tensorflow as tf
from tensorflow import python
import warnings
warnings.simplefilter(action = 'ignore', category=FutureWarning)
python.util.deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from .Base import MolUtils as utils
from .Base.PropCalc import PropCalc
from .Base import MLUtils as mlutils
from matplotlib import pyplot as plt
import openbabel as ob
import pandas as pd
import re
from datetime import datetime
from sklearn.linear_model import LinearRegression
import signal
import numpy as np
import progressbar
from sklearn import cluster
from sklearn.decomposition import PCA
import sklearn
from copy import copy, deepcopy

class Bina:

    def __init__(self, mols = None):
        self.mols = mols
        self._mol_names = None
        self._mol_class = utils.Mol
        self._PropCalc_class = None
        self.results = pd.DataFrame()
        self.rate_constants = pd.DataFrame()
        self._reps = None
        self.embeded_mols = None
        self._tstamp = datetime.now()
        self.encoding_model = None
        self.predictive_model = None
        self.filter_strategy = None
        self.mols_for_comp = None
        self._embeded = False
        self.encoding_opt_iterations = 1
        self.voting_model = None
        self.train_test_data = {
            'voting_train': None,
            'voting_test': None,
            'prediction_train': None,
            'prediction_test': None
        }
        self._normalization_params = None
        self.normalization_method = 'Z_score'
        self.PropCalc_timeout = None
        self.num_of_test_mols = 1000

    def set_mol_class(self, mol_class):
        self._mol_class = mol_class

    def load_mols_from_dir(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError("Supplied path doesn't exist !")
        else:
            allowed_fmts = [fmt.split("--")[0].strip() for fmt in ob.OBConversion().GetSupportedInputFormat()]
            allowed_fmts.remove('mop'); allowed_fmts.remove('mopout'); allowed_fmts.remove('arc')
            files = []
            for file_ in os.listdir(path):
                name, ext = os.path.splitext(file_)
                if ext[1:] in allowed_fmts and not ext[1:] in ['mop', 'out', 'arc', 'txt']:
                    files.append(os.path.join(path, file_))
                if len(file_) == 0:
                    raise RuntimeError("No supported formats in selected path")
            
            mols = []; names = []
            for file_ in files:
                name = os.path.split(file_)[-1]
                name = os.path.splitext(name)[0]
                names.append(name)
                mol = self._mol_class()
                mol.from_file(file_)
                mols.append(mol)
            
            self.mols = mols; self._mol_names = names

    def load_mols_from_files(self, paths):
        allowed_fmts = [fmt.split("--")[0].strip() for fmt in ob.OBConversion().GetSupportedInputFormat()]
        allowed_fmts.remove('mop'); allowed_fmts.remove('mopout'); allowed_fmts.remove('arc')
        names = []; mols = []
        if type(paths) == list:
            for path in paths:
                if not os.path.exists(path):
                    raise FileNotFoundError("Some files in list don't exist")
                else:
                    name, ext = os.path.splitext(path)
                    if ext[1:] in allowed_fmts and not ext[1:] in ['mop', 'out', 'arc']:
                        names.append(name)
                        mol = self._mol_class()
                        mol.from_file(path)
                        mols.append(mol)
                    else:
                        raise RuntimeError("Some files in list don't have supported format")
        else:
            if not os.path.exists(path):
                raise FileNotFoundError("Some files in list don't exist")
            else:
                name, ext = os.path.splitext(path)
                if ext[1:] in allowed_fmts and not ext[1:] in ['mop', 'out', 'arc']:
                    names.append(name)
                    mol = self._mol_class()
                    mol.from_file(path)
                    mols.append(mol)
                else:
                    raise RuntimeError("Some files in list don't have supported format")
        self.mols = mols; self._mol_names = names

    def load_mol_from_smiles(self, smiles, load_text_only=True):
        if not load_text_only:
            mol = self._mol_class()
            mol.from_smiles(smiles)
            if self.mols is None:
                self.mols = [mol]
                self._mol_names = [1]
            else:
                self.mols.append(mol)
                self._mol_names.append(len(self.mols))
        else:
            if self.mols is None:
                self.mols = [smiles]
                self._mol_names = [1]
            else:
                self.mols.append(smiles)
                self._mol_names.append(len(self.mols))
        if " " in smiles:
            smiles = smiles.split(" ")
        if self._reps is None:
            self._reps = [smiles]
        else:
            self._reps.append(smiles)

    def load_mols_from_smiles_file(self, path, verbose=1, embed=True):
        '''Method to load mols from files of smiles. Every smiles (or smiles vec) should be in a new line'''
        if not os.path.isfile(path):
            raise FileNotFoundError("The specified file doesn't exist")
        if verbose == 1:
            print("Loading SMILES from file...")
        with open(path, "r") as f:
            for line in f.readlines():
                self.load_mol_from_smiles(line[:-1])
        if embed is True:
            self.embed_smiles(use_encoding_model=False, verbose=verbose)
            self._embeded = True

    def load_encoding_model(self, model: mlutils.Model, EPOCHS=10, verbose=1):
        if model._autoencoder is None:
            raise Warning("No support for autotrain with non-autoencoder models")
        self.encoding_model = model

    def load_filtering_strategy(self, filter_strategy: mlutils.FilterStrategies):
        self.filter_strategy = lambda reps: filter_strategy.strategy(reps)

    def save_reps_to_file(self, filename):
        with open(filename, "w") as f:
            for rep in self._reps:
                for r in rep:
                    f.write(str(r))
                    f.write(" ")
                f.write("\n")

    def load_reps_from_file(self, filename):
        self._reps = []
        with open(filename, 'r') as f:
            for line in f.readlines():
                self._reps.append([float(s) for s in line.split(" ")[:-1]])
        self._reps = np.array(self._reps)
  
    def tokenize_smiles(self, verbose=1):
        if self._reps is None:
            raise Exception("Must load smiles before tokenization !")

        max_length = 0; chars = set()
        if verbose == 1:
            bar = progressbar.ProgressBar(maxval=len(self._reps) * 2, widgets=[progressbar.Bar('=', '[', ']', '.'), ' ', progressbar.Percentage()], term_width=50)
            prog_count = 0
            bar.start()
        if type(self._reps[0]) == str:
            for smi in self._reps:
                if verbose == 1:
                    prog_count += 1
                    bar.update(prog_count)
                if len(smi) > max_length:
                    max_length = len(smi)
                for c in smi:
                    chars.add(c)
        else:
            for vec in self._reps:
                if verbose == 1:
                    prog_count += 1
                    bar.update(prog_count)
                for smi in vec:
                    if len(smi) > max_length:
                        max_length = len(smi)
                    for c in smi:
                        chars.add(c)

        Dict = dict([(c, i + 2) for i,c in enumerate(chars)])

        reps = []
        if type(self._reps[0]) == str:
            for smi in self._reps:
                if verbose == 1:
                    prog_count += 1
                    bar.update(prog_count)
                rep = []
                for c in smi:
                    rep.append(Dict[c])
                rep.append(1)
                while True:
                    if len(rep) == max_length + 1:
                        break
                    rep.append(0)
                reps.append(rep)
        else:
            for vec in self._reps:
                if verbose == 1:
                    prog_count += 1
                    bar.update(prog_count)
                rep = []
                for smi in vec:
                    r = []
                    for c in smi:
                        r.append(Dict[c])
                    r.append(1)
                    while True:
                        if len(r) == max_length + 1:
                            break
                        r.append(0)
                    rep.append(r)
                reps.append(rep)
        reps = np.array(reps)
        self._reps = reps.reshape(len(reps), int(np.prod(reps.shape[1:])))
        if verbose == 1:
            bar.finish()

    def _normalize(self, vecs, axis=None, norm_type=None, use_params=True, set_params=False):
        vecs = np.array(vecs)
        if norm_type is None:
            norm_type = self.normalization_method
        if not use_params or self._normalization_params is None:
            if norm_type == 'Z_score':
                p1 = np.mean(vecs, axis=axis)
                p2 = np.std(vecs, axis=axis)
            elif norm_type == 'Unit_scale':
                p1 = np.array(np.min(vecs, axis=axis))
                p2 = np.array(np.max(vecs, axis=axis)) - p1
            if set_params:
                self._normalization_params = (axis, p1, p2)
        else:
            axis, p1, p2 = self._normalization_params
        if axis is None:
            p1 = np.array([p1 for _ in range(len(vecs))])
            p2 = np.array([p2 for _ in range(len(vecs))])
        if not axis == 0:
            p1 = np.reshape(p1, (p1.shape[-1], 1))
            p2 = np.reshape(p2, (p2.shape[-1], 1))
        return (vecs - p1) / p2

    def _reverse_normalize(self, vecs):
        vecs = np.array(vecs)
        axis, p1, p2 = self._normalization_params
        if axis is None:
            p1 = np.array([p1 for _ in range(len(vecs))])
            p2 = np.array([p2 for _ in range(len(vecs))])
        if not axis == 0:
            p1 = np.reshape(p1, (p1.shape[-1], 1))
            p2 = np.reshape(p2, (p2.shape[-1], 1))
        return vecs * p2 + p1

    def embed_smiles(self, use_encoding_model=True, normalize=True, verbose=1):
        if verbose == 1:
            print("Embedding SMILES...")
        if not self._embeded:
            if type(self._reps[0][0]) is str:
                self.tokenize_smiles(verbose=verbose)
            self._reps = np.array(self._normalize(self._reps, norm_type='Unit_scale', use_params=False))
            if use_encoding_model:
                if not self.encoding_model is None:
                    if self.encoding_model._autoencoder is None:
                        self._reps = self.encoding_model.predict(self._reps)
                    else:
                        self._reps = self.encoding_model.encode(self._reps, norm_type='Unit_scale', normalize=normalize)
                    self._reps = self._normalize(self._reps, use_params=False)
                else:
                    raise Exception("Must load an encoding model before encoding")
            self.embeded_mols = copy(self._reps)
            self._embeded = True

    def costume_embed(self, embed_func):
        '''Method for custume molecule embedding.
        ARGS:
            - embed_func: a function that takes a Mol object and returns a vector embeding of it.'''
        self._reps = []
        for mol in self.mols:
            self._reps.append(embed_func(mol))
        self._reps = np.array(self._reps)
        self.embeded_mols = copy(self._reps)
        self._embeded = True

    def load_PropCalc_functions(self, PropCalc_Instance):
        if not isinstance(PropCalc_Instance, PropCalc):
            raise ValueError("PropCalc_class must be an instance of PropCalc")
                
        self._PropCalc_class = PropCalc_Instance

    def compare_to_reference_data(self, path, outliers = None):
        '''method comparison between results from data file and comp results.
        The file must be with the same column titles as the output of the computation.
        ARGUMENTS:
            - path (str): path to the data file (should be space delimited)
            - outliers (list): outliers to ignore in comparision'''

        ref_results = pd.read_csv(path, sep = ' ')
        ref_results = ref_results.set_index("name")
        prop_dict = dict()
        for key in ref_results.columns:
            if not key == "name":
                preds = []; refs = []
                for name in list(self.results.index):
                    if name in outliers:
                        continue
                    preds.append(self.results.loc[name, key])
                    refs.append(ref_results.loc[name, key])
                    prop_dict[key] = [preds, refs]

        for idx, key in enumerate(list(prop_dict.keys())):
            preds, lits = prop_dict[key]
            preds = np.array(preds); lits = np.array(lits)
            model = LinearRegression()
            model.fit(preds.reshape(-1, 1), lits.reshape(-1, 1))
            corrs = model.predict(preds.reshape(-1, 1))
            corrs = corrs.flatten()
            error = round(np.mean(np.abs(lits - corrs)), 2)
            print(f"Average absolute fit error for {key}: {error} %")
            
            plt.figure(idx + 1)
            plt.title(key)
            plt.scatter(preds, lits, c='black')
            plt.plot(preds, corrs, '-', color='r')

        plt.show()

    def export_to_csv(self, path, add_tstamp=False, export_reps=False):
        if not os.path.isdir(path):
            raise ValueError("path must be a directory")
        if add_tstamp is True:
            tstamp = self._tstamp.strftime("%H-%M_%d-%m-%Y")
        for key, df in self.train_test_data.items():
            if not df is None:
                if add_tstamp:
                    filename = key + "_" + tstamp + ".csv"
                else:
                    filename = key + ".csv"
                filename = os.path.join(path, filename)
                if export_reps:
                    df.to_csv(filename)
                else:
                    df.drop(columns='reps').to_csv(filename)
        if not self.results is None:
            if add_tstamp:
                filename = "Results_" + tstamp + ".csv"
            else:
                filename = "Results.csv"
            filename = os.path.join(path, filename)
            self.results.to_csv(filename)
        if not self.rate_constants is None:
            if add_tstamp:
                filename = "Rate_Constants_" + tstamp + ".csv"
            else:
                filename = "Rate_Constants.csv"
            filename = os.path.join(path, filename)
            self.rate_constants.to_csv(filename)

    def load_from_csv(self, filename, load_to='prediction_train_test', test_size=0.25, add_reps=False, use_prop='all'):
        '''Method for loading data from csv files.
        ARGS:
            - filename: path for the csv file. The csv file *must* contain a 'name' column with the molecules' names.
            - load_to: where to load the csv data.
                options: results, rate_constants, prediction_train, prediction_test, prediction_train_test (splits data to train and test automatically)
            - test_size (float): size of test set, in case of a split.
            - add_reps (bool): if names of molecules match, adds the reps of the moleules in the current instance to the selected dataframe.'''
        df = pd.read_csv(filename)
        if add_reps:
            names = df['name'].values.tolist()
            if type(names[0]) is int or type(names[0]) is float:
                reps = []
                for name in names:
                    reps.append(self._reps[int(name) - 1])
            else:
                reps = []
                for name in names:
                    idx = self._mol_names.index(name)
                    reps.append(self._reps[idx])
            df['reps'] = reps
        elif 'reps' in df.columns:
            # reading the reps properly
            reps = df['reps']
            # checks if reps is list of lists
            if '[' in reps[0]:
                new_reps = []
                for rep in reps:
                    r = re.split(r" |\n", rep[1:-1])
                    r = list(filter(lambda x: not x == '', r))
                    r = [float(v) for v in r]
                    new_reps.append(r)
                df['reps'] = new_reps
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns='Unnamed: 0')
        if not use_prop == 'all':
            if not use_prop in df.columns:
                raise ValueError(f"specified prop is not in dataset, props in dataset: {df.columns}")
            if 'reps' in df.columns:
                df = df[['name', 'reps', use_prop]]
            else:
                df = df[['name', use_prop]]
        if load_to == 'results':
            df = df.set_index('name')
            self.results = df
        elif load_to == 'rate_constants':
            df = df.set_index('name')
            self.rate_constants = df
        elif load_to in list(self.train_test_data.keys()):
            self.train_test_data[load_to] = df.set_index('name')
        elif load_to == 'prediction_train_test':
            df_dict = df.to_dict(orient='records')
            train_idxs, test_idxs, y1, y2 = sklearn.model_selection.train_test_split(range(len(df_dict)), range(len(df_dict)), test_size=test_size)
            self.train_test_data['prediction_train'] = pd.DataFrame([df_dict[i] for i in train_idxs])
            self.train_test_data['prediction_test'] = pd.DataFrame([df_dict[i] for i in test_idxs])
            self.train_test_data['prediction_train'] = self.train_test_data['prediction_train'].set_index('name')
            self.train_test_data['prediction_test'] = self.train_test_data['prediction_test'].set_index('name')
        else:
            raise ValueError("Unrecognized \'load_to\' option")

    def load_predictive_model(self, model, n_voters=1, voting_model=None):
        if not self.encoding_model is None:
            if isinstance(self.encoding_model.model, mlutils.Architectures.AutoEncoders) and self.encoding_model.variational and not model.variational_input:
                raise Warning("Encoding model is variational but prediction model is not, this might lead to unwanted results.")
        if type(model) is mlutils.Model:
            if n_voters == 1:
                self.predictive_model = [model]
                self.voting_model = None
            else:
                self.predictive_model = [model.__copy__() for _ in range(n_voters)]
                if not voting_model is None:
                    self.voting_model = voting_model
                else:
                    layers = model.model.layers[1:-1] + [tf.keras.layers.Dense(n_voters, activation='sigmoid')]
                    Input = model.model.input
                    voting_model = mlutils.Architectures.GeneralANN(layers=layers, Input=Input)
                    self.voting_model = mlutils.Model(voting_model, model.optimizer, tf.keras.losses.categorical_crossentropy, 
                                                        variational_input=model.variational_input, training_epochs=model.training_epochs)
        if type(model) is list:
            self.predictive_model = model
            if self.voting_model is None:
                raise ValueError("voting_model must be defined when defining predictive_model as list.")
            self.voting_model = voting_model

    def run_PropCalc(self, verbose = 0, timeout = None, mols_for_comp=None):
        self.results = pd.DataFrame()
        self.rate_constants = pd.DataFrame()
        class TimeoutError (RuntimeError):
            pass

        def timeout_handler (signum, frame):
            raise TimeoutError()

        signal.signal (signal.SIGALRM, timeout_handler)
        if verbose > 1:
            tstamp = self._tstamp.strftime("%H-%M_%d-%m-%Y")
            print("=" * 100)
            print(" " * 25 + f"PropCalc computation started successfuly at {tstamp}")
            print("=" * 100)
            print("Preparing for Computation...")
        GlobalDict = self._PropCalc_class.GlobalPreCalc()
        counter = 0
        if mols_for_comp is None:
            mols_for_comp = range(len(self.mols))
        if verbose == 1:
            tstamp = self._tstamp.strftime("%H-%M_%d-%m-%Y")
            bar = progressbar.ProgressBar(maxval=len(mols_for_comp), widgets=[progressbar.Bar('=', '[', ']', '.'), ' ', progressbar.Percentage()], term_width=50)
            bar.start()
            ErrorsEncountered = False
        for m, mol_name in zip([self.mols[i] for i in mols_for_comp], [self._mol_names[i] for i in mols_for_comp]):
            counter += 1
            if verbose == 1:
                bar.update(counter)
            if verbose > 1:
                print(f"Calculating for {mol_name} ({counter} out of {len(mols_for_comp)})")

            if type(m) is str:
                mol = self._mol_class()
                mol.from_smiles(m)
            else:
                mol = m
            try:
                if not timeout is None:
                    signal.alarm(timeout)
                
                self._PropCalc_class.PreRunExternalSoftware(mol, mol_name, GlobalDict)
                RateDict = self._PropCalc_class.RateDict(mol, mol_name, GlobalDict)    
                PropDict = self._PropCalc_class.PropDict(RateDict)

                if verbose > 2:
                    print(" ".join([key + "=" + str(round(val, 5)) for key, val in PropDict.items()]))
                
                if verbose > 3:
                    print(" ".join([key + "=" + str(round(val, 5)) for key, val in RateDict.items()]))

                RateDict.update({"name": mol_name})
                self.rate_constants = self.rate_constants.append(RateDict, ignore_index=True)
                PropDict.update({"name": mol_name})
                self.results = self.results.append(PropDict, ignore_index=True)
            except TimeoutError:
                if verbose > 1:
                    print(f"Timeout at {mol_name}, skipping to the next compound")
            except Exception as error:
                if verbose == 1:
                    ErrorsEncountered = True
                if verbose > 1:
                    print(f"Error at {mol_name}, skipping to the next compound")
                err = error

        if verbose == 1:
            bar.finish()
            if ErrorsEncountered:
                print("Errors Encountered in calculation")
        if len(self.results) == 0:
            raise err
        
        if verbose > 1:
            tstamp = datetime.now().strftime("%H-%M_%d-%m-%Y")
            print("=" * 100)
            print(" " * 48 + "DONE" + " "*48)
            print(" " * 36 + f"Finished at {tstamp}" + " " * 35)
            print("=" * 100)
        self.rate_constants = self.rate_constants.set_index('name')
        self.results = self.results.set_index('name')

    def _gen_predictive_dataframes(self, mols_for_comp, dtype, verbose, timeout, include_reps=True):
        '''method for generating dataframes for storing predictive model train and test reps and labels (props).
        Always overwrites existing dataframes !'''
        # getting reps of mols
        if include_reps:
            if not self.encoding_model is None:
                reps = self.encoding_model.encode(self.embeded_mols)
            else:
                reps = copy(self._reps)
        # generates dataframe
        self.run_PropCalc(timeout=timeout, verbose=verbose, mols_for_comp=mols_for_comp)
        df = copy(self.results)
        self.results = None
        self.rate_constants = None
        # getting reps
        if include_reps:
            df['reps'] = [np.array(reps[idx]) for idx in mols_for_comp]
        self.train_test_data['prediction_' + dtype] = df
        
    def _gen_voting_dataframes(self):
        '''method for generating dataframes for voting model train and test reps and labels.
        Always overwrites existing dataframes'''
        if self.train_test_data['prediction_train'] is None or self.train_test_data['prediction_test'] is None:
            raise RuntimeError("Must generation prediction_train and prediction_test dataframes before generating voting data.")
        # getting data from predictive dataframes
        names_list = self.train_test_data['prediction_train'].index.values.tolist() + self.train_test_data['prediction_test'].index.values.tolist()
        train_reps, train_props = self._gen_data_from_dataframes(dtype='prediction_train', normalize=False)
        test_reps, test_props = self._gen_data_from_dataframes(dtype='prediction_test', normalize=False)
        unnormalized_props_list = np.array(train_props.tolist() + test_props.tolist()) # must use unormalized props, otherwise it writes normalized props to df
        props_list = self._normalize(unnormalized_props_list, axis=0)
        if type(train_reps) is list:
            m_list = np.array(train_reps[0].tolist() + test_reps[0].tolist())
            s_list = np.array(train_reps[1].tolist() + test_reps[1].tolist())
            reps_list = [m_list, s_list]
        else:
            reps_list = np.array(train_reps.tolist() + test_reps.tolist())
        # generating labels
        loss_vecs = []
        true_vals = tf.constant(props_list, shape=props_list.shape, dtype='float32')
        for model in self.predictive_model:
            pred_vals = tf.constant(model.predict(reps_list), shape=np.array(props_list).shape, dtype='float32')
            with tf.compat.v1.Session():
                loss_vecs.append(model.loss(true_vals, pred_vals).eval())
        loss_vecs = np.transpose(np.array(loss_vecs))
        labels_list = []
        for loss_vec in loss_vecs:
            loss_vec = loss_vec.tolist()
            labels_list.append(loss_vec.index(max(loss_vec)))
        labels_list = tf.keras.utils.to_categorical(labels_list, num_classes=loss_vecs.shape[-1])
        # reformatting reps
        if type(reps_list) is list:
            reps_list = [[m, s] for m, s in zip(m_list, s_list)]
        # writing to dataframes
        df_dict = []
        for name, rep, label, prop in zip(names_list, reps_list, labels_list, unnormalized_props_list):
            d = {'name': name, 'reps': rep, 'labels': label, 'props': prop}
            df_dict.append(d)
        train_idxs, test_idxs, y1, y2 = sklearn.model_selection.train_test_split(range(len(names_list)), range(len(names_list)), test_size=0.25)
        self.train_test_data['voting_train'] = pd.DataFrame([df_dict[i] for i in train_idxs])
        self.train_test_data['voting_test'] = pd.DataFrame([df_dict[i] for i in test_idxs])
        self.train_test_data['voting_train'] = self.train_test_data['voting_train'].set_index('name')
        self.train_test_data['voting_test'] = self.train_test_data['voting_test'].set_index('name')
        
    def _gen_data_from_dataframes(self, dtype=None, normalize=True, output_props=False):
        '''method for reading the dataframes in self.train_test_data and generating data for further manipulation (model training, visualisation...)
        ARGS:
            dtype: type of data to generate, options: prediction_train, prediction_test, voting_train, voting_test (like the keys in self.train_test_data)
            normalize: normalize the outputed properties
            output_props: for voting models, weather to output labels or properties'''
        df = self.train_test_data[dtype]
        # getting reps
        if not 'reps' in df.columns:
            if not self.encoding_model is None:
                if not self.encoding_model._trained:
                    raise Warning("Encoding model is not trained, it might cause to unwanted results.")
                self._reps = self.encoding_model.encode(self.embeded_mols)
            else:
                self._reps = self.embeded_mols
            reps = []
            for name, rep in zip(self._mol_names, self._reps):
                if name in df.index:
                    reps.append(np.array(rep))
            reps = np.array(reps)
        else:
            reps = np.array([np.array(r) for r in df['reps'].values.tolist()])
        if len(reps.shape) > 2:
            if self.predictive_model.variational_input:
                m = [np.array(rep[0]) for rep in reps]
                s = [np.array(rep[1]) for rep in reps]
                reps = [m, s]
            else:
                reps = [rep[0] for rep in reps]
                reps = np.array(reps)
        else:
            reps = np.array(reps)
        # getting labels / props
        if 'voting' in dtype:
            if output_props:
                props = np.array(df['props'].values.tolist())
            else:
                props = np.array(df['labels'].values.tolist())
        else:
            props = np.array(df.drop(columns='reps').values.tolist())
        # normalizing
        if normalize:
            # getting normalization params
            if self._normalization_params is None:
                if self.train_test_data['prediction_train'] is None:
                    raise RuntimeError("Could not normalize properties without prediction_train data. Please load prediction_train data or generate one.")
                p = np.array(self.train_test_data['prediction_train'].drop(columns='reps').values.tolist())
                if props.shape[-1] == 1:
                    self._normalize(p, set_params=True)
                else:
                    self._normalize(p, axis=0, set_params=True)
            # normalizing props
            if props.shape[-1] == 1:
                props = self._normalize(props)
            else:
                props = self._normalize(props, axis=0)
        return reps, np.array(props)
            
    def _eval_encoding(self, reps, num_samples: int=1000):
        if not num_samples is None:
            reps = np.array([reps[np.random.randint(0, len(reps) - 1)] for _ in range(num_samples)])
        dist_mat = np.sum((reps[:, np.newaxis, :] - reps[np.newaxis, :, :]) ** 2, axis = -1)
        dist_mat = np.sqrt(dist_mat)
        dists = np.tril(dist_mat, 0)[np.nonzero(np.tril(dist_mat, 0))].flatten()
        avg = np.mean(dists)
        std = np.std(dists)
        return avg + 0.1 / std

    def predict(self, rep, verbose=1):
        if len(rep.shape) == 1:
            rep = [rep]
        if self.voting_model is None:
            res = self.predictive_model[0].predict(rep)
        else:
            res = []
            if verbose > 0:
                bar = progressbar.ProgressBar(maxval=len(rep), widgets=[progressbar.Bar('=', '[', ']', '.'), ' ', progressbar.Percentage()], term_width=50)
                bar.start()
                counter = 0
            for r in rep:
                if verbose > 0:
                    counter += 1
                    bar.update(counter)
                model_idx = self.voting_model.predict(np.array([r]))[0].tolist().index(max(self.voting_model.predict(np.array([r]))[0]))
                res.append(self.predictive_model[model_idx].predict(np.array([r])))
            if verbose > 0:
                bar.finish()
        return res
    
    def run_ml(self, verbose=1):
        if not self.encoding_model is None:
            if not self._embeded:
                raise RuntimeError("Mols must be embedded before autoencoder training. Please use embed_smiles or costume_embed.")
            if not self.encoding_model._trained:
                if verbose > 0:
                    print("Optimizing AutoEncoder...")
                self.encoding_model.train(self.embeded_mols, self.embeded_mols, verbose=min(max(0, verbose - 1), 1))
                if self.encoding_opt_iterations > 1:
                    encoding_descr = self._eval_encoding(self.encoding_model.encode(self.embeded_mols))
                    if verbose > 0:
                        print("Iteration 1 descriptor value:", encoding_descr)
                    for _ in range(self.encoding_opt_iterations - 1):
                        if verbose > 0:
                            print("Running iteration", _ + 2)
                        new_model = self.encoding_model.__copy__()
                        new_model.reinitialize()
                        new_model.train(self.embeded_mols, self.embeded_mols, verbose=min(max(0, verbose - 1), 1))
                        new_descr = self._eval_encoding(new_model.encode(self.embeded_mols))
                        if verbose > 0:
                            print("Iteration", _ + 2, "descriptor value:", new_descr)
                        if new_descr < encoding_descr:
                            encoding_descr = new_descr
                            self.encoding_model = new_model
            self._reps = self.encoding_model.encode(self.embeded_mols)

        if self.mols_for_comp is None and not self.filter_strategy is None:
            if verbose > 0:
                print("Filtering Molecules...")
            self.mols_for_comp = self.filter_strategy(self._reps)
            if verbose > 0:
                print("Number of filtered mols:", len(self.mols_for_comp))

        if not self.predictive_model is None and not all([self.predictive_model[i]._trained for i in range(len(self.predictive_model))]):
            # data generation for train and test
            if self.train_test_data['prediction_train'] is None:
                if verbose > 0:
                    print("Generating train data...")
                self._gen_predictive_dataframes(self.mols_for_comp, 'train', verbose - 1 if verbose > 1 else verbose, self.PropCalc_timeout)
            if self.train_test_data['prediction_test'] is None:
                if verbose > 0:
                    print("Generating test data...")
                self._gen_predictive_dataframes([np.random.randint(0, len(self.mols) - 1) for _ in range(self.num_of_test_mols)], 'test', verbose - 1 if verbose > 1 else verbose, self.PropCalc_timeout)
            train_reps, train_props = self._gen_data_from_dataframes(dtype='prediction_train', normalize=True)
            test_reps, test_props = self._gen_data_from_dataframes(dtype='prediction_test', normalize=True)
            # training predictive models
            for idx, model in enumerate(self.predictive_model):
                if not model._trained:
                    if verbose > 0:
                        print("Training predictive model", idx + 1)
                    model.train(train_reps, train_props, verbose=min(max(0, verbose - 1), 1))
                    if verbose == 1:
                        print("Final model loss:", model.test(train_reps, train_props, verbose=0))
            # training voting model
            if not self.voting_model is None:
                # vote_train_reps, vote_train_labels = self.gen_data(gen_voting_data=True, _type='train', verbose=verbose)
                self._gen_voting_dataframes()
                vote_train_reps, vote_train_labels = self._gen_data_from_dataframes(dtype='voting_train', normalize=False)
                if verbose > 0:
                    print("Training voting model...")
                self.voting_model.train(vote_train_reps, vote_train_labels, verbose=min(max(0, verbose - 1), 1))
                if verbose == 1:
                    print("Final model loss:", self.voting_model.test(vote_train_reps, vote_train_labels, verbose=0))
            self.results = pd.DataFrame(self._reverse_normalize(self.predict(self._reps)), columns=['prop'], index=self._mol_names)
    
    def visualize(self, plot_encoding=False, plot_filtered_mols=False, add_random_points=False, plot_prediction_train=False, plot_prediction_test=False, save_to_path=None, show=True):
        if not save_to_path is None and not os.path.isdir(save_to_path):
            raise ValueError("save_to_path must be a directory path")
        if plot_encoding or plot_filtered_mols:
            if not self._embeded:
                raise RuntimeError("Mols must be embeded to before encoding. Please use embed_smiles or custume_embed methods and try again.")
            xencoded = self.encoding_model.encode(self.embeded_mols)
            xencoded = self.encoding_model.plot_encoding(xencoded, alpha=min(500 / len(self.embeded_mols), 1), encode=False, show=False, return_x=True)
            plt.title("Encoding and Filtering")
            if plot_filtered_mols:
                if self.mols_for_comp is None:
                    mols_for_comp = self.filter_strategy(xencoded)
                else:
                    mols_for_comp = self.mols_for_comp
                points = np.array([xencoded[i] for i in mols_for_comp])
                self.encoding_model.plot_encoding(points, alpha=1, prop='r', encode=False, show=False, new_fig=False)
            if add_random_points:
                rnd_ps = np.array([xencoded[np.random.randint(0, len(xencoded) - 1)] for _ in range(len(mols_for_comp))])
                self.encoding_model.plot_encoding(rnd_ps, alpha=1, prop='g', encode=False, show=False, new_fig=False)
            if not save_to_path is None:
                plt.savefig(os.path.join(save_to_path, 'encoding_plot.png'), transparent=True)
        if plot_prediction_train:
            if self.voting_model is None:
                x, true_vals = self._gen_data_from_dataframes(dtype='prediction_train', normalize=True)                
                true_vals = np.reshape(true_vals, (true_vals.shape[0], true_vals.shape[-1]))
            else:
                x, true_vals = self._gen_data_from_dataframes(dtype='voting_train', normalize=True, output_props=True)
            pred_vals = np.array(self.predict(x))
            pred_vals = np.reshape(pred_vals, (pred_vals.shape[0], pred_vals.shape[-1]))
            if pred_vals.shape[-1] == 1:
                plt.figure()
                plt.title("Train Set Results")
                plt.scatter(pred_vals, true_vals, c='b', alpha=min(1, 500 / len(pred_vals)))
                plt.plot(pred_vals, pred_vals, c='r')
                if not save_to_path is None:
                    plt.savefig(os.path.join(save_to_path, 'train_fit_plot.png'), transparent=True)
            else:
                pred_vals = np.transpose(pred_vals)
                true_vals = np.transpose(true_vals)
                for i in range(pred_vals.shape[0]):
                    preds = pred_vals[i]
                    trues = true_vals[i]
                    plt.figure()
                    plt.title("Prop " + str(i + 1) + " Train Set Results")
                    plt.scatter(preds, trues, c='b', alpha=min(1, 500 / len(pred_vals)))
                    plt.plot(preds, preds, c='r')
                    if not save_to_path is None:
                        plt.savefig(os.path.join(save_to_path, 'train_fit_prop' + str(i + 1) + '.png'), transparent=True)
        if plot_prediction_test:
            if self.voting_model is None:
                x, true_vals = self._gen_data_from_dataframes(dtype='prediction_test', normalize=True)                
                true_vals = np.reshape(true_vals, (true_vals.shape[0], true_vals.shape[-1]))
            else:
                x, true_vals = self._gen_data_from_dataframes(dtype='voting_test', normalize=True, output_props=True)
            pred_vals = np.array(self.predict(x))
            pred_vals = np.reshape(pred_vals, (pred_vals.shape[0], pred_vals.shape[-1]))
            if pred_vals.shape[-1] == 1:
                plt.figure()
                plt.title("Test Set Results")
                plt.scatter(pred_vals, true_vals, c='b', alpha=min(1, 500 / len(pred_vals)))
                plt.plot(pred_vals, pred_vals, c='r')
                if not save_to_path is None:
                    plt.savefig(os.path.join(save_to_path, 'test_fit_plot.png'), transparent=True)
            else:
                pred_vals = np.transpose(pred_vals)
                true_vals = np.transpose(true_vals)
                for i in range(pred_vals.shape[0]):
                    preds = pred_vals[i]
                    trues = true_vals[i]
                    plt.figure()
                    plt.title("Prop " + str(i + 1) + " Test Set Results")
                    plt.scatter(preds, trues, c='b', alpha=min(1, 500 / len(pred_vals)))
                    plt.plot(preds, preds, c='r')
                    if not save_to_path is None:
                        plt.savefig(os.path.join(save_to_path, 'train_fit_prop' + str(i + 1) + '.png'), transparent=True)
        if show:
            plt.show()

    def calc_statistics(self, show=True, write_to_file=None, format_columns=True):
        def calc_rmse(true, pred):
            pred = np.array(pred)
            true = np.array(true)
            return np.sqrt(np.mean(np.square(pred - true)))
        
        def calc_mare(true, pred):
            '''calculate the mean absolute relative error (mare)'''
            pred = np.array(pred)
            true = np.array(true)
            return np.mean(np.abs((pred - true) / true))

        def calc_mae(true, pred):
            '''calculate the mean absolute error (mae)'''
            pred = np.array(pred)
            true = np.array(true)
            return np.mean(np.abs(pred - true))

        predictive_models_stats = pd.DataFrame()
        train_x, train_true_vals = self._gen_data_from_dataframes(dtype='prediction_train')
        test_x, test_true_vals = self._gen_data_from_dataframes(dtype='prediction_test')
        for idx, model in enumerate(self.predictive_model):
            Dict = dict()
            train_pred_vals = model.predict(train_x)
            test_pred_vals = model.predict(test_x)
            Dict['model_id'] = 'model ' + str(idx + 1)
            Dict['rmse, train'] = calc_rmse(train_true_vals, train_pred_vals)
            Dict['rmse, test'] = calc_rmse(test_true_vals, test_pred_vals)
            Dict['mean absolute error, train'] = calc_mae(train_true_vals, train_pred_vals)
            Dict['mean absolute error, test'] = calc_mae(test_true_vals, test_pred_vals)
            Dict['mean absolute relative error, train'] = calc_mare(train_true_vals, train_pred_vals)
            Dict['mean absolute relative error, test'] = calc_mare(test_true_vals, test_pred_vals)
            predictive_models_stats = predictive_models_stats.append(Dict, ignore_index=True)
        if not self.voting_model is None:
            train_x, train_true_vals = self._gen_data_from_dataframes(dtype='voting_train', output_props=True)
            test_x, test_true_vals = self._gen_data_from_dataframes(dtype='voting_test', output_props=True)
            train_pred_vals = self.predict(train_x)
            test_pred_vals = self.predict(test_x)
            Dict = dict()
            Dict['model_id'] = 'voting model'
            Dict['rmse, train'] = calc_rmse(train_true_vals, train_pred_vals)
            Dict['rmse, test'] = calc_rmse(test_true_vals, test_pred_vals)
            Dict['mean absolute error, train'] = calc_mae(train_true_vals, train_pred_vals)
            Dict['mean absolute error, test'] = calc_mae(test_true_vals, test_pred_vals)
            Dict['mean absolute relative error, train'] = calc_mare(train_true_vals, train_pred_vals)
            Dict['mean absolute relative error, test'] = calc_mare(test_true_vals, test_pred_vals)
            predictive_models_stats = predictive_models_stats.append(Dict, ignore_index=True)
        # formating dataframe
        if format_columns:
            a = predictive_models_stats.columns.str.split(', ', expand=True).values
            predictive_models_stats.columns = pd.MultiIndex.from_tuples([('', x[0]) if pd.isnull(x[1]) else x for x in a])
            cols = copy(predictive_models_stats.columns.values.tolist())
            cols.remove(('', 'model_id'))
            cols = [('', 'model_id')] + cols
            predictive_models_stats = predictive_models_stats[cols]
        if show:
            print(predictive_models_stats)
        if not write_to_file is None:
            predictive_models_stats.to_csv(write_to_file)
        return predictive_models_stats
            