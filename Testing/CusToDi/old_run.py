import sys; sys.path.append('../')
import numpy as np
import os
from sklearn.linear_model import Lasso as LinearModel
from rdkit.Chem.rdMolDescriptors import CalcLabuteASA as prop_func
from copy import copy 
import warnings; warnings.simplefilter(action = 'ignore', category=FutureWarning)
import tensorflow as tf
import pandas as pd

from Torina.Data.Objects import SmilesVec
from Torina.Data.Base import flatten, generate_data_using_comp_protocol
from Torina.Molecule.Objects import SubsMol, IsoGen
from Torina.CompProtocol.Objects import RDKitPropCalc
from Torina.Data.utils import *
from Torina.Model.utils import KerasArchitectures
from Torina.Model.Objects import *

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

def find_idx_dict(inputs):
    char_set = set()
    coupling_set = set()
    for vec in inputs:
        vec = flatten(vec)
        for idx in range(len(vec)):
            char_set.add(vec[idx])
            try:
                coupling_set.add(vec[idx] + vec[idx + 1])
            except IndexError:
                pass
    idx_dict = {}
    for idx, char in enumerate(list(char_set)):
        idx_dict[char] = idx
    for idx, chars in enumerate(list(coupling_set)):
        idx_dict[chars] = idx + len(char_set)
    return idx_dict

def find_costume_dict(inputs, labels, idx_dict):
    '''optimize parameters for costume tokenization dict'''
    # generating X data
    X = []
    for vec in inputs:
        x = np.zeros(len(idx_dict))
        vec = flatten(vec)
        for idx in range(len(vec)):
            x[idx_dict[vec[idx]]] += 1
            try:
                x[idx_dict[vec[idx] + vec[idx + 1]]] += 1
            except IndexError:
                pass
        X.append(x)
    reg = LinearModel(fit_intercept=False, alpha=0.05)
    reg.fit(X, labels)
    d = {}
    for key, c in zip(idx_dict.keys(), reg.coef_):
        d[key] = c
    return d

def tokenize_from_optimized_dict(tokenization_dict, vecs):
    tokenized_vecs = []
    for vec in vecs:
        vec = flatten(vec)
        tokenized_vec = []
        for idx in range(len(vec) - 1):
            tokenized_vec.append(tokenization_dict[vec[idx]] + tokenization_dict[vec[idx] + vec[idx + 1]])
        tokenized_vec.append(tokenization_dict[vec[-1]])
        tokenized_vecs.append(tokenized_vec)
    return np.array(tokenized_vecs)

class SubsBenzene (SubsMol):

    core_struct = '*c1c(*)c(*)c(*)c(*)c1*'
    binding_positions = {'pos1': [0, 1, 2, 3, 4, 5]}
    core_charge = 0

class SubsButene (SubsMol):

    core_struct = "*C(*)(*)C(/*)=C(/*)C(*)(*)*"
    binding_positions = {'pos1': [0, 1, 2, 3]}

def iso_gen(iso_file, parent_specie, position_args, *args, **kwargs):
    if not os.path.isfile(iso_file):
        mol = parent_specie(position_args)
        isos = IsoGen(mol, *args, **kwargs)
    with open(iso_file, "r") as f:
        print("Number of generated isomers: %s" % len(f.readlines()))
        return len(f.readlines())

def run_calc(test_set_size, train_set_size, parent_specie, iso_file):
    # setting data 
    data = SmilesVec(parent_specie)
    data.load_inputs_from_file(iso_file)
    data.vectorized_inputs = data.pad_data("_", " ", pad='inputs')
    train_idxs, test_idxs = choose(len(data.inputs), [train_set_size, test_set_size])
    train_data = generate_data_using_comp_protocol(data, RDKitPropCalc(prop_func), input_idxs=train_idxs, nprocs=1)
    train_data.vectorized_labels = train_data.vectorize_labels()
    train_data.noramlize_vectors(normalize='labels', method='z_score')
    idx_dict = find_idx_dict(data.vectorized_inputs)
    tokenization_dict = find_costume_dict([data.vectorized_inputs[i] for i in train_idxs], train_data.vectorized_labels, idx_dict)
    data.vectorized_inputs = tokenize_from_optimized_dict(tokenization_dict, data.vectorized_inputs)
    # CHECK WITHOUT INPUT NORMALIZATION !!!
    # data.noramlize_vectors(normalize='inputs', method='z_score')
    test_data = generate_data_using_comp_protocol(data, RDKitPropCalc(prop_func), input_idxs=test_idxs, nprocs=1)
    test_data.vectorized_labels = test_data.vectorize_labels()
    test_data.noramlize_vectors(normalize='labels', method='z_score')
    input_shape = np.array(data.vectorized_inputs[0]).shape
    input_size = np.prod(input_shape)
    encoding_size = 2 * int(np.floor(input_size / 3))
    data_reps = np.array(data.vectorized_inputs)
    predictive_model = KerasArchitectures.GeneralANN(
        input_shape=input_shape,
        layers=[
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(20, activation='linear'),
            tf.keras.layers.Dense(20, activation='relu'),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1, activation='tanh'),
            tf.keras.layers.Dense(1, activation='linear')
        ])
    NN = KerasNN(predictive_model, tf.keras.optimizers.Adam(lr=0.001), tf.keras.losses.mean_squared_error)
    NN.train(np.array([data_reps[i] for i in train_idxs]), train_data.vectorized_labels, epochs=500, verbose=0)
    test_err = NN.test(np.array([data_reps[i] for i in test_idxs]), test_data.vectorized_labels, verbose=0)
    print("Test set error is:", round(test_err, 4), "\n")
    # plotting results
    # NN.plot_fit(np.array([data_reps[i] for i in train_idxs]), train_data.vectorized_labels, show=False) # taining set
    # NN.plot_fit(np.array([data_reps[i] for i in test_idxs]), test_data.vectorized_labels, show=False) # testing set
    return test_err

def main():
    df = pd.DataFrame()
    print('Running for Substituted Benzene 1')
    iso_file = "./SubsBenzene1.csv"
    num_isos = iso_gen(iso_file, SubsBenzene, {'pos1': (0, 'max', 2, '[*H]')}, {'pos1': [['*O', '*CC', '*Cl', '*Br', '*F', '*CO', '*N']]}, {'pos1': [[0, 0, 0, 0, 0, 0, 0]]}, 0, export_to=iso_file, add_title=False)
    for train_size in [100, 200, 500, 1000]:
        print("Running with train size:", train_size)
        test_err = run_calc(1000, train_size, SubsBenzene, iso_file)
        df = df.append({'parent_specie': "SubsBenezene", 'num_isos': num_isos, 'train_set_size': train_size, 'test_err': test_err}, ignore_index=True)
    
    print("Running for Substituted Benzene 2")
    iso_file = "./SubsBenzene2.csv"
    num_isos = iso_gen(iso_file, SubsBenzene, {'pos1': (0, 'max', 2, '[*H]')}, {'pos1': [['*O', '*CC', '*Cl', '*Br', '*F', '*CO', '*N', "*[N+]([O-])=O", "*C=O", "*C=C", "*I"]]}, {'pos1': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]}, 0, export_to=iso_file, add_title=False)
    for train_size in [100, 200, 500, 1000]:
        print("Running with train size:", train_size)
        test_err = run_calc(1000, train_size, SubsBenzene, iso_file)
        df = df.append({'parent_specie': "SubsBenezene", 'num_isos': num_isos, 'train_set_size': train_size, 'test_err': test_err}, ignore_index=True)
    
    print("Running for substituted Butene 1")
    iso_file = "./SubsButene1.csv"
    num_isos = iso_gen(iso_file, SubsBenzene, {'pos1': (0, 'max', 2, '[*H]')}, {'pos1': [['*O', '*CC', '*Cl', '*Br', '*F', '*CO', '*N']]}, {'pos1': [[0, 0, 0, 0, 0, 0, 0]]}, 0, export_to=iso_file, add_title=False)
    for train_size in [100, 200, 500, 1000]:
        print("Running with train size:", train_size)
        test_err = run_calc(1000, train_size, SubsBenzene, iso_file)
        df = df.append({'parent_specie': "SubsButene", 'num_isos': num_isos, 'train_set_size': train_size, 'test_err': test_err}, ignore_index=True)

    print("Running for susbtituted Butene 2")
    iso_file = "./SubsButene2.csv"
    num_isos = iso_gen(iso_file, SubsBenzene, {'pos1': (0, 'max', 2, '[*H]')}, {'pos1': [['*O', '*CC', '*Cl', '*Br', '*F', '*CO', '*N', "*C=O", "*CCC", "*C#N"]]}, {'pos1': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]}, 0, export_to=iso_file, add_title=False)
    for train_size in [100, 200, 500, 1000]:
        print("Running with train size:", train_size)
        test_err = run_calc(1000, train_size, SubsBenzene, iso_file)
        df = df.append({'parent_specie': "SubsButene", 'num_isos': num_isos, 'train_set_size': train_size, 'test_err': test_err}, ignore_index=True)

    print("Finnished Computation !!!")
    df.to_csv("./results.csv")

if __name__ == '__main__':
    main()