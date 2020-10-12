import sys; sys.path.append('../')
import numpy as np
import os
from copy import copy 
import warnings; warnings.simplefilter(action = 'ignore', category=FutureWarning)
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from Torina.Data.Objects import SmilesVec
from Torina.Data.Base import flatten, generate_data_using_comp_protocol
from Torina.Molecule.Objects import SubsMol, IsoGen
from Torina.CompProtocol.Objects import RDKitPropCalc
from Torina.Data.utils import *
from Torina.Model.utils import KerasArchitectures
from Torina.Model.Objects import *

class SubsBenzene (SubsMol):

    core_struct = '*c1c(*)c(*)c(*)c(*)c1*'
    binding_positions = {'pos1': [0, 1, 2, 3, 4, 5]}
    core_charge = 0

class SubsButene (SubsMol):

    core_struct = "*C(*)(*)C(/*)=C(/*)C(*)(*)*"
    binding_positions = {'pos1': [0, 1, 2, 3]}

class Corrole (SubsMOl):

    core_struct = "*c1c(*)c2n3c1C1=[N+]4C(=C(*)c5c(*)c(*)c6n5*43(*)(*)N3C(C(*)=C(*)C3=C6*)=C2*)C(*)=C1*"
    binding_positions = {
        'meso': [],
        'beta': [],
        'metal': [],
        'ligand': []
    }
    core_charge = -3

def iso_gen(iso_file, parent_specie, position_args, *args, **kwargs):
    if not os.path.isfile(iso_file):
        mol = parent_specie(position_args)
        isos = IsoGen(mol, *args, **kwargs)
    with open(iso_file, "r") as f:
        print("Number of generated isomers: %s" % len(f.readlines()))
        return len(f.readlines())

def ml_run(input_shape, train_inputs, train_labels, lr=0.001, nn_depth=3, epochs=500):
    input_size = np.prod(input_shape)
    
    model_layers = [tf.keras.layers.Flatten(), tf.keras.layers.Dense(input_size, activation='linear')]
    for i in range(nn_depth - 1):
        model_layers = model_layers + [tf.keras.layers.Dense(round(input_size / nn_depth) * (i + 1), activation='relu')]
    model_layers = model_layers + [tf.keras.layers.Dense(1, activation='tanh'), tf.keras.layers.Dense(1, activation='linear')]
    
    predictive_model = KerasArchitectures.GeneralANN(
    input_shape=input_shape,
    layers=model_layers)
    NN = KerasNN(predictive_model, tf.keras.optimizers.Adam(lr=0.001), tf.keras.losses.mean_squared_error)
    NN.train(train_inputs, train_labels, epochs=epochs, verbose=1)
    return NN

def data_prep(parent_specie, iso_file, prop_func, subset_size='all', nprocs=1):
    data = SmilesVec(parent_specie)
    data.load_inputs_from_file(iso_file)
    data.vectorized_inputs = data.pad_data("_", " ", pad='inputs')
    if subset_size == 'all':
        subset_idxs = [i for i in range(len(data.inputs))]
    else:
        subset_idxs = choose(len(data.inputs), [subset_size])
    labeled_data = generate_data_using_comp_protocol(data, RDKitPropCalc(prop_func), input_idxs=subset_idxs, nprocs=nprocs)
    labeled_data.vectorized_labels = train_data.vectorize_labels()
    labeled_data.noramlize_vectors(normalize='labels', method='z_score')
    return labeled_data

def run_calc(labeled_data, tokenization_method, train_idxs, test_idxs, image_path, image_prefix='', lr=0.001, nn_depth=3, epochs=500):
    if tokenization_method == 'custodi':
        labeled_data.set_custodi_tokenization_func(use_idxs=train_idxs)
    elif tokenization_method == 'word':
        labeled_data.set_word_tokenization_func()
    else:
        raise ValueError("Unrecognized tokenization method. Allowed methods \'custodi\', \'word\'")
    labeled_data.tokenize('vectorized_inputs')
    input_shape = np.array(labeled_data.vectorized_inputs[0]).shape
    NN = ml_run(input_shape, np.array([labeled_data.vectorized_inputs[i] for i in train_idxs]), [labeled_data.vectorized_labels[i] for i in train_idxs], lr, nn_depth, epochs)
    test_err = NN.test([labeled_data.vectorized_inputs[i] for i in test_idxs], [labeled_data.vectorized_labels[i] for i in test_idxs])
    train_err = NN.test([labeled_data.vectorized_inputs[i] for i in train_idxs], [labeled_data.vectorized_labels[i] for i in train_idxs])
    NN.plot_fit(np.array([labeled_data.vectorized_inputs[i] for i in train_idxs]), [labeled_data.vectorized_labels[i] for i in train_idxs], show=False) # taining set
    plt.gcf()
    plt.savefig(os.path.join(image_path, prefix + 'train.png'))
    NN.plot_fit(np.array([labeled_data.vectorized_inputs[i] for i in test_idxs]), [labeled_data.vectorized_labels[i] for i in test_idxs], show=False) # testing set
    plt.gcf()
    plt.savefig(os.path.join(image_path, prefix + 'test.png'))
    return test_err, train_err
    