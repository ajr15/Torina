import sys; sys.path.append('../../')
import numpy as np
import os
from copy import copy 
import warnings; warnings.simplefilter(action = 'ignore', category=FutureWarning)
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd

from Torina.Data.Objects import SMILES
from Torina.Data.Base import flatten, generate_data_using_comp_protocol
from Torina.Data.utils import *
from Torina.Model.utils import KerasArchitectures
from Torina.Model.Objects import *

def ml_run(input_shape, train_inputs, train_labels, lr=0.001, nn_depth=3, epochs=500):
    input_size = np.prod(input_shape)
    
    model_layers = [tf.keras.layers.Flatten(), tf.keras.layers.Dense(input_size, activation='linear')]
    for i in range(nn_depth - 1):
        model_layers = model_layers + [tf.keras.layers.Dense(round(input_size / nn_depth) * (i + 1), activation='relu')]
    model_layers = model_layers + [tf.keras.layers.Dense(1, activation='sigmoid')]
    
    predictive_model = KerasArchitectures.GeneralANN(input_shape=input_shape, layers=model_layers)
    NN = KerasNN(predictive_model, tf.keras.optimizers.SGD(lr=lr), tf.keras.losses.binary_crossentropy)
    NN.train(train_inputs, train_labels, epochs=epochs, verbose=1)
    return NN

def select_idxs(max_idx, sample_size):
    idxs = []
    while True:
        i = np.random.randint(0, max_idx)
        if not i in idxs:
            idxs.append(i)
        if len(idxs) == sample_size:
            break
    return idxs

def data_prep(sample_size=15000):
    df = pd.read_csv('~/Documents/PYTHON_PROJECTS/CoSToDi_PAPER/database.csv')
    idxs = select_idxs(len(df) - 1, sample_size)
    df = df.iloc[idxs, :]
    data = SMILES()
    data.load_inputs_from_text('\n'.join(df['SMILES'].values), sep='chars') 
    data.vectorized_labels = df['Success Rate'].values
    return data

def get_binary_descriptors(pred, true):
    fp = 0
    tn = 0
    tp = 0
    fn = 0
    correct = 0
    count0 = 0
    count1 = 0
    for p, t in zip(pred, true):
        p = int(round(p[0]))
        t = int(t)
        if t == 1 and p == 1:
            tp += 1
            correct += 1
            count1 += 1
        elif t == 1 and p == 0:
            fn += 1
            count1 += 1
        elif t == 0 and p == 1:
            fp += 1
            count0 += 1
        elif t == 0 and p == 0:
            tn += 1
            correct += 1
            count0 += 1

    return {"0 count": count0,
            "1 count": count1,
            "accuracy": correct / len(true),
            "fp_rate": fp / (fp + tn),
            "fn_rate": fn / (fn + tp)}

def run_calc(labeled_data, tokenization_method, train_idxs, test_idxs, lr=0.001, nn_depth=3, epochs=500):
    labeled_data.vectorized_inputs = labeled_data.pad_data("_", " ", pad='inputs')
    print("setting tokenization function...")
    if tokenization_method == 'custodi':
        labeled_data.set_custodi_tokenization_func(use_idxs=train_idxs)
    elif tokenization_method == 'word':
        labeled_data.set_word_tokenization_func()
    else:
        raise ValueError("Unrecognized tokenization method. Allowed methods \'custodi\', \'word\'")
    print('tokenizing...')
    # from Torina.Data.utils import tokenize_from_function
    # print(labeled_data.tokenization_dict)
    # print(tokenize_from_function(labeled_data.vectorized_inputs[0], labeled_data.tokenization_func, keep_shape=False))
    # import sys; sys.exit(0)
    labeled_data.vectorized_inputs = labeled_data.tokenize('vectorized_inputs', keep_shape=False)
    print("running ML...")
    input_shape = np.array(labeled_data.vectorized_inputs[0]).shape
    NN = ml_run(input_shape, np.array([labeled_data.vectorized_inputs[i] for i in train_idxs]), np.array([labeled_data.vectorized_labels[i] for i in train_idxs]), lr, nn_depth, epochs)
    train_pred = NN.predict(np.array([labeled_data.vectorized_inputs[i] for i in train_idxs]))
    test_pred = NN.predict(np.array([labeled_data.vectorized_inputs[i] for i in test_idxs]))
    print("calculating descriptors...")
    
    train_descrps = get_binary_descriptors(train_pred, [labeled_data.vectorized_labels[i] for i in train_idxs])
    test_descrps = get_binary_descriptors(test_pred, [labeled_data.vectorized_labels[i] for i in test_idxs])

    return train_descrps, test_descrps

def main():
    print("preparing data...")
    data = data_prep()
    
    tokenization_methods = ['custodi', 'word']
    results = pd.DataFrame()

    #train_sizes = [0.001, 0.005, 0.01, 0.05, 0.1]
    train_sizes = [0.1]
    train_idxs = []
    for train_size in train_sizes:
        print("setting train idxs...")
        while True:
            i = np.random.randint(0, len(data.inputs))
            if not i in train_idxs:
                train_idxs.append(i)
            if len(train_idxs) == round(train_size * len(data.inputs)):
                break
        print("setting test idxs...")
        test_idxs = [i for i in range(len(data.inputs)) if not i in train_idxs]
        for tokenization_method in tokenization_methods:
            print(f"running cacluation with train size {train_size} and {tokenization_method} tokenization method...")
            train_descrps, test_descrps = run_calc(data, tokenization_method, train_idxs, test_idxs, nn_depth=0, epochs=500)
            Dict = {"tokenization method": tokenization_method, "train size": train_size}
            train_descrps = dict([("train_" + k, v) for k, v in train_descrps.items()])
            Dict.update(train_descrps)
            test_descrps = dict([("test_" + k, v) for k, v in test_descrps.items()])
            Dict.update(test_descrps)
            results = results.append(Dict, ignore_index=True)
    results.to_csv(os.path.join('~/Documents/PYTHON_PROJECTS/CoSToDi_PAPER/', 'results.csv'))
            
if __name__ == '__main__':
    main()
    