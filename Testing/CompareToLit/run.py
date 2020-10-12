import sys; sys.path.append('/home/shachar/Documents')
from Torina.Data.Objects import SMILES
from Torina.Data.utils import inverse_zscore_normalization
from Torina.Molecule.Base import BaseMol
from Torina.Molecule.utils import calculate_ecfp4
from Torina.Model.Objects import KernalRidge

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def select_idxs(max_idx, sample_size):
    idxs = []
    while True:
        i = np.random.randint(0, max_idx)
        if not i in idxs:
            idxs.append(i)
        if len(idxs) == sample_size:
            break
    return idxs

def gen_train_test_idxs(train_size, data_size, train_idxs):
    print("setting train idxs...")
    while True:
        if len(train_idxs) == round(train_size * data_size):
            break
        i = np.random.randint(0, data_size)
        if not i in train_idxs:
            train_idxs.append(i)
    print("setting test idxs...")
    test_idxs = [i for i in range(data_size) if not i in train_idxs]
    return train_idxs, test_idxs

def _safe_calc_diff_with_func(pred, true, func):
    s = 0
    for p, t in zip(pred, true):
        val = func(p, t)
        if not val == [np.inf] and not val == np.inf:
            s = s + val
    return s / len(pred)

def calc_descps(pred, true, prefix='', un_normalize_params=None):
    if not un_normalize_params == None:
        pred = inverse_zscore_normalization(pred, *un_normalize_params)
        true = inverse_zscore_normalization(true, *un_normalize_params)
    descps = {}
    rmse_func = lambda p, t: np.square(p - t)
    descps[prefix + 'rmse'] = np.sqrt(_safe_calc_diff_with_func(pred, true, rmse_func))
    mae_func = lambda p, t: np.abs(p - t)
    descps[prefix + 'mae'] = _safe_calc_diff_with_func(pred, true, mae_func)
    mare_func = lambda p, t: np.abs((p - t) / t)
    descps[prefix + 'mare'] = _safe_calc_diff_with_func(pred, true, mare_func)
    # corrects shape of descrps values in case a numpy array is returned
    for k, v in descps.items():
        try:
            descps[k] = v[0]
        except IndexError:
            continue
    return descps

def data_prep(label, sample_size, normalization_method='z_score'):
    print("Reading csv...")
    df = pd.read_csv('../CoSToDi_PAPER/QM9/database.csv')
    if not sample_size == None:
        idxs = select_idxs(len(df) - 1, sample_size)
        df = df.iloc[idxs, :]
    df = df[df['SMILES'] != 'C[C@H](C=C=C(O)=O)C[NH3]']
    df = df[df['SMILES'] != 'C[C@@H]([NH3])c1noc(=O)n1']
    df = df[df['SMILES'] != 'NC[C@H]([NH3])c1nnnn1']
    print("Parsing data...")
    data = SMILES()
    data.load_inputs_from_text('\n'.join(df['SMILES'].values), sep='chars')
    data.vectorized_labels = df[label].values
    data.noramlize_vectors(normalize='labels', method=normalization_method)
    data.vectorized_labels = data.vectorized_labels.tolist()
    print("Vectorizing inputs...")
    data.vectorized_inputs = []
    for i, smiles in enumerate(data.inputs):
        mol = BaseMol()
        mol.from_str(''.join(smiles))
        try:
            data.vectorized_inputs.append(calculate_ecfp4(mol))
        except Exception as err:
            print("Errors encountered with", ''.join(smiles))
            data.vectorized_labels.pop(i)
    return data

def run_ml(data, train_idxs, test_idxs, prefix, un_normalize=True, alpha=0.01, kernel='laplacian'):
    model = KernalRidge(alpha, kernel=kernel)
    model.train([data.vectorized_inputs[i] for i in train_idxs], [data.vectorized_labels[i] for i in train_idxs])
    pred = model.predict(data.vectorized_inputs)
    if un_normalize:
        descrps = calc_descps([pred[i] for i in train_idxs], [data.vectorized_labels[i] for i in train_idxs], 'train_', data._norm_params)
        descrps.update(calc_descps([pred[i] for i in test_idxs], [data.vectorized_labels[i] for i in test_idxs], 'test_', data._norm_params))
    else:
        descrps = calc_descps([pred[i] for i in train_idxs], [data.vectorized_labels[i] for i in train_idxs], 'train_')
        descrps.update(calc_descps([pred[i] for i in test_idxs], [data.vectorized_labels[i] for i in test_idxs], 'test_'))
    print("*" * 30 + " " + prefix + " " + "*" * 30)
    for k, v in descrps.items():
        print(k, '\t', v)
    model.plot_fit([data.vectorized_inputs[i] for i in train_idxs], [data.vectorized_labels[i] for i in train_idxs], show=False, alpha=0.5)
    plt.savefig('./' + prefix + '_train.png')
    model.plot_fit([data.vectorized_inputs[i] for i in test_idxs], [data.vectorized_labels[i] for i in test_idxs], show=False, alpha=0.5)
    plt.savefig('./' + prefix + '_test.png')
    return descrps

def main():
    labels = ['dipole moment' ,' isotropic polarizability','homo' ,'electronic spatial extent' ,'heat capacity']
    sample_size = 7000
    train_size = 0.9
    df = pd.DataFrame()
    for label in labels:
        print("=" * 50)
        print(" " * 10 + "Running for " + label)
        print("=" * 50)
        data = data_prep(label, sample_size)
        train_idxs, test_idxs = gen_train_test_idxs(train_size, len(data.inputs), [])
        descrps = run_ml(data, train_idxs, test_idxs, label)
        descrps['label'] = label
        df = df.append(descrps, ignore_index=True)
        df.to_csv('./results.csv')

if __name__ == '__main__':
    main()