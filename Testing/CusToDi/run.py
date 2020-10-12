import os
import sys; sys.path.append('../../')
import pandas as pd
from rdkit.Chem.rdMolDescriptors import CalcLabuteASA, CalcKappa1, CalcChi0n, CalcExactMolWt
prop_funcs = [CalcLabuteASA, CalcKappa1, CalcChi0n, CalcExactMolWt]
prop_names = ["CalcLabuteASA", "CalcKappa1", "CalcChi0n", "CalcExactMolWt"]

from utils import *
from Data.utils import choose

def mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def stdout(msg):
    print(msg)
    with open('./stdout.txt', "a") as f:
        f.write(msg)

mkdir('./isos')
mkdir('./images')

f = open('./stdout.txt', "w")
f.close()

parent_species = [SubsBenzene, SubsButene]
specie_names = ["SubsBenzene", "SubsButene"]
subs_dict = {'pos1': [['*O', '*CC', '*Cl', '*Br', '*F', '*CO', '*N', "*C=O", "*CCC", "*C#N", "*I", "*C(C)(C)C", "*C(F)(F)F"]]}
subs_charges = {'pos1': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]}
iso_gen_arguments = {"SubsBenzene": ({'pos1': (0, 'max', 2, '[*H]')}, subs_dict, subs_charges, 0),
                    "SubsButene": ({'pos1': (0, 'max', 2, '[*H]')}, subs_dict, subs_charges, 0)}

nprocs = 1
df = pd.DataFrame()
for prop_func, prop_name in zip(prop_funcs, prop_names):
    for parent_specie, specie_name in zip(parent_species, specie_names):
        stdout("Running with %s, %s" % (prop_name, specie_name))
        for idx, iso_gen_args in enumerate(iso_gen_arguments[specie_name]):
            # iso generation
            # iso_gen_args = position_args, position_charges, target_charge
            iso_file = os.path.join('./isos', specie_name + " " + str(idx) + '.csv')
            iso_gen(iso_file, parent_specie, *iso_gen_args, export_to=iso_file, add_title=False)
            with open(iso_file, "r") as f:
                stdout("Number of generated isomers: %s" % len(f.readlines()))
            stdout("Calculating labels...")
            labeled_data = data_prep(parent_specie, iso_file, prop_func, 'all', nprocs)
            test_idxs = choose(len(labeled_data.inputs), 0.1, rel_sizes=True)
            stdout("Finished label calculation!")
            image_dir = os.path.join('./images', prop_name + "_" + specie_name + "_" + str(len(labeled_data.inputs)))
            mkdir(image_dir)
            for tokenization_method in ['custodi', 'word']:
                for train_size in [0.001, 0.005, 0.01, 0.05, 0.1]:
                    train_size = round(train_size * len(labeled_data.inputs))
                    train_idxs = []
                    while True:
                        num = np.random.randint(0, len(labeled_data.inputs))
                        if num in test_idxs:
                            continue
                        else:
                            train_idxs.append(num)
                        if len(train_idxs) == train_size:
                            break
                    stdout("Running NN on train size %s with %s tokenization" % (train_size, tokenization_method))
                    prefix = str(train_size) + "_" + tokenization_method
                    test_err, train_err = run_calc(labeled_data, tokenization_method, train_idxs, test_idxs, image_dir, image_prefix=prefix, lr=0.001, nn_depth=3, epochs=500)
                    stdout("Finished! Final test error: %.4f" % test_err)
                    df = df.append({'prop': prop_name, 'specie': specie_name, 
                                    'dataset_size': len(labeled_data.inputs), 
                                    'tokenization': tokenization_method,
                                    'train_set_size': train_size,
                                    'test_err': test_err,
                                    'train_err': train_err})
                    df.to_csv('./results.csv')