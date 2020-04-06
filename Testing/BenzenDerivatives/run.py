import sys; sys.path.append("../../")
import warnings
warnings.simplefilter(action = 'ignore', category=FutureWarning)
import tensorflow as tf
from Molecule.Objects import SubsMol, IsoGen
from Data.Objects import SmilesVec
from Data.Base import generate_data_using_comp_protocol
from Model.utils import KerasArchitectures
from Model.Objects import *
from CompProtocol.Objects import RDKitPropCalc
from rdkit.Chem.rdMolDescriptors import CalcKappa1 as prop_func
import os
import numpy as np

print("importing tf")


# define mol object
class SubsBenzene (SubsMol):

    core_struct = '*c1c(*)c(*)c(*)c(*)c1*'
    binding_positions = {'pos1': [0, 1, 2, 3, 4, 5]}
    core_charge = 0

# generate isomers
if not os.path.isfile('./isos.csv'):
    mol = SubsBenzene(position_args={'pos1': (0, 'max', 2, '[*H]')})
    isos = IsoGen(mol, {'pos1': [['*O', '*CC', '*Cl', '*Br', '*F', '*CO', '*N']]}, {'pos1': [[0, 0, 0, 0, 0, 0, 0]]}, 0, export_to='./isos.csv', add_title=False)
with open('./isos.csv', "r") as f:
    print("Number of generated isomers: %s" % len(f.readlines()))

# setting data 
data = SmilesVec(parent_specie=SubsBenzene)
data.load_inputs_from_file('./isos.csv')
data.vectorize_inputs()
print("data is vectorized")
input_idxs = [np.random.randint(0, len(data.inputs) - 1) for itr in range(1000)]
labeled_data = generate_data_using_comp_protocol(data, RDKitPropCalc(prop_func), input_idxs=input_idxs)
print("labels are computed")
labeled_data.vectorized_labels = labeled_data.vectorize_labels()
labeled_data.noramlize_vectors(normalize='labels')
print("data is ready")

# setting AE
input_shape = np.array(data.vectorized_inputs[0]).shape
input_size = np.prod(input_shape)
encoding_size = 2 * int(np.floor(input_size / 3))
print("encoding size is: %s, input size is %s" % (encoding_size, input_size))
autoencoder = KerasArchitectures.AutoEncoders(variational=False)
autoencoder.set_encoder(input_shape, layers=[
    tf.keras.layers.Dense(input_size, activation='relu'),
    tf.keras.layers.Dense(encoding_size, activation='relu'),
    tf.keras.layers.Dense(encoding_size, activation='tanh')
])

autoencoder.set_decoder(layers=[
    tf.keras.layers.Dense(encoding_size, activation='relu'),
    tf.keras.layers.Dense(input_size, activation='relu'),
    tf.keras.layers.Reshape(input_shape)
])

AE = KerasAE(autoencoder, tf.keras.optimizers.Adam(lr = 0.003), tf.keras.losses.binary_crossentropy)
AE.train(data.vectorized_inputs, epochs=20)
data_reps = AE.encode(data.vectorized_inputs)
print("autoencoder is trained")
# setting NN
predictive_model = KerasArchitectures.GeneralANN(
    input_shape=(encoding_size, ),
    layers=[
        tf.keras.layers.Dense(encoding_size, activation='linear'),
        tf.keras.layers.Dense(encoding_size, activation='relu'),
        tf.keras.layers.Dense(encoding_size, activation='relu'),
        tf.keras.layers.Dense(encoding_size, activation='relu'),
        tf.keras.layers.Dense(1, activation='tanh'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

NN = KerasNN(predictive_model, tf.keras.optimizers.Adam(lr=0.001), tf.keras.losses.mean_squared_error)
NN.train(np.array([data_reps[i] for i in input_idxs[:750]]), labeled_data.vectorized_labels[:750], epochs=50)
print("NN is trained")
# plotting results
NN.plot_fit(np.array([data_reps[i] for i in input_idxs[:750]]), labeled_data.vectorized_labels[:750], show=False)
NN.plot_fit(np.array([data_reps[i] for i in input_idxs[750:]]), labeled_data.vectorized_labels[750:], show=True)