import pandas as pd
from pathos.multiprocessing import Pool
import os
from .. import CompProtocol
from .commons import *

def convert_inputs(data, inputs=None, input_idxs=None, verbose=0, nprocs=os.cpu_count()):
    '''Recommended function for conversion on multiple inputs/input_idxs to species. Supports parallel computing'''
    # TODO: add verbosity mode support
    if not inputs is None:
        if nprocs > 1:
            with Pool(processes=nprocs) as pool:
                species = pool.map(data.to_specie, inputs)
        else:
            species = []
            for Input in inputs:
                species.append(data.to_specie(Input))
    elif not input_idxs is None:
        if nprocs > 1:
            with Pool(processes=nprocs) as pool:
                species = pool.map(data.to_specie, [data.inputs[i] for i in input_idxs])
        else:
            species = []
            for Input in [data.inputs[i] for i in input_idxs]:
                species.append(data.to_specie(Input))
    return species

def generate_data_using_comp_protocol(data, comp_protocol, input_idxs=None, inputs=None, verbose=0, nprocs=os.cpu_count(), timeout=None):
    if not isinstance(comp_protocol, CompProtocol.Base.CompProtocol):
        raise ValueError("computation protocol must be an instance of a CompProtocol object")
    if not isinstance(data, Data):
        raise ValueError("data must be an instance of a Data object")

    species = convert_inputs(data, inputs, input_idxs, verbose, nprocs)
    aux_df, res_df = CompProtocol.Base.run_protocol(comp_protocol, species, verbose, timeout, nprocs)
    labels = [vec[:-1] for vec in res_df.to_numpy()]
    new_data = copy(data)
    new_data.inputs = inputs if not inputs == None else [data.inputs[i] for i in input_idxs]
    if not new_data.vectorized_inputs is None:
        new_data.vectorized_inputs = None if input_idxs == None else [data.vectorized_inputs[i] for i in input_idxs]
    new_data.labels = labels
    return new_data