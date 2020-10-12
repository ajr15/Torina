from abc import ABC, abstractclassmethod, abstractproperty
import pandas as pd
from ..Molecule.Base import Specie
from multiprocessing import TimeoutError, Pool
from multiprocessing.pool import ThreadPool
import signal
import json
import os
import numpy as np

class CompProtocol (ABC):
    """Class to hold data on a general computational procedure"""

    def __init__(self):
        pass
    
    @property
    def cpus_per_task(self):
        return self.__cpus_per_task

    @cpus_per_task.setter
    def cpus_per_task(self, n_cpus=1):
        self.__cpus_per_task = n_cpus

    def GlobalPreCalc(self) -> dict:
        '''Function for calculating a global properties to be used in further computations'''
        pass

    def AuxDict(self, specie: Specie) -> dict:
        """Function to calculate auxilary data, to be saved apart from main results after computation"""
        return {}

    def ErrorHandler(self, specie, exception):
        """Method to handel errors that rise during computation. Handles errors and return aux and res dicts."""
        raise exception

    @abstractclassmethod
    def PropDict(self, specie: Specie, GlobalDict, aux_dict) -> dict:
        '''Function of the rate constants in rate dict to define different properties for the calculation. Function returns a dict with property names (keys) and functions to calculate them (values)'''
        raise NotImplementedError("PropDict is not implemented. Impossible to calculate properties.")

# defining TimeOut handler
def timeout_handler(signum, frame):
    raise TimeoutError

signal.signal(signal.SIGALRM, timeout_handler)

# Function writes data to a json results and aux file, to be later analyzed.
def _single_run(verbose, protocol, specie, timeout=None):
    # TODO: *** make verbosity modes ! ***
    if not timeout is None:
        signal.alarm(timeout)
    try:
        aux_dict = protocol.AuxDict(specie)
        res_dict = protocol.PropDict(specie)
    except TimeoutError:
        aux_dict, res_dict = {}, {"Errors": "Timeout"}
    except Exception as err:
        aux_dict, res_dict = protocol.ErrorHandler(specie, err)
        res_dict["Errors"] = str(err)
    aux_dict['name'] = specie.name
    res_dict['name'] = specie.name
    return aux_dict, res_dict

def run_protocol(protocol: CompProtocol, comp_set, verbose=0, timeout=None, nprocs='max'):
    '''Function to run a calculation based protocol instructions. The job could be ran in parallel'''
    # getting number of procs for calc
    if nprocs == 'max':
        nprocs = np.floor(os.cpu_count() / protocol.cpus_per_task)
    # calculating GlobDict
    protocol.GlobalPreCalc()
    # running computation
    try:
        with Pool(processes=nprocs) as pool:
            output = pool.starmap(_single_run, [(verbose, protocol, specie, timeout) for specie in comp_set])
    except TypeError:
        with ThreadPool(processes=nprocs) as pool:
            output = pool.starmap(_single_run, [(verbose, protocol, specie, timeout) for specie in comp_set])
    # parsing output to dataframes
    res_df = pd.DataFrame()
    aux_df = pd.DataFrame()
    for i, out in enumerate(output):
        a, r = out
        aux = pd.DataFrame(a, index=[i])
        res = pd.DataFrame(r, index=[i])
        aux_df = res_df.append(aux, ignore_index=True)
        res_df = res_df.append(res, ignore_index=True)
    return aux_df, res_df