import pandas as pd
from time import time
from .commons import kw_cartesian_prod
from ..Data.Data.Data import Data

def _get_changing_kwds(kw_dict):
    l = []
    for k, v in kw_dict.items():
        if type(v) is list and len(v) > 1:
            l.append(k)
    return l

def grid_estimation(model_type, 
                    train_inputs: list, 
                    train_labels: list,
                    estimation_sets: list,
                    estimators=[], 
                    estimators_names=[],
                    additional_descriptors={},
                    write_to=None,
                    train_kwargs={},
                    init_kwargs={},
                    verbose=1):
    """Method to calculate estimators on combinations of training kwargs.
    ARGS:
        - model_type (Model): model type to train
        - train_inputs: training inputs
        - train_labels: training labels
        - estimation_sets: list of 3-tuples with (set name, inputs, labels)
        - estimators: list of estimators (for estimate_fit method)
        - estimators_names: list of estimator names (for estimate_fit method)
        - additional_descriptors (dict): dictionary with additional information to be added to results
        - write_to (str): (optional) csv file to write online results to
        - train_kwargs (dict): kwargs dict for training. each argument that will be given as a list will be included in combinations
        - init_kwargs (dict): kwargs dict for instancing the model. each argument that will be given as a list will be included in combinations
    RETURNS:
        Dataframe with all estimator values and results"""
    train_kw_list = kw_cartesian_prod(train_kwargs)
    init_kw_list = kw_cartesian_prod(init_kwargs)
    res = pd.DataFrame()
    if verbose == 1:
        train_changing_kwargs = _get_changing_kwds(train_kwargs) 
        init_changing_kwargs = _get_changing_kwds(init_kw_list)

    for init_kwds in init_kw_list:
        # creating model instance
        m = model_type(**init_kwds)
        for train_kwds in train_kw_list:
            if verbose == 1:
                print("Running with {}".format(', '.join([str(train_kwds[w]) for w in train_changing_kwds] + [str(init_kwds[w]) for w in init_changing_kwds])))
            # measuring execution time
            ti = time()
            # training model
            m.train(train_inputs, train_labels, **train_kwds)
            tf = time()
            # getting results dictionary
            d = {'computation_time': round(tf - ti, 2)}
            d.update(init_kwds) # add model parameters
            d.update(train_kwds) # add train parameters
            d.update(additional_descriptors) # additional value
            d.update(m.estimate_fit(train_inputs, train_labels, estimators, estimators_names, 'train_')) # train estimators
            for name, inputs, labels in estimation_sets:
                d.update(m.estimate_fit(inputs, labels, estimators, estimators_names, name + '_')) # other sets estimators
            res = res.append(d, ignore_index=True) # add dictionary to results
            if write_to is not None:
                res.to_csv(write_to)
    return res

def grid_estimation(model_type, 
                    train_data: Data,
                    estimation_datas: list,
                    estimators=[], 
                    estimators_names=[],
                    additional_descriptors={},
                    write_to=None,
                    train_kwargs={},
                    init_kwargs={},
                    verbose=1):
    """Method to calculate estimators on combinations of training kwargs.
    ARGS:
        - model_type (Model): model type to train
        - train_data (Data): data to train on
        - estimation_datas: list of 2-tuples with (set name, data)
        - estimators: list of estimators (for estimate_fit method)
        - estimators_names: list of estimator names (for estimate_fit method)
        - additional_descriptors (dict): dictionary with additional information to be added to results
        - write_to (str): (optional) csv file to write online results to
        - train_kwargs (dict): kwargs dict for training. each argument that will be given as a list will be included in combinations
        - init_kwargs (dict): kwargs dict for instancing the model. each argument that will be given as a list will be included in combinations
    RETURNS:
        Dataframe with all estimator values and results"""
    train_kw_list = kw_cartesian_prod(train_kwargs)
    init_kw_list = kw_cartesian_prod(init_kwargs)
    res = pd.DataFrame()
    if verbose == 1:
        train_changing_kwds = _get_changing_kwds(train_kwargs) 
        init_changing_kwds = _get_changing_kwds(init_kwargs)
    for init_kwds in init_kw_list:
        # creating model instance
        m = model_type(**init_kwds)
        for train_kwds in train_kw_list:
            if verbose == 1:
                print("Running with {}".format(', '.join([w + "=" + str(train_kwds[w]) for w in train_changing_kwds] + [w + "=" + str(init_kwds[w]) for w in init_changing_kwds])))
            # measuring execution time
            ti = time()
            # training model
            m.train(train_data.vectorized_inputs, train_data.vectorized_labels, **train_kwds)
            tf = time()
            # getting results dictionary
            d = {'computation_time': round(tf - ti, 2)}
            d.update(init_kwds) # add model parameters
            d.update(train_kwds) # add train parameters
            d.update(additional_descriptors) # additional value
            d.update(m.estimate_fit(train_data, estimators, estimators_names, 'train_')) # train estimators
            for name, data in estimation_datas:
                d.update(m.estimate_fit(data, estimators, estimators_names, name + '_')) # other sets estimators
            res = res.append(d, ignore_index=True) # add dictionary to results
            if write_to is not None:
                res.to_csv(write_to)
    return res