from abc import ABC, abstractclassmethod
from copy import deepcopy
from ..commons import *

class Data (ABC):
    '''Abstract Data object, to handle general Data formats (both inputs and labels).

    ATTRIBUTES:
        - inputs (list): list of input objects
        - labels (list): list of label objects
        - vectorized_inputs (list): list of vectorized inputs
        - vectorized_labels (list): list of vectorized labels
    
    PRIVATE ATTRIBUTES:
        - _input_norm_params (tuple): tuple of normalization parameters for inputs
        - _label_norm_params (tuple): tuple of normalization parameters for labels'''

    # ====================================
    #           Abstract methods
    # ====================================
 
    @abstractclassmethod
    def __init__(self):
        self.inputs = None
        self.labels = None
        self.vectorized_inputs = None
        self.vectorized_labels = None
        self._input_norm_params = None
        self._label_norm_params = None

    @abstractclassmethod
    def vectorize_inputs(self):
        '''Method to convert inputs to vectors (machine readable)'''
        raise NotImplementedError

    @abstractclassmethod
    def vectorize_labels(self):
        '''Method to convert labels to vectors (machine readable)'''
        raise NotImplementedError

    # ====================================
    #        Not implemented methods
    # ====================================

    def load_from_dataframe(self, df, input_columns=None, label_columns=None, reps_columns=None):
        # TODO: implement this methods, to load data from dataframe to Data instance
        raise NotImplementedError

    def export_to_dataframe(self, include='all'):
        # TODO: implement this method, to export Data instance to df.
        raise NotImplementedError

    # ====================================
    #    Methods for Specie-based inputs
    # ====================================

    @property
    def parent_specie(self):
        return self._parent_specie

    @parent_specie.setter
    def parent_specie(self, specie):
        self._parent_specie = specie

    def to_specie(self, x):
        '''Method to convert a single input entry to a \'Mol\' like object. For example converting smiles input to Mol'''
        raise NotImplementedError("to_specie method is not implemented in this subclass")
    
    # ====================================
    #       Data preperation methods
    # ====================================
    
    def vectorized_attributes_to_nparrays(self):
        self.vectorized_inputs = np.array([np.array(x) for x in self.vectorized_inputs])
        self.vectorized_labels = np.array([np.array(x) for x in self.vectorized_labels])

    # padding
    
    def _padd_attr(self, attr, pad_char, end_char):
        return padd_vecs(getattr(self, attr), end_char, pad_char)

    def pad_data(self, pad_char=0, end_char=None, pad='all'):
        """Method to pad vectors in data.
            ARGS:
                - pad_char: character used for padding. default=0
                - end_char: character to add to the end of the vector before padding. default=None
                - pad (str): what attribute to pad? (all, inputs, labels, vectorized_inputs, vectorized_labels)
            RETURNS:
                padded vectors"""
        if pad == 'all' or pad == 'inputs':
            return self._padd_attr('inputs', pad_char, end_char)
        if pad == 'all' or pad == 'labels':
            return self._padd_attr('labels', pad_char, end_char)
        if pad == 'all_vecs' or pad == 'vectorized_inputs':
            return self._padd_attr('vectorized_inputs', pad_char, end_char)
        if pad == 'all_vecs' or pad == 'vectorized_labels':
            return self._padd_attr('vectorized_labels', pad_char, end_char)

    # normalization

    def noramlize_vectors(self, normalize='all', method='unit_scale', axis=None, batch_size=128):
        """Method to normalize vectors in data.
            ARGS:
                - normalize (str): specify what to normalize (inputs, labels, all). default=\'all\'.
                - method (str): name of normalization fuction to be used (unit_scale, z_score, positive_z_score). default=\'unit_scale\'.
                - axis (int): axis to perform the normalization on. default=None.
                - batch_size (int): size of batch for batch normalization. default=128"""
        
        normalization_methods = {
                                    'unit_scale': unit_normalization,
                                    'z_score': zscore_normalization,
                                    'positive_z_score': positive_zscore_normalization
                                }

        if not method in normalization_methods.keys():
            raise ValueError("Unknown method. Recognized methods are only %s" % (normalization_methods.keys()))
        norm_func = normalization_methods[method]
        if normalize == 'inputs' or normalize == 'all':
            self.vectorized_inputs, params = norm_func(self.vectorized_inputs, axis, batch_size)
            self._input_norm_params = {'method': method, 'params': params}
        if normalize == 'labels' or normalize == 'all':
            self.vectorized_labels, params = norm_func(self.vectorized_labels, axis, batch_size)
            self._label_norm_params = {'method': method, 'params': params}

    def unnormalize_vectors(self, vectors, params='labels'):
        """Method to un-normalize vectors in data.
            ARGS:
                - normalize (list): specify what to normalize (inputs, labels)
                - params (str): which normalization parameters to use (inputs or labels). default=labels
            RETURNS:
                un-normalized vectors"""

        inverse_normalization_methods = {
                                            'unite_scale': inverse_unit_normalization,
                                            'z_score': inverse_zscore_normalization
                                        }

        if params == 'inputs':
            if not self._input_norm_params is None:
                unnorm_func = inverse_normalization_methods[self._input_norm_params['method']]
                return unnorm_func(vectors, *self._input_norm_params['params'])
            else:
                return vectors
        if params == 'labels':
            if not self._label_norm_params is None:
                unnorm_func = inverse_normalization_methods[self._label_norm_params['method']]
                return unnorm_func(vectors, *self._label_norm_params['params'])
            else:
                return vectors

    def remove_entries(self, entry_values: list):
        """Method to remove all entries in entry_values from \'vectorized_inputs\' and \'vectorized_labels\'. 
        ARGS:
            - entry_values (list): values of entries to remove. if \'empty_arrays\' is entered filters out empty arrays"""
        inps = []
        labels = []
        for inp, label in zip(self.vectorized_inputs, self.vectorized_labels):
            if not inp in entry_values and not label in entry_values:
                if 'empty_arrays' in entry_values:
                    if hasattr(inp, "__len__"):
                        if len(inp) > 0:
                            inps.append(inp)
                    else:
                        inps.append(inp)
                    if hasattr(label, "__len__"):
                        if len(label) > 0:
                            labels.append(label)
                    else:
                        labels.append(label)
                else:
                    inps.append(inp)
                    labels.append(labels)
        self.vectorized_inputs = inps
        self.vectorized_labels = labels

    # ====================================
    #        Data retrival methods
    # ====================================

    @staticmethod
    def _choose_idxs_if_not_empty(vecs, idxs):
        """take idxs from a list if the list is not None"""
        # helper function to select data from idxs
        if vecs is not None:
            return np.array([vecs[i] for i in idxs])
        else:
            return None

    def data_from_idxs(self, idxs):
        """Method to take data from selected entries.
        ARGS:
            - idxs (list): list of indecis of entries
        RETURNS:
            A Data object with changed inputs/labels according to the indecis (every other parameter is saved)"""
        d = deepcopy(self) # copy all attributes/variables of current object
        # change inputs/labels according to idxs
        try:
            d.inputs = self._choose_idxs_if_not_empty(self.inputs, idxs)
            d.labels = self._choose_idxs_if_not_empty(self.labels, idxs)
        except IndexError:
            print("ERROR IN COPYING INPUTS AND LABELS, PROCEEDING WITHOUT THEM...")
            # TODO: stop being lazy  and fix remove_entries method to not cause this error !
        d.vectorized_inputs = self._choose_idxs_if_not_empty(self.vectorized_inputs, idxs)
        d.vectorized_labels = self._choose_idxs_if_not_empty(self.vectorized_labels, idxs)
        return d

    def split_to_groups(self, group_sizes, relative_sizes=True, add_fill_group=False, random_seed=None):
        idx_groups = choose(len(self.vectorized_inputs), group_sizes, rel_sizes=relative_sizes, add_fill_group=add_fill_group, random_seed=random_seed)
        ds = []
        for idxs in idx_groups:
            ds.append(self.data_from_idxs(idxs))
        return ds

    def sample(self, sample_size, relative_size=False, random_seed=None):
        idxs = choose(len(self.inputs), [sample_size], rel_sizes=relative_size, random_seed=random_seed)[0]
        return self.data_from_idxs(idxs)

    def __len__(self):
        return len(self.inputs)