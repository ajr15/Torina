import numpy as np
from copy import copy

def make_binary_substitution_idx_vecs(min_subs, max_subs, total_l, with_order=False):
    '''makes a binary vector of all posibble substitutions on total_l chain'''
    vecs = [[]]
    fvecs = []
    if min_subs == 0:
        fvecs = [[]]
    stop_cond = False
    for v in vecs:
        max_v = max(v) + 1 if not v == [] else 0
        for i in range(max_v, total_l):
            vecs.append(v + [i])
            if with_order and len(v) > 0:
                vecs.append([i] + v)
            if len(v + [i]) > max_subs:
                stop_cond = True
                break
            elif len(v + [i]) >= min_subs:
                fvecs.append(v + [i])
                if with_order and len(v) > 0:
                    fvecs.append([i] + v)
        if stop_cond:
            break
    return fvecs

def add_single_sub_to_idxs_vecs(vecs):
    subs_vecs_dict = dict()
    fvecs = []
    for i in range(len(vecs)):
        if vecs[i] == [] or type(vecs[i][0]) == int:
            vecs[i] = [vecs[i]]
        if not len(vecs[i][-1]) in subs_vecs_dict.keys():
            try:
                subs_vecs_dict[len(vecs[i][-1])] = make_binary_substitution_idx_vecs(1, len(vecs[i][-1]) - 1, len(vecs[i][-1]))
            except IndexError:
                continue
        for idx_vec in subs_vecs_dict[len(vecs[i][-1])]:
            vec = vecs[i] + [[vecs[i][-1][j] for j in idx_vec]]
            fvecs.append(vec)
    return fvecs

def make_multi_sub_idx_vecs(l, min_subs, max_subs, nsubs):
    '''make all possible substitutions with given amount of substituents. returns a dict with number of subs (keys) and susbs_idx_vecs (vals)'''
    d = dict()
    d[1] = make_binary_substitution_idx_vecs(min_subs, max_subs, l)
    for i in range(nsubs - 1):
        d[i + 2] = add_single_sub_to_idxs_vecs(d[i + 1])
    if nsubs == 1:
        d[1] = [[v] for v in d[1]]
    return d

def make_multi_sub_vecs_from_idx_vecs(l, multisub_idx_dict, subs_list, null_symbol=0, nsubs=1, has_zero=True):
    vecs = []
    if subs_list == []:
        return [[null_symbol for i in range(l)]]
    for all_vecs in multisub_idx_dict.values():
        for idx_vecs in all_vecs:
            # option to regulate the number of susbstituents, meant to prevenet duplicates in generation on isomers.
            if has_zero and idx_vecs == [[]]:
                continue
            if len(idx_vecs) < nsubs:
                continue
            elif len(idx_vecs) > nsubs:
                break
            vec = [null_symbol for i in range(l)]
            for i, idx_vec in enumerate(idx_vecs):
                for idx in idx_vec:
                    vec[idx] = subs_list[i]
            vecs.append(vec)
    return vecs

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

def cartesian_prod(vecs1, vecs2):
    prod = []
    for vec1 in vecs1:
        for vec2 in vecs2:
            prod.append(vec1 + list(vec2))
    return prod

def only_different_cartesian(vecs1, vecs2):
    '''Function to generate cartesian product where only different members of vec1 and vec2 are taken'''
    prod = []
    for vec1 in vecs1:
        for vec2 in vecs2:
            if not any([v1 == v2 for v1, v2 in zip(vec1, vec2)]):
                prod.append(vec1 + list(vec2))
    return prod

def vec_choose_n(vec, n):
    '''Function for generating all possible ways to take n objects out of a vector'''
    idx_vecs = make_binary_substitution_idx_vecs(0, n, len(vec), with_order=False)
    final_vecs = []
    for idxs in idx_vecs:
        final_vecs.append([vec[i] for i in idxs])
    return final_vecs

class base_position:

    def __init__(self, l, min_subs, max_subs, n_substitutions, null_symbol):
        self.l = l
        self.n_substitutions = n_substitutions
        self.null_symbol = null_symbol
        self.has_zero = (min_subs == 0)
        if max_subs == 'max':
            max_subs = l
        self.idx_vecs = make_multi_sub_idx_vecs(l, min_subs, max_subs, n_substitutions)

    def gen_subs_vecs(self, subs):
        return vec_choose_n(subs, self.n_substitutions)

    def gen_isos_with_subs(self, substituents: list):
        '''Makes substituents vectors based on idx_vecs and supplied substituents.
        ARGS:
            - substituents (list): substituents to use for generation. note that it only replace indeces with substituents and doesnt generate all possible combinations of subs.
        NOTE: this method is mainly for inner uses, it is not recommended for end use.'''
        if not len(substituents) <= self.n_substitutions:
            raise ValueError("Wrong number of substituents specified (%s). Max value %s" % (len(substituents), self.n_substitutions))
        return make_multi_sub_vecs_from_idx_vecs(self.l, self.idx_vecs, substituents, self.null_symbol, len(substituents), self.has_zero)
            

class Position:
    '''Position object to hold data and generate substitution vecs for multiple and basic positions with more complex susbtitution schemes.
    ARGS:
        - basic_position (list): list of basic_position objects to describe the position
        - basic_positions_types (list): a list with integer types of the different basic positions, for example [1, 2, 1] for basic positions 0 and 1, but for 2 there are different subs. 
                                        its usful when one wants to limit the different substituents on a position, for example, making A2B corroles 
                                        (5, 15 has the same sub and 10 has different sub). The list must consist integers in a rising order, i.e. position types in the form [1, 3, 1] is
                                        illegal and should be written as [1, 2, 1]. The lowest integer should always be 1.'''

    def __init__(self, basic_positions, basic_position_types):
        self.basic_positions = basic_positions
        self.l = sum([pos.l for pos in basic_positions])
        self.basic_positions_types = basic_position_types
        # TODO: make input checks for basic_positions and basic_position_types
    
    def gen_subs_vecs(self, subs):
        '''Method to generate substituents vectors for substitution. In the appropriate form for the position'''
        types_dict = {}
        ntypes = len(set(self.basic_positions_types))
        for t, pos, sub in zip(self.basic_positions_types, self.basic_positions, subs):
            types_dict[t] = pos.gen_subs_vecs(sub)
            if len(types_dict) == ntypes:
                break
        final_vecs = [[]]
        for subs_vec in types_dict.values():
            final_vecs = only_different_cartesian(final_vecs, subs_vec)
        # correcting dimentions of subs vecs in case of 1 basic position type
        if ntypes == 1:
            final_vecs = [[v] for v in final_vecs]
        for i in range(len(final_vecs)):
            vec = [final_vecs[i][j - 1] for j in self.basic_positions_types]
            final_vecs[i] = vec
        return final_vecs

    def gen_isos_with_subs(self, substituents):
        if type(substituents) is dict:
            corr = []
            for t in self.basic_positions_types:
                corr.append(substituents[t])
            substituents = corr
        if len(set(self.basic_positions_types)) == 1:
            substituents = [substituents]
        vecs = [[]]
        for pos, sub in zip(self.basic_positions, substituents):
            vecs = cartesian_prod(vecs, pos.gen_isos_with_subs(sub))
        return vecs

class iso_base_struct:
    '''Main class to handle susbstitutions of a core structure.
    
    ARGS:
        - positions (dict): dict with Position objects for each binding position.
        - core_struct_charge (int): charge of the core struct'''
    
    def __init__(self, positions, core_struct_charge=0, position_names=None):
        if type(positions) == dict:
            self._pos_names = positions.keys()
            self.positions = positions.values()
        else:
            self._pos_names = position_names
            self.positions = positions
        self.core_struct_charge = core_struct_charge

    def check_charges(self, charges, target_charge):
        if type(charges) == dict:
            tot_c = 0
            for vec in charges.values():
                tot_c += sum(vec)
        else:
            tot_c = sum(charges)
        return tot_c + self.core_struct_charge == target_charge
    
    def gen_subs_vecs(self, subs):
        vecs = [[]]
        for pos, sub in zip(self.positions, subs):
            vecs = cartesian_prod(vecs, pos.gen_subs_vecs(sub))
        return vecs

    def gen_isos_with_subs(self, substituents):
        if type(substituents) == dict:
            substituents = [substituents[name] for name in self._pos_names]
        vecs = [[]]
        for pos, sub in zip(self.positions, substituents):
            # print(sub, pos.gen_isos_with_subs(sub))
            vecs = cartesian_prod(vecs, pos.gen_isos_with_subs(sub))
        return vecs