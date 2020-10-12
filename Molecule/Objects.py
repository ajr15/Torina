from .Base import *
from .SubsMol import *
from rdkit.Chem import rdchem, rdFingerprintGenerator, AllChem
from rdkit import Chem, DataStructs
from rdkit.Geometry.rdGeometry import Point3D

def estimate_geometry_of_mol(rdmol):
    '''Wrapper to estimate the geometry of general molecule using rdkit. The method is mainly meant for dealing with 
    dummy atoms and metals in the molecule'''
    # TODO: generalize method for metals / dummy atoms
    rdmol.UpdatePropertyCache(strict=False)
    rdmol = Chem.rdchem.RWMol(rdmol)
    # checks for dative bonds in struct and removing atoms
    remove_atoms = []
    neighbor_atoms = []
    for atom in rdrdmol.GetAtoms():
        if len(atom.GetBonds()) > 3 and atom.GetAtomicNum() > 10:
            remove_atoms.append(atom)
            neighbor_atoms.append(atom.GetNeighbors())
    for atom in remove_atoms:
        rdmol.RemoveAtom(atom.GetIdx())
    # calculating geometry
    AllChem.EmbedMolecule(rdmol)
    # fixing coordinates for fragments
    frags_idxs = list(Chem.rdmolops.GetMolFrags(rdmol))
    conf = rdmol.GetConformer()
    if len(frags_idxs) > 1:
        for frag_idxs in frags_idxs:
            diff = (0.5 - np.random.rand()) * 5 * np.ones(3)
            for idx in frag_idxs:
                vec = conf.GetAtomPos(idx)
                vec = np.array([vec.x, vec.y, vec.z]) + diff
                conf.SetAtomPosition(i, Point3D(vec[0], vec[1], vec[2]))
    # adding dative atoms and setting coords
    for atom, nbors in zip(remove_atoms, neighbor_atoms):
        coords = np.zeros(0)
        for neighbor in nbors:
            vec = conf.GetAtomPosition(neighbor.GetIdx())
            coord += np.array([vec.x, vec.y, vec.z])
            rdmol.AddAtom(atom)
            rdmol.AddBond(atom.GetIdx(), neighbor.GetIdx(), order=Chem.rdchem.BondType.DATIVE)
        coords = coords / len(nbors)
        conf.SetAtomPosition(atom.GetIdx(), Point3D(coords[0], coords[1], coords[2]))
        rdmol.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(rdmol)
    return rdmol

def IsoGen(mol, substituents_dict, subs_charges, target_charge, connectivity_dict=None, export_to=None, sep=" ", add_title=True):   
    '''Method to generate substituent vectors for SubsMol object. The method takes all the substituents in the substituents dict and makes susbsitution vectors
    
    ARGS:
        - mol (SubsMol): Molecule to generate substituents vecs to.
        - substituents_dict (dict): dictionary with substituents. The keys of the dictionary must fit the position names in the molecule.
        - subs_charges (dict): dict in a similar form of substituents dict, with charges of the substituents in substituetns_dict.
        - target_charge (int): target charge of the substituted specie.
        - connectivity_dict (dict): optional dictionary specifying connections between different substituents. This is meant to prevent a case of non-bounded
                                    substituents. For example, the dict {\'pos1\': \'pos2\'} means that position 1 is connected to positions 2 and if None is
                                    within the substituents in position 2 then, position 1 is also None.
        - export_to (str): path to a file for export the subs vecs
        - sep (str): separator for writing the file
        - add_title (bool): add title to exported file, telling the position of a substituent
        - return_vecs (bool): weather to return all the generated vectors (memory intensive)'''

    if not isinstance(mol, SubsMol):
        raise ValueError("mol must be a SubsMol")

    # setting parameters for calculation
    substituents_vec = [substituents_dict[key] for key in mol._iso_base_struct._pos_names]
    subs_charges = [subs_charges[key] for key in mol._iso_base_struct._pos_names]
    all_subs_vecs = mol._iso_base_struct.gen_subs_vecs(substituents_vec)
    charge_vecs = mol._iso_base_struct.gen_subs_vecs(subs_charges)
    if not connectivity_dict is None:
        d = {}
        for key, val in connectivity_dict.items():
            d[mol._iso_base_struct._pos_names.index(key)] = mol._iso_base_struct._pos_names.index(val)
        connectivity_dict = copy(d)
        del d
    
    # setting export formats
    final_vecs = []
    if not export_to == None:
        with open(export_to, "w") as f:
            if add_title:
                title = ''
                for i, name in enumerate(mol._iso_base_struct._pos_names):
                    title = name + str(i + 1) + sep
                f.write(title + '\n')

    # generating isos
    for vec, charges in zip(all_subs_vecs, charge_vecs):
        if not mol._iso_base_struct.check_charges(flatten(charges), target_charge):
            continue
        if not connectivity_dict == None:
            if any([len(vec[val]) < mol._iso_base_struct.positions[val].l and not len(vec[key]) == 0 for key, val in connectivity_dict.items()]):
                continue
        subs_vecs = mol._iso_base_struct.gen_isos_with_subs(vec)
        if subs_vecs in final_vecs:
            continue
        final_vecs = final_vecs + subs_vecs
        if not export_to == None:
            with open(export_to, "a") as f:
                for vec in subs_vecs:
                    f.write(sep.join(vec) + '\n')

    return final_vecs

class SubsMol (BaseMol):

    # core structure props, must be specified when inherited.
    core_struct = None
    binding_positions = {}
    core_charge = None
    __iso_base_struct = None

    def __init__(self, position_args=None, atoms=None, coords=None, bondmap=None):
        super().__init__(atoms, coords, bondmap)
        if self.core_struct == None:
            raise RuntimeError("Coul'd not init class without specified core_struct")
        if self.binding_positions == {}:
            raise RuntimeError("Coul'd not init class without specified binding_positions")
        if self.core_charge == None:
            raise RuntimeError("Coul'd not init class without specified core_charge")
        self.position_args = position_args

    @property
    def _iso_base_struct(self):
        return self.__iso_base_struct

    @_iso_base_struct.setter
    def _iso_base_struct(self, struct=None):
        self.__iso_base_struct = struct

    @_iso_base_struct.getter
    def _iso_base_struct(self):
        if self.__iso_base_struct == None:
            if self.position_args == None:
                position_args = [(0, 'max', 1, 0) for i in range(len(self.binding_positions))]
            else:
                if not len(self.position_args.keys()) == len(self.binding_positions.keys()):
                    position_args = self.position_args
                    for key in self.binding_positions.keys():
                        if not key in position_args.keys():
                            position_args[key] = (0, 'max', 1, 0)
                position_args = [self.position_args[key] for key in self.position_args.keys()]
            positions = []
            for vec, args in zip(self.binding_positions.values(), position_args):
                if any([type(v) == list for v in vec]):
                    basic_positions = []
                    basic_positions_types = []
                    for v in vec:
                        if type(v) == list:
                            base_pos = base_position(len(v), *args)
                        else:
                            base_pos = base_position(1, *args)
                        basic_positions.append(base_pos)
                        basic_positions_types.append(1)
                    positions.append(Position(basic_positions, basic_positions_types))
                else:
                    positions.append(Position([base_position(len(vec), *args)], [1]))
            position_names = self.binding_positions.keys()
            self.__iso_base_struct = iso_base_struct(positions, self.core_charge, list(position_names))
            return self.__iso_base_struct
        else:
            return self.__iso_base_struct
    
    @property
    def binding_positions(self):
        return self._binding_positions

    @binding_positions.setter
    def binding_positions(self, binding_positions={}):
        if binding_positions == {}:
            raise RuntimeError("Must set binding_positions for Substituted Mol befor init")
        core = Chem.MolFromSmiles(self.core_struct)
        bounded_positions = {}
        l = flatten(binding_positions.values())
        for idx in flatten(binding_positions.values()):
            atom = core.GetAtomWithIdx(idx)
            if not atom.GetAtomicNum() == 0:
                raise ValueError("All positions in atomic positions must be designated by dummy atom in structure. atom number %s is not dummy" % idx)
            for n in [a.GetIdx() for a in atom.GetNeighbors()]:
                if n in l:
                    bounded_positions[l.index(n)] = l.index(idx)
        self._binding_positions = binding_positions
        self._bounded_positions = bounded_positions

    def _vec_to_subs_vec(self, vec):
        if type(vec) == list:
            d = {}
            begin_idx = 0
            end_idx = 0
            for key, val in self.binding_positions.items():
                end_idx += len(flatten(val))
                d[key] = vec[begin_idx:end_idx]
                begin_idx += len(flatten(val))
        ordered_positions = sorted(flatten(list(self.binding_positions.values())))
        subs_vec = [None for i in ordered_positions]
        for key in self.binding_positions.keys():
            for sub, pos in zip(d[key], self.binding_positions[key]):
                subs_vec[ordered_positions.index(pos)] = sub
        return subs_vec
        
    def from_vec(self, vec, estimate_geometry=False, convert_to_subs_vec=True):
        '''Method to get a molecule from a substituents vector.'''
        # TODO: implement estimate geometry methods.
        core = Chem.RWMol(Chem.MolFromSmiles(self.core_struct))

        subs_vec = vec
        if convert_to_subs_vec:
            subs_vec = self._vec_to_subs_vec(vec)
        for sub in subs_vec:
            if sub == "0":
                continue
            try:
                sub = Chem.RWMol(Chem.MolFromSmiles(sub, sanitize=False))
            except:
                print("Error occured while converting %s susbs in %s" % (sub, subs_vec))
                raise
            # finding binding positions on sub
            sub_binding_pos = []
            dummy_atom_count = 0
            for idx, atom in enumerate(sub.GetAtoms()):
                if atom.GetAtomicNum() == 0:
                    dummy_atom_count += 1
                    neighbors = [a.GetIdx() for a in atom.GetNeighbors()]
                    sub_binding_pos += neighbors
            # find binding position on core
            core_binding_pos = []
            core_bond_types = []
            for idx, atom in enumerate(sub.GetAtoms()):
                if atom.GetAtomicNum() == 0:
                    neighbors = [a.GetIdx() for a in atom.GetNeighbors()]
                    core_binding_pos += neighbors
                    for n in neighbors:
                        btype = core.GetBondBetweenAtoms(idx, n).GetBondType()
                        if btype == rdchem.BondType.AROMATIC:
                            btype = rdchem.BondType.SINGLE
                        core_bond_types.append(btype)
                    break
            combo = Chem.RWMol(Chem.CombineMols(core, sub))
            for core_idx, sub_idx, bondtype in zip(core_binding_pos, sub_binding_pos, core_bond_types):
                combo.AddBond(core_idx, core.GetNumAtoms() + sub_idx, order=bondtype)
            remove_idxs = []
            for idx, atom in enumerate(combo.GetAtoms()):
                if atom.GetAtomicNum() == 0:
                    if len(remove_idxs) == 0:
                        remove_idxs.append(idx)
                    elif idx > core.GetNumAtoms() - 1:
                        remove_idxs.append(idx)
            for i, idx in enumerate(remove_idxs):
                combo.RemoveAtom(idx - i)
            core = copy(combo)
        try:
            # tries to convert rdkit mol to Basic mol with standard method, otherwise uses SMILES to bridge the formats
            self.from_rdkit_mol(core, kekulize=True)
        except ValueError:
            print(Chem.MolToSmiles(core))
            self.from_str(Chem.MolToSmiles(core))

    def estimate_geometry(self):
        # TODO: implement !
        raise NotImplementedError()

