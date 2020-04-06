from pymatgen.core.structure import Structure, Molecule, Site
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import JmolNN as env_strategy
from pymatgen.core.periodic_table import Element
import openbabel as ob
from rdkit.Chem import rdchem, rdFingerprintGenerator, AllChem
from rdkit import Chem, DataStructs
from rdkit.Geometry.rdGeometry import Point3D
import os
import re
from copy import copy
import numpy as np
from abc import ABC, abstractclassmethod
import sys; sys.path.append('../../')
from Data.Base import Specie

# =========================================
#             Utils Functions
# =========================================

def neighbors(indicis, bondmap):
    '''Finds indicis of neighbors within one bond from the specified atoms in inicis list.
    ARGS:
    - indicis: list of intigers, for neighbor search.
    - bondmap: list of len 3 lists with neighbor data between indicis.
    
    RETURNS: a list of intigers corrisponding to the neighbors of the indicis.'''
    neighbors = set()
    if type(indicis) == int:
        for bond in bondmap:
            if indicis in bond:
                index = bond.index(indicis)
                neighbors.add(bond[1 - index])
        if indicis in neighbors: neighbors.remove(indicis)
    elif type(indicis) == list:
        for bond in bondmap:
            for j in indicis:
                if j in bond:
                    index = bond.index(j)
                    neighbors.add(bond[1 - index])
        for i in indicis:
            if i in neighbors: neighbors.remove(i)
    else:
        raise ValueError("Indicis must be intiger or list !")
    return list(neighbors)

def multi_neighbors(i, n, bondmap):
    '''Finds neighbors withing n bonds from the atom at index i'''
    seed = neighbors(i, bondmap); pseed = set(); pseed.add(i)
    for i in range(n - 1):
        for s in seed:
            pseed.add(s)
        seed = neighbors(seed, bondmap)
        for k in list(pseed):
            if k in seed:
                seed.remove(k)
    return seed

def to_rdkit_Mol(mol):
    rdmol = rdchem.EditableMol(rdchem.Mol())
    for atom in mol.atoms:
        rdmol.AddAtom(rdchem.Atom(Element(atom).Z))
    for bond in mol.bondmap:
        rdmol.AddBond(bond[0], bond[1], rdchem.BondType.values[bond[2]])
    return rdmol.GetMol()

def to_openbabel_Mol(mol, CalcBondmap = False):
    obmol = ob.OBMol()
    if not mol.coords == None:
        for atom, coord in zip(mol.atoms, mol.coords):
            obatom = ob.OBAtom()
            obatom.SetAtomicNum(ob.OBElementTable().GetAtomicNum(atom))
            coord_vec = ob.vector3(coord[0], coord[1], coord[2])
            obatom.SetVector(coord_vec)
            obmol.InsertAtom(obatom)
    else:
        for atom in mol.atoms:
            obatom = ob.OBAtom()
            obatom.SetAtomicNum(ob.OBElementTable().GetAtomicNum(atom))
            obmol.InsertAtom(obatom)
    if CalcBondmap == True:
        obmol.ConnectTheDots()
        obmol.PerceiveBondOrders()
    else:
        for bond in mol.bondmap:
            obmol.AddBond(bond[0] + 1, bond[1] + 1, bond[2])
    return obmol

def to_pymatgen_Molecule(mol):
    sites = []
    for atom, coord in zip(mol.atoms, mol.coords):
        sites.append(Site(atom, coord))
    return Molecule.from_sites(sites)

def add_hydrogens(mol):
    obmol = to_openbabel_Mol(mol)
    obmol.AddHydrogens()
    mol.from_openbabel_mol(obmol)
    return mol

def remove_hydrogens(self):
    obmol = self.to_openbabel_Mol()
    obmol.DeleteHydrogens()
    mol.from_openbabel_mol(obmol)
    return mol

def calculate_fingerprint(mol, method = 'morgan'):
    rdmol = to_rdkit_Mol(mol)
    rdmol.UpdatePropertyCache(strict = False)
    Chem.GetSymmSSSR(rdmol)
    Dict = {
        'rdkit': rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=64),
        'morgan': rdFingerprintGenerator.GetMorganGenerator(fpSize=64),
        'topological-torsion': rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=64),
        'atom-pairs': rdFingerprintGenerator.GetAtomPairGenerator(fpSize=64),
    }
    rep = []
    fp = [Dict[method].GetFingerprint(rdmol, fromAtoms=[i]) for i in range(len(mol.atoms))]
    for atomic_fp in fp:
        arr = np.zeros((1, ))
        DataStructs.ConvertToNumpyArray(atomic_fp, arr)
        rep.append(arr)
    return np.array(rep)

def check_smiles(string):
    '''Function to check if smiles string is syntetically correct. Returns True / False accordingly.'''
    try:
        Chem.MolFromSmiles(string, sanitize=False)
        return True
    except:
        return False

# =========================================
#           The Main Mol Object
# =========================================

class BaseMol (Specie):

    def __init__(self, atoms=None, coords=None, bondmap=None):
        self.atoms = atoms
        self.coords = coords
        self.bondmap = bondmap

    def from_file(self, filename):
        '''Method to read Mols from file. Must be implemented for every subclass'''
        mol = ob.OBMol(); conv = ob.OBConversion()
        conv.ReadFile(mol, filename)
        self.from_openbabel_mol(mol)

    def from_str(self, string):
        '''Method to read Mols from strings, strings could be SMILES, SMILES vectors... Mainly used for database reading.
        Must be implemented for every subclass'''
        try:
            mol = Chem.MolFromSmiles(string, sanitize=False)
            self.from_rdkit_mol(mol)
        except ValueError:
            mol = ob.OBMol(); conv = ob.OBConversion()
            conv.SetInFormat("smi")
            conv.ReadString(mol, string)
            self.from_openbabel_mol(mol)

    
    def to_str(self):
        '''Method to write Mols to strings, strings could be SMILES, SMILES vectors... Mainly used for database generation.
        Must be implemented for every subclass'''
        return Chem.MolToSmiles(to_rdkit_Mol(self))

    @property
    def core_structure(self):
        return self.__core_structure

    @core_structure.setter
    def core_structure(self, structure):
        if not type(structure) is core_structure:
            raise ValueError("Core structure must be a core_structure object")
        self.__core_structure = structure

    def from_openbabel_mol(self, mol):
        '''Method to convert an openbabel Mol object to Torina Mol object.'''
        atoms = []; coords = []
        for atom in ob.OBMolAtomIter(mol):
            element = ob.OBElementTable().GetSymbol(atom.GetAtomicNum())
            coord = [atom.GetX(), atom.GetY(), atom.GetZ()]
            atoms.append(element); coords.append(coord)
        bondmap = []
        for bond in ob.OBMolBondIter(mol):
            bondmap.append([bond.GetBeginAtomIdx() - 1, bond.GetEndAtomIdx() - 1, bond.GetBO()])
        self.atoms = atoms
        self.coords = coords
        self.bondmap = bondmap

    def from_rdkit_mol(self, rdmol, kekulize=True):
        '''Method to convert an rdkit Mol object to Torina Mol object.'''
        Chem.rdmolops.GetSymmSSSR(rdmol)
        rdmol.UpdatePropertyCache(strict=False)
        if kekulize is True:
            Chem.rdmolops.Kekulize(rdmol)
        HasConf = False
        if not rdmol.GetNumConformers() == 0:
            HasConf = True
            conf = rdmol.GetConformer()
            coords = []
        else:
            coords = None
        atoms = []
        for atom in rdmol.GetAtoms():
            atoms.append(atom.GetSymbol())
            if HasConf:
                vec = conf.GetAtomPosition(atom.GetIdx())
                coords.append([vec.x, vec.y, vec.z])
        bonds = []
        for bond in rdmol.GetBonds():
            bonds.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), int(bond.GetBondTypeAsDouble())])
        self.atoms = atoms
        self.coords = coords
        self.bondmap = bonds

    def write_to_file(self, filename):
        '''Method to write molecule to file'''
        obmol = to_openbabel_Mol(self)
        conv = ob.OBConversion()
        conv.WriteFile(obmol, filename)

    def estimate_bondmap(self, Set=False):
        '''Method to estimate the bondmap of the molecule. Returns the calculated bondmap.'''
        obmol = to_openbabel_Mol(self, CalcBondmap = True)
        bondmap = []
        for bond in ob.OBMolBondIter(obmol):
            vec = [bond.GetBeginAtomIdx() - 1, bond.GetEndAtomIdx() - 1, bond.GetBO()]
            bondmap.append(vec)
        if Set == True:
            self.bondmap = bondmap
        return bondmap

    def estimate_geometry(self):
        '''Method to estimate the geometry of the molecule. Returns the calculated bondmap.'''
        rdmol = to_rdkit_Mol(self)
        rdmol.UpdatePropertyCache(strict = False)
        Chem.GetSymmSSSR(rdmol)
        Chem.AllChem.EmbedMolecule(rdmol)
        self.from_rdkit_mol(rdmol)

    def neighbors_of(self, i):
        '''Method to find indicis of neighboring atoms to the i-est atom in the molecule. Returns the indicis of the neighboring atoms.'''
        return neighbors(i, self.bondmap)

    def get_bond(self, i, j):
        '''Method to get the bond between the i-th and j-th atoms. Returns None if there is no such bond.'''
        for bond in self.bondmap:
            if set([bond[0], bond[1]]) == set([i, j]):
                return bond
        return None

    def MMGeoOpt(self, ForceField = "UFF", StepNum = 1000):
        '''Method to run a molecular mechanics geometry optimization. Sets the new geometry to the molecule.'''
        obmol = to_openbabel_Mol(self)
        # optimization
        OBFF = ob.OBForceField.FindForceField(ForceField)
        suc = OBFF.Setup(obmol)
        if not suc == True:
            raise ValueError("Could not set up force field for molecule")
        OBFF.ConjugateGradients(StepNum)
        OBFF.GetCoordinates(obmol)
        self.from_openbabel_mol(obmol)