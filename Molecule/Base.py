import openbabel as ob
from rdkit.Chem import AllChem
from rdkit import Chem
from .utils import *

class BaseMol:

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