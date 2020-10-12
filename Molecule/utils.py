from pymatgen.core.structure import Molecule, Site
from pymatgen.core.periodic_table import Element
import openbabel as ob
from rdkit.Chem import rdchem, rdFingerprintGenerator, AllChem
from rdkit import Chem, DataStructs
import numpy as np

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

def calculate_fingerprint(mol, method='morgan'):
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

def calculate_ecfp4(mol, nBits=1024):
    rdmol = to_rdkit_Mol(mol)
    Chem.rdmolops.SanitizeMol(rdmol)
    fp = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(rdmol, 2, nBits=nBits)
    arr = np.array([])
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def check_smiles(string):
    '''Function to check if smiles string is syntetically correct. Returns True / False accordingly.'''
    try:
        Chem.MolFromSmiles(string, sanitize=False)
        return True
    except:
        return False


