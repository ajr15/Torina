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

def neighbors(indicis, bondmap):
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

# finds neighbors at n bonds distance
def multi_neighbors(i, n, bondmap):
    seed = neighbors(i, bondmap); pseed = set(); pseed.add(i)
    for i in range(n - 1):
        for s in seed:
            pseed.add(s)
        seed = neighbors(seed, bondmap)
        for k in list(pseed):
            if k in seed:
                seed.remove(k)
    return seed

class Mol:

    def __init__(self, atoms = None, coords = None, bondmap = None):
        self.atoms = atoms
        self.coords = coords
        self.bondmap = bondmap

    def from_file(self, filename):
        mol = ob.OBMol(); conv = ob.OBConversion()
        conv.ReadFile(mol, filename)

        atoms = []; coords = []
        for atom in ob.OBMolAtomIter(mol):
            element = ob.OBElementTable().GetSymbol(atom.GetAtomicNum())
            coord = [atom.GetX(), atom.GetY(), atom.GetZ()]
            atoms.append(element); coords.append(coord)

        bondmap = []
        for bond in ob.OBMolBondIter(mol):
            vec = [bond.GetBeginAtomIdx() - 1, bond.GetEndAtomIdx() - 1, bond.GetBO()]
            bondmap.append(vec)

        self.atoms = atoms; self.coords = coords; self.bondmap = bondmap

    def from_smiles(self, string, calc_geometry = True):
        mol = Chem.MolFromSmiles(string, sanitize = False)

        if calc_geometry == True:
            mol.UpdatePropertyCache(strict = False)
            Chem.GetSymmSSSR(mol)
            Chem.AllChem.EmbedMolecule(new_mol)
        
        self.from_rdkit_mol(mol)

    def from_rdkit_mol(self, rdmol, kekulize = True):
        if kekulize is True:
            Chem.rdmolops.Kekulize(rdmol)

        HasConf = False
        if not rdmol.GetNumConformers() == 0:
            HasConf = True
            conf = rdmol.GetConformer()
            coords = []
        atoms = []
        for atom in rdmol.GetAtoms():
            atoms.append(atom.GetSymbol())
            if HasConf:
                vec = conf.GetAtomPosition(atom.GetIdx())
                coords.append([vec.x, vec.y, vec.z])

        bonds = []
        for bond in rdmol.GetBonds():
            bonds.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), int(bond.GetBondTypeAsDouble())])

        self.atoms = atoms; self.bondmap = bonds

        if HasConf:
            self.coords = coords

    def get_biggest_mol(self):
        rdmol = self.to_rdkit_Mol()
        rs = Chem.GetMolFrags(rdmol)

        idx = [len(a) for a in rs].index(max([len(a) for a in rs]))

        atom_idx_dict = dict()
        for index, i in enumerate(rs[idx]):
            atom_idx_dict[i] = index

        self.atoms = [self.atoms[i] for i in rs[idx]]
        self.coords = [self.coords[i] for i in rs[idx]]

        for bond in copy(self.bondmap):
            if not ((bond[0] in rs[idx]) and (bond[1] in rs[idx])):
                self.bondmap.remove(bond)
            else:
                self.bondmap[self.bondmap.index(bond)] = [atom_idx_dict[bond[0]], atom_idx_dict[bond[1]], bond[2]]

    def to_pymatgen_Molecule(self):
        sites = []
        for atom, coord in zip(self.atoms, self.coords):
            sites.append(Site(atom, coord))

        return Molecule.from_sites(sites)

    def to_rdkit_Mol(self):
        rdmol = rdchem.EditableMol(rdchem.Mol())

        for atom in self.atoms:
            rdmol.AddAtom(rdchem.Atom(Element(atom).Z))

        for bond in self.bondmap:
            rdmol.AddBond(bond[0], bond[1], rdchem.BondType.values[bond[2]])

        return rdmol.GetMol()

    def to_openbabel_Mol(self, CalcBondmap = False):
        obmol = ob.OBMol()
        if not self.coords == None:
            for atom, coord in zip(self.atoms, self.coords):
                obatom = ob.OBAtom()
                obatom.SetAtomicNum(ob.OBElementTable().GetAtomicNum(atom))
                coord_vec = ob.vector3(coord[0], coord[1], coord[2])
                obatom.SetVector(coord_vec)
                obmol.InsertAtom(obatom)
        else:
            for atom in self.atoms:
                obatom = ob.OBAtom()
                obatom.SetAtomicNum(ob.OBElementTable().GetAtomicNum(atom))
                obmol.InsertAtom(obatom)


        if CalcBondmap == True:
            obmol.ConnectTheDots()
            obmol.PerceiveBondOrders()
        else:
            for bond in self.bondmap:
                obmol.AddBond(bond[0] + 1, bond[1] + 1, bond[2])

        return obmol

    def write_to_file(self, filename):
        obmol = self.to_openbabel_Mol()
        conv = ob.OBConversion()
        conv.WriteFile(obmol, filename)

    def add_hydrogens(self):
        obmol = self.to_openbabel_Mol()
        obmol.AddHydrogens()

        atoms = []; coords = []
        for atom in ob.OBMolAtomIter(obmol):
            element = ob.OBElementTable().GetSymbol(atom.GetAtomicNum())
            coord = [atom.GetX(), atom.GetY(), atom.GetZ()]
            atoms.append(element); coords.append(coord)

        bondmap = []
        for bond in ob.OBMolBondIter(obmol):
            vec = [bond.GetBeginAtomIdx() - 1, bond.GetEndAtomIdx() - 1, bond.GetBO()]
            bondmap.append(vec)

        self.atoms = atoms; self.coords = coords; self.bondmap = bondmap

    def remove_hydrogens(self):
        obmol = self.to_openbabel_Mol()
        obmol.DeleteHydrogens()

        atoms = []; coords = []
        for atom in ob.OBMolAtomIter(obmol):
            element = ob.OBElementTable().GetSymbol(atom.GetAtomicNum())
            coord = [atom.GetX(), atom.GetY(), atom.GetZ()]
            atoms.append(element); coords.append(coord)

        bondmap = []
        for bond in ob.OBMolBondIter(obmol):
            vec = [bond.GetBeginAtomIdx() - 1, bond.GetEndAtomIdx() - 1, bond.GetBO()]
            bondmap.append(vec)

        self.atoms = atoms; self.coords = coords; self.bondmap = bondmap



    def EstimateBondmap(self, Set = False):
        obmol = self.to_openbabel_Mol(CalcBondmap = True)
        bondmap = []
        for bond in ob.OBMolBondIter(obmol):
            vec = [bond.GetBeginAtomIdx() - 1, bond.GetEndAtomIdx() - 1, bond.GetBO()]
            bondmap.append(vec)

        if Set == True:
            self.bondmap = bondmap
            return bondmap
        else:
            return bondmap

    def neighbors_of(self, i):
        return neighbors(i, self.bondmap)

    def multi_neighbors_of(self, i, n):
        return multi_neighbors(i, n, self.bondmap)

    def get_bond(self, i, j):
        for bond in self.bondmap:
            if set([bond[0], bond[1]]) == set([i, j]):
                return bond

        return None

    def to_smiles_str(self):
        return Chem.MolToSmiles(self.to_rdkit_Mol())

    def MMGeoOpt(self, ForceField = "UFF", StepNum = 1000):
        obmol = self.to_openbabel_Mol()

        # optimization
        OBFF = ob.OBForceField.FindForceField(ForceField)
        suc = OBFF.Setup(obmol)
        if not suc == True:
            raise ValueError("Could not set up force field for molecule")
        OBFF.ConjugateGradients(StepNum)
        OBFF.GetCoordinates(obmol)

        # load to Mol object
        atoms = []; coords = []
        for atom in ob.OBMolAtomIter(obmol):
            element = ob.OBElementTable().GetSymbol(atom.GetAtomicNum())
            coord = [atom.GetX(), atom.GetY(), atom.GetZ()]
            atoms.append(element); coords.append(coord)

        bondmap = []
        for bond in ob.OBMolBondIter(obmol):
            vec = [bond.GetBeginAtomIdx() - 1, bond.GetEndAtomIdx() - 1, bond.GetBO()]
            bondmap.append(vec)

        self.atoms = atoms; self.coords = coords; self.bondmap = bondmap

    def rep_from_dict(self, dict):
        rep = []
        for s in self.to_smiles_str():
            rep.append(dict[s])

        return rep

    def rep_from_fingerprints(self, method = 'morgan'):
        # rdmol = Chem.rdmolops.RemoveHs(self.to_rdkit_Mol(), sanitize=False)
        rdmol = self.to_rdkit_Mol()
        rdmol.UpdatePropertyCache(strict = False)
        Chem.GetSymmSSSR(rdmol)
        Dict = {
            'rdkit': rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=64),
            'morgan': rdFingerprintGenerator.GetMorganGenerator(fpSize=64),
            'topological-torsion': rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=64),
            'atom-pairs': rdFingerprintGenerator.GetAtomPairGenerator(fpSize=64),
        }

        rep = []
        fp = [Dict[method].GetFingerprint(rdmol, fromAtoms=[i]) for i in range(len(self.atoms))]
        for atomic_fp in fp:
            arr = np.zeros((1, ))
            DataStructs.ConvertToNumpyArray(atomic_fp, arr)
            rep.append(arr)

        return np.array(rep)

    def generate_mopac_input_file(self, filename, top_keywords = None, bottom_keywords = None, run = False, check_for_calc = True):
        if not filename.endswith(".mop"):
            filename = filename + ".mop"
        
        with open(filename, "w") as f:
            # writing top part
            if not top_keywords is None:
                top_string = ""
                for word in top_keywords:
                    top_string += " " + word
                top_string += "\r\n"
                f.write(top_string)
                f.write("title\r\n")
                f.write("\n")

            # writing coords and atoms
            for atom, coord in zip(self.atoms, self.coords):
                s = atom
                for c in coord:
                    s += " " + str(c) + " 1"
                s += "\r\n"
                f.write(s)

            # writing bottom part
            if not bottom_keywords is None:
                bottom_string = ""
                for word in bottom_keywords:
                    bottom_string += " " + word
                bottom_string += "\r\n"
                f.write("\n")
                f.write(bottom_string)

        if run is True:
            if not check_for_calc:
                os.system(f'/opt/mopac/MOPAC2016.exe {filename}')
            elif not os.path.isfile(filename[:-3] + "out"):
                os.system(f'/opt/mopac/MOPAC2016.exe {filename}')

    def has_geometry(self):
        return not self.coords is None

class Tetrapyrrole (Mol):
    
    def get_tetrapyrrole_atoms(self):
        if not self.is_tetrapyrrole():
            raise ValueError("Molecule must be a tetrapyrrole !")

        filtered_bondmap = []
        for bond in self.bondmap:
            atom1 = self.atoms[bond[0]]; atom2 = self.atoms[bond[1]]
            if (atom1 == "C" or atom1 == "N") and (atom2 == "C" or atom2 == "N"):
                filtered_bondmap.append(bond)

        Nindicis = []
        for i, atom in enumerate(self.atoms):
            if atom == "N": Nindicis.append(i)

        if len(Nindicis) == 4:
            CenterNs = Nindicis
        else:
            NNeighborsDict = {}
            for n in Nindicis:
                NNeighbors = []
                for i in multi_neighbors(n, 4, filtered_bondmap):
                    if i in Nindicis: NNeighbors.append(i)
                NNeighborsDict[n] = NNeighbors

            Stop = False
            for n1 in Nindicis:
                if len(self.neighbors_of(n1)) < 3:
                    continue
                forbidden = []; forbidden.append(n1)
                for n2 in [i for i in NNeighborsDict[n1] if (not i in forbidden)]:
                    forbidden.append(n2)
                    for n3 in [i for i in NNeighborsDict[n2] if (not i in forbidden)]:
                        forbidden.append(n3)
                        for n4 in [i for i in NNeighborsDict[n3] if (not i in forbidden)]:
                            if (n1 in multi_neighbors(n4, 3, filtered_bondmap)) or (n1 in NNeighborsDict[n4]):
                                Stop = True
                            if Stop == True:
                                CenterNs = [n1, n2, n3, n4]
                                break
                        if Stop: break
                    if Stop: break
                if Stop: break

        betas = set(); mesos = set()
        for i in CenterNs:
            nbs = multi_neighbors(i, 2, filtered_bondmap)
            for nb in nbs:
                Break = False; beta = False
                for n in CenterNs:
                    if n in neighbors(nb, filtered_bondmap): Break = True
                if Break: break
                for k in neighbors(nb, filtered_bondmap):
                    if k in nbs: beta = True
                if beta == True: betas.add(nb)
                else: mesos.add(nb)

                if Break: break

        return list(CenterNs), list(betas), list(mesos)

    def as_smiles_vec(self):
        rdmol = self.to_rdkit_Mol()
        CenterNs, betas, mesos = self.get_tetrapyrrole_atoms()
        Nneighbors = self.neighbors_of(CenterNs)

        ligands = []; CentralMetal = None
        for nb in Nneighbors:
            if not self.atoms[nb] == "C":
                if self.atoms[nb] == "H":
                    CentralMetal = 1
                    break
                else:
                    neighbors = self.neighbors_of(nb)
                    if all((el in neighbors) for el in CenterNs): # checks if neighbours of central metal are all pyrrolic nitrogens.
                        CentralMetal = Element(self.atoms[nb]).Z

                        for n in neighbors:
                            if not n in CenterNs:

                                frags = Chem.rdmolops.FragmentOnBonds(copy(rdmol), [self.bondmap.index(self.get_bond(n, nb))])
                                frags_idx = Chem.GetMolFrags(frags)
                                frags = Chem.GetMolFrags(frags, asMols=True, sanitizeFrags=False)

                                if not len(frags) == 2:
                                    print("Unsupported tetrapyrrole structure")
                                    return None

                                for f, i in zip(frags, frags_idx):
                                    if n in i:
                                        ligands.append(Chem.MolToSmiles(f))

        meso_subs = []
        for meso in mesos:
            for nb in self.neighbors_of(meso):
                if not nb in Nneighbors:
                    frags = Chem.rdmolops.FragmentOnBonds(copy(rdmol), [self.bondmap.index(self.get_bond(meso, nb))])
                    frags_idx = Chem.GetMolFrags(frags)
                    frags = Chem.GetMolFrags(frags, asMols=True, sanitizeFrags=False)

                    if not len(frags) == 2:
                        print("Unsupported tetrapyrrole structure")
                        return None

                    for f, i in zip(frags, frags_idx):
                        if nb in i:
                            meso_subs.append(Chem.MolToSmiles(f))

        beta_subs = []
        for beta in betas:
            for nb in self.neighbors_of(beta):
                if (not nb in Nneighbors) and (not nb in betas):
                    frags = Chem.rdmolops.FragmentOnBonds(copy(rdmol), [self.bondmap.index(self.get_bond(beta, nb))])
                    frags_idx = Chem.GetMolFrags(frags)
                    frags = Chem.GetMolFrags(frags, asMols=True, sanitizeFrags=False)

                    if len(frags) == 2:
                        for f, i in zip(frags, frags_idx):
                            if nb in i:
                                beta_subs.append(Chem.MolToSmiles(f))
                    else:
                        for n in self.neighbors_of(beta):
                            if n in betas:
                                for bn in self.neighbors_of(n):
                                    if (not bn in Nneighbors) and (not bn in betas):
                                        frags = Chem.rdmolops.FragmentOnBonds(copy(rdmol), [self.bondmap.index(self.get_bond(n, bn)), self.bondmap.index(self.get_bond(beta, nb))])
                                        frags_idx = Chem.GetMolFrags(frags)
                                        frags = Chem.GetMolFrags(frags, asMols=True, sanitizeFrags=False)

                                        if not len(frags) == 2:
                                            print("unsupported tetrapyrrole structure")
                                            return None

                                        for f, i in zip(frags, frags_idx):
                                            if bn in i:
                                                beta_subs.append(Chem.MolToSmiles(f))

        return CentralMetal, ligands, meso_subs, beta_subs

    def is_tetrapyrrole(self):
        rdmol = self.to_rdkit_Mol()
        pattlist = ["C1C=CC=N1", "C1=CC=CN1", "C1C=CCN1"]
        match_counter = 0
        for smtpatt in pattlist:
            patt = Chem.MolFromSmarts(smtpatt)
            match_counter += len(rdmol.GetSubstructMatches(patt))

        return match_counter == 4

    def from_smiles(self, string: str):
        if not " " in string:
            mol = Chem.MolFromSmiles(string, sanitize = False)
        else:
            vec = string.split(" ")
            if not '0' in vec[0]:
                metal = vec[0]
                ligands = []
                for i in range(13, 15):
                    if not '0' in vec[i]:
                        ligands.append(vec[i])
            else:
                metal = None
                ligands = None

            betas = []
            for i in range(1, 9):
                if not '*H' in vec[i]:
                    betas.append(vec[i])
                else:
                    betas.append(None)
            
            if '0' in vec[12]:
                tetrapyrrole_type = 'corrole'
            else:
                tetrapyrrole_type = 'porphyrin'
            
            mesos = []
            for i in range(4):
                if not '0' in vec[i + 9]:
                    mesos.append(vec[i + 9])

            self.from_subtituents(tetrapyrrole_type, betas, mesos, metal, ligands)

    def estimate_geometry(self):
        mol = self.to_rdkit_Mol()
        mol.UpdatePropertyCache(strict = False)
        Chem.GetSymmSSSR(mol)

        new_mol = Chem.rdchem.RWMol(mol); neighbors = []; centerM = None
        for atom in mol.GetAtoms():
            if len(atom.GetBonds()) > 3 and atom.GetAtomicNum() > 11:
                centerM = atom; new_mol.RemoveAtom(atom.GetIdx())
                m_idx = centerM.GetIdx()

                for n in atom.GetNeighbors():
                    idx = n.GetIdx()
                    if idx < m_idx:
                        neighbors.append(idx)
                    else:
                        neighbors.append(idx - 1)

        AllChem.EmbedMolecule(new_mol)

        if not centerM == None:
            new_mol = Chem.rdchem.RWMol(new_mol)
            frags_idxs = list(Chem.rdmolops.GetMolFrags(new_mol))
            if len(frags_idxs) > 1:
                frags_idxs.pop([len(f) for f in frags_idxs].index(max([len(f) for f in frags_idxs])))

            idx = new_mol.AddAtom(centerM)
            conf = new_mol.GetConformer()
            for index, ligand in enumerate(frags_idxs):
                diff = (np.mod(index + 1, 2) + 1) * np.ones(3)
                for i in list(ligand):
                    vec = conf.GetAtomPosition(i)
                    vec = np.array([vec.x, vec.y, vec.z])
                    vec = vec + diff
                    conf.SetAtomPosition(i, Point3D(vec[0], vec[1], vec[2]))

            coord = np.zeros(3)

            for n in neighbors:
                vec = conf.GetAtomPosition(n)
                coord += np.array([vec.x, vec.y, vec.z])
                new_mol.AddBond(n, idx, order=Chem.rdchem.BondType.DATIVE)

            coord = coord / len(neighbors)
            conf.SetAtomPosition(idx, Point3D(coord[0], coord[1], coord[2]))
            new_mol.UpdatePropertyCache(strict = False)
            Chem.GetSymmSSSR(new_mol)

        self.from_rdkit_mol(new_mol)
        self.add_hydrogens()
        self.MMGeoOpt(StepNum=1000)


    def from_subtituents(self, tetrapyrrole_type, betas = None, mesos = None, metal = None, ligands = None):
        # ** Add feature for more than single bond for ligands ! **
        '''
            Function takes vectors of smiles strings and inserts them to a tetrapyrrolic form.
            The vectors are of beta, meso, central metal and ligand subs.
            The smiles (except metal smiles) has to have a dummy atom connected to the atom that connects to the ring. 
            Hydrogens should be written as None.
        '''
        porphyrin_tup = (Chem.rdmolops.RemoveHs(Chem.MolFromSmiles('[NH]1c2cc3nc(cc4ccc(cc5C=Cc(n5)cc1cc2)[NH]4)C=C3')), [9, 10, 14, 15, 20, 21, 24, 23], [12, 18, 3, 7])
        corrole_tup = (Chem.rdmolops.RemoveHs(Chem.MolFromSmiles('N=1C=2C=CC=1c1[NH]c(C=C3C=CC(=Cc4ccc([NH]4)C=2)N3)cc1')), [3, 4, 23, 22, 11, 12, 16, 17], [9, 14, 20])
        tetrapyrrole_tup = {'porphyrin': porphyrin_tup, 'corrole': corrole_tup}
        mol, beta_idxs, meso_idxs = tetrapyrrole_tup[tetrapyrrole_type]
        beta_idxs = [idx - 1 for idx in beta_idxs]
        meso_idxs = [idx - 1 for idx in meso_idxs]
        Chem.rdmolops.Kekulize(mol)

        if not metal == None:
            metal = Chem.MolFromSmiles(metal)
            combo = Chem.EditableMol(Chem.CombineMols(mol, metal))
            for idx, atom in enumerate(mol.GetAtoms()):
                if atom.GetAtomicNum() == 7:
                    combo.AddBond(idx, mol.GetNumAtoms(), order = Chem.rdchem.BondType.SINGLE)
            mol = combo.GetMol()

        if not ligands == None:
            metal_idx = mol.GetNumAtoms() - 1
            for ligand in ligands:
                if not ligand == None:
                    ligand = Chem.MolFromSmiles(ligand, sanitize = False)
                    Chem.rdmolops.Kekulize(ligand)
                    atoms = [atom.GetAtomicNum() for atom in ligand.GetAtoms()]
                    binding_atom_idx = ligand.GetAtomWithIdx(atoms.index(0)).GetNeighbors()[0].GetIdx()

                    combo = Chem.EditableMol(Chem.CombineMols(mol, ligand))
                    combo.AddBond(metal_idx, binding_atom_idx + mol.GetNumAtoms(), order = Chem.rdchem.BondType.SINGLE)
                    combo.RemoveAtom(atoms.index(0) + mol.GetNumAtoms())
                    mol = combo.GetMol()

        if not betas == None:
            for beta, beta_idx in zip(betas, beta_idxs):
                if not beta == None:
                    beta = Chem.MolFromSmiles(beta)
                    Chem.rdmolops.Kekulize(beta)
                    atoms = [atom.GetAtomicNum() for atom in beta.GetAtoms()]
                    binding_atom_idx = beta.GetAtomWithIdx(atoms.index(0)).GetNeighbors()[0].GetIdx()

                    combo = Chem.rdchem.EditableMol(Chem.CombineMols(mol, beta))
                    combo.AddBond(beta_idx, binding_atom_idx + mol.GetNumAtoms(), order = Chem.rdchem.BondType.SINGLE)
                    combo.RemoveAtom(atoms.index(0) + mol.GetNumAtoms())
                    mol = combo.GetMol()
        
        if not mesos == None:
            for meso, meso_idx in zip(mesos, meso_idxs):
                if not meso == None:
                    meso = Chem.MolFromSmiles(meso)
                    Chem.rdmolops.Kekulize(meso)
                    atoms = [atom.GetAtomicNum() for atom in meso.GetAtoms()]
                    binding_atom_idx = meso.GetAtomWithIdx(atoms.index(0)).GetNeighbors()[0].GetIdx()

                    combo = Chem.EditableMol(Chem.CombineMols(mol, meso))
                    combo.AddBond(meso_idx, binding_atom_idx + mol.GetNumAtoms(), order = Chem.rdchem.BondType.SINGLE)
                    combo.RemoveAtom(atoms.index(0) + mol.GetNumAtoms())
                    mol = combo.GetMol()

        self.from_rdkit_mol(mol, kekulize=False)

def get_mol_from_CIF(path):
    my_structure = Structure.from_file(path)

    # begin ordering algorithm

    disordered_elements = []; disordered_sites = []; tot_nums = []
    for site in my_structure.sites:
        if site.is_ordered:
            continue
        else:
            element = str(site.species.elements[0])
            if element in disordered_elements:
                index = disordered_elements.index(element)
                disordered_sites[index].append(site)
                tot_nums[index] = tot_nums[index] + site.species.num_atoms
            else:
                disordered_sites.append([site])
                disordered_elements.append(element)
                tot_nums.append(site.species.num_atoms)

    for num, sites in zip(tot_nums, disordered_sites):
        pop_indicis = []
        for i in sorted(range(len(sites)), key=lambda i: sites[i].species.num_atoms)[int(num):]:
            pop_indicis.append(my_structure.sites.index(sites[i]))

        keep_indicis = []
        for i in sorted(range(len(sites)), key=lambda i: sites[i].species.num_atoms)[:int(num)]:
            keep_indicis.append(my_structure.sites.index(sites[i]))

        for i in keep_indicis:
            element = str(my_structure.sites[i].species.elements[0])
            my_structure.replace(i, element)

        my_structure.remove_sites(pop_indicis)

    # molecule finding
    sg = StructureGraph.with_local_env_strategy(my_structure, env_strategy())
    my_molecules = sg.get_subgraphs_as_molecules()

    # convert to Mol objects
    my_Mols = []
    for molecule in my_molecules:
        coords = []; atoms = []
        for site in molecule.sites:
            element = str(site.species.elements[0])
            atoms.append(element); coords.append(site.coords)
        mol = Mol()
        mol.atoms = atoms; mol.coords = coords; mol.EstimateBondmap(Set = True)
        my_Mols.append(mol)

    return my_Mols

def get_tetrapyrroles_from_cif(path, export_path, ExportFitting = True):
    if ExportFitting == True:
        my_structure = Structure.from_file(path)
        if my_structure.composition.contains_element_type("transition_metal"):
                raise Exception("File consists transition metals and can not be calculated")

    mols_in_cif = get_mol_from_CIF(path); tetrapyrrole_counter = 0
    for mol in mols_in_cif:
        if mol.is_tetrapyrrole():
            tetrapyrrole_counter += 1
            name, ext = os.path.splitext(os.path.split(path)[1])
            print(name, "is a tetrapyrrole !")
            mol.write_to_file(export_path + name + f'_{tetrapyrrole_counter}.mol')

def read_mopac_file(filename):
    atoms = []; coords = []; OxyEnergy = None; ElecAffinity = None; ES = None; ET = None; TotalE = None; FreqVec = []; RateVec = []
    with open(filename, "r") as f:
        CoordsBlock = False; ESandETBlock = False; OxyStateBlock = False; VibBlock = False; VibVec = []
        for line in f.readlines():
            wordsvec = re.split(r" |\t|\n", line)
            wordsvec = list(filter(lambda a: a != '', wordsvec))
            if len(wordsvec) == 0:
                continue

            if "".join(wordsvec) == "CARTESIANCOORDINATES":
                CoordsBlock = True
                coords = []
                continue

            if CoordsBlock and not len(wordsvec) == 5:
                CoordsBlock = False
                continue

            if CoordsBlock and not wordsvec[0] == "NO.":
                atoms.append(wordsvec[1])
                coords.append([float(x) for x in wordsvec[2:]])
                continue

            if line == "  STATE       ENERGY (EV)        Q.N.  SPIN   SYMMETRY              POLARIZATION\n":
                ESandETBlock = True
                continue
            if ESandETBlock and (len(wordsvec) == 6 or len(wordsvec) == 9):
                if wordsvec[4] == "SINGLET":
                    ES = float(wordsvec[2])
                    continue
                elif wordsvec[4] == "TRIPLET":
                    ET = float(wordsvec[2])
                    continue

            if not ES == None and not ET == None and ESandETBlock:
                ESandETBlock = False
                continue

            try:
                if wordsvec[0] + wordsvec[1] == "IONIZATIONPOTENTIAL":
                    OxyEnergy = float(wordsvec[3])
                    continue
                elif wordsvec[0] + wordsvec[1] == "HOMOLUMO":
                    ElecAffinity = float(wordsvec[-1])
                    continue
                elif wordsvec[0] + wordsvec[1] == "ALPHASOMO":
                    ElecAffinity = float(wordsvec[-1])
                    continue
                elif wordsvec[0] + wordsvec[1] == "BETASOMO":
                    if float(wordsvec[-1]) < ElecAffinity:
                        ElecAffinity = float(wordsvec[-1])
                    continue
                elif wordsvec[0] + wordsvec[1] == "TOTALENERGY":
                    TotalE = float(wordsvec[3])
                    continue
            except IndexError:
                continue
            
            if line == "          DESCRIPTION OF VIBRATIONS\n":
                VibBlock = True
                continue

            if VibBlock and line == "           FORCE CONSTANT IN CARTESIAN COORDINATES (Millidynes/A)\n":
                VibBlock = False
                continue

            if VibBlock:
                if wordsvec[0] == "FREQUENCY":
                    VibVec = []
                    VibVec.append(float(wordsvec[1]))
                elif wordsvec[0] + wordsvec[1] == "EFFECTIVEMASS":
                    try:
                        VibVec.append(float(wordsvec[2]))
                    except:
                        continue
                elif wordsvec[0] + wordsvec[1] == "FORCECONSTANT":
                    VibVec.append(float(wordsvec[2]))

                if len(VibVec) == 3 and VibVec[2] > 0 and not VibVec[0] in FreqVec:
                    FreqVec.append(VibVec[0])
                    RateVec.append(1 / np.power((VibVec[1] * VibVec[2]), 0.25))

    return {
        "atoms": atoms,
        "coords": coords,
        "ES": ES,
        "ET": ET,
        "OxyEnergy": OxyEnergy,
        "ElecAffinity": ElecAffinity,
        "TotalE": TotalE,
        "FreqVec": FreqVec,
        "RateVec": RateVec
    }

