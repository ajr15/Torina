
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
        rdmol = to_rdkit_Mol(self)
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
        rdmol = to_rdkit_Mol(self)
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
        mol = to_rdkit_Mol(self)
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