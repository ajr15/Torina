from Molecule.Base import BaseMol
import os

class XYZ:

    def __init__(self):
        self.atoms = None
        self.coords = None
    
    @classmethod
    def read_file(cls, path):
        if not os.path.isfile(path):
            raise ValueError("Supplied XYZ file doesn't exist")
        atoms = []
        coords = []
        with open(path, "r") as f:
            for line in f.readlines():
                v = line.split()
                if len(v) == 4:
                    atoms.append(v[0])
                    coords.append([float(x) for x in v[1:]])
        return XYZ(atoms, coords)

    def get_mol(self):
        return BaseMol(atoms=self.atoms, coords=self.coords)