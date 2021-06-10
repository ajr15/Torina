from .Base import CompProtocol
<<<<<<< HEAD
import sys; sys.path.append('../')
from Molecule.Base import to_rdkit_Mol
=======
from ..Molecule.Base import to_rdkit_Mol
>>>>>>> dev
from rdkit import Chem

class RDKitPropCalc (CompProtocol):

    def __init__(self, prop_func, prop_names=None):
        if callable(prop_func):
            self.prop_func = [prop_func]
        elif not type(prop_func) == list:
            raise ValueError("Property function must be an rdkit property function or a list of rdkit property functions. Not %s" % type(prop_func))
        if prop_names == None:
            prop_names = []
            for idx in range(len(self.prop_func)):
                prop_names.append('prop' + str(idx + 1))
        self.prop_names = prop_names

    def PropDict(self, specie):
        rdmol = to_rdkit_Mol(specie)
        rdmol.UpdatePropertyCache(strict = False)
        Chem.GetSymmSSSR(rdmol)
        prop_dict = {}
        for name, func in zip(self.prop_names, self.prop_func):
            prop_dict[name] = func(rdmol)
        return prop_dict