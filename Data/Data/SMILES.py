from .Text import Text
from ...Molecule.Base import BaseMol

class SMILES (Text):

    def __init__(self, parent_specie=BaseMol):
        super().__init__()
        self.parent_specie = BaseMol

    def to_specie(self, x):
        return self.parent_specie.from_str(''.join(x))