from .TextVec import TextVec

class SmilesVec (TextVec):

    def __init__(self, parent_specie, vec_sep=" "):
        super().__init__()
        self.parent_specie = parent_specie
        self.vec_sep = vec_sep

    def to_specie(self, x):
        mol = self.parent_specie()
        mol.from_vec([''.join(v) for v in x])
        return mol
