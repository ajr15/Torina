from IoWrapper import IoWrapper
from Molecule.Base import BaseMol
from XYZ import XYZ
import os

class ORCA (IoWrapper):

    """General structure of kw_dict:
            {
                "kw_top_line": [LIST OF TOP KEY WORDS],
                "BLOCK_NAME1": [BLOCK_PARAMETER_SENTENCE]
                ...
                "XYZ": {"CHARGE": CHARGE OF MOL,
                        "MULT": MULTIPLICITY OF MOL}
                "COORDS": [XYZ COORDS OF ATOMS IN MOLECULE] - optional (can be supplide by Molecule object),
                "ATOMS": [ATOMIC SYMBOLS OF ATOMS IN MOLECULE] - optional (can be supplide by Molecule object)
            }

        Atributes:
            - kw_dict: dict: Dictionary with input keywords parameters
            - scf_converged: bool: did the scf computation converge?
            - geo_converged: bool: did the geometry optimization converge?
            - terminated_normally: bool: did the computation ended normally?
            - molecule: Molecule: Molecule object of the molecule in input/output file"""

    def __init__(self):
        self.scf_converged = False
        self.geo_converged = False
        self.terminated_normally = False
        self.molecule = BaseMol()
        
        self._kw_dict = {}
        self._file_path = None
        self._final_energy = None # hidden variable to store energy output of calculation

    @property
    def kw_dict(self):
        return self._kw_dict

    @kw_dict.setter
    def kw_dict(self, kw_dict: dict):
        # set a kw_dict attr and update Mol attr
        d = {k: v for k, v in self._correct_kw_dict(kw_dict).items() if not k in ["coords", "atoms"]}
        self._kw_dict = d
        self._molecule.atoms = self._correct_kw_dict(kw_dict)["atoms"]
        self._molecule.coords = self._correct_kw_dict(kw_dict)["coords"]

    @kw_dict.getter
    def kw_dict(self):
        # get XYZ data from Mol attr and return dict
        d = {k: v for k, v in self._kw_dict.items()}
        d["atoms"] = self._molecule.atoms
        d["coords"] = self._molecule.coords
        return d

    @staticmethod
    def _correct_kw_dict(kw_dict):
        d = {k.lower(): v for k, v in kw_dict.items()}
        if any([k not in d.keys() for k in ["kw_top_line", "xyz", "coords", "atoms"]]):
            raise ValueError("kw_dict is missing some mandatory keywords, please refere to the docs to see the kw_dict format.")
        if not len(d["atoms"]) == len(d["coords"]):
            raise ValueError("Number of atoms is not equal to number of coords")
        return d

    def write_input_str(cls, path, kw_dict: dict={}):
        d = cls._correct_kw_dict(kw_dict)
        block_names = [k for k in d.keys() if k not in ["kw_top_line", "xyz", "coords", "atoms"]]
        s = ""
        # writing top keywords
        s += "! " + " ".join(d["kw_top_line"]) + "\n\n"
        # writing blocks
        for k in block_names:
            s += "%" + k + "\n"
            for sentence in d[k]:
                s += "\t" + sentence + "\n"
            s += "end" + "\n\n"
        # writing xyz block
        s += "* xyz " + str(d["xyz"]["charge"]) + " " + str(d["xyz"]["mult"]) + "\n"
        for a, c in zip(d['atoms'], d['coords']):
            s += "\t".join(['', a] + [str(x) for x in c]) + "\n"
        s += "*"
        return s

    def read_input_str(self, string):
        """Method to parse input string to an ORCA object"""
        pass

    def read_output(self, path):
        """Method to parse output file to an ORCA object
        RETURNS:
            An ORCA object with data from output file. Data could be retreived by the class methods."""
        if not os.path.isfile(path):
            raise FileNotFoundError("Supplide output file doesn't exist")
        
        self._file_path = path
        # reads molecule, energy, comp. time and convergence data
        with open(path, "r") as f:
            InputBlock = False
            input_str = ""
            for line in f.readlines():
                if "INPUT FILE" in line:
                    InputBlock = True
                if "====" in line:
                    continue
                if InputBlock:
                    if ">" in line:
                        t = line.split("> ")[-1]
                        if "END OF INPUT" in t:
                            InputBlock = False
                        else:
                            input_str += t
                if "FINAL SINGLE POINT ENERGY" in line:
                    self._energy = float(line.split()[-1])
                    continue
                if "*** OPTIMIZATION RUN DONE ***" in line:
                    self.geo_converged = True
                    continue
                if "****ORCA TERMINATED NORMALLY****" in line:
                    self.terminated_normally = True
                    continue
                if "SCF CONVERGED AFTER" in line:
                    self.scf_converged = True
                    continue
                if "TOTAL RUN TIME" in line:
                    v = [line.split()[2 * i + 3] for i in range(5)]
                    self._comp_time = 24 * 60 * 60 * v[0] + \
                                           60 * 60 * v[1] + \
                                                60 * v[2] + \
                                                     v[4] + \
                                             0.001 * v[5]
                    continue
            # parsing input string
            self.read_input_str(input_str)
            # checking for final geometry
            xyz_file = os.path.splitext(path)[0] + '.xyz'
            self.molecule = XYZ.read_file(xyz_file).get_mol()

    def get_energy(self):
        return self._final_energy
    
    def get_comp_time(self):
        return self._comp_time

    def get_mol(self):
        return self.molecule