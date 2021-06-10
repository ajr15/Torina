import os

class IoWrapper:

    def __init__(self):
        pass

    @classmethod
    def write_input_str(cls, kw_dict: dict={}):
        """Method to write input files of a software.
        RETURNS: a string of input to a software"""
        pass
    
    @classmethod
    def write_input_file(cls, path, kw_dict: dict={}):
        """"Method to write input files of a software"""
        with open(path, "w") as f:
            f.write(write_input_str(kw_dict))
    
    @classmethod
    def read_input_str(cls, string):
        """Method to read input string to an IoWrapper object"""
        pass

    def read_input_file(cls, path):
        if not os.path.isfile(path):
            raise FileNotFoundError("Supplide input file doesn't exist")
        with open(path, "r") as f:
            s = f.read()
            return cls.read_input_str(s)

    @classmethod
    def read_output(cls, path):
        """Method to read output file to an IoWrapper object"""
        pass

    def run(self):
        """Method to run a computation with the desired software"""
        pass