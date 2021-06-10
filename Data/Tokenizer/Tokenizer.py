from abc import ABC, abstractclassmethod

class Tokenizer (ABC):
    """Abstract class to handel general input tokenizations"""

    @abstractclassmethod
    def tokenize_vector(self, vec):
        """Method to tokenize a string"""
        return NotImplementedError

    @abstractclassmethod
    def translate_vector(self, vec):
        """Method to translate a tokenized vector back to a string"""
        return NotImplementedError