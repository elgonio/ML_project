#Dont need this class anymore
import numpy as np

class Attribute:

    def __init__(self, name):
        self.ID = name
        self.occurences = 0

    def get_ID(self):
        return self.ID
    def get_Occurances(self):
        return self.occurences
    def set_Occurances(self, count):
        self.occurences = count
    def __eq__(self, attr):
        return (isinstance(attr, Attribute) and (attr.getID == self.ID))


