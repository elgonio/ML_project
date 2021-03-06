from math import log
from DecisionTree.feature import *
from DecisionTree.attribute import *

class Dataset:
    def __init__(self, labelData, data):
        self.labelData = labelData
        # Create a 2D array to store labels and occurances of labels
        # Labels will either be 1 or -1
        self.labelDict = dict()
        self.data = data
        for label in self.labelData:
            if label not in self.labelDict:
                self.labelDict[label] = 1
            else:
                self.labelDict[label] += 1
        self.entropy = self.calcEntropy()
        #print(self.labelDict)

        f = Feature(self,1)
        #print("Num Features = "+str(self.get_NumFeatures()))
        #print(self.get_FeatureGains())

    def minusPLog2P(self, p):
        return (-1)*p*log(p,2)
    def calcEntropy(self):
        e = 0
        for label in self.labelDict:
            e += self.minusPLog2P(self.labelDict[label]/(len(self.labelData)))
        return e

    def get_Entropy(self):
        return self.entropy
    def get_Data(self):
        return self.data

    def get_FeatureGains(self):
        fGains = []
        for feature in range(0, self.get_NumFeatures()):
            fGains.append(Feature(self, feature).get_FeatureGain())
        return fGains

    def get_NumFeatures(self):
        return len(self.data[0])

    def get_majorityLabel(self):
        max = 0
        indexLabel = ""
        for label in self.labelDict:
            if self.labelDict[label] > max:
                max = self.labelDict[label]
                indexLabel = label
        return indexLabel

    def get_valuesOfFeature(self, featureID):
        f = Feature(self, featureID)
        f.get_Values()
        return f
    """
    Since the dota data set has no labelled features in the actual data set,
    numerical values will represent each feature
    """
    def get_Features(self):
        featureArray = []
        for f in range(0, self.get_NumFeatures()):
            featureArray.append(f)

        return f




