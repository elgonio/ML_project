from math import log
from DecisionTree.dataset import *

class Feature:
    def __init__(self, dataset, columnNum):
        self.dataset = dataset
        self.labelDict  = dataset.labelDict
        self.labelData = dataset.labelData
        self.columnID = columnNum
        self.values = []
        for row in range(0, len(dataset.data)):
            self.values.append(int(dataset.data[row][columnNum]))
        self.attributeDict = dict()
        for pos in range(0, len(self.values)):
            #Unfortunately I had to hard code checking for unique labels as it wouldnt work through iteration
            if self.values[pos] not in self.attributeDict:
                self.attributeDict[self.values[pos]] = {-1: 0, 1: 0}
            if self.labelData[pos] == -1:
                self.attributeDict[self.values[pos]][-1] += 1
            elif self.labelData[pos] == 1:
                self.attributeDict[self.values[pos]][1] += 1

            ''' Code for iteration method that failed to recognise a key value even though it should add it if not in the sub dictionary
            if self.values[pos] not in self.attributeDict:
                self.attributeDict[self.values[pos]] = dict()
            for label in self.labelDict:
                if labelData[pos] not in self.attributeDict[self.values[pos]]:
                    self.attributeDict[self.values[pos]] = 1
                else:
                    self.attributeDict[self.values[pos]] += 1
            '''

        #print(self.attributeDict)
        self.featureGain = self.calcFeatureGain()
        #print(self.featureGain)

    def get_Values(self):
        return self.values
    def get_ColumnID(self):
        return self.columnID
    def minusPLog2P(self, p):
        if p != 0:
            return (-1)*p*log(p,2)
        else:
            return 0
    def calcAttrTotal(self, attribute):
        t = 0
        for label in self.attributeDict[attribute]:
            t += self.attributeDict[attribute][label] #Calculating total labels under specific feature attribute
        return t
    def calcAttrEntropy(self, attribute):
        e = 0
        total = 0
        total = self.calcAttrTotal(attribute)
        for label in self.attributeDict[attribute]:
            e += self.minusPLog2P(self.attributeDict[attribute][label] / total)
        return e
    def calcFeatureGain(self):
        fG = 0.0
        fG = self.dataset.get_Entropy() #Starting entropy of entire dataset
        for attribute in self.attributeDict:
            fG += (-1)*(1/len(self.values))*(self.calcAttrTotal(attribute)*self.calcAttrEntropy(attribute))
        return fG
    def get_FeatureGain(self):
        return self.featureGain


