#Main function for decision tree algorithm

#tree = LearnTree(training_dataset)
#accuracy = DetermineAccuracy(tree, testing_dataset)

import numpy as np
import pandas as pd
from DecisionTree.data_loader import *
from DecisionTree.dataset import Dataset
from DecisionTree.node import Node





class DecisionTree:
    def __init__(self, labelData, data):
        self.labelData = labelData
        self.data = data
        self.dataset = Dataset(self.labelData, self.data)
        self.featureGainsArray = self.dataset.get_FeatureGains()
        self.largestGain = 0
        self.featureIndexID = 0
        counter = 0
        for gain in self.featureGainsArray:
            if (gain > self.largestGain):
                self.largestGain = gain
                self.featureIndexID = counter
            counter += 1

        print(self.featureGainsArray)
        print("Largest Feature Gain = " + str(self.largestGain) + " ID: "+str(self.featureIndexID))




labelData, data = load_file("DhevansData.txt")
tree = DecisionTree(labelData, data)




