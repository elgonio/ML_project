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
        self.root = None
        self.labelData = labelData
        self.data = data
        self.dataset = Dataset(self.labelData, self.data)
        self.features = []
        for f in range(0, len(data[0])):
            self.features.append(f)

        self.entropy = self.dataset.get_Entropy()
        self.featureGainsArray = self.dataset.get_FeatureGains()
        self.largestGain = 0
        self.featureGainIndexID = 0
        counter = 0
        for gain in self.featureGainsArray:
            if (gain > self.largestGain):
                self.largestGain = gain
                self.featureGainIndexID = counter
            counter += 1

        print(self.featureGainsArray)
        print("Largest Feature Gain = " + str(self.largestGain) + " ID: "+str(self.featureGainIndexID))

    def singleLabelled(self):
        label = self.labelData[0]
        for id in range(0, len(self.labelData)):
            if self.labelData[id] != label:
                return False
        return True

    def buildTree(self):
        dataIds = [x for x in range(len(self.data))]
        self.root = self.id3Alg(dataIds, self.features, self.root)


    def id3Alg(self, dataIDs, features, root):
        root = Node()
        if self.singleLabelled():
            root.value = self.labels[dataIDs[0]]
            return root
        if len(features) == 0:
            root.value = self.dataset.get_majorityLabel()
            return root
        root.value = self.featureGainIndexID
        root.children = []






labelData, data = load_file("DhevansData.txt")
tree = DecisionTree(labelData, data)
tree.buildTree()




