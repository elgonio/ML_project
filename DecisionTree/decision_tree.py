#Main function for decision tree algorithm

#tree = LearnTree(training_dataset)
#accuracy = DetermineAccuracy(tree, testing_dataset)

import numpy as np
import pandas as pd
from DecisionTree.data_loader import *
from DecisionTree.dataset import Dataset


labelData, data = load_file("dota2Toy.csv")

dataset = Dataset(labelData, data)

fgArray = dataset.get_FeatureGains()
largestGain = 0
count = 0
for gain in fgArray:
    if(gain > largestGain):
        largestGain = gain
        print(count)
    count += 1
print(fgArray)
print("Largest Feature Gain = " + str(largestGain))



