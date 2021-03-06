import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import random
from pprint import pprint


def check_purity(data):
    label_column = data[:, -1]
    unique_classes = np.unique(label_column)

    if len(unique_classes) == 1:
        return True
    else:
        return False


def classify_data(data):
    label_column = data[:, -1]
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)

    index = counts_unique_classes.argmax()
    classification = unique_classes[index]

    return classification


def get_potential_splits(data):
    potential_splits = {}
    _, n_columns = data.shape
    for column_index in range(n_columns - 1):  # excluding the last column which is the label
        values = data[:, column_index]
        unique_values = np.unique(values)

        type_of_feature = FEATURE_TYPES[column_index]
        if type_of_feature == "continuous":
            potential_splits[column_index] = []
            for index in range(len(unique_values)):
                if index != 0:
                    current_value = unique_values[index]
                    previous_value = unique_values[index - 1]
                    potential_split = (current_value + previous_value) / 2

                    potential_splits[column_index].append(potential_split)

        # feature is categorical
        # (there need to be at least 2 unique values, otherwise in the
        # split_data function data_below would contain all data points
        # and data_above would be empty)
        elif len(unique_values) > 1:
            potential_splits[column_index] = unique_values

    return potential_splits


def split_data(data, split_column, split_value):
    split_column_values = data[:, split_column]

    type_of_feature = FEATURE_TYPES[split_column]
    if type_of_feature == "continuous":
        data_below = data[split_column_values <= split_value]
        data_above = data[split_column_values > split_value]

    # feature is categorical
    else:
        data_below = data[split_column_values == split_value]
        data_above = data[split_column_values != split_value]

    return data_below, data_above


def calculate_entropy(data):
    label_column = data[:, -1]
    _, counts = np.unique(label_column, return_counts=True)

    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))

    return entropy


def calculate_overall_entropy(data_below, data_above):
    n = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n
    p_data_above = len(data_above) / n

    overall_entropy = (p_data_below * calculate_entropy(data_below)
                       + p_data_above * calculate_entropy(data_above))

    return overall_entropy


def determine_best_split(data, potential_splits):
    overall_entropy = 9999
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_below, data_above = split_data(data, split_column=column_index, split_value=value)
            current_overall_entropy = calculate_overall_entropy(data_below, data_above)

            if current_overall_entropy <= overall_entropy:
                overall_entropy = current_overall_entropy
                best_split_column = column_index
                best_split_value = value

    return best_split_column, best_split_value


def determine_type_of_feature(df):
    feature_types = []
    n_unique_values_treshold = 15
    for feature in df.columns:
        if feature != "label":
            unique_values = df[feature].unique()
            example_value = unique_values[0]

            if (isinstance(example_value, str)) or (len(unique_values) <= n_unique_values_treshold):
                feature_types.append("categorical")
            else:
                feature_types.append("continuous")

    return feature_types


def decision_tree_algorithm(df, counter=0, min_samples=2, max_depth=5):
    # data preparations
    if counter == 0:
        global COLUMN_HEADERS, FEATURE_TYPES
        COLUMN_HEADERS = df.columns
        FEATURE_TYPES = determine_type_of_feature(df)
        data = df.values
    else:
        data = df

        # base cases
    if (check_purity(data)) or (len(data) < min_samples) or (counter == max_depth):
        classification = classify_data(data)

        return classification


    # recursive part
    else:
        counter += 1

        # helper functions
        potential_splits = get_potential_splits(data)
        split_column, split_value = determine_best_split(data, potential_splits)
        data_below, data_above = split_data(data, split_column, split_value)

        # determine question
        feature_name = COLUMN_HEADERS[split_column]
        type_of_feature = FEATURE_TYPES[split_column]
        if type_of_feature == "continuous":
            question = "{} <= {}".format(feature_name, split_value)

        # feature is categorical
        else:
            question = "{} = {}".format(feature_name, split_value)

        # instantiate sub-tree
        sub_tree = {question: []}

        # find answers (recursion)
        yes_answer = decision_tree_algorithm(data_below, counter, min_samples, max_depth)
        no_answer = decision_tree_algorithm(data_above, counter, min_samples, max_depth)

        # If the answers are the same, then there is no point in asking the qestion.
        # This could happen when the data is classified even though it is not pure
        # yet (min_samples or max_depth base case).
        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)

        return sub_tree


def classify_example(example, tree):
    question = list(tree.keys())[0]
    feature_name, comparison_operator, value = question.split(" ")

    # ask question
    if comparison_operator == "<=":
        if example[feature_name] <= float(value):
            answer = tree[question][0]
        else:
            answer = tree[question][1]

    # feature is categorical
    else:
        if str(example[feature_name]) == value:
            answer = tree[question][0]
        else:
            answer = tree[question][1]

    # base case
    if not isinstance(answer, dict):
        return answer

    # recursive part
    else:
        residual_tree = answer
        return classify_example(example, residual_tree)


def calculate_accuracy(df, tree):
    df["classification"] = df.apply(classify_example, args=(tree,), axis=1)
    df["classification_correct"] = df["classification"] == df["label"]

    labels = df["label"]
    predictions = df["classification"]
    confusionmatrix = {"TruePositive": 0,"TrueNegative": 0,"FalsePositive":0,"FalseNegative":0}
    for x in range(0, len(labels)):
        if labels[x] == predictions[x] and (labels[x] == 1 ):
            confusionmatrix['TruePositive'] += 1
        elif labels[x] == predictions[x] and (labels[x] == -1):
            confusionmatrix['TrueNegative'] += 1
        elif labels[x] != predictions[x] and (labels[x] == 1 and predictions[x] == -1):
            confusionmatrix['FalseNegative'] += 1
        elif labels[x] != predictions[x] and (labels[x] == -1 and predictions[x] == 1):
            confusionmatrix['FalsePositive'] += 1

    accuracy = df["classification_correct"].mean()

    return accuracy, confusionmatrix

if __name__ == '__main__':


    trainingDf = pd.read_csv("dota2TrainSmall.csv")
    trainingDf['label'] = trainingDf['labels']
    trainingDf = trainingDf.drop(['labels', 'clusterID'], axis=1)  # drop column 1 and 2
    testDf = pd.read_csv("dota2TestSmall.csv")
    testDf['label'] = testDf['labels']
    testDf = testDf.drop(['labels', 'clusterID'], axis=1)  # drop column 1 and 2

    treeDepth = 31
    x = []
    y = []
    for case in range(1, 2):
        x.append(treeDepth)
        tree = decision_tree_algorithm(trainingDf, max_depth=treeDepth)
        testAcc, confusionDict = calculate_accuracy(testDf, tree)
        testAcc = np.around(testAcc*100,2)
        y.append(testAcc)
        print("Case {}: Tree Depth = {}     Accuracy on Test Data: {}%".format(case,treeDepth,testAcc))
        print((confusionDict))
        treeDepth += 2

    trainingDf = pd.read_csv("dota2TrainSmall.csv")
    trainingDf['label'] = trainingDf['labels']
    trainingDf = trainingDf.drop(['labels', 'clusterID'], axis=1)  # drop column 1 and 2
    testDf = pd.read_csv("dota2TestSmall.csv")
    testDf['label'] = testDf['labels']
    testDf = testDf.drop(['labels', 'clusterID'], axis=1)  # drop column 1 and 2

    '''
       treeDepth = 3
    y2 = []
    for case in range(1, 31):
        tree2 = decision_tree_algorithm(trainingDf, max_depth=treeDepth)
        trainingAcc = np.around(calculate_accuracy(trainingDf, tree2)*100,2)
        y2.append(trainingAcc)
        print("Case {}: Tree Depth = {}     Accuracy on Training Data: {}%".format(case,treeDepth,trainingAcc))
        treeDepth += 2 
    '''


    #print(x)
    #print(y)
    #Plot on graph
    plt.plot(x,y, linestyle='--', marker='o', color='r', label="Test Data")
    #plt.plot(x,y2, linestyle='--', marker='o', color='b', label="Training Data")
    plt.axis([0, 70, 51, 56])
    plt.xlabel('Tree Depth')
    plt.ylabel('Accuracy (%)')
    plt.title('Line Graph showing the relationship between tree depth and accuracy for test data')
    plt.grid()
    plt.legend()
    plt.show()

