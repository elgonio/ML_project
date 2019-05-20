import numpy as np

"""
reads in a csv file, converts labels into the correct form
then splits the labels from the data and returns 2 seperate arrays
"""
def load_file(filename):
    data = np.genfromtxt(filename,delimiter=',')
    # convert the labels into the correct form
    #for row in data:
    #    if row[0] == -1.0:
    #        row[0] = 0
    #    elif row[0] == 1.0:
    #        row[0] = 1
            

    # y is the first column i.e the labels
    y = data[:,0]
    # X is the remainder of the data
    X = data[:,2:]
 
    return y,X