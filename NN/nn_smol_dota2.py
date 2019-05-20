import numpy as np

from nn_network import Network 
from nn_fc_layer import FCLayer 
from nn_activation_layer import ActivationLayer 
from nn_activation_tan import tanh, tanh_prime 
from nn_loss import mse, mse_prime
# Data loader
from data_loader import load_file


# load the data
print( "loading training data..." )
y_train, x_train = load_file("D:/2019/Semester1/COMS3007 - ML/Project/Repo/NN/dota2nnTrain.csv")
# we have to do funny things with the shape due to the way the NN is set up
x_train = np.reshape( x_train, [x_train.shape[0], 1, x_train.shape[1]] )
print( "loading training data completed \n" )

# network
net = Network()
# number of layers
num_layers = 2
num_nodes = 115
# init layer
net.add( FCLayer( x_train.shape[2], num_nodes ) )
net.add( ActivationLayer( tanh, tanh_prime ) )
# hidden layer(s)
for n in range( num_layers - 2 ):
    net.add( FCLayer( num_nodes, num_nodes ) )
    net.add( ActivationLayer( tanh, tanh_prime ) )
# final layer
net.add( FCLayer( num_nodes, 1 ) )
net.add( ActivationLayer( tanh, tanh_prime ) )

# train
net.use( mse, mse_prime )
net.fit( x_train, y_train, epochs = 500, learning_rate = 0.005 )

# test training results
out = np.array( net.predict_hr( x_train ) )
accuracy = 0

for i in range( len( out ) ):
    if out[i] == y_train[i]:
        accuracy += 1
    # print output to check
    #print( "Current Accuracy: " )
    #print( "pred: ", out[i], " truth: ", y_train[i] )

accuracy = accuracy/len( out )
print( "Final accuracy: ", accuracy )

print( "loading testing data..." )
y_test, x_test = load_file("D:/2019/Semester1/COMS3007 - ML/Project/Repo/NN/dota2nnTest.csv")
x_test = np.reshape( x_test, [x_test.shape[0], 1, x_test.shape[1]] )
print( "loading testing data completed \n" )

# testing
test = np.array( net.predict_hr( x_test ) )
test_accuracy = 0

# confusion matrix things
cm_tp = 0
cm_tn = 0
cm_fp = 0
cm_fn = 0

for i in range( len( test ) ):
    if test[i] == y_test[i]:
        test_accuracy += 1

        # confusion matrix
        if y_test[i] == 1 and test[i] == 1:
            tp += 1
        elif y_test[i] == -1 and test[i] == -1:
            tn += 1
        elif y_test[i] == -1 and test[i] == 1:
            fp += 1
        elif y_test[i] == 1 and test[i] == -1:
            fn += 1  
    #print( "pred: ", test[i], "truth: ", y_test[i] )

test_accuracy = test_accuracy/len( test )
print( "Final accuracy: ", test_accuracy, "\n" )
print( "TP: ", cm_tp, "TN: ", cm_tn, "FP: ", cm_fp, "FN: ", cm_fn  )


# OBSOLETE
# training data
#x_train = np.array( [ [[2,2,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,-1,0,0,0,-1,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]], 
#                    [[8,2,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,-1,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0]], 
#                    [[2,2,1,0,0,0,-1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,-1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]], 
#                    [[2,2,-1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,-1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,-1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]] ] )
#y_train = np.array( [ [[-1]], [[1]], [[-1]], [[1]] ] )