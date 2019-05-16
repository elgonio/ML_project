import numpy as np 

from nn_network import Network 
from nn_fc_layer import FCLayer 
from nn_activation_layer import ActivationLayer 
from nn_activation_tan import tanh, tanh_prime 
from nn_loss import mse, mse_prime
# Data loader
from data_loader import load_file


# training data
#x_train = np.array( [ [[2,2,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,-1,0,0,0,-1,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]], 
#                    [[8,2,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,-1,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0]], 
#                    [[2,2,1,0,0,0,-1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,-1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]], 
#                    [[2,2,-1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,-1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,-1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]] ] )
#y_train = np.array( [ [[-1]], [[1]], [[-1]], [[1]] ] )

# load the data
print( "loading data..." )
y_train, x_train = load_file( "dota2Test.csv" )
# we have to do funny things with the shape due to the way the NN is set up
x_train = np.reshape( x_train, [x_train.shape[0], 1, x_train.shape[1]] )
print( "loading completed \n" )

# network
net = Network()
# 3 layers (1 hidden layer)
num_layers = 3
num_nodes = 115
# init layer
net.add( FCLayer( 115, num_nodes ) )
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
net.fit( x_train, y_train, epochs = 1000, learning_rate = 0.02 )

# test
out = net.predict( x_train )  # replace with test when testing, then check outputs against the validation
accuracy = 0
for i in range( len( out ) ):
    if out[i] == y_train[i]:
        accuracy += 1
    # print output to check
    print( out )

accuracy = accuracy/len( out )
print( "Final accuracy: ", accuracy )

# OBSOLETE
# creating hidden layers
#for n in range( num_layers - 2 ):
#    net.add( FCLayer( num_nodes, num_nodes ) )
#    net.add( ActivationLayer( tanh, tanh_prime ) )

# writing out to file
# f = open( "results.txt", "w+" )
# print( out )
# f.write( str( out ) )
# f.close()
