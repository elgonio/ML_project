import numpy as np 

from nn_network import Network 
from nn_fc_layer import FCLayer 
from nn_activation_layer import ActivationLayer 
from nn_activation_tan import tanh, tanh_prime 
from nn_loss import mse, mse_prime
from data_loader import load_file

# training data
#x_train = np.array( [ [[2,2,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,-1,0,0,0,-1,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]], 
#                    [[8,2,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,-1,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0]], 
#                    [[2,2,1,0,0,0,-1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,-1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]], 
#                    [[2,2,-1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,-1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,-1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]] ] )
#y_train = np.array( [ [[0]], [[1]], [[0]], [[1]] ] )

print("loading data ...")
y_train,x_train = load_file("dota2Test.csv")

 # we have to do funny stuff with the shape due to the way the NN is set up
x_train = np.reshape(x_train,[x_train.shape[0],1,x_train.shape[1]])

print("loading completed \n")


# network
net = Network()
# number of layers
num_layers = 1
num_nodes = 115
# init layer
net.add( FCLayer( x_train.shape[2], num_nodes ) )
net.add( ActivationLayer( tanh, tanh_prime ) )

# create amount of layers
for n in range( num_layers - 2 ):
    net.add( FCLayer( num_nodes, num_nodes ) )
    net.add( ActivationLayer( tanh, tanh_prime ) )

# final layer
net.add( FCLayer( num_nodes, 1 ) )
net.add( ActivationLayer( tanh, tanh_prime ) )

# train
net.use( mse, mse_prime )
net.fit( x_train, y_train, epochs = 100, learning_rate = 0.003 )

# test
out = np.array(net.predict_hr( x_train ))
accuracy = 0
for i in range(len(out)):
    if out[i] == y_train[i]:
        accuracy += 1

    #print("pred:    ", out[i], "   truth: ", y_train[i])

accuracy = accuracy/len(out)
print("Final accuracy: ", accuracy)
# f = open( "results.txt", "w+" )
#print( out )
# f.write( str( out ) )
# f.close()