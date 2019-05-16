import numpy as np 

from nn_network import Network 
from nn_fc_layer import FCLayer 
from nn_activation_layer import ActivationLayer 
from nn_activation_tan import tanh, tanh_prime 
from nn_loss import mse, mse_prime

# training data
x_train = np.array( [ [[0,0]], [[0,1]], [[1,0]], [[1,1]] ] )
y_train = np.array( [ [[0]], [[1]], [[1]], [[0]] ] )

# network
net = Network()
net.add( FCLayer( 2, 3 ) )
net.add( ActivationLayer( tanh, tanh_prime ) )
net.add( FCLayer( 3, 1 ) )
net.add( ActivationLayer( tanh, tanh_prime ) )

# train
net.use( mse, mse_prime )
net.fit( x_train, y_train, epochs = 1000, learning_rate = 0.1 )

# test
out = net.predict( x_train )
f = open( "results.txt", "w+" )
print( out )
f.write( str( out ) )
f.close()