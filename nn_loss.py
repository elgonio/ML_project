# error, loss of the network
# mean squared error
import numpy as np 

# loss function
def mse( y_true, y_pred ):
    return np.mean( np.power( y_true - y_pred, 2 ))

# derivative
def mse_prime( y_true, y_pred ):
    return 2 * ( y_pred - y_true ) / y_true.size