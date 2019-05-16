# fully connected neural network
# every input neuron is connected to every output neuron
# activation function is in a separate layer
from nn_layer import Layer
import numpy as np

# inherit from base layer class
class FCLayer( Layer ):
    # input_size -> number of input neurons
    # output_size -> number of output neurons
    def __init__( self, input_size, output_size ):
        self.weights = np.random.rand( input_size, output_size ) - 0.5
        self.bias = np.random.rand( 1, output_size ) - 0.5

    # return output for given input
    def fwd_prop( self, input_data ):
        self.input = input_data
        self.output = np.dot( self.input, self.weights ) + self.bias
        return self.output

    # compute dE/dW, dE/dB for a given output_error = dE/dY, return input_error = dE/dX
    def back_prop( self, output_error, learning_rate ):
        input_error = np.dot( output_error, self.weights.T )
        weights_error = np.dot( self.input.T, output_error )
        # Bias = output_error

        # update the parameters!
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error