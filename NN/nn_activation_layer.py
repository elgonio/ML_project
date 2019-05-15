# activation layer
# adds non-linearity to the system to be able to learn and computes dE/dX
# input and output have same dimensions
from nn_layer import Layer 

# inherit from base layer class
class ActivationLayer( Layer ):
    def __init__( self, activation, activation_prime ):
        self.activation = activation
        self.activation_prime = activation_prime

    # return activated input
    def fwd_prop( self, input_data ):
        self.input = input_data
        self.output = self.activation( self.input )
        return self.output

    # return input_error=dE/dX for a given output_error=dE/dY
    def back_prop( self, output_error, learning_rate ):
        return self.activation_prime( self.input ) * output_error
