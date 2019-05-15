# base class for a layer
class Layer:
    def __init__( self ):
        self.input = None
        self.output = None

    # forward propagation, compute output y for a given input x
    def fwd_prop( self, input ):
        raise NotImplementedError

    # back propagation, compute error and adjust itself using the learning rate(optimiser), dE/dX for a given dE/dY. update parameters
    def back_prop( self, output_error, learn_rate ):
        raise NotImplementedError