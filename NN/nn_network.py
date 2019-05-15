class Network:
    def __init__( self ):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # adding a layer to the network
    def add( self, layer ):
        self.layers.append( layer )

    # setting the loss
    def use( self, loss, loss_prime ):
        self.loss = loss
        self.loss_prime = loss_prime

    # predict output for given input
    def predict( self, input_data ):
        # sample the dimension first
        samples = len( input_data )
        result = []

        # run network over all samples
        for i in range( samples ):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.fwd_prop( output )
            result.append( output )

        return result

    # training the network
    def fit( self, x_train, y_train, epochs, learning_rate ):
        # sample dimension first
        samples = len( x_train )

        # training loop
        for i in range( epochs ):
            err = 0
            for j in range( samples ):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.fwd_prop( output )

                # get loss and display
                err += self.loss( y_train[j], output )

                # back propagation
                error = self.loss_prime( y_train[j], output )
                for layer in reversed( self.layers ):
                    error = layer.back_prop( error, learning_rate )

            # calculate average error on all samples
            err /= samples
            print( 'epoch %d/%d error=%f' % ( i+1, epochs, err ) )