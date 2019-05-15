# ReLu activation function
import numpy as np 

def relu( x, w ):
    return np.clip( (x * w), 0, None, out=None )