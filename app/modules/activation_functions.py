import numpy as np

class Activation_Function:
    def __init__(self):
        pass

    def forward(self, x):
        self.output = x

    def backward(self, x):
        self.gradient = x


class ReLu(Activation_Function):
    def forward(self, x):
        self.input = x
        self.output = np.maximum(0, x)

    def backward(self, x):
        self.gradient = x.copy()
        self.gradient[self.input <= 0] = 0


class Softmax(Activation_Function):
    def forward(self, x):
        # normalize and take exponent
        vals = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.output = vals / np.sum(vals, axis=1, keepdims=True)

    # Backward pass not implemented because jacobian matrix 
    # see softmax_crossentropy_class for combined implementation of 
    # softmax and crossentropy
