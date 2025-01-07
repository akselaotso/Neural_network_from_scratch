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
    
    
class Sigmoid(Activation_Function):
    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))

    def backward(self, x):
        sigmoid = self.forward(x)
        self.gradient = sigmoid * (1 - sigmoid)


class Softmax(Activation_Function):
    def forward(self, x):
        # vals = np.exp(x - np.max(x, axis=1, keepdims=True))
        vals = np.exp(x)
        self.output = vals / np.sum(vals, axis=1, keepdims=True)

