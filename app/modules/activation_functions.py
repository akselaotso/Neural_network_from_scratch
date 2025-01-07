import numpy as np

class Activation_Function:
    def __init__(self):
        pass

    def calculate(self, x):
        return x

    def derivate(self, x):
        return x


class ReLu(Activation_Function):
    def calculate(self, x):
        return np.maximum(0, x)

    def derivate(self, x):
        return np.where(x > 0, 1, 0)
    
    
class Sigmoid(Activation_Function):
    def calculate(self, x):
        return 1 / (1 + np.exp(-x))

    def derivate(self, x):
        sigmoid = self.calculate(x)
        return sigmoid * (1 - sigmoid)


class Softmax(Activation_Function):
    def calculate(self, x):
        # vals = np.exp(x - np.max(x, axis=1, keepdims=True))
        vals = np.exp(x)
        return vals / np.sum(vals, axis=1, keepdims=True)
    
    def derivate(self, x):
        return x

