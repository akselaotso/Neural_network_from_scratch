import numpy as np
from .loss_functions import Cross_Entropy_Loss

class Softmax_Cross_Entropy:
    def forward(self, input, y):
        # Softmax activation function into self.output
        vals = np.exp(input - np.max(input, axis=1, keepdims=True))
        self.output = vals / np.sum(vals, axis=1, keepdims=True)
        loss = Cross_Entropy_Loss()
        self.loss = loss.forward(self.output, y)
    
    def backward(self, y_pred, y_true):
        if len(y_true.shape) == 2:
            # returns the index of the 1 for each item
            y_true = np.argmax(y_true, axis=1)

        self.gradient = y_pred.copy()
        self.gradient[range(len(y_pred)), y_true] -= 1
        self.gradient = self.gradient / len(y_pred)

