import numpy as np
from .activation_functions import *

class Layer:
    def __init__(self, output_count, input_count, activation_function: Activation_Function) -> None:
        self.input_count = input_count
        self.weights = 0.01 * np.random.randn(input_count, output_count)
        self.biases = np.zeros((1, output_count))
        self.activation_function = activation_function

    def forward(self, inputs):
        self.inputs = inputs
        self.output = self.activation_function.calculate(np.dot(inputs, self.weights) + self.biases)
        return self.output
    
    def backward(self, dL_da, stepsize):
        da_dz = self.output
        dL_dz = dL_da * da_dz
        self.weights -= stepsize * np.dot(np.transpose(self.inputs), dL_dz)
        self.biases -= stepsize * np.sum(dL_da, axis=0, keepdims=True)
        self.dL_dx = np.dot(dL_dz, np.transpose(self.weights))

