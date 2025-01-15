import numpy as np
from .activation_functions import *

class Layer:
    def __init__(self, output_count, input_count, optimizer = "gradient descent") -> None:
        self.input_count = input_count
        self.weights = 0.01 * np.random.randn(input_count, output_count)
        self.biases = np.zeros((1, output_count))
        self.optimizer = optimizer

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    
    def backward(self, dL_dz, stepsize):
        self.dL_dw = np.dot(np.transpose(self.inputs), dL_dz)
        self.dL_db = np.sum(dL_dz, axis=0, keepdims=True)
        self.gradient = np.dot(dL_dz, np.transpose(self.weights))
        
        match self.optimizer:
            case "gradient descent":
                self.weights = self.weights - stepsize * self.dL_dw
                self.biases = self.biases - stepsize * self.dL_db
            case _:
                raise Exception("Optimizer is not recognized.")
