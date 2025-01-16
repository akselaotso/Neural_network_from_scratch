import numpy as np
from .activation_functions import *

class Layer:
    def __init__(self, 
                output_count, 
                input_count, 
                optimizer = "Adam", 
                beta_1 = 0.9,
                beta_2 = 0.999
        ) -> None:

        self.input_count = input_count
        self.weights = 0.01 * np.random.randn(input_count, output_count)
        self.biases = np.zeros((1, output_count))
        self.optimizer = optimizer

        self.epsilon = 1e-7
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    
    def backward(self, dL_dz, stepsize, iterations):
        self.dL_dw = np.dot(np.transpose(self.inputs), dL_dz)
        self.dL_db = np.sum(dL_dz, axis=0, keepdims=True)
        self.gradient = np.dot(dL_dz, np.transpose(self.weights))
        
        match self.optimizer:
            case "grad":
                self.weights = self.weights - stepsize * self.dL_dw
                self.biases = self.biases - stepsize * self.dL_db
            case "Adam":
                if not hasattr(Layer, "weight_momentum"):
                    self.weight_momentum = np.zeros_like(self.weights)
                    self.weight_cache = np.zeros_like(self.weights)
                    self.bias_momentum = np.zeros_like(self.biases)
                    self.bias_cache = np.zeros_like(self.biases)

                momentum_correction_term = 1 - self.beta_1 ** (iterations + 1)
                cache_correction_term = 1 - self.beta_2 ** (iterations + 1)
                
                self.weight_momentum = self.beta_1 * self.weight_momentum + (1 - self.beta_1) * self.dL_dw
                self.weight_cache = self.beta_2 * self.weight_cache + (1 - self.beta_2) * self.dL_dw ** 2

                self.bias_momentum = self.beta_1 * self.bias_momentum + (1 - self.beta_1) * self.dL_db
                self.bias_cache = self.beta_2 * self.bias_cache + (1 - self.beta_2) * self.dL_db ** 2

                self.weights = self.weights - stepsize * (self.weight_momentum / momentum_correction_term) / np.sqrt(self.weight_cache / cache_correction_term + self.epsilon)

                
                self.biases = self.biases - stepsize * (self.bias_momentum / momentum_correction_term) / np.sqrt(self.bias_cache / cache_correction_term + self.epsilon)

            
            
            case _:
                raise Exception("Optimizer is not recognized.")
