import numpy as np
from .layer import Layer

class Network:
    def __init__(self, input_count, output_count, layer_count):
        self.input_count = input_count
        self.output_count = output_count
        self.layer_count = layer_count