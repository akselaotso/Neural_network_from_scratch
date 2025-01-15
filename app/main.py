import numpy as np
from nnfs.datasets import spiral_data
from .modules.activation_functions import *
from .modules.layer import Layer
from .modules.loss_functions import * 
from .modules.softmax_crossentropy_class import Softmax_Cross_Entropy
from .utils.accuracy_calculator import accuracy_calculator

    
def main():
    print("Starting progam.\n")
    stepsize = initial_stepsize = 0.03
    stepsize_decay = 0
    samples = 100
    classes = 3
    X, y = spiral_data(samples=samples, classes=classes)

    layer1 = Layer(output_count=64, input_count=2)
    activation1 = ReLu()
    layer2 = Layer(output_count=3, input_count=64)
    loss_function = Softmax_Cross_Entropy()

    for i in range(200000):
        if stepsize_decay > 0:
            stepsize = initial_stepsize / (1  + stepsize_decay * i)

        layer1.forward(X)
        activation1.forward(layer1.output)
        layer2.forward(activation1.output)
        loss_function.forward(layer2.output, y)

        loss_function.backward(loss_function.output, y)
        layer2.backward(loss_function.gradient, stepsize)
        activation1.backward(layer2.gradient)
        layer1.backward(activation1.gradient, stepsize)

        if not i % 100:
            print(f'i: {i}, accuracy: {accuracy_calculator(loss_output=loss_function.output, y=y):.3f}, loss: {np.mean(loss_function.loss):.3f}')
    
    print(f'\nFinal values:\ni: {i}, accuracy: {accuracy_calculator(loss_output=loss_function.output, y = y):.3f}, loss: {np.mean(loss_function.loss):.3f}')    
    print("\nProgram complete.")


if __name__ == "__main__":
    main()

