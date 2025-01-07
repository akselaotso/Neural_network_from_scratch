import numpy as np
from nnfs.datasets import spiral_data
from .modules.activation_functions import *
from .modules.layer import Layer
from .modules.loss_functions import * 
from .modules.softmax_crossentropy_class import Softmax_Cross_Entropy


def accuracy_calculator(prediction, true):
    prediction = np.argmax(prediction, axis=1)
    return np.mean(prediction == true)

    
def main():
    print("Starting progam.\n")
    stepsize = 0.03
    samples = 100
    classes = 3
    X, y = spiral_data(samples=samples, classes=classes)

    layer1 = Layer(output_count=64, input_count=2)
    activation1 = ReLu()
    layer2 = Layer(output_count=3, input_count=64)
    loss_function = Softmax_Cross_Entropy()

    for i in range(int(1e6)):
        layer1.forward(X)
        activation1.forward(layer1.output)
        layer2.forward(activation1.output)
        loss_function.forward(layer2.output, y)

        loss_function.backward(loss_function.output, y)
        layer2.backward(loss_function.gradient, stepsize)
        activation1.backward(layer2.gradient)
        layer1.backward(activation1.gradient, stepsize)
        
        predictions = np.argmax(loss_function.output, axis=1)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == y)

        if not i % 100:
            print(f'i: {i}, accuracy: {accuracy:.3f}, loss: {np.mean(loss_function.loss):.3f}')
    
    print("\nProgram complete.")


if __name__ == "__main__":
    main()

