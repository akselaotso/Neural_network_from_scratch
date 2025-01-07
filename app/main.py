import numpy as np
from nnfs.datasets import spiral_data
from .modules.activation_functions import *
from .modules.layer import Layer


def cross_entropy_loss(prediction, true):
    prediction = np.clip(prediction, 0.0000001, 1 - 0.0000001)
    return -np.log(prediction[range(len(prediction)), true])

def one_hot_cross_entropy_loss(prediction, true):
    prediction = np.clip(prediction, 0.0000001, 1 - 0.0000001)
    return -np.log(np.sum(np.array(prediction) * np.array(true), axis=1))

def accuracy(prediction, true):
    prediction = np.argmax(prediction, axis=1)
    return np.mean(prediction == true)

    
def main():
    stepsize = 0.001
    samples = 100
    classes = 3
    X, y = spiral_data(samples=samples, classes=classes)

    layer1 = Layer(output_count=3, input_count=2, activation_function=ReLu())

    for _ in range(100):
        output = layer1.forward(X)

        loss = cross_entropy_loss(prediction=output, true=y)

        dL_dy = 0
        dy_da = 0
        
        dL_da = dL_dy * dy_da

        layer1.backward(dL_da=dL_da, stepsize=stepsize)

        average_loss = np.mean(loss)

        accuracy = accuracy(output, y)

        print(accuracy)

        print(average_loss)

    print("\nProgram complete.")


if __name__ == "__main__":
    main()

