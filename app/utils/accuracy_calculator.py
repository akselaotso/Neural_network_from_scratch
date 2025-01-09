import numpy as np

def accuracy_calculator(loss_output):
    predictions = np.argmax(loss_output, axis=1)
    
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    
    return np.mean(predictions == y)
