import numpy as np

class Loss_Function:
    def __init__(self):
        pass

    def forward(self, x):
        return x

    def backward(self, x):
        return x
    
class Cross_Entropy_Loss(Loss_Function):
    def forward(self, y_pred, y):
        prediction = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y.shape) == 1:
            prediction = prediction[range(len(y_pred)), y]
        elif len(y.shape) == 2:
            prediction = np.sum(prediction * y, axis=1)

        return -np.log(prediction)
    
    def backward(self, y_hat, y):
        y_hat = np.clip(y_hat, 0.0000001, 1 - 0.0000001)

        if len(y.shape) == 1:
            y = np.eye(len(y_hat[0]))[y]

        self.gradient =  - (y / y_hat) / len(y_hat)
        # self.gradient =  - y / y_hat
    
