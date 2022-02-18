import numpy as np




# Dense layer
class DenseLayer:

    # Layer initialization
    def __init__(self, inputShape, NoOfNeurons):
        # Initialize W and b
        self.W = 0.01 * np.random.randn(inputShape, NoOfNeurons)
        self.b = np.zeros((1, NoOfNeurons))

    # ForwardPass pass
    def ForwardPass(self, DataInput):
        # Remember input values
        self.DataInput = DataInput
        # Calculate DataOutput values from DataInput, W and b
        self.DataOutput = np.dot(DataInput, self.W) + self.b

    # BackProp pass
    def BackProp(self, derivativevalues):
        # Gradients on parameters
        self.dW = np.dot(self.DataInput.T, derivativevalues)
        self.db = np.sum(derivativevalues, axis=0, keepdims=True)
        # Gradient on values
        self.dDataInput = np.dot(derivativevalues, self.W.T)


# ReLU activation
class ReLU:

    def ForwardPass(self, DataInput):
        self.DataInput = DataInput
        self.DataOutput = np.maximum(0, DataInput)

    # BackProp pass
    def BackProp(self, derivativevalues):
        self.dDataInput = derivativevalues.copy()
        self.dDataInput[self.DataInput <= 0] = 0



class Softmax:
    def ForwardPass(self, DataInput):
        self.DataInput = DataInput
        ExponentialValues = np.exp(DataInput - np.max(DataInput, axis=1,  keepdims=True))
        Prob = ExponentialValues / np.sum(ExponentialValues, axis=1, keepdims=True)
        self.DataOutput = Prob

    def BackProp(self, derivativevalues):

        self.dDataInput = np.empty_like(derivativevalues)

        for index, (s_dataOut, single_derivativevalues) in \
                enumerate(zip(self.DataOutput, derivativevalues)):
            s_dataOut = s_dataOut.reshape(-1, 1)
            J_Matrix = np.diagflat(s_dataOut) - np.dot(s_dataOut, s_dataOut.T)
            self.dDataInput[index] = np.dot(J_Matrix,single_derivativevalues)












class momentumSgd():
    def __init__(self, lr, momentum):
        self.lr = lr
        self.momentum = momentum

    def update_params(self, layer):

        if not hasattr(layer, 'weight_momentums'):
            layer.weight_momentums = np.zeros_like(layer.W)
            layer.bias_momentums = np.zeros_like(layer.b)

        UpdateW = self.momentum * layer.weight_momentums -  self.lr * layer.dW
        layer.weight_momentums = UpdateW

        Updateb = self.momentum * layer.bias_momentums - self.lr * layer.db
        layer.bias_momentums = Updateb
    
        layer.W += UpdateW
        layer.b += Updateb


# SGD optimizer
class SGD:
    def __init__(self, lr=1.):
        self.lr = lr


    # Update parameters
    def update_params(self, layer):

        UpdateW = -self.lr * layer.dW
        Updateb = -self.lr *   layer.db

        layer.W += UpdateW
        layer.b += Updateb






# RMSprop optimizer
class RMSprop:

    # Initialize optimizer - set settings
    def __init__(self, lr=0.001, decay=0., eps=1e-7,
                 rho=0.9):
        self.lr = lr
        self.eps = eps
        self.rho = rho


    # Update parameters
    def update_params(self, layer):

        if not hasattr(layer, 'CacheW'):
            layer.CacheW = np.zeros_like(layer.W)
            layer.Cacheb = np.zeros_like(layer.b)

        layer.CacheW = self.rho * layer.CacheW + (1 - self.rho) * layer.dW**2
        layer.Cacheb = self.rho * layer.Cacheb + (1 - self.rho) * layer.db**2

        layer.W += -self.lr *  layer.dW /  (np.sqrt(layer.CacheW) + self.eps)
        layer.b += -self.lr *  layer.db /  (np.sqrt(layer.Cacheb) + self.eps)


class Loss:
    def calculate(self, DataOutput, y):
        sample_losses = self.ForwardPass(DataOutput, y)
        data_loss = np.mean(sample_losses)
        return data_loss




# Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):

    def ForwardPass(self, y_pred, y_true):

        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]

        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


    def BackProp(self, derivativevalues, y_true):
        samples = len(derivativevalues)
        labels = len(derivativevalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        self.dDataInput = -y_true / derivativevalues
        self.dDataInput = self.dDataInput / samples



class Softmax_Loss_CategoricalCrossentropy():

    def __init__(self):
        self.activation = Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def ForwardPass(self, DataInput, y_true):
        self.activation.ForwardPass(DataInput)
        self.DataOutput = self.activation.DataOutput
        return self.loss.calculate(self.DataOutput, y_true)

    def BackProp(self, derivativevalues, y_true):
        samples = len(derivativevalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dDataInput = derivativevalues.copy()
        self.dDataInput[range(samples), y_true] -= 1
        self.dDataInput = self.dDataInput / samples



