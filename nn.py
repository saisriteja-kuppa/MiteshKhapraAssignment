class SGD:
    def __init__(self, lr):
        self.lr = lr
    
    def update(self, layer):
        updateW =- layer.DerivativeWeights * self.lr
        updateb =- layer.DerivativeBiases * self.lr


        layer.W += updateW
        layer.b += updateb


class MomentumSGD:
    def __init__(self, lr, momentum):
        self.lr =lr
        self. momentum = momentum


    def update(self,layer):
        layer.momentumW = np.zeros_like(layer.W)
        layer.momentumb = np.zeros_like(layer.b)

        updateW = self.momentum * layer.momentumW - self.lr * layer.DerivativeWeights
        updateb = self.momentum * layer.momentumb - self.lr * layer.DerivativeBiases

        layer.momentumW = updateW
        layer.momentumb = updateb

        layer.W += updateW
        layer.b += updateb


class Rmsprop:
    def __init__(self, lr, eps= 1e-7, rho = 0.9):
        self.lr = lr
        self.eps = eps
        self.rho = rho


    def update(self, layer):
        if not hasattr(layer, 'Wcache'):
            layer.Wcache = np.zeros_like(layer.W)
            layer.bcache = np.zeros_like(layer.b)

        layer.Wcache = self.rho * layer.Wcache + (1 - self.rho) * layer.DerivativeWeights ** 2
        layer.bcache = self.rho * layer.bcache + (1 - self.rho) * layer.DerivativeBiases ** 2

        layer.W += -self.lr * layer.DerivativeWeights / (np.sqrt(layer.Wcache) + self.eps)
        layer.b += -self.lr * layer.DerivativeBiases / (np.sqrt(layer.bcache) + self.eps)
    



import numpy as np


class Loss:
    def computeloss(self, predicted, original):
        sample_losses = self.forward(predicted, original)
        loss = np.mean(sample_losses)
        return loss


class CategoricalCrossentropyLoss(Loss):
    def ForwardPro(self, predicted, original):

        samples = len(predicted)
        y_pred_clipped = np.clip(predicted, 1e-15, 1 - 1e-15)
        if len(predicted.shape) == 1:
            Confidence = y_pred_clipped[list(range(samples)), original]

        Nlog = -np.log(Confidence)
        return Nlog

    
    def BackwardPro(self, derivativevalues, original):

        samples = len(derivativevalues)
        lables = len(derivativevalues[0])
        if len(original.shape) == 1:
            original = np.eye(lables)[original]
        self.derivativeInput = - original / derivativevalues
        self.derivativeInput = self.derivativeInput / samples


class softmaxCCEloss():
    def __init__(self):
        self.activation = SoftmaxActivation()
        self.loss = CategoricalCrossentropyLoss()
    
    def ForwardPro(self, predicted, original):
        self.activation.ForwardPro(original)
        self.output = self.activation.DataOutput(original)
        return self.loss.calculate(self.output, predicted)

    def BackwardPro(self, derivativevalues, original):
        samples = len(derivativevalues)
        self.derivativeinputs  =derivativevalues.copy()
        self.derivativeinputs[range(samples), original] -= 1
        self.derivativeinputs = self.derivativeinputs / samples
    

import  numpy as np

class DenseLayer:

    def __init__(self,InputShape, NoOfNeurons):

        self.W = 0.01 * np.random.rand(InputShape, NoOfNeurons)
        self.b = np.zeros((1, NoOfNeurons))

    
    def ForwardPro(self, DataInput):
        self.DataInput = DataInput
        self.Output = np.dot(DataInput, self.W) + self.b


    def BackwardPro(self, DerivativeValues):
        self.DerivativeWeights = np.dot(self.DataInput.T, DerivativeValues)
        self.DerivativeBiases = np.sum(DerivativeValues , axis = 0, keepdims= True)
        self.DerivativeInputs = np.dot(DerivativeValues , self.W.T)

class ReLUActivation:

    def ForwardPro(self,DataInput):
        self.DataInput = DataInput
        self.DataOutput = np.maximum(0, DataInput)

    
    def BackwardPro(self, DerivativeValues):
        self.DerivativeValues = DerivativeValues.copy()
        self.DerivativeValues[self.DataInput <=0] = 0



class SoftmaxActivation:
    def ForwardPro(self, DataInput):
        self.DataInput = DataInput
        ExponentialValues = np.exp(DataInput - np.max(DataInput, axis =1, keepdims = True))
        Probs  = ExponentialValues / np.sum(ExponentialValues, axis = 1, keepdims= True)
        self.DataOutput = Probs

    
    def BackwardPro(self, DerivativeValues):
        self.DerivativeInputs = np.empty_like(DerivativeValues)
        for No, (Output, DerivativeValue) in enumerate(zip(self.DataOutput, DerivativeValues)):
            Output = Output.reshape(-1,1)
            JMatrix = np.diagflat(Output) - np.dot(Output, Output.T)
            self.DerivativeInputs[No] = np.dot(JMatrix, DerivativeValue)


class Activation_Softmax:

    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs

        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1,
                                            keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1,
                                            keepdims=True)

        self.output = probabilities

