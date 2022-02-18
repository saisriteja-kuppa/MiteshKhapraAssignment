import numpy as np


class Loss:
    def computeloss(self, predicted, original):
        sample_losses = self.forward(predicted, original)
        loss = np.mean(sample_losses)
        return loss


class CategoricalCrossentropyLoss(Loss):
    def ForwardProp(self, predicted, original):

        samples = len(predicted)
        y_pred_clipped = np.clip(predicted, 1e-15, 1 - 1e-15)
        if len(predicted.shape) == 1:
            Confidence = y_pred_clipped[list(range(samples)), original]

        Nlog = -np.log(Confidence)
        return Nlog

    
    def BackwardProp(self, derivativevalues, original):

        samples = len(derivativevalues)
        lables = len(derivativevalues[0])
        if len(original.shape) == 1:
            original = np.eye(lables)[original]
        self.derivativeInput = - original / derivativevalues
        self.derivativeInput = self.derivativeInput / samples


from nnactivations import softmax
class softmaxCCEloss():
    def __init__(self):
        self.activation = softmax()
        self.loss = CategoricalCrossentropyLoss()
    
    def forward(self, predicted, original):
        self.activation.ForwardProp(original)
        self.output = self.activation.DataOutput(original)
        return self.loss.calculate(self.output, predicted)

    def backward(self, derivativevalues, original):
        samples = len(derivativevalues)
        self.derivativeinputs  =derivativevalues.copy()
        self.derivativeinputs[range(samples), original] -= 1
        self.derivativeinputs = self.derivativeinputs / samples
    

