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


    

