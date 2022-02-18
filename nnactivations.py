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
    


