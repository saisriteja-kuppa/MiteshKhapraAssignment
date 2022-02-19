from nn_update import DenseLayer, Softmax, ReLU, tanH, Sigmoid
from nn_update import  RMSprop, Softmax_Loss_CategoricalCrossentropy, SGD
import numpy as np


def main(epochs, NoOfHiddenLayers, SizeOfHiddenLayers, WeightDecay, LearningRate, Optimizer, BatchSize, WeightInit, ActivationFunc):


    X  = np.random(10)
    y = np.random(10)
    ActivationFunc = ReLU
    loss_Activation = Softmax_Loss_CategoricalCrossentropy()
    optimizer = Optimizer(lr = LearningRate)



    if NoOfHiddenLayers == 3:

        for i in range(epochs):

            X = X[0: BatchSize]
            y = y[0: BatchSize]

            # Forward pass
            d1 = DenseLayer(784, SizeOfHiddenLayers, initialization  = WeightInit, W_regu_l2 = WeightDecay, b_regu_l2 = WeightDecay)
            a1 = ActivationFunc
            # -------------
            d2 = DenseLayer(SizeOfHiddenLayers,SizeOfHiddenLayers, initialization  = WeightInit, W_regu_l2 = WeightDecay, b_regu_l2 = WeightDecay)
            a2 = ActivationFunc
            d3 = DenseLayer(SizeOfHiddenLayers,SizeOfHiddenLayers, initialization  = WeightInit,  W_regu_l2 = WeightDecay, b_regu_l2 = WeightDecay)
            a3 = ActivationFunc
            d4 = DenseLayer(SizeOfHiddenLayers,SizeOfHiddenLayers, initialization  = WeightInit,  W_regu_l2 = WeightDecay, b_regu_l2 = WeightDecay)
            a4 = ActivationFunc
            # --------------
            d5 = DenseLayer(SizeOfHiddenLayers,10, initialization  = WeightInit,  W_regu_l2 = WeightDecay, b_regu_l2 = WeightDecay)
            a5 = Softmax()
            # --------------

            # --------------
            d1.ForwardPass(X)
            a1.ForwardPass(d1.DataOutput)
            d2.ForwardPass(a1.DataOutput)
            a2.ForwardPass(d2.DataOutput)
            d3.ForwardPass(a2.DataOutput)
            a3.ForwardPass(d3.DataOutput)
            d4.ForwardPass(a3.DataOutput)
            a4.ForwardPass(d4.DataOutput)
            d5.ForwardPass(a4.DataOutput)


            train_loss = loss_Activation.ForwardPass(d5.DataOutput, y)
            preds = np.argmax(loss_Activation.DataOutput, axis =1)
            train_accuracy = np.mean(preds == y)


            # Backward pass
            loss_Activation.BackProp(loss_Activation.DataOutput, y)
            d5.BackProp(loss_Activation.dDataInput)
            a4.BackProp(d5.dDataInput)
            d4.BackProp(a4.dDataInput)
            a3.BackProp(d4.dDataInput)
            d3.BackProp(a3.dDataInput)
            a2.BackProp(d3.dDataInput)
            d2.BackProp(a2.dDataInput)
            a1.BackProp(d2.dDataInput)
            d1.BackProp(a1.dDataInput)

            optimizer.update(d1)
            optimizer.update(d2)
            optimizer.update(d3)
            optimizer.update(d4)
            optimizer.update(d5)



    




    if NoOfHiddenLayers == 4:

        for i in range(epochs):

            X = X[0: BatchSize]
            y = y[0: BatchSize]

            # Forward pass
            d1 = DenseLayer(784, SizeOfHiddenLayers, initialization  = WeightInit, W_regu_l2 = WeightDecay, b_regu_l2 = WeightDecay)
            a1 = ActivationFunc
            # -------------
            d2 = DenseLayer(SizeOfHiddenLayers,SizeOfHiddenLayers, initialization  = WeightInit, W_regu_l2 = WeightDecay, b_regu_l2 = WeightDecay)
            a2 = ActivationFunc
            d3 = DenseLayer(SizeOfHiddenLayers,SizeOfHiddenLayers, initialization  = WeightInit,  W_regu_l2 = WeightDecay, b_regu_l2 = WeightDecay)
            a3 = ActivationFunc
            d4 = DenseLayer(SizeOfHiddenLayers,SizeOfHiddenLayers, initialization  = WeightInit,  W_regu_l2 = WeightDecay, b_regu_l2 = WeightDecay)
            a4 = ActivationFunc
            d41 = DenseLayer(SizeOfHiddenLayers,SizeOfHiddenLayers, initialization  = WeightInit,  W_regu_l2 = WeightDecay, b_regu_l2 = WeightDecay)
            a41 = ActivationFunc
            # --------------
            d5 = DenseLayer(SizeOfHiddenLayers,10, initialization  = WeightInit,  W_regu_l2 = WeightDecay, b_regu_l2 = WeightDecay)
            a5 = Softmax()

            # --------------



            # --------------
            d1.ForwardPass(X)
            a1.ForwardPass(d1.DataOutput)
            d2.ForwardPass(a1.DataOutput)
            a2.ForwardPass(d2.DataOutput)
            d3.ForwardPass(a2.DataOutput)
            a3.ForwardPass(d3.DataOutput)
            d4.ForwardPass(a3.DataOutput)
            a4.ForwardPass(d4.DataOutput)
            d41.ForwardPass(a4.DataOutput)
            a41.ForwardPass(d41.DataOutput)
            d5.ForwardPass(a4.DataOutput)


            train_loss = loss_Activation.ForwardPass(d5.DataOutput, y)
            preds = np.argmax(loss_Activation.DataOutput, axis =1)
            train_accuracy = np.mean(preds == y)


            loss_Activation.BackProp(loss_Activation.DataOutput, y)
            d5.BackProp(loss_Activation.dDataInput)
            a41.BackProp(d5.dDataInput)
            d41.BackProp(a41.dDataInput)
            a4.BackProp(d41.dDataInput)
            d4.BackProp(a4.dDataInput)
            a3.BackProp(d4.dDataInput)
            d3.BackProp(a3.dDataInput)
            a2.BackProp(d3.dDataInput)
            d2.BackProp(a2.dDataInput)
            a1.BackProp(d2.dDataInput)
            d1.BackProp(a1.dDataInput)

            optimizer.update(d1)
            optimizer.update(d2)
            optimizer.update(d3)
            optimizer.update(d4)
            optimizer.update(d41)
            optimizer.update(d5)










    if NoOfHiddenLayers == 5:

        for i in range(epochs):

            X = X[0: BatchSize]
            y = y[0: BatchSize]

            # Forward pass
            d1 = DenseLayer(784, SizeOfHiddenLayers, initialization  = WeightInit, W_regu_l2 = WeightDecay, b_regu_l2 = WeightDecay)
            a1 = ActivationFunc
            # -------------
            d2 = DenseLayer(SizeOfHiddenLayers,SizeOfHiddenLayers, initialization  = WeightInit, W_regu_l2 = WeightDecay, b_regu_l2 = WeightDecay)
            a2 = ActivationFunc
            d3 = DenseLayer(SizeOfHiddenLayers,SizeOfHiddenLayers, initialization  = WeightInit,  W_regu_l2 = WeightDecay, b_regu_l2 = WeightDecay)
            a3 = ActivationFunc
            d4 = DenseLayer(SizeOfHiddenLayers,SizeOfHiddenLayers, initialization  = WeightInit,  W_regu_l2 = WeightDecay, b_regu_l2 = WeightDecay)
            a4 = ActivationFunc
            d41 = DenseLayer(SizeOfHiddenLayers,SizeOfHiddenLayers, initialization  = WeightInit,  W_regu_l2 = WeightDecay, b_regu_l2 = WeightDecay)
            a41 = ActivationFunc
            d42 = DenseLayer(SizeOfHiddenLayers,SizeOfHiddenLayers, initialization  = WeightInit,  W_regu_l2 = WeightDecay, b_regu_l2 = WeightDecay)
            a42 = ActivationFunc
            # --------------
            d5 = DenseLayer(SizeOfHiddenLayers,10, initialization  = WeightInit,  W_regu_l2 = WeightDecay, b_regu_l2 = WeightDecay)
            a5 = Softmax()

            # --------------



            # --------------
            d1.ForwardPass(X)
            a1.ForwardPass(d1.DataOutput)
            d2.ForwardPass(a1.DataOutput)
            a2.ForwardPass(d2.DataOutput)
            d3.ForwardPass(a2.DataOutput)
            a3.ForwardPass(d3.DataOutput)
            d4.ForwardPass(a3.DataOutput)
            a4.ForwardPass(d4.DataOutput)
            d41.ForwardPass(a4.DataOutput)
            a41.ForwardPass(d41.DataOutput)
            d42.ForwardPass(a41.DataOutput)
            a42.ForwardPass(d42.DataOutput)
            d5.ForwardPass(a42.DataOutput)


            train_loss = loss_Activation.ForwardPass(d5.DataOutput, y)
            preds = np.argmax(loss_Activation.DataOutput, axis =1)
            train_accuracy = np.mean(preds == y)


            loss_Activation.BackProp(loss_Activation.DataOutput, y)
            d5.BackProp(loss_Activation.dDataInput)
            a42.BackProp(d5.dDataInput)
            d42.BackProp(a42.dDataInput)
            a41.BackProp(d42.dDataInput)
            d41.BackProp(a41.dDataInput)
            a4.BackProp(d41.dDataInput)
            d4.BackProp(a4.dDataInput)
            a3.BackProp(d4.dDataInput)
            d3.BackProp(a3.dDataInput)
            a2.BackProp(d3.dDataInput)
            d2.BackProp(a2.dDataInput)
            a1.BackProp(d2.dDataInput)
            d1.BackProp(a1.dDataInput)

            optimizer.update(d1)
            optimizer.update(d2)
            optimizer.update(d3)
            optimizer.update(d4)
            optimizer.update(d41)
            optimizer.update(d42)
            optimizer.update(d5)







    


      



    






# epochs, NoOfHiddenLayers, SizeOfHiddenLayers, WeightDecay, LearningRate, Optimizer, BatchSize, WeightInit, ActivationFunc = 10, 4, 64, _,_,_,_,_,ReLU

# main(epochs, NoOfHiddenLayers, SizeOfHiddenLayers, WeightDecay, LearningRate, Optimizer, BatchSize, WeightInit, ActivationFunc)