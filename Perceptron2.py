import numpy as np
import random
from Perceptron import Perceptron

class Perceptron2(Perceptron):
    
    def __init__(self, data, dataLength, learningRate = 0.1, uniformBound = 0.5):
        self.data = data
        self.dataLength = dataLength
        self.epoch = 0
        self.learningRate = learningRate
        self.uniformBound = uniformBound
        self.numberOfFitsInTheWeightVector = 0

    def getNextWeightsVector(self, previousWeightsVector, error, inputAttributesVector):
        return (previousWeightsVector + self.learningRate * error * inputAttributesVector)

    def getWeightsVector(self):
        weightsVector = np.array([])
        for i in range(3):
            randomNumber = random.uniform(-self.uniformBound, self.uniformBound)
            weightsVector = np.append(weightsVector, randomNumber)
        return (weightsVector)

    def teachPerceptron(self, verbose=True):
        deducedOutputVector = np.array([])
        outputVector = self.getOutputVector(self.data)
        weightsVector = self.getWeightsVector()
        inputVectorGroup = self.getInputVectorGroup(self.data, self.dataLength)

        self.numberOfFitsInTheWeightVector = 0
        self.epoch = 0

        if (verbose): print("Pesos iniciais: ", weightsVector)

        while (not self.checkArrayEquality(deducedOutputVector, outputVector)):
            self.epoch += 1
            localFits = 0

            for i in range(len(inputVectorGroup)):
                inputVector = inputVectorGroup[i]
                realOutput = outputVector[i]

                weightedSum = self.getWeightedSum(inputVector, weightsVector)
                outputDeducted = self.step(weightedSum)
                
                if (len(deducedOutputVector) == self.dataLength):
                    deducedOutputVector[i] = outputDeducted
                else:
                    deducedOutputVector = np.insert(deducedOutputVector, i, outputDeducted)

                error = self.getError(realOutput, outputDeducted)

                if (error):
                    weightsVector = self.getNextWeightsVector(weightsVector, error, inputVector)
                    self.numberOfFitsInTheWeightVector += 1
                    localFits += 1
        
        if (verbose): 
            print("Total de Épocas: ", self.epoch)
            print("Total de ajustes: ", self.numberOfFitsInTheWeightVector)
            print(f"Pesos finais: {weightsVector}\n")