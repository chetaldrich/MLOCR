import util
from loadFeatures import *

class Perceptron:

    def __init__(self, legalLabels, iterations):
        self.legalLabels = legalLabels
        self.iterations = iterations
        self.weights = {}
        for label in legalLabels:
            self.weights[label] = util.Counter()

    def train(self, trainingData, trainingLabels, validationData, validationLabels, tune):
        """
        train() updates the perceptron prototype vectors over the training values given.  
        """
        # TODO: figure out what to pass in here
        self.features = None

        if (tune):
            iterationValues = [ 1,3, 5, 10, 15, 20]
            tune(iterationValues, validationData, validationLabels)
        else:
            iterationValues = [self.iterations]

        for i in range(iterationValues[0]):
            for j in range(len(trainingData)):
                values = util.Counter()

                # Go over each label, and create the value from the training data and current vectors
                for label in self.legalLabels:
                    activation = 0
                    for key, value in trainingData[j].items():
                        activation += self.weights[label][key] * value
                    values[label] = activation

                # Here, we update values in weight vectors if we reach an incorrect conclusion.
                if values.argMax() != trainingLabels[j]:
                    for key, value in trainingData[j].items():
                        self.weights[trainingLabels[j]][key] += value
                        self.weights[values.argMax()][key] -= value


    def tune(self, validationData, validationLabels):
        """
        tune() tunes the data to the best number of iterations over the validation data.

        TODO: Actually make this work.

        """
        pass

    def classify(self, data):
        """
        classify() classifies each data item in the input by finding the best prototype vector.

        """
        guesses = []
        for entry in data:
            values = util.Counter()

            # for each label, compute activation
            for label in self.legalLabels:
                activation = 0
                # sum over the weights * values to get activation
                for key, value in entry.items():
                    activation += self.weights[label][key] * value
                values[label] = activation

            # add classification guess for data by getting the argmax
            guesses.append(values.argMax())
        return guesses
