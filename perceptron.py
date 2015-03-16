# perceptron.py
# -----------------
# Main file for the Perceptron algorithm methods.
# Thanks to the Berkeley AI projects for general code structure.
#
# Chet Aldrich, Laura Biester


import util
import perceptronWeights
from loadFeatures import *
from copy import deepcopy
from progressBar import ProgressBar


class Perceptron:
    """
    A Perceptron Classifier.
    """
    def __init__(self, legalLabels, iterations):
        """
        Initializes the information required for the perceptron.

        Keyword Arguments:
        legalLabels -- the possible labels in the dataset
        iterations -- the desired iteration value during training
        """
        self.legalLabels = legalLabels
        self.iterations = iterations
        self.initWeights()

    def printToFile(self):
        """
        printToFile() prints trained weights to a separate file. Simply place this method after the training sequence if you want to save the values for later use.
        """
        file = open("perceptronWeights.py", "w+")
        print >>file, "def getSavedWeights():\n\treturn {0}".format(self.weights)
        file.close()

    def initWeights(self):
        """
        initWeights() initializes the weight dictionary with Counters (see util.py).
        """
        self.weights = {}
        for label in self.legalLabels:
            self.weights[label] = util.Counter()

    def useTrainedWeights(self):
        """
        useTrainedWeights() gets a dictionary of previously trained weights from
        the perceptronWeights.py file.
        """
        self.weights = perceptronWeights.getSavedWeights()

    def train(self, trainingData, trainingLabels, validationData, validationLabels, tune):
        """
        train() updates the perceptron prototype vectors over the training values given.

        Keyword Arguments:
        trainingData -- training data for the perceptron
        trainingLabels -- labels for the associated training data
        validationData -- validation data for the perceptron tuning function
        validationLabels -- labels for the associated validation data
        tune -- a boolean indicating whether to tune over iterations.
        """

        if (tune):
            iterationValues = [1, 3, 5, 10, 15, 20]
            self.tune(trainingData, trainingLabels, validationData, validationLabels, iterationValues)
        else:
            self.trainingHelper(trainingData, trainingLabels, self.iterations)

    def trainingHelper(self, trainingData, trainingLabels, iterations):
        """
        trainingHelper() classifies training data using the perceptron weights
        and updates the perceptron weights if the perceptron is incorrect.

        Keyword Arguments:
        trainingData -- training data for the perceptron
        trainingLabels -- labels for the associated training data
        iterations -- desired number of iterations over the training dataset.
        """
        for i in range(iterations):
            progressBar = ProgressBar(100, len(trainingData), "Learning Weights, Iteration {0} of {1}"
                .format(i + 1, iterations))
            for j in range(len(trainingData)):
                progressBar.update(j)

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
            progressBar.clear()

    def tune(self, trainingData, trainingLabels, validationData, validationLabels, iterationValues):
        """
        tune() tunes the data to the best number of iterations over the validation data.

        Keyword Arguments:
        trainingData -- training data for the perceptron
        trainingLabels -- labels for the associated training data
        validationData -- validation data for the perceptron tuning function
        validationLabels -- labels for the associated validation data
        iterationValues -- a list of different numbers of iterations to try
        """
        tuningWeights = Counter()
        correctlyClassified = Counter()

        # classify for each number of iterations
        for index, iteration in enumerate(iterationValues):
            # just do the number of extra iterations required
            if index != 0:
                extraIterations = iterationValues[index] - iterationValues[index - 1]
            else:
                extraIterations = iteration

            # train extra iterations
            self.trainingHelper(trainingData, trainingLabels, extraIterations)

            # save weights for each number of iterations
            tuningWeights[iteration] = deepcopy(self.weights)

            # classify data
            classified = self.classify(validationData)
            # check if classification matches label
            for i in range(len(validationLabels)):
                if validationLabels[i] == classified[i]:
                    correctlyClassified[iteration] += 1

        # find iteration value with most correct classifications
        # use those weights when classifying
        bestIterationValue = correctlyClassified.argMax()
        self.weights = tuningWeights[bestIterationValue]


    def classify(self, data):
        """
        classify() classifies each data item in the input by finding the best prototype vector.

        Keyword Arguments:
        data -- the test data to classify
        """
        guesses = []
        progressBar = ProgressBar(100, len(data), "Classifying Data")
        for index, entry in enumerate(data):
            progressBar.update(index)
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
        progressBar.clear()
        return guesses
