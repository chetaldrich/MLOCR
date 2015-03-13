import util
from loadFeatures import *
from copy import deepcopy


class Perceptron:

    def __init__(self, legalLabels, iterations):
        self.legalLabels = legalLabels
        self.iterations = iterations
        self.initWeights()

    def initWeights(self):
        self.weights = {}
        for label in self.legalLabels:
            self.weights[label] = util.Counter()


    def train(self, trainingData, trainingLabels, validationData, validationLabels, tune):
        """
        train() updates the perceptron prototype vectors over the training values given.
        """

        if (tune):
            iterationValues = [1, 3, 5, 10, 15, 20]
            self.tune(trainingData, trainingLabels, validationData, validationLabels, iterationValues)
        else:
            self.trainingHelper(trainingData, trainingLabels, self.iterations)

    def trainingHelper(self, trainingData, trainingLabels, iterations):
        """
        trainingHelper finds the classification using the perceptron weights
        and updates weights if needed
        """
        for i in range(iterations):
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

    def tune(self, trainingData, trainingLabels, validationData, validationLabels, iterationValues):
        """
        tune() tunes the data to the best number of iterations over the validation data.

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

    def test(self, testingData, testingLabels):
        """
        test() gets a classification for the training data points and checks
        if it matches the labels
        """
        # get classification of data
        classified = self.classify(testingData)

        countCorrect = 0
        # check if classification matches label
        for i in range(len(testingLabels)):
            if testingLabels[i] == classified[i]:
                countCorrect += 1

        print "Number of Correct Classifications"
        print "================================="
        print countCorrect

        print "Percent of Correct Classifications"
        print "=================================="
        print float(countCorrect) / len(testingLabels) * 100.0
