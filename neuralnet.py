# perceptron.py
# -----------------
# Main file for the Neural Net algorithm methods.
# Currently just a shell for further development.
# Thanks to the Berkeley AI projects for general code structure.
#
# Chet Aldrich

from loadFeatures import *
import numpy
import random

class NeuralNet:
    """
    A Neural Net classifier with one hidden layer.
    """

    def __init__(self, legalLabels):
        self.legalLabels = legalLabels
        self.initWeights()

    def initWeights(self):
        return 1

    def classify(self, data):
        guesses = []
        progressBar = ProgressBar(100, len(data), "Classifying Data")
        for index, entry in enumerate(data):
            progressBar.update(index)
            # temporary to get the network to return
            guesses.append(1)

        progressBar.clear()
        return guesses
