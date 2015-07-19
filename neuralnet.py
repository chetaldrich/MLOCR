# perceptron.py
# -----------------
# Main file for the Neural Net algorithm methods.
# Currently just a shell for further development.
# Thanks to the Berkeley AI projects for general code structure.
#
# Chet Aldrich

from loadFeatures import *
import numpy as np
import random

class NeuralNet:
    """
    A Neural Net classifier with one hidden layer.
    """

    def __init__(self, legalLabels):
        self.legalLabels = legalLabels
        self.layer_sizes, self.biases, self.weights = self.initWeights()

    def initWeights(self):
        layer_sizes = [10, 15, 10]
        back_layers = layer_sizes[1:]
        front_layers = layer_sizes[:-1]

        biases = []

        for layer_size in back_layers:
            # Generates layer_size number of random values
            # from Gaussian distribution with mean 0 and
            # standard deviation of 1
            biases.append(np.random.randn(layer_size, 1))

        weights = []
        # Builds the weights that create connections between
        # the neurons in the network.
        for layer_size_input, layer_size_output in zip(front_layers, back_layers):
            weights.append(np.random.randn(layer_size_output, layer_size_input))

        return layer_sizes, biases, weights

    def classify(self, data):
        guesses = []
        progressBar = ProgressBar(100, len(data), "Classifying Data")
        for index, entry in enumerate(data):
            progressBar.update(index)
            # temporary to get the network to return
            guesses.append(1)

        progressBar.clear()
        return guesses
