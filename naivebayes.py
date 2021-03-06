# naivebayes.py
# -----------------
# Main file for the Naive Bayes algorithm methods.
# Thanks to the Berkeley AI projects for general code structure.
#
# Chet Aldrich, Laura Biester

from loadFeatures import *
from naiveBayesProbs import getSavedProbs
from progressBar import ProgressBar
import util
import math
import copy
import time

class NaiveBayes:
    """
    A Naive Bayes classifier.
    """
    def __init__(self, legalLabels):
        """
        Initializes the information required for Naive Bayes.

        Keyword Arguments:
        legalLabels -- the possible labels in the dataset
        """
        self.legalLabels = legalLabels
        self.k = 1 # this is the smoothing parameter
        self.automaticTuning = False # Flat for automatic tuning of the parameters
        self.probLabel = util.Counter()
        self.probFeature = util.Counter()
        self.features = None
        
    def printToFile(self):
        """
        printToFile() prints trained probabilities to a separate file. Simply place this method after the training sequence if you want to save the values for later use.
        """
        file = open("naiveBayesProbs.py", "w+")
        print >>file, "def getSavedProbs():\n\treturn {0}, {1}".format(self.probLabel, self.probFeature)
        file.close()

    def useTrainedProbs(self, features):
        """
        useTrainedProbs() gets a dictionary of previously trained probabilities from
        the naiveBayesProbs.py file.
        
        Keyword Arguments:
        featurs -- features needed for the saved data
        """
        self.probLabel, self.probFeature = getSavedProbs()
        self.features = features

    def train(self, trainingData, trainingLabels, validationData, validationLabels, allFeatures, tune):
        """
        train() acts as an outside shell to call the method.

        Keyword Arguments:
        trainingData -- training data for the perceptron
        trainingLabels -- labels for the associated training data
        validationData -- validation data for the perceptron tuning function
        validationLabels -- labels for the associated validation data
        allFeatures -- a list of all of the possible features in the dataset
        tune -- a boolean indicating whether to tune over iterations.
        """
        self.features = allFeatures # list with all valid features
        self.automaticTuning = tune


        if (self.automaticTuning):
            kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
        else:
            kgrid = [self.k]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
        """
        trainAndTune() trains the classifier by collecting counts over the training data and choosing the smoothing parameter among the choices in kgrid by
        using the validation data. This method should store the right parameters
        as a side-effect and should return the best smoothing parameters.

        Keyword Arguments:
        trainingData -- training data for the perceptron
        trainingLabels -- labels for the associated training data
        validationData -- validation data for the perceptron tuning function
        validationLabels -- labels for the associated validation data
        kgrid -- a list of possible k values to try for smoothing
        """

        # We begin by creating the prior probabilities for each of the labels
        # and the features based on the counts in the training data.
        countLabel = util.Counter() # in form k = label, v = numOfL
        countFeature = util.Counter()
        # We begin looking over the training data here.
        progressBar = ProgressBar(100, len(trainingData), "Counting Data")
        for i in range(len(trainingData)):
            # update our progress bar
            progressBar.update(i)

            label = trainingLabels[i]
            # Labels are counted at each point they are seen here.
            countLabel[label] += 1
            # Then, if we haven't seen the label, we add it to the feature counter.
            if label not in countFeature:
                countFeature[label] = util.Counter()
            # Finally, we loop over the features for each datum and add each feature once
            # for each occurrence.
            for feature in trainingData[i]:
                countFeature[label][feature] += trainingData[i][feature]
        progressBar.clear()

        self.probLabel = copy.deepcopy(countLabel)
        self.probLabel.normalize()


        # At this point we have the counts, and we want to see what level of smoothing
        # increases our accuracy the most over the training set. Essentially, we just
        # create all of the probabilities from the feature counts while adding the smoothing
        # and classify the training data each time and pick whatever was most accurate.
        kClassifications = util.Counter()
        probForK = util.Counter()
        numCorrectK = util.Counter()
        print "Validation Accuracy"
        print "==================="
        for k in kgrid:
            # make counter for probabilities for each k
            probForK[k] = util.Counter()
            # make counters for probabilities for each label
            for label in self.legalLabels:
                probForK[k][label] = util.Counter()
                # find probability of each feature given each label
                progressBar = ProgressBar(100, len(self.features), "Getting Probabilities for Features, Label {0}".format(label))
                for index, feature in enumerate(self.features):
                    progressBar.update(index)
                    if countFeature[label] != 0:
                        probForK[k][label][feature] = float(countFeature[label][feature] + k) / (countLabel[label] + k * len(self.features))
                progressBar.clear()

            # set probabilities for features and classify validation data
            self.probFeature = probForK[k]
            classificationLabels = self.classify(validationData)

            # check how much of the data was classified correctly
            correct = 0
            for i in range(len(classificationLabels)):
                if classificationLabels[i] == validationLabels[i]:
                    correct += 1

            # print accuracy for each k
            print "k = {0}, number of correct classifications = {1}".format(k, correct)
            # store the number of correct classifications for k value
            numCorrectK[k] = correct

        # pick k from our list of possible k values
        self.k = None
        for k in numCorrectK:
            # find k with the highest number of correct classifications
            # if there is a tie, use a lower k value
            if (self.k == None or numCorrectK[self.k] < numCorrectK[k] or
                (numCorrectK[self.k] == numCorrectK[k] and k < self.k)):
                self.k = k
        self.probFeature = probForK[self.k]

        # print final choice for k
        print "K chosen = {0}".format(self.k)
        
        return self.k

    def classify(self, testData):
        """
        classify() classifies each data item in the input by finding the best prototype vector.

        Keyword Arguments:
        testData -- the test data to classify
        """
        guesses = []
        self.posteriors = [] # Log posteriors are stored for later data analysis.
        counter = 0
        size = len(testData)
        progressBar = ProgressBar(100, len(testData), "Classifying Data")
        for index, datum in enumerate(testData):
            progressBar.update(index)
            posterior = self.calculateLogJointProbabilities(datum)
            guesses.append(posterior.argMax())
            self.posteriors.append(posterior)
        progressBar.clear()
        return guesses

    def calculateLogJointProbabilities(self, datum):
        """
        calculateLogJointProbabilities() returns the log-joint distribution over
        legal labels and the datum.

        Keyword Arguments:
        datum -- a dictionary with 0 or 1 values depending on whether a value is
                 colored or not. 
        """
        logJoint = util.Counter()
        for label in self.legalLabels:
            logJoint[label] += math.log(self.probLabel[label])
            for feature in self.features:
                # probability is the stored probability if the feature is on
                # otherwise, it is 1 - that probability, because the distribution
                # sums to 0
                if datum[feature] != 0:
                    probability = self.probFeature[label][feature]
                else:
                    probability = 1 - self.probFeature[label][feature]
                # if probability is 0, break to avoid log being undefined
                # any probability in the joint being 0 will lead to an
                # overall probability of 0
                if probability == 0:
                    logJoint[label] = 0
                    break
                # otherwise, sum up the logs to get values proportional
                # to multiplying the probabilities
                logJoint[label] += math.log(probability)
        return logJoint
