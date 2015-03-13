import naivebayes
import perceptron
import mnist
import util
import sys
import loadFeatures
import argparse


def readCommand():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", type = str, choices=["naivebayes", "perceptron"], help="selects the classifier for use with the MNIST data")
    args = parser.parse_args()

    if args.c == "naivebayes":
        runNaiveBayes()
    elif args.c == "perceptron":
        runPerceptron()

def runPerceptron():
    perceptronClassifier = perceptron.Perceptron(range(10), 3)

    print "Loading Testing Data....\n"
    trainingData, trainingLabels, validationData, validationLabels, features = loadFeatures.loadTrainingData(600)

    print "Training Perceptron....\n"
    perceptronClassifier.train(trainingData, trainingLabels, validationData, validationLabels, False)

    print "Loading Testing Data....\n"
    testingData, testingLabels = loadFeatures.loadTestingData(100)

    print "Testing Perceptron....\n"
    perceptronClassifier.test(testingData, testingLabels)


def runNaiveBayes():
    naiveBayesClassifier = naivebayes.NaiveBayes(range(10))

    print "Loading Training Data....\n"
    trainingData, trainingLabels, validationData, validationLabels, features = loadFeatures.loadTrainingData(50)

    print "Training Naive Bayes Classifier....\n"
    naiveBayesClassifier.train(trainingData, trainingLabels, validationData, validationLabels, features, False)

    print "Loading Testing Data....\n"
    testingData, testingLabels = loadFeatures.loadTestingData(50)

    print "Testing Naive Bayes Classifier....\n"
    naiveBayesClassifier.test(testingData, testingLabels)


if __name__ == "__main__":
    readCommand()
