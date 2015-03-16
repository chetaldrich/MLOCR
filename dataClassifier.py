# dataClassifier.py
# -----------------
# Main execution file for machine learning algorithms in MLOCR.
#
# Chet Aldrich, Laura Biester

import naivebayes
import perceptron
import loadFeatures
import argparse
import time


def readCommand():
    """
    readCommand() begins the machine learning program based on parameters given by the user. The command line arguments detailed in comments below.
    """
    # default values
    numTestValues = 100
    numTrainValues = 100
    pixels = 0

    parser = argparse.ArgumentParser()

    # `-c` selects the classifier (`naivebayes` and `perceptron` are options)
    parser.add_argument("-c", type = str, choices=["naivebayes", "perceptron"], help="selects the classifier for use with the MNIST data")

    # `-a` Tune iterations/Laplace smoothing while training
    parser.add_argument("-a", action="store_true", default=False, help="tune while training")

    # `-u` Use pre-learned weights for the perceptron. These were created after training the program for about 3 hours on all of the training data, and this option is available so you can see the results of the classifier without having to wait for the perceptron to train.
    parser.add_argument("-u", action="store_true", default=False, help="use pre-learned weights for perceptron")

    # `--pixels` Chops off some pixels from the edges of each image to improve processing time. Sometimes leads to a slight loss in accuracy. Options are 1-12 pixels.
    parser.add_argument("--pixels", type = int, choices=range(13), help="remove this many pixels from outside of photo for faster training")

    # `--train` selects the number of training data samples to be used by the classifier
    parser.add_argument("--train", type = int, help="selects the number of training data samples to be used by the classifier")

    # `--test` selects the number of test data samples to be used by the classifier
    parser.add_argument("--test", type = int,  help="selects the number of testing data samples to be used by the classifier")

    args = parser.parse_args()

    # Here, we set arguments if they are given by optional params.
    if args.train != None:
        numTrainValues = args.train
    if args.test != None:
        numTestValues = args.test
    if args.pixels != None:
        pixels = args.pixels

    # Here, we determine which algorithm to run based on input on
    # the -c parameter.
    if args.c == "naivebayes":
        runNaiveBayes(numTrainValues, numTestValues, pixels, args.a)
    elif args.c == "perceptron":
        runPerceptron(numTrainValues, numTestValues, pixels, args.a, args.u)

def runPerceptron(numTrainValues, numTestValues, pixels, tune, useTrainedWeights):
    """
    runPerceptron() runs the perceptron learning algorithm on the MNIST dataset.
    It also prints associated analytics, including the accuracy and time taken
    to run.

    Keyword arguments:
    numTrainValues -- number of training values to train the perceptron
    numTestValues -- number of test values to test the trained perceptron
    pixels -- number of pixels to chop from the margins of the image
    tune -- a boolean for whether to tune to find the optimal number of iterations
    useTrainedWeights -- boolean to use pretrained weights
    """
    t = time.clock()
    perceptronClassifier = perceptron.Perceptron(range(10), 3)

    if useTrainedWeights:
        perceptronClassifier.useTrainedWeights()
    else:
        print "Loading Testing Data....\n"
        trainingData, trainingLabels, validationData, validationLabels, features = loadFeatures.loadTrainingData(numTrainValues, pixels, tune)

        print "Training Perceptron....\n"
        perceptronClassifier.train(trainingData, trainingLabels, validationData, validationLabels, tune)


    print "Loading Testing Data....\n"
    testingData, testingLabels = loadFeatures.loadTestingData(numTestValues, pixels)

    print "Testing Perceptron....\n"
    classifiedData = perceptronClassifier.classify(testingData)
    test(classifiedData, testingLabels)

    print "Total Time {0}".format(time.clock() - t)


def runNaiveBayes(numTrainValues, numTestValues, pixels, tune):
    """
    runNaiveBayes() runs the Naive Bayes learning algorithm on the MNIST dataset.
    It also prints associated analytics, including the accuracy and time taken
    to run.

    Keyword arguments:
    numTrainValues -- number of training values to train the perceptron
    numTestValues -- number of test values to test the trained perceptron
    pixels -- number of pixels to chop from the margins of the image
    tune -- a boolean for whether to tune to find the optimal number of iterations
    """
    t = time.clock()
    naiveBayesClassifier = naivebayes.NaiveBayes(range(10))

    print "Loading Training Data....\n"
    trainingData, trainingLabels, validationData, validationLabels, features = loadFeatures.loadTrainingData(numTrainValues, pixels, tune)

    print "Training Naive Bayes Classifier....\n"
    naiveBayesClassifier.train(trainingData, trainingLabels, validationData, validationLabels, features, tune)

    print "Loading Testing Data....\n"
    testingData, testingLabels = loadFeatures.loadTestingData(numTestValues, pixels)

    print "Testing Naive Bayes Classifier....\n"
    classifiedData = naiveBayesClassifier.classify(testingData)
    test(classifiedData, testingLabels)

    print "Total Time {0}".format(time.clock() - t)

def test(classifiedData, testingLabels):
    """
    test() gets a classification for the test data and checks
    if it matches the labels. It then returns a performance metric
    on the test set.

    Keyword Arguments:
    classifiedData -- the labels outputted by the trained algorithm on the test set
    testingLabels -- the correct labels associated with test set
    """
    countCorrect = 0
    # check if classification matches label
    for i in range(len(testingLabels)):
        if testingLabels[i] == classifiedData[i]:
            countCorrect += 1

    print "Number of Correct Classifications"
    print "================================="
    print countCorrect

    print "Percent of Correct Classifications"
    print "=================================="
    print float(countCorrect) / len(testingLabels) * 100.0



if __name__ == "__main__":

    # begin program
    readCommand()
