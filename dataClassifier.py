import naivebayes
import perceptron
import loadFeatures
import argparse


def readCommand():

    # default values
    numTestValues = 100
    numTrainValues = 100
    pixels = 0

    parser = argparse.ArgumentParser()

    parser.add_argument("-c", type = str, choices=["naivebayes", "perceptron"], help="selects the classifier for use with the MNIST data")

    parser.add_argument("--pixels", type = int, choices=range(13), help="remove this many pixels from outside of photo for faster training")

    parser.add_argument("--train", type = int, help="selects the number of training data samples to be used by the classifier")

    parser.add_argument("--test", type = int,  help="elects the number of testing data samples to be used by the classifier")

    args = parser.parse_args()

    if args.train != None:
        numTrainValues = args.train
    if args.test != None:
        numTestValues = args.test
    if args.pixels != None:
        pixels = args.pixels


    if args.c == "naivebayes":
        runNaiveBayes(numTrainValues, numTestValues)
    elif args.c == "perceptron":
        runPerceptron(numTrainValues, numTestValues)

def runPerceptron(numTrainValues, numTestValues):
    perceptronClassifier = perceptron.Perceptron(range(10), 3)

    print "Loading Testing Data....\n"
    trainingData, trainingLabels, validationData, validationLabels, features = loadFeatures.loadTrainingData(numTrainValues, pixels)

    print "Training Perceptron....\n"
    perceptronClassifier.train(trainingData, trainingLabels, validationData, validationLabels, False)

    print "Loading Testing Data....\n"
    testingData, testingLabels = loadFeatures.loadTestingData(numTestValues, pixels)

    print "Testing Perceptron....\n"
    perceptronClassifier.test(testingData, testingLabels)


def runNaiveBayes(numTrainValues, numTestValues):
    naiveBayesClassifier = naivebayes.NaiveBayes(range(10))

    print "Loading Training Data....\n"
    trainingData, trainingLabels, validationData, validationLabels, features = loadFeatures.loadTrainingData(numTrainValues)

    print "Training Naive Bayes Classifier....\n"
    naiveBayesClassifier.train(trainingData, trainingLabels, validationData, validationLabels, features, False)

    print "Loading Testing Data....\n"
    testingData, testingLabels = loadFeatures.loadTestingData(numTestValues)

    print "Testing Naive Bayes Classifier....\n"
    naiveBayesClassifier.test(testingData, testingLabels)


if __name__ == "__main__":

    # begin program
    readCommand()
