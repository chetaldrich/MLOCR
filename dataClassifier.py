import naivebayes
import perceptron
import loadFeatures
import argparse
import time
import neuralnet


def readCommand():

    # default values
    numTestValues = 100
    numTrainValues = 100
    pixels = 0
    numNeurons = 30
    batchSize = 10

    parser = argparse.ArgumentParser()

    parser.add_argument("-c", type = str, choices = ["naivebayes", "perceptron", "neuralnet"], help = "selects the classifier for use with the MNIST data")

    parser.add_argument("-a", action = "store_true", default=False, help = "tune while training")

    parser.add_argument("-u", action = "store_true", default=False, help = "use pre-learned weights for perceptron")

    parser.add_argument("--pixels", type = int, choices = range(13), help = "remove this many pixels from outside of photo for faster training")

    parser.add_argument("--train", type = int, help = "selects the number of training data samples to be used by the classifier")

    parser.add_argument("--test", type = int,  help = "selects the number of testing data samples to be used by the classifier")

    parser.add_argument("-n", type = int, help = "designates the number of neurons in the hidden layer of the neural net")

    parser.add_argument("-b", type = int, help = "selects the batch size for the stochastic gradient descent algorithm in the neural net")

    args = parser.parse_args()

    if args.train != None:
        numTrainValues = args.train
    if args.test != None:
        numTestValues = args.test
    if args.pixels != None:
        pixels = args.pixels

    if args.c == "naivebayes":
        runNaiveBayes(numTrainValues, numTestValues, pixels, args.a)
    elif args.c == "perceptron":
        runPerceptron(numTrainValues, numTestValues, pixels, args.a, args.u)
    elif args.c == "neuralnet":
        runNeuralNet(numTrainValues, numTestValues, numNeurons, batchSize)

def runPerceptron(numTrainValues, numTestValues, pixels, tune, useTrainedWeights):
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
    perceptronClassifier.test(testingData, testingLabels)

    print "Total Time {0}".format(time.clock() - t)


def runNaiveBayes(numTrainValues, numTestValues, pixels, tune):
    naiveBayesClassifier = naivebayes.NaiveBayes(range(10))

    print "Loading Training Data....\n"
    trainingData, trainingLabels, validationData, validationLabels, features = loadFeatures.loadTrainingData(numTrainValues, pixels, tune)

    print "Training Naive Bayes Classifier....\n"
    naiveBayesClassifier.train(trainingData, trainingLabels, validationData, validationLabels, features, tune)

    print "Loading Testing Data....\n"
    testingData, testingLabels = loadFeatures.loadTestingData(numTestValues, pixels)

    print "Testing Naive Bayes Classifier....\n"
    naiveBayesClassifier.test(testingData, testingLabels)

def test(classifiedData, testingLabels, ):
    pass

def runNeuralNet(numTrainValues, numTestValues, numNeurons, batchSize):
    pass


if __name__ == "__main__":

    # begin program
    readCommand()
