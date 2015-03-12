from mnist import *
from util import Counter

def loadTrainingData(n=None):
    """
    loadTrainingData() pulls trainig data from MNIST training set, splits it into training and
    validation data, then parses the data into features
    """
    # load data from MNIST files
    images, labels = load_mnist('training')

    # only return n data points
    if n:
        images = images[:n]
        labels = labels[:n]

    # find out where to split so that 5/6 of data is training
    # and 1/6 is validation
    split = float(len(labels) * 5 / 6)

    # split training and validation images/labels
    trainingImages, trainingLabels = images[:split], labels[:split]
    validationImages, validationLabels = images[split:], labels[split:]

    # get features for data
    trainingData = defineFeatures(trainingImages)
    validationData = defineFeatures(validationImages)

    return trainingData, trainingLabels, validationData, validationLabels

def loadTestingData(n=None):
    """
    loadTestingData() pulls testing data from MNIST training set,
    then parses the data into features
    """
    # load data from MNIST files
    images, labels = load_mnist('testing')

    # only return n data points
    if n:
        images = images[:n]
        labels = labels[:n]

    # get features for data
    testingData = defineFeatures(images)

    return testingData, labels


def defineFeatures(imageList):
    """
    defineFeatures() defines a simple feature of a pixel either being white (0)
    or not (1) for a list of images and pixel values
    """
    featureList = []
    for image in imageList:
        # create feature of on/off for (x, y) positions in image
        imgFeature = Counter()
        for x in range(len(image)):
            for y in range(len(image[x])):
                if image[x][y] == 0:
                    imgFeature[(x, y)] = 0
                else:
                    imgFeature[(x, y)] = 1
        featureList.append(imgFeature)

    return featureList


def main():
    loadData()


if __name__=="__main__":
    main()
