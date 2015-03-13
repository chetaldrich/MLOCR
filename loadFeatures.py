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

    # print "label length:", len(labels)


    # find out where to split so that 5/6 of data is training
    # and 1/6 is validation
    split = float(len(labels) * 5 / 6)

    # split training and validation images/labels
    trainingImages, trainingLabels = images[:split], labels[:split]
    validationImages, validationLabels = images[split:], labels[split:]

    # get features for data
    trainingData, trainingFeatures = defineFeatures(trainingImages)
    validationData, validationFeatures = defineFeatures(validationImages)

    return trainingData, trainingLabels, validationData, validationLabels, trainingFeatures

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
    testingData = defineFeatures(images)[0]

    return testingData, labels


def defineFeatures(imageList):
    """
    defineFeatures() defines a simple feature of a pixel either being white (0)
    or not (1) for a list of images and pixel values
    """
    featureList = []
    features = []
    for image in imageList:
        # create feature of on/off for (x, y) positions in image
        imgFeature = Counter()
        for x in range(len(image)):
            for y in range(len(image[x])):
                features.append((x,y))
                if image[x][y] == 0:
                    imgFeature[(x, y)] = 0
                else:
                    imgFeature[(x, y)] = 1
        featureList.append(imgFeature)

    return featureList, features


def main():
    loadData()


if __name__=="__main__":
    main()
