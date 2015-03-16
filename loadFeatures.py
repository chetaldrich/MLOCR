# loadFeatures.py
# -----------------
# Main file for parsing data from the MNIST dataset, and passing it onto the
# machine learning algorithms.
# 
# Chet Aldrich, Laura Biester

from mnist import *
from util import Counter
from progressBar import ProgressBar
import random

def loadTrainingData(n=None, pixels=0, tune=False):
    """
    loadTrainingData() pulls trainig data from MNIST training set, splits it into training and
    validation data, then parses the data into features
    """
    # load data from MNIST files
    images, labels = load_mnist('training')


    random.seed(5)
    imageLabels = zip(images, labels)
    random.shuffle(imageLabels)
    imageLabels = [list(t) for t in zip(*imageLabels)]
    images, labels = imageLabels[0], imageLabels[1]

    # split off 100 training instances if we are tuning
    split = -100 if tune else len(labels)

    # split training and validation images/labels
    trainingImages, trainingLabels = images[:split], labels[:split]
    validationImages, validationLabels = images[split:], labels[split:]

    # get point where we should split to have n training images
    n = n if n <= len(trainingLabels) else len(trainingLabels)

    trainingImages, trainingLabels = trainingImages[:n], trainingLabels[:n]

    # get features for data
    trainingData, trainingFeatures = defineFeatures(trainingImages, pixels)
    validationData, validationFeatures = defineFeatures(validationImages, pixels)

    return trainingData, trainingLabels, validationData, validationLabels, trainingFeatures

def loadTestingData(n=None, chop=0, tune=False):
    """
    loadTestingData() pulls testing data from MNIST training set,
    then parses the data into features
    """
    # load data from MNIST files
    images, labels = load_mnist('testing')

    # randomize data to be tested on
    random.seed(5)
    imageLabels = zip(images, labels)
    random.shuffle(imageLabels)
    imageLabels = [list(t) for t in zip(*imageLabels)]
    images, labels = imageLabels[0], imageLabels[1]

    # only return n data points
    if n:
        images = images[:n]
        labels = labels[:n]

    # get features for data
    testingData = defineFeatures(images, chop)[0]

    dict = {}
    for label in labels:
        if label in dict:
            dict[label] += 1
        else:
            dict[label] = 1

    print "labels in testing data: ", dict

    return testingData, labels


def defineFeatures(imageList, chop):
    """
    defineFeatures() defines a simple feature of a pixel either being white (0)
    or not (1) for a list of images and pixel values

    chops off pixels on outside of image for faster (but less accurate) classification
    """
    featureList = []
    features = []
    progressBar = ProgressBar(100, len(imageList), "Getting Features for Images")
    for index, image in enumerate(imageList):
        # update progress
        progressBar.update(index)

        # create feature of on/off for (x, y) positions in image
        imgFeature = Counter()
        for x in range(chop, len(image) - chop):
            for y in range(chop, len(image[x]) - chop):
                # features.append((x,y))
                if image[x][y] == 0:
                    imgFeature[(x, y)] = 0
                else:
                    imgFeature[(x, y)] = 1

        featureList.append(imgFeature)

    progressBar.clear()

    if len(imageList) > 0:
        image = imageList[0]
        for x in range(chop, len(image) - chop):
            for y in range(chop, len(image[x]) - chop):
                features.append((x,y))


    return featureList, features


def main():
    loadData()


if __name__=="__main__":
    main()
