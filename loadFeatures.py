from mnist import *
from util import Counter
from progressBar import ProgressBar

def loadTrainingData(n=None, pixels=0, tune=False):
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
    if tune:
        split = float(len(labels) * 5 / 6)
    # if we are not tuning, use all validation data
    else:
        split = len(labels)

    # split training and validation images/labels
    trainingImages, trainingLabels = images[:split], labels[:split]
    validationImages, validationLabels = images[split:], labels[split:]

    print len(validationImages)

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

    # only return n data points
    if n:
        images = images[:n]
        labels = labels[:n]

    # get features for data
    testingData = defineFeatures(images, chop)[0]

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
                features.append((x,y))
                if image[x][y] == 0:
                    imgFeature[(x, y)] = 0
                else:
                    imgFeature[(x, y)] = 1

        featureList.append(imgFeature)
    progressBar.clear()

    return featureList, features


def main():
    loadData()


if __name__=="__main__":
    main()
