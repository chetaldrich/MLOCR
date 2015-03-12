from numpy import *
from mnist import *
from util import Counter

def loadData():
    images, labels = load_mnist('training')
    featureList = defineFeatures(images)

    return featureList, labels

def defineFeatures(imageList, n):
    imageList = imageList[0:]
    featureList = []
    for image in imageList:
        imgFeature = Counter()
        for i in range(len(image)):
            for j in range(len(image[i])):
                if image[i][j] == 0:
                    imgFeature[(i, j)] = 0
                else:
                    imgFeature[(i, j)] = 1
                # imgFeature[(i, j)] = image[i][j]
        featureList.append(imgFeature)
    return featureList
    

def main():
    pass

if __name__=="__main__":
    main()
