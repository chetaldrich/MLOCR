# MLOCR
Optical Character Recognition of digits using machine learning.

We will be using the MNIST database of handwritten digits to make use of machine learning algorithms.

We currently have implementations of the following machine learning techniques.

* Single Perceptron
* Naive Bayes

In order to run the program, you'll need the [MNIST data](http://yann.lecun.com/exdb/mnist/), which you should place in a directory called `mnist`.

There is a script to perform this included with the repository, which can be run as follows:

* `$ chmod +x loadMNISTData.sh`
* `$ ./loadMNISTData.sh`

The following commands should allow you to try each of the classification techniques we have implemented:

* Single Perceptron
  `python dataClassifier.py -c perceptron`
* Naive Bayes `python dataClassifier.py -c naivebayes`
