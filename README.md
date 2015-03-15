# MLOCR

### Optical Character Recognition of digits using machine learning.

We will be using the MNIST database of handwritten digits to make use of machine learning algorithms.

We currently have implementations of the following machine learning techniques.

* Single Perceptron
* Naive Bayes

In order to run the program, you'll need the [MNIST data](http://yann.lecun.com/exdb/mnist/), which you should place in a directory called `mnist`.

There is a script to perform this included with the repository, which can be run as follows:

* `$ chmod +x loadMNISTData.sh`
* `$ ./loadMNISTData.sh`

The following commands should allow you to try each of the classification techniques we have implemented (We recommend having a maximizing the size of your terminal window for the best experience with the progress bars):

* Single Perceptron
  `$ python dataClassifier.py -c perceptron`
* Naive Bayes `$ python dataClassifier.py -c naivebayes`

The commands above will run the classifiers over a smaller amount of data by default so that it doesn't take an exorbitant amount of time to run. However, you can run the program on all of the data (and achieve higher accuracy) as follows:

* `$ python dataClassifier.py -a -c naivebayes --train 60000 --test 10000`
* `$ python dataClassifier.py -a -c perceptron --train 60000 --test 10000`

Naturally, you can change these parameters in order to vary the amount of training and test data.

All parameters that are available for use:

* `-h, --help` show the help message and exit

* `-c` selects the classifier (`naivebayes` and `perceptron` are options)

* `-a` Tune iterations/Laplace smoothing while training

* `-u` Use pre-learned weights for the perceptron. These were created after training the program for about 3 hours on all of the training data, and this option is available so you can see the results of the classifier without having to wait for the perceptron to train.

* `--pixels` Chops off some pixels from the edges of each image to improve processing time. Sometimes leads to a slight loss in accuracy. Options are 1-12 pixels.

* `--train` selects the number of training data samples to be used by the classifier

* `--test` selects the number of test data samples to be used by the classifier
