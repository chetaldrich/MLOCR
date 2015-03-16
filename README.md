# MLOCR

### Optical Character Recognition of digits using machine learning.

The program in this repository uses the MNIST database of handwritten digits and attempts to classify them using machine learning algorithms.

We currently have implementations of the following machine learning techniques:

* Single Perceptron
* Naive Bayes

#### Dependencies

In order to run the program, you'll need the [MNIST data](http://yann.lecun.com/exdb/mnist/), which you should place in a directory called `mnist`.

There is a script to perform this included with the repository, which can be run as follows:

* `$ chmod +x loadMNISTData.sh`
* `$ ./loadMNISTData.sh`

Additionally, you'll need NumPy. If you don't have this Python package already, you can get it with pip, the installation instructions for which can be found [here](https://pip.pypa.io/en/latest/installing.html). After installing pip, you can get NumPy with the following command:

`$ sudo pip install numpy`

#### Usage

The following commands should allow you to try each of the classification techniques we have implemented (We recommend maximizing the size of your terminal window for the best experience with the progress bars):

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

* `-u` Use pre-learned weights for the chosen classifier. These were created after training the program over all of the training data. This is available so you can see the results of an optimized classifier without having to wait for the classifiers to train.

* `--pixels` Chops off some pixels from the edges of each image to improve processing time. Sometimes leads to a slight loss in accuracy. Options are 1-12 pixels.

* `--train` selects the number of training data samples to be used by the classifier

* `--test` selects the number of test data samples to be used by the classifier

* `-i` shows the most frequent incorrect classifications
