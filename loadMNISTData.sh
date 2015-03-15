############
# loadMNISTData.sh
#
# Once downloaded, use the following commands to execute inside the repository to load # the MNIST data for use in the application:
#
# $ chmod +x loadMNISTData.sh
# $ ./loadMNISTData.sh
#
############


mkdir mnist/
curl http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz > mnist/train-images-idx3-ubyte.gz
curl http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz > mnist/train-labels-idx1-ubyte.gz
curl http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz > mnist/t10k-images-idx3-ubyte.gz
curl http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz > mnist/t10k-labels-idx1-ubyte.gz

echo "MNIST data loaded"
