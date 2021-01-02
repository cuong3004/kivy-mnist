import numpy as np
from mnist import MNIST


def get_data_mnist():
    print("Load data")
    mndata = MNIST(r'GoMNIST\data\mnist')
    mndata.load_testing()
    X_test = np.asarray(mndata.test_images)
    X_test = X_test.reshape((X_test.shape[0], 28, 28))
    y_test = np.asarray(mndata.test_labels)

    return X_test, y_test

