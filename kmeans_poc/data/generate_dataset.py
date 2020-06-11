# generate dataset : random data in a two-dimensional space

#https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#%matplotlib inline


def generate_x():
    X = -2 * np.random.rand(100,2)
    X1 = 1 + 2 * np.random.rand(50,2)
    X[50:100, :] = X1
    return X


def plot_x(X):
    plt.scatter(X[ : , 0], X[ :, 1], s = 50) #, c = ‘b’)
    plt.show()
    pass


def save_x(X):
    np.savetxt("x_poc.csv", X, delimiter=",")
    pass


def main():
    X = generate_x()
    plot_x(X)
    save_x(X)


if __name__ == '__main__':
    main()
