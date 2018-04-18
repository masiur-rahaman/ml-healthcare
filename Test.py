import numpy as np
import matplotlib.pyplot as plt
import Data
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def plot(x, y, color):
    plt.scatter(x, y, color=color)


def show():
    plt.show()


def plot_data(x, y, color):
    plot(x, y, color)
    plt.show()


def pca_fn():
    x = Data.X
    x_centered = x - x.mean(axis=0)
    # plot_data(X[:, 0], X[:, 1])
    # plot_data(X_centered[:, 0], X_centered[:, 1])
    # U,s,V = np.linalg.svd(X_centered)
    # plot_data(V[:, 0], V[:, 1])
    # print(V)
    pca = PCA(n_components=2)
    x2_d = pca.fit_transform(x)
    # plot_data(X2D[:, 0], X2D[:, 1])
    plot(x2_d[:, 0], x2_d[:, 1], 'red')
    # plot(X[:, 0], X[:, 1], 'red')
    plot(0, 0, 'green')
    plot(pca.components_[:, 0], pca.components_[:, 1], 'blue')
    show()
    # plot_data(pca.components_[:, 0], pca.components_[:, 1], 'red')
    print(pca.components_)
    # print(pca.explained_variance_)


def plot_a_point(x, label):
    if label == 0:
        plot(x[:, 0], x[:, 1], 'red')
    elif label == 1:
        plot(x[:, 0], x[:, 1], 'green')
    elif label == 2:
        plot(x[:, 0], x[:, 1], 'blue')
    elif label == 3:
        plot(x[:, 0], x[:, 1], 'violet')


def plot_colored_clusters(x, labels):
    for i, v in enumerate(x):
        plot_a_point(v, labels[i])
    show()


def cluster():
    x = Data.X
    np.random.shuffle(x)
    plot_data(x[:, 0], x[:, 1], 'red')
    kmeans = KMeans(n_clusters=3).fit(x)
    labels = kmeans.predict(x)
    print(labels)
    centroids = kmeans.cluster_centers_
    # plot(x[:, 0], x[:, 1], 'red')
    # plot(centroids[:, 0], centroids[:, 1], 'blue')
    # show()
    plot_colored_clusters(x, labels)
    print(centroids)


cluster()
