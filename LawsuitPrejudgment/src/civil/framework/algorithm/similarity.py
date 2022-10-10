# -*- coding: utf-8 -*-
import numpy as np

"""
Several similarity operators
"""


def manhattan(x, y):
    """
    Manhattan distance of vectors x and y. Shapes of x and y must be the same.
    :param x: 1-D or 2-D vector
    :param y: 1-D or 2-D vector
    :return: manhattan distance
    """
    x = np.array(x)
    y = np.array(y)
    if x.shape != y.shape:
        raise Exception('Shapes are not the same!')
    if len(x.shape) > 2:
        raise Exception('Only 1-D or 2-D is supported!')

    return np.sum(np.abs(x - y), axis=-1)


def euclidean(x, y):
    """
    Euclidean distance of vectors x and y. Shapes of x and y must be the same.
    d(x, y) = sqrt((x1-y1)^2+(x2-y2)^2+...+(xn-yn)^2)
    :param x: 1-D or 2-D vector
    :param y: 1-D or 2-D vector
    :return: euclidean distance
    """
    x = np.array(x)
    y = np.array(y)
    if x.shape != y.shape:
        raise Exception('Shapes are not the same!')
    if len(x.shape) > 2:
        raise Exception('Only 1-D or 2-D is supported!')

    return np.sqrt(np.sum(np.square(x - y), axis=-1))


def cosine(x, y):
    """
    Cosine distance of vectors x and y. Shapes of x and y must be the same.
    :param x: 1-D vector
    :param y: 1-D vector
    :return: Cosine distance
    """
    x = np.array(x)
    y = np.array(y)
    if x.shape != y.shape:
        raise Exception('Shapes are not the same!')
    if len(x.shape) > 1:
        raise Exception('Only 1-D is supported!')

    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def cosine_matrix(X, Y):
    """
    Cosine distance of matrix X and Y. Shapes of X and Y must be the same.
    :param X: 2-D vector
    :param Y: 2-D vector
    :return: Cosine distance of each vector pair
    """
    X = np.mat(X)
    Y = np.mat(Y)
    if X.shape != Y.shape:
        raise Exception('Shapes are not the same!')
    if len(X.shape) != 2:
        raise Exception('Only 2-D is supported!')

    XY = X * Y.transpose()
    _X = np.sqrt(np.multiply(X, Y).sum(axis=1))
    _Y = np.sqrt(np.multiply(Y, Y).sum(axis=1))
    return np.divide(XY, _X * _Y.transpose())


def pearson(x, y):
    """
    Pearson correlation coefficient of vectors x and y. Shapes of x and y must be the same.
    Bettern result than euclidean distance will be given when there are outliers.
    :param x: 1-D or 2-D vector
    :param y: 1-D vector
    :return: Pearson correlation coefficient
    """
    x = np.array(x)
    y = np.array(y)
    if x.shape != y.shape:
        raise Exception('Shapes are not the same!')
    if len(x.shape) > 1:
        raise Exception('Only 1-D is supported!')

    x_ = x - np.mean(x)
    y_ = y - np.mean(y)
    return np.dot(x_, y_) / (np.linalg.norm(x_) * np.linalg.norm(y_))


def jaccard(text_x, text_y):
    """
    Jaccard similarity coefficient of text x and y.
    :param text_x: text
    :param text_y: text
    :return: Jaccard correlation coefficient
    """
    x = set(text_x)
    y = set(text_y)
    return len(x & y) / len(x | y)


def cooccurence(x, y):
    '''
    Similarity of text x and y calculated by word cooccurence.
    :param law_name1:
    :param law_name2:
    :return:
    '''
    log1 = np.log(len(x)) if len(x) > 1 else 1
    log2 = np.log(len(y)) if len(y) > 1 else 1
    return len(set(x) & set(y)) / (log1 + log2)


if __name__ == '__main__':
    x = np.random.random(10)
    y = np.random.random(10)
    print(manhattan(x, y))
    print(euclidean(x, y))
    print(cosine_matrix(x, y))
    print(pearson(x, y))
    print(jaccard('你怎么看', '元芳怎么看'))
    print(cooccurence('你怎么看', '元芳怎么看'))
