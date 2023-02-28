from matplotlib import pyplot as plt
import numpy as np
import parameters
import scipy.stats as stats
import pandas as pd


def _set_color(color):
    if color == 'tb':
        color = '#0abab5'
    elif color == 'kb':
        color = '#002FA7'
    return color


def cal_mse(hist, est):
    try:
        diff = hist - est
        return np.sum(diff**2)/len(diff)
    except:
        return -1


def cal_norm(hist, est, ord=2):
    diff = hist - est
    norm = np.linalg.norm(diff, ord=ord)
    return norm


def draw_bar(hist, title = 'default', color = 'tb'):
    color = _set_color(color)
    d = len(hist)
    plt.bar(range(d), hist, color = color) #Tiffany blue:#0abab5； Klein blue：#002FA7
    plt.title(title)
    plt.show()

def draw_hist(points, bins=None, title = 'default', color = 'tb'):
    color = _set_color(color)
    plt.hist(points, color = color, bins=bins)
    plt.title(title)
    plt.show()


def cal_hist_c(X, d):
        hist = np.zeros(d)
        unique, counts = np.unique(X, return_counts=True)
        for i in range(len(unique)):
            hist[int(unique[i])] = counts[i]
        return hist


def cal_hist_f(X, d):
    hist = np.zeros(d)
    unique, counts = np.unique(X, return_counts=True)
    for i in range(len(unique)):
        hist[int(unique[i])] = counts[i]
    return hist/len(X)


def read_data(name, path = parameters.data_path):
    with open(path + name + '.txt', 'r') as rfile:
        data = rfile.readlines()
        para_dict = eval(data[0])
        d = para_dict['d']
        n = para_dict['n']
        X = eval(data[1])
    return X, d, n

