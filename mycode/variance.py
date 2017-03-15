from collections import Counter
from linear_algebra import sum_of_squares, dot
import math


# this isn't right if you don't from __future__ import division
# 计算均值，数组元素总和 / 数组长度
def mean(x):
    return sum(x) / len(x)

# 计算每一个元素与均值的差
def de_mean(x):
    """translate x by subtracting its mean (so the result has mean 0)"""
    x_bar = mean(x)
    print(x_bar)
    return [x_i - x_bar for x_i in x]


#print(de_mean([1,4,5]))

def correlation(x, y):
    stdev_x = standard_deviation(x)
    stdev_y = standard_deviation(y)
    if stdev_x > 0 and stdev_y > 0:
        return covariance(x, y) / stdev_x / stdev_y
    else:
        return 0 # if no variation, correlation is zero

def standard_deviation(x):
    return math.sqrt(variance(x))


def variance(x):
    """assumes x has at least two elements"""
    n = len(x)
    deviations = de_mean(x)
    return sum_of_squares(deviations) / (n - 1)