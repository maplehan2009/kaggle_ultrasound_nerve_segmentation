# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 18:38:19 2016

@author: Jingtao
"""

from scipy import misc
import numpy as np

def read_img(name):
    path = 'data/train/' + name + '.tif'
    image = misc.imread(path)
    return (image.T).flatten()
    
def read_training_set():
    X = np.zeros([5635, 243600])
    Y = np.zeros([5635, 243600])
    bizarre_people = [3, 7, 17, 34, 44]
    idx = 0
    for i in range(1, 48):
        for j in range(1, 121):
            if i in bizarre_people and j == 120:
                continue
            name_x = str(i) + '_' + str(j)
            name_y = name_x + '_mask'
            x_temp = read_img(name_x)
            y_temp = read_img(name_y)
            X[idx] = x_temp
            Y[idx] = y_temp
            idx += 1
    return X, Y

def RLE(y):
    position = 0
    N = y.size
    ans = []
    while (position < N):
        left = np.argmax(y[position:])
        right = np.argmin(y[(position + left):])
        if (left != right):
            ans.append((position+left+1, right))
            position = position + left + right
        else:
            break
    return ans

if __name__=='__main__':
    X, Y = read_training_set()
