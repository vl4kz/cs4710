from scipy.special import expit
from scipy.optimize import minimize
import math
import numpy as np
import pickle
import time
import json
import sys


def g(x):
    '''
    Calculates the sigmoid function for a vector
    '''
    vsigmoid = np.vectorize(expit)
    return vsigmoid(x)


def forwardPropagate(features, thetas):
    '''
    features = single vector of features for a test case
    thetas = array of theta matrices
    '''
    a_vec_list = []
    a_vec = features

    a_vec_list.append(a_vec)

    for theta_matrix in thetas:
        z_vec = theta_matrix * a_vec
        a_vec = g(z_vec)
        a_vec = np.insert(a_vec, 0, 1)
        a_vec = a_vec.transpose()
        a_vec_list.append(a_vec)

    return a_vec_list

def main():
    i = str(sys.argv[1])
    with open('parameters'+i+'.p', 'rb') as f:
        data = pickle.load(f)
    thetas = data['params']
    curr_testing_set = data['test']
    set_size = len(curr_testing_set)
    num_correct = 0
    for example in curr_testing_set:
        X = example['input']
        Y = example['output']
        result_vec = forwardPropagate(X, thetas)[-1]
        max_index = np.argmax(result_vec[1:]) + 1
        if Y[max_index] == 1:
            num_correct += 1
    print(num_correct / set_size)


if __name__ == '__main__':
    main()
