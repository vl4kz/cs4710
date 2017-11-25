from scipy.special import expit
import math
import numpy
import json

LAMBDA = 1
NUM_HIDDEN_LAYERS = 1


def main():
    pass



def g(x):
    '''
    Calculates the sigmoid function for a vector
    '''
    vsigmoid = np.vectorize(expit)
    return vsigmoid(x)


def costFunction(thetas, results, answers, m, K):
    '''
    thetas = array of theta matrices
    results = array of vectors (of results from neural network)
    answers = array of vectors (of given answers))
    m = # of training examples
    K = # of classes
    (assumes bias is included in the matrices given)
    '''
    doubleSum = 0
    tripleSum = 0
    for i in range(1, m):
        for k in range(1, K):
            y_curr = answers.get(i).get(k)
            h_curr = results.get(i).get(k)
            doublesum += y_curr  * math.log(h_curr) + (1-y_curr) * math.log(1-h_curr)

    L = len(thetas) # iterate through all thetas except the last single vector one
    for i in range(1, L-1)
        theta_matrix_curr = thetas[i]
        tripleSum += numpy.square(theta_matrix).sum()
    return (-1/m) * doubleSum + (LAMBDA/(2*m)) * tripleSum


def forwardPropagate(features, thetas):
    '''
    features = single vector of features for a test case
    thetas = array of theta matrices
    '''
    a_vec = features
    for theta_matrix in np.nditer(thetas):
        z_vec = theta_matrix * a_vec
        a_vec = g(z_vec)
    return a_vec

if __name__ == '__main__':
    main()
