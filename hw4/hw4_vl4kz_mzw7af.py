from scipy.special import expit
from scipy.optimize import minimize
import math
import numpy as np
import time
import json

LAMBDA = 1
NUM_HIDDEN_LAYERS = 1
NUM_FEATURES = 2398
NUM_NEURONS_PER_LAYER = 1209
EPSILON = 0.06987143837
NUM_CLASSES = 20
NUM_TRAINING_EXAMPLES = 2

def getIngredients():
    with open('ingredients.json') as json_data:
        data = json.load(json_data)
        ingredientsList = data["ingredients"]
        ingredientsDict = {k: v for v, k in enumerate(ingredientsList)}

    return ingredientsDict


def getCuisines():
    cuisineList = []

    with open('training.json') as f:
        for line in f:
            data = json.loads(line)
            cuisineList.append(data["cuisine"])

    cuisines = np.unique(cuisineList)
    cuisinesDict = {k: v for v, k in enumerate(cuisines)}

    return cuisinesDict


def formatTrainingSet(cuisines, ingredients):
    ishape = (2399, 1)
    cshape = (21, 1)
    trainingSet = []

    with open('training.json') as f:
        for line in f:
            ingredientMatrix = np.zeros(ishape)
            cuisineMatrix = np.zeros(cshape)

            # add in biases
            ingredientMatrix[0] = 1
            cuisineMatrix[0] = 1

            data = json.loads(line)

            for item in data["ingredients"]:
                ingredIndex = ingredients[item]
                ingredientMatrix[ingredIndex+1] = 1 # shift by 1 because of bias
            cuisineIndex = cuisines[data["cuisine"]]
            cuisineMatrix[cuisineIndex+1] = 1

            trainingDict = {}
            trainingDict['output'] = cuisineMatrix
            trainingDict['input'] = ingredientMatrix
            trainingSet.append(trainingDict)

        return trainingSet


def g(x):
    '''
    Calculates the sigmoid function for a vector
    '''
    vsigmoid = np.vectorize(expit)
    return vsigmoid(x)


def costFunction(thetas, X, Y):
    '''
    thetas = flat theta parameters
    X = list of training inputs
    Y = list of training answers
    theta_struct = list of theta matrices
    returns the cost and list of gradients
    '''
    m = NUM_TRAINING_EXAMPLES
    K = NUM_CLASSES
    thetas = reshape_matrices(thetas)
    forward_prop_results = [forwardPropagate(x, thetas) for x in X]
    results = [x[-1] for x in forward_prop_results]

    # calculate cost Function
    doubleSum = 0
    tripleSum = 0
    for i in range(1, m):
        for k in range(1, K):
            y_curr = Y[i][k]
            h_curr = results[i][k]
            doubleSum += y_curr  * np.log(h_curr) + (1-y_curr) * np.log(1-h_curr)

    L = len(thetas) # iterate through all thetas except the last single vector one
    for i in range(L):
        theta_matrix_curr = thetas[i]
        tripleSum += np.square(theta_matrix_curr).sum()
    cost = (-1/m) * doubleSum + (LAMBDA/(2*m)) * tripleSum

    ################### Calculate gradients ####################
    gradients_struct = backwardPropagate(forward_prop_results, Y, thetas)
    gradient_flat = unroll_matrices(gradients_struct)
    return cost, gradient_flat


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


def backwardPropagate(forward_prop_results, answers, thetas):
    '''
    Perform back propagation on a training Set (inputs and outputs)
    and thetas matrices
    forward_prop_results = list of a_vec_lists from performing forward propogate at each layers
    answers = answers from training set
    thetas = theta structs matrices
    Returns a matrix of deltas (for the cost function gradient descent)
    '''
    delta1 = np.zeros((NUM_NEURONS_PER_LAYER, NUM_FEATURES + 1))
    delta2 = np.zeros((NUM_CLASSES, NUM_NEURONS_PER_LAYER + 1))

    for idx in range(NUM_TRAINING_EXAMPLES):
        a_vec_list = forward_prop_results[idx]
        L = len(a_vec_list)
        a_vec_list.insert(0, "dummy") # insert dummy variable to make it match coursera
        delta_list = [[] for x in range(len(a_vec_list))]
        delta_list[L] = (a_vec_list[L]- answers[idx])[1:]

        #compute delta l-1 => 2
        for l in range(L-1, 1, -1): #for one hidden layer, only does one iteration lolz
            curr_theta = thetas[l-1].transpose() # use l-1 instead of l because thetas is 0 indexed
            predelta = curr_theta * delta_list[l+1]
            curr_delta = np.multiply(predelta, a_vec_list[l])
            complement_a = np.subtract(1, a_vec_list[l])
            curr_delta = np.multiply(curr_delta, complement_a)
            delta_list[l] = curr_delta[1:] # we ignore bias unit in delta vectors

        a_1_transpose = a_vec_list[1].transpose()
        a_2_transpose = a_vec_list[2].transpose()
        delta1 = delta1 + (delta_list[2] * a_1_transpose)
        delta2 = delta2 + (delta_list[3] * a_2_transpose)
    m = NUM_TRAINING_EXAMPLES
    theta1 = thetas[0]
    theta2 = thetas[1]

    # FINAL DELTA 1 CALCULATIONS
    for i in range(NUM_NEURONS_PER_LAYER):
        for j in range(NUM_FEATURES + 1):
            if j == 0:
                delta1[i, j] = (1/m) * (delta1[i, j])
            else:
                delta1[i, j] = (1/m) * (delta1[i, j] + LAMBDA * theta1[i, j])

    # FINAL DELTA 2 CALCULATIONS
    for i in range(NUM_CLASSES):
        for j in range(NUM_NEURONS_PER_LAYER + 1):
            if j == 0:
                delta2[i, j] = (1/m) * (delta2[i, j])
            else:
                delta2[i, j] = (1/m) * (delta2[i, j] + LAMBDA * theta2[i, j])
    return [delta1, delta2]


def initializeThetas():
    theta_results = []
    theta1 = np.random.rand(NUM_NEURONS_PER_LAYER, NUM_FEATURES + 1) # theta1 x input = (1209 x 2039) * (2039 x 1) = 1209 x 1 (plus 1 from forward propogate)
    theta1 = np.multiply(theta1, 2*EPSILON)
    theta1 = np.subtract(theta1, EPSILON)
    theta1 = np.asmatrix(theta1)

    theta2 = np.random.rand(NUM_CLASSES, NUM_NEURONS_PER_LAYER + 1) # theta2 x activation = (20 x 1210) * (1210 x 1) = 20 x 1 (plus 1 from forward propogate)
    theta2 = np.multiply(theta2, 2*EPSILON)
    theta2 = np.subtract(theta2, EPSILON)
    theta2 = np.asmatrix(theta2)

    return [theta1, theta2]


def gradientChecking(thetas, gradients, results, answers):
    epsilon = 1e-4
    theta_1_flat = np.hstack(thetas[0].flat)
    theta_2_flat = np.hstack(thetas[1].flat)
    gradient_1_flat = np.hstack(gradients[0].flat)
    gradient_2_flat = np.hstack(gradients[1].flat)
    print(theta_1_flat)
    print(theta_2_flat)
    theta_flat = np.concatenate([theta_1_flat, theta_2_flat])
    gradient_flat = np.concatenate([gradient_1_flat,gradient_2_flat])
    n = len(theta_flat)

    gradApprox = [0 for i in range(n)]

    for idx in range(n):
        thetaPlus = theta_flat.copy()
        thetaPlus[idx] = np.add(thetaPlus[idx], epsilon)
        thetaPlus_1 = thetaPlus[:len(theta_1_flat)]
        thetaPlus_2 = thetaPlus[len(theta_1_flat):]
        thetaPlus = [np.reshape(thetaPlus_1, thetas[0].shape), np.reshape(thetaPlus_2, thetas[1].shape)]
        thetaMinus = theta_flat.copy()
        thetaMinus[idx] = np.subtract(thetaMinus[idx], epsilon)
        thetaMinus_1 = thetaMinus[:len(theta_1_flat)]
        thetaMinus_2 = thetaMinus[len(theta_1_flat):]
        thetaMinus = [np.reshape(thetaMinus_1, thetas[0].shape), np.reshape(thetaMinus_2, thetas[1].shape)]
        gradApprox[idx] = (costFunction(thetaPlus, results, answers) - costFunction(thetaMinus, results, answers))/(2*epsilon)
    for idx, grad in gradApprox:
        if math.abs(grad - gradient_flat[idx]) > 1e-9:
            print(str(grad) + "   " + str(gradient_flat[idx]))


def unroll_matrices(matrix_list):
    flat_list = [np.hstack(matrix).flat for matrix in matrix_list]
    return np.concatenate(flat_list)


def reshape_matrices(flat_matrix):
    theta_1_len = NUM_NEURONS_PER_LAYER * (NUM_FEATURES + 1)
    theta_1 = flat_matrix[:theta_1_len]
    theta_2 = flat_matrix[theta_1_len:]
    theta_1_shape = (NUM_NEURONS_PER_LAYER, NUM_FEATURES + 1)
    theta_2_shape = (NUM_CLASSES, NUM_NEURONS_PER_LAYER + 1)
    return [np.asmatrix(np.reshape(theta_1, theta_1_shape)), np.asmatrix(np.reshape(theta_2, theta_2_shape))]


def main():
    ingredients = getIngredients()
    cuisines = getCuisines()
    trainingSet = formatTrainingSet(cuisines, ingredients)[:NUM_TRAINING_EXAMPLES]
    thetas = initializeThetas()
    # gradients = backwardPropagate(trainingSet, thetas)
    X = [x['input'] for x in trainingSet]
    Y = [y['output'] for y in trainingSet]
    # results = [forwardPropagate(x, thetas)[-1] for x in X]
    theta_flat = unroll_matrices(thetas)
    t0 = time.time()
    result = minimize(costFunction, theta_flat, args=(X, Y), method='TNC', jac=True)
    t1 = time.time()
    with open('parameters.json', 'w') as f:
        json.dump({'params' : result.x.tolist()}, f)
    total_time = t1-t0
    print(str(total_time))

if __name__ == '__main__':
    main()



# TEST CODE
# import os
# import pandas as pd
# path = 'ex2data1.txt'
# data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
#
# # add a ones column - this makes the matrix multiplication work out easier
# data.insert(0, 'Ones', 1)
#
# # set X (training data) and y (target variable)
# cols = data.shape[1]
# X = data.iloc[:,0:cols-1]
# y = data.iloc[:,cols-1:cols]
#
# # convert to numpy arrays and initalize the parameter array theta
# X = list(X.values)
# X = [np.matrix(x).transpose() for x in X]
# Y = list(y.values)
# print(Y)
# y = [np.matrix(y_item).transpose() for y_item in Y]
# theta = np.zeros(3)
# results = [forwardPropagate(x, [theta]) for x in X]
# #print(results)
#
# print(costFunction(theta, results, y, 100, 2))


# X = [
#     np.matrix([[1], [34.623660], [78.024693]]),
#     np.matrix([[1], [30.286711], [43.894998]]),
#     np.matrix([[1], [35.847409], [72.902198]]),
#     np.matrix([[1], [60.182599], [86.308552]]),
#     np.matrix([[1], [79.032736], [75.344376]]),
# ]
# y = [
#     np.matrix([[1], [1]]),
#     np.matrix([[1], [1]]),
#     np.matrix([[1], [0]]),
#     np.matrix([[1], [1]]),
#     np.matrix([[1], [1]]),
#
# ]
# theta = np.zeros(3)
# results = [forwardPropagate(x, [theta]) for x in X]
# print(results)
# print(costFunction(theta, results, y, len(X), 2))
# '''
#
# '''
# theta_1 = np.matrix([[-30, 20, 20], [10, -20, -20]])
# theta_2 = np.matrix([-10, 20, 20])
# inputvector = np.matrix([[1],[0],[1]])
# thetaArray = [theta_1, theta_2]
# print(forwardPropagate(inputvector, thetaArray))
#
