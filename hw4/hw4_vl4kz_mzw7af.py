from scipy.special import expit
import math
import numpy as np
import json

LAMBDA = 1
NUM_HIDDEN_LAYERS = 1
NUM_FEATURES = 2398
NUM_NEURONS_PER_LAYER = 1209
EPSILON = 0.06987143837

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
            y_curr = answers[i][k]
            h_curr = results[i][k]
            doubleSum += y_curr  * np.log(h_curr) + (1-y_curr) * np.log(1-h_curr)

    L = len(thetas) # iterate through all thetas except the last single vector one
    for i in range(1, L-1):
        theta_matrix_curr = thetas[i]
        tripleSum += np.square(theta_matrix_curr).sum()
    return (-1/m) * doubleSum + (LAMBDA/(2*m)) * tripleSum

def forwardPropagate(features, thetas):
    '''
    features = single vector of features for a test case
    thetas = array of theta matrices
    '''
    a_vec = features
    for theta_matrix in thetas:
        z_vec = theta_matrix * a_vec
        a_vec = g(z_vec)
        a_vec = np.insert(a_vec, 0, 1)
        a_vec = a_vec.transpose()
    return a_vec


def initializeThetas():
    theta_results = []
    theta1 = np.random.rand(NUM_NEURONS_PER_LAYER, NUM_FEATURES + 1)
    theta1 = np.multiply(theta1, 2*EPSILON)
    theta1 = np.subtract(theta1, EPSILON)

    theta2 = np.random.rand(1, 20 + 1)
    theta2 = np.multiply(theta2, 2*EPSILON)
    theta2 = np.subtract(theta2, EPSILON)

    return [theta1, theta2]


def main():
    ingredients = getIngredients()
    cuisines = getCuisines()
    trainingSet = formatTrainingSet(cuisines, ingredients)
    thetas = initializeThetas()


if __name__ == '__main__':
    main()
