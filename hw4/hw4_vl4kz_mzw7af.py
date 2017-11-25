from scipy.special import expit
import math
import numpy as np
import json

LAMBDA = 1
NUM_HIDDEN_LAYERS = 1


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
            y_curr = answers.get(i).get(k)
            h_curr = results.get(i).get(k)
            doublesum += y_curr  * math.log(h_curr) + (1-y_curr) * math.log(1-h_curr)

    L = len(thetas) # iterate through all thetas except the last single vector one
    for i in range(1, L-1):
        theta_matrix_curr = thetas.get(i)
        tripleSum += np.square(theta_matrix).sum()
    return (-1/m) * doubleSum + (LAMBDA/(2*m)) * tripleSum


def forwardPropagate(features, thetas):
    '''
    features = single vector of features for a test case
    thetas = array of theta matrices
    '''
    a_vec = features
    for theta_matrix in np.nditer(thetas):
        print(theta_matrix)
        z_vec = theta_matrix * a_vec
        a_vec = g(z_vec)
    return a_vec


def initializeThetas():
    pass

def getEpsilon(L_in, L_out):
    '''
    L_in = # of features
    L_out = # of classes for classification
    '''
    return math.sqrt(6) / math.sqrt(L_in + L_out)

def main():
    ingredients = getIngredients()
    cuisines = getCuisines()
    trainingSet = formatTrainingSet(cuisines, ingredients)

    theta = np.matrix([-30, 20, 20])
    inputvector = np.matrix([[1],[0],[0]])
    thetaArray = np.array([theta])
    print(forwardPropagate(inputvector, thetaArray))

if __name__ == '__main__':
    main()
