from scipy.special import expit
from scipy.optimize import minimize
import math
import numpy as np
import pickle
from random import shuffle
import time
import json

TOTAL_NUM_TRAINING_EXAMPLES = 100#1794


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


def main():
    ingredients = getIngredients()
    cuisines = getCuisines()
    trainingSet = formatTrainingSet(cuisines, ingredients)[:TOTAL_NUM_TRAINING_EXAMPLES]
    shuffle(trainingSet) #shuffle training set for kfold cross validation
    split_training_set = np.array_split(trainingSet, 6) # list of training set lists
    split_training_set = [s.tolist() for s in split_training_set]
    with open('split_training_set.p', 'wb') as f:
        pickle.dump(split_training_set, f)


if __name__ == '__main__':
    main()
