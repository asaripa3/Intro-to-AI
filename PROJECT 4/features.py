# features.py
# -----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import numpy as np
import util
import samples
#import matplotlib.pylab as plt # Yantian
from threading import Thread # Yantian
import time

DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28

def basicFeatureExtractor(datum):
    """
    Returns a binarized and flattened version of the image datum.

    Args:
        datum: 2-dimensional numpy.array representing a single image.

    Returns:
        A 1-dimensional numpy.array of features indicating whether each pixel
            in the provided datum is white (0) or gray/black (1).
    """
    features = np.zeros_like(datum, dtype=int)
    features[datum > 0] = 1
    return features.flatten()


def enhancedFeatureExtractor(datum):
    """
    Returns a feature vector of the image datum.

    Args:
        datum: 2-dimensional numpy.array representing a single image.
colorbar()
    Returns:
        A 1-dimensional numpy.array of features designed by you. The features
            can have any length.

    ## DESCRIBE YOUR ENHANCED FEATURES HERE...
    I have taken the hint from the question where encoding techniques are used for white region classes
    The enhanced features in the code captures blocks of white regions
    Connected White Components: Identify and quantify clusters of connected white pixels in the image.
    Run-Length Encoding for White and Black Regions: Encode the lengths of consecutive sequences of white and black pixels in the image.
    Normalization of Encodings: Ensure uniformity and consistency in the measurements across different images.
    Run-Length Encoding Function: Perform run-length encoding on a 1-dimensional binary array.
    ##
    """
    features = basicFeatureExtractor(datum)

    "*** YOUR CODE HERE ***"
    enhanced_feature_1 = calculate_and_encode_white_regions(datum)
    enhanced_feature_2 = calculate_and_encode_black_pixels_ratio(datum)

    enhanced_features = np.concatenate((enhanced_feature_1, enhanced_feature_2))
    return np.concatenate((features, enhanced_features))

def calculate_and_encode_white_regions(datum):
    # Calculate the number of white regions and perform encoding
    white_regions = custom_white_regions_calculation(datum)
    return custom_encoding(white_regions, [1, 2, 3])

def calculate_and_encode_black_pixels_ratio(datum):
    # Calculate black pixels ratio and perform encoding
    black_pixels_ratio = custom_black_pixels_ratio_calculation(datum)
    return custom_encoding(black_pixels_ratio, np.arange(0, 1.1, 0.1))

def custom_white_regions_calculation(datum):
    # Custom logic to calculate white regions
    visited = set()
    regions = 0
    for i in range(DIGIT_DATUM_HEIGHT):
        for j in range(DIGIT_DATUM_WIDTH):
            if (i, j) not in visited and datum[i][j] == 0:
                dfs(datum, i, j, visited)
                regions += 1
    return regions

def custom_black_pixels_ratio_calculation(datum):
    # Custom logic to calculate black pixels ratio
    black_pixels = np.sum(datum > 0)
    bounding_box_pixels = calculate_bounding_box_pixels(datum)
    return black_pixels / bounding_box_pixels

def calculate_bounding_box_pixels(datum):
    min_i, max_i, min_j, max_j = DIGIT_DATUM_HEIGHT - 1, 0, DIGIT_DATUM_WIDTH - 1, 0
    for i in range(DIGIT_DATUM_HEIGHT):
        for j in range(DIGIT_DATUM_WIDTH):
            if datum[i][j] > 0:
                min_i, max_i, min_j, max_j = min(i, min_i), max(i, max_i), min(j, min_j), max(j, max_j)
    return (max_j - min_j + 1) * (max_i - min_i + 1)

def custom_encoding(value, encoding_scheme):
    # Custom encoding logic
    encoded_feature = np.zeros(len(encoding_scheme), dtype=int)
    index = np.digitize(value, encoding_scheme) - 1
    encoded_feature[index] = 1
    return encoded_feature

def dfs(datum, i, j, visited):
    neigbours= [(i-1, j-1), (i-1, j), (i-1, j+1), (i, j-1), (i, j+1), (i+1, j-1), (i+1, j), (i+1, j+1)]
    
    if 0 <= i < DIGIT_DATUM_HEIGHT and 0 <= j < DIGIT_DATUM_WIDTH and datum[i][j] == 0:
        visited.add((i, j))
        for x, y in neigbours:
            if (x, y) not in visited:
                dfs(datum, x, y, visited)


def analysis(model, trainData, trainLabels, trainPredictions, valData, valLabels, validationPredictions):
    """
    This function is called after learning.
    Include any code that you want here to help you analyze your results.

    Use the print_digit(numpy array representing a training example) function
    to the digit
    
    An example of use has been given to you.

    - model is the trained model
    - trainData is a numpy array where each row is a training example
    - trainLabel is a list of training labels
    - trainPredictions is a list of training predictions
    - valData is a numpy array where each row is a validation example
    - valLabels is the list of validation labels
    - valPredictions is a list of validation predictions

    This code won't be evaluated. It is for your own optional use
    (and you can modify the signature if you want).
    """

    # Put any code here...
    # Example of use:
    # for i in range(len(trainPredictions)):
    #     prediction = trainPredictions[i]
    #     truth = trainLabels[i]
    #     if (prediction != truth):
    #         print "==================================="
    #         print "Mistake on example %d" % i
    #         print "Predicted %d; truth is %d" % (prediction, truth)
    #         print "Image: "
    #         print_digit(trainData[i,:])


## =====================
## You don't have to modify any code below.
## =====================

def print_features(features):
    str = ''
    width = DIGIT_DATUM_WIDTH
    height = DIGIT_DATUM_HEIGHT
    for i in range(width):
        for j in range(height):
            feature = i*height + j
            if feature in features:
                str += '#'
            else:
                str += ' '
        str += '\n'
    print(str)

def print_digit(pixels):
    width = DIGIT_DATUM_WIDTH
    height = DIGIT_DATUM_HEIGHT
    pixels = pixels[:width*height]
    image = pixels.reshape((width, height))
    datum = samples.Datum(samples.convertToTrinary(image),width,height)
    print(datum)

def _test():
    import datasets
    train_data = datasets.tinyMnistDataset()[0]
    for i, datum in enumerate(train_data):
        print_digit(datum)

if __name__ == "__main__":
    _test()
