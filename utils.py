import cv2, glob, os, sys, csv
import numpy as np
import tensorflow as tf
import pandas as pd 

# Read and preprocess image data
# Preprocess: Normalization
def readImageFile(path):
    try:
        I = cv2.imread(path)
        I = I / 255
        I = np.reshape(I, (1, I.shape[0], I.shape[1], I.shape[2]))
    except Exception as e:
        print("\nError reading image file")
        print("\nError Message: " + str(e))
        sys.exit(-1)
    return I

# Extract normalized image data, label and filename from a given path
def getDataAndLabels(path):
    classes = glob.glob(os.path.join(path, "*"))
    labels = []
    data = []
    filename = []
    for label in classes:
        classPath = os.path.join(path, label)
        files = glob.glob(os.path.join(classPath, "*"))
        for imageFile in files:
            data.append(readImageFile(imageFile))
            filename.append(os.path.split(imageFile)[1])
            labels.append(os.path.split(label)[1])

    return data, labels, filename

# Get InceptionV3 model from tensorflow
def getModel():
    print("\nFetching InceptionV3 model")
    try:
        model = tf.keras.applications.InceptionV3(
                include_top=False,
                weights="imagenet",
                input_shape=(1532, 2048, 3),
                pooling="avg")
    except Exception as e:
        print("\nError loading the model")
        print("\nError: " + str(e))
        sys.exit(-1)

    return model

# Extract features from Inception
def extractFeatures(data, labels, filenames):
    model = getModel()
    featureDict = {}
    print("\nExtracting features from Model")
    for index in range(0, len(data)):
        featureDict[filenames[index]] = []
        featureDict[filenames[index]] = model.predict(data[index])[0].tolist()
        featureDict[filenames[index]].append(labels[index])
    return featureDict

# Extract and write features to a file
def writeFeatures(data, labels, filenames):
    featureDict = extractFeatures(data, labels, filenames)
    print("\nWriting extracted features to file: features.csv")
    with open('features.csv', 'w') as f:
        dataFrame = pd.DataFrame(featureDict)
        dataFrame = dataFrame.T
        dataFrame.to_csv("features.csv")
    f.close()
    print("\nFeatue file written")

# Extract one image features
def getOneImageFeatures(path):
    I = readImageFile(path)
    model = getModel()
    features = model.predict(I)
    return features

def getName(val):
    mapping = {
               0:'Acinetobacter.baumanii', \
               1:'Actinomyces.israel', \
               2:'Bacteroides.fragilis', \
               3:'Bifidobacterium.spp', \
               4:'Candida.albicans'
              }
    return mapping[val]