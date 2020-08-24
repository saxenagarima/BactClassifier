import argparse
from utils import *
from classifier import *

def extractFeature(dataPath):
    data, labels, filenames = getDataAndLabels(dataPath)
    writeFeatures(data, labels, filenames)
    return

def evaluate(model, testPath):
    testFiles = glob.glob(os.path.join(testPath, "*.tif"))
    for testFile in testFiles:
        features = getOneImageFeatures(testFile)
        prediction = model.predict(features)
        print("Image file "+ os.path.split(testFile)[1] + "is predicted as :" + getName(prediction[0]))
    return

def runProject(parser):
    args = parser.parse_args()
    dataPath = args.data
    testPath = args.testData
    modelPath = os.path.join(os.path.split(os.path.realpath(__file__))[0], "svmClassifier.pkl")
    
    if(args.useSavedModel):
        if(os.path.exists(modelPath)):
            with open(modelPath, 'rb') as f:
                model = pickle.load(f)
            evaluate(model, testPath)
            sys.exit(0)
        
    if (not args.useDumpedFeature):
        extractFeature(dataPath)
        
    csvFilePath = os.path.join(os.path.split(os.path.realpath(__file__))[0], "features.csv")
    
    _classifier = svmClassifier(csvFilePath)
    _classifier.saveToDisk(modelPath)

    model = _classifier.model

    evaluate(model, testPath)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Arguments to be passed with this script
    parser.add_argument("-data", help="Path of data folder for train phase", required = False, \
        default=os.path.join(os.path.split(os.path.realpath(__file__))[0], "data"))
    parser.add_argument("-testData", help="Path of data folder for test phase", required = False, \
        default=os.path.join(os.path.split(os.path.realpath(__file__))[0], "test"))
    parser.add_argument("--useDumpedFeature", help="Use already extracted features", \
        action= "store_true")
    parser.add_argument("--useSavedModel", help="Use saved model", action= "store_true")

    runProject(parser)