from utils import *
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle

class svmClassifier:

    def saveToDisk(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        f.close()        

    def initializeModel(self):
        dataset = pd.read_csv(self.dataPath)
        X = dataset.iloc[:, 1:-1].values
        Y = dataset.iloc[:, -1].values
        Y = LabelEncoder().fit_transform(Y)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y,\
                                         test_size = 0.10, random_state = 0)
        classifier = SVC(kernel = 'linear', random_state = 0)
        print(X_train.shape)
        print(Y_train.shape)
        classifier.fit(X_train, Y_train)
        Y_pred = classifier.predict(X_test)
        cm = confusion_matrix(Y_test, Y_pred)
        print("\nConfusion Matrix for the trained model is :")
        print(cm)
        print("\nAccuracy of trained model is :")
        print(accuracy_score(Y_test, Y_pred))
        return classifier

    def __init__(self, csvFilePath):
        self.dataPath = csvFilePath
        self.model = self.initializeModel()