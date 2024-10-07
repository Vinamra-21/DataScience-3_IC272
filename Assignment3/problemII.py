import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

#mean
def meanCalc(arr):
    return sum(arr) / len(arr)

#covariance
def covCalc(X):
    n = X.shape[0]
    mn = np.mean(X, axis=0)
    XCentered = X - mn
    return (1 / (n - 1)) * np.dot(XCentered.T, XCentered)

# load
IrisTrain = pd.read_csv("iris_train.csv")
IrisTest = pd.read_csv("iris_test.csv")

XTrain = IrisTrain[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
YTrain = IrisTrain['Species'].values
XTest = IrisTest[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
YTest = IrisTest['Species'].values

#classes
classes = np.unique(YTrain)

#params
params = {}
def paramCalc(XTrain,classes):
    for c in classes:
        classData = XTrain[YTrain == c]
        mn = np.mean(classData, axis=0)
        covMat = covCalc(classData) 
        prior = len(classData) / len(XTrain)
        params[c] = {
            "mean": mn,
            "covariance": covMat,
            "prior": prior
        }

paramCalc(XTrain,classes)

#bays multivariate classification
predictions = []

def BaysMultivarClassifier(XTest):
    for x in XTest:
        posterior = {}
        for c in classes:
            likelihood = multivariate_normal.pdf(x, mean=params[c]["mean"], cov=params[c]["covariance"])
            posterior[c] = likelihood * params[c]["prior"]
        
        predictions.append(max(posterior, key=posterior.get))

BaysMultivarClassifier(XTest)

#confusion matrix 
confusion_mat = confusion_matrix(YTest, predictions, labels=classes)
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=classes)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

#accuracy
def accuracy_score(original, predicted):
    # return np.mean(original == predicted) * 100
    return meanCalc(original == predicted) * 100

accuracy = accuracy_score(YTest, predictions) 
print(f"Accuracy of (second) the model: {accuracy:.2f}%")

