import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

#median
def medianCalc(arr):
    sortArr = sorted(arr)
    n = len(sortArr)
    mid = n // 2
    if n % 2 == 0:
        return (sortArr[mid - 1] + sortArr[mid]) / 2
    else:
        return sortArr[mid]

#mean
def meanCalc(arr):
    return sum(arr) / len(arr)

#variance
def varCalc(arr):
    mean = sum(arr) / len(arr)
    sqdiff = [(x - mean) ** 2 for x in arr]
    variance = sum(sqdiff) / len(arr)
    return variance

#outlier
def outlieReplace(X):
    for col in range(X.shape[1]):
        colmnData = X[:, col]
        
        q1 = np.percentile(colmnData, 25)
        q3 = np.percentile(colmnData, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        for i in range(len(colmnData)):
            if colmnData[i] < lower_bound or colmnData[i] > upper_bound:
                colmnData[i] = medianCalc(colmnData)
    return X

#PCA
def PCAfn(X):
    Xmeaned = X - np.mean(X, axis=0)
    # Xmeaned = X - meanCalc(X, axis=0)

    corrMat = np.dot(Xmeaned.T, Xmeaned)

    eigenValues, eigenVectors = np.linalg.eig(corrMat)
    sortIndices = np.argsort(eigenValues)[::-1]
    sortEignVals = eigenValues[sortIndices]
    sortEignVtor = eigenVectors[:, sortIndices]
    sortEignVtor = sortEignVtor / np.linalg.norm(sortEignVtor, axis=0)
    Q = sortEignVtor[:, :1]

    Xreduced = np.dot(Xmeaned, Q)
    return Xreduced

#standardisation
def stdrd(X):
    XStd = np.copy(X)
    for col in range(X.shape[1]):
        # mean = np.mean(XStd[:, col])
        mean = meanCalc(XStd[:, col])
        # stdDev = np.std(XStd[:, col])
        stdDev = np.sqrt(varCalc(XStd[:, col]))
        if stdDev != 0: 
            XStd[:, col] = (XStd[:, col] - mean) / stdDev
        else:
            XStd[:, col] = 0 
    return XStd

#data
IrisTrain = pd.read_csv("iris_train.csv")  
XTrain = IrisTrain[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
YTrain = IrisTrain['Species'].values
IrisTest = pd.read_csv("iris_test.csv")  
XTest = IrisTest[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
YTest = IrisTest['Species'].values

XTrainRed = PCAfn(stdrd(outlieReplace(XTrain)))
XTestRed = PCAfn(stdrd(outlieReplace(XTest)))

#mean and variance
def mean_var(data, labels):
    classes = np.unique(labels)
    params = {}
    for c in classes:
        class_data = data[labels == c]
        # mean = np.mean(class_data)
        mean = meanCalc(class_data)
        # variance = np.var(class_data)
        variance = varCalc(class_data)
        params[c] = (mean, variance)
    return params

params = mean_var(XTrainRed, YTrain)

#probabilities
def pdf(x, mean, variance):
    return (1.0 / np.sqrt(2 * np.pi * variance)) * np.exp(-((x - mean) ** 2) / (2 * variance))

#classify
def classification(sample, params):
    posterior={}
    for class_label, (mean, variance) in params.items():
        posterior[class_label]=pdf(sample, mean, variance)*len(XTrainRed[YTrain==class_label])/len(XTrainRed)
    return max(posterior, key=posterior.get)

predLabels = np.array([classification(x, params) for x in XTestRed])

#confusion matrix
def confMatrix(original, predicted, classes):
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for i, original_class in enumerate(classes):
        for j, predicted_class in enumerate(classes):
            matrix[i, j] = np.sum((original == original_class) & (predicted == predicted_class))
    return matrix

#accuracy
def accuracy_score(original, predicted):
    # return np.mean(original == predicted) * 100
    return meanCalc(original == predicted) * 100

classes = np.unique(YTrain)
conf_matrix = confMatrix(YTest, predLabels, classes)
accuracy = accuracy_score(YTest, predLabels)

confMat_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classes)
confMat_display.plot(cmap=plt.cm.Blues)

plt.show()
print(f"Accuracy of (first) the model: {accuracy:.2f}%")
