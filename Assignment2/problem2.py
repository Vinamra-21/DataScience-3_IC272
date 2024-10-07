import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter

irisFile = pd.read_csv('Iris.csv')
Xreduce = pd.read_csv('dimRedData.csv')

le = LabelEncoder()
y_encoded = le.fit_transform(irisFile['Species'])

Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xreduce.values, y_encoded, test_size=0.2, random_state=104, shuffle=True)

def kNNClass(Xtrain, Ytrain, Xtest, K=5):
    yPred = []
    for testpt in Xtest:
        dists = np.sqrt(np.sum((Xtrain - testpt) ** 2, axis=1))
        nn = Ytrain[np.argsort(dists)[:K]]
        majorClass = Counter(nn).most_common(1)[0][0]
        yPred.append(majorClass)
    return yPred

predict = kNNClass(Xtrain, Ytrain, Xtest, K=5)

confMat = confusion_matrix(Ytest, predict)
confMat_display = ConfusionMatrixDisplay(confusion_matrix=confMat, display_labels=le.classes_)

confMat_display.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.savefig('problem2.png')
plt.show()
