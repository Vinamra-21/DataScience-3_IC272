import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# read CSV
IrisFile = pd.read_csv("Iris.csv")  
X = IrisFile[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
y = IrisFile['Species'].values

# outlier detection
def medianCalc(arr):
    sortArr = sorted(arr)
    n = len(sortArr)
    mid = n // 2
    if n % 2 == 0:
        return (sortArr[mid - 1] + sortArr[mid]) / 2
    else:
        return sortArr[mid]

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

# PCA
Xmeaned = X - np.mean(X, axis=0)

corrMat = np.dot(Xmeaned.T, Xmeaned)

eigenValues, eigenVectors = np.linalg.eig(corrMat)
sortIndices = np.argsort(eigenValues)[::-1]
sortEignVals = eigenValues[sortIndices]
sortEignVtor = eigenVectors[:, sortIndices]
print(sortEignVals)
sortEignVtor=sortEignVtor/np.linalg.norm(sortEignVtor, axis=0)
Q = sortEignVtor[:, :2]

Xreduced = np.dot(Xmeaned, Q)

# plot
plt.figure(figsize=(8, 6))
for species in np.unique(y):
    indices = np.where(y == species)
    plt.scatter(Xreduced[indices, 0], Xreduced[indices, 1], label=species)

pd.DataFrame(Xreduced, columns=['an1', 'an2']).to_csv('dimRedData.csv', index=False)

origin = np.zeros((2, 2))
Q1=np.dot(Q.T,Q)
plt.quiver(*origin, Q1[0, :], Q1[1, :], scale=3, color=['r', 'b'])

plt.title('PCA: Dimension Reduction and Eigen Directions')
plt.legend()
plt.grid()
plt.savefig('problem1.png')
plt.show()

Xreconst = np.dot(Xreduced, Q.T)

Xreconst += np.mean(X, axis=0)

Rmse = np.sqrt(np.mean((X - Xreconst) ** 2, axis=0))

attriName = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
for attr, rmse in zip(attriName, Rmse):
    print(f"RMSE for {attr}: {rmse}")
