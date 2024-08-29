import pandas as pd
import math

#mean
def calc_mean(data):
    return sum(data) / len(data)

# Pearson correlation coefficient
def calcPearsonCor(x, y):
    mean_x = calc_mean(x)  
    mean_y = calc_mean(y)  
    
    # numerator
    num = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x)))
    
    # denominator
    denX = math.sqrt(sum((x[i] - mean_x) ** 2 for i in range(len(x))))
    denY = math.sqrt(sum((y[i] - mean_y) ** 2 for i in range(len(y))))
    
    return num / (denX * denY)

# Load the CSV
df = pd.read_csv("landslide_data_original.csv")

# correlation matrix
def corRelMatrix(data):
    attri = data.columns.tolist()[2:]  
    corMatrix = {}

    #pairwise correlation for each pair of attributes
    for first in attri:
        corMatrix[first] = {}
        for second in attri:
            clean_x = data[first].dropna().tolist()
            clean_y = data[second].dropna().tolist()
            correlation = calcPearsonCor(clean_x, clean_y)
            corMatrix[first][second] = correlation

    return corMatrix

def printMatrix(corMatrix):
    attr = list(corMatrix.keys())
    print(" " * 10 + " ".join(f"{a:>10}" for a in attr))
    for first in attr:
        row = [f"{corMatrix[first][second]:>10.3f}" for second in attr]
        print(f"{first:>10} {' '.join(row)}")

# attributes correlated with 'lightavg'
def highcorRel_lightavg(corMatrix, threshold=0.6):
    redundant_attri = []
    for attr, correlation in corMatrix.get("lightavg", {}).items():
        if abs(correlation) >= threshold and attr != "lightavg":
            redundant_attri.append(attr)
    return redundant_attri

if __name__ == "__main__":
    corMatrix = corRelMatrix(df)
    print("Correlation Matrix:")
    printMatrix(corMatrix)

    redundant_attrs = highcorRel_lightavg(corMatrix)
    if redundant_attrs:
        print("\nRedundant attributes with respect to 'lightavg':")
        for attr in redundant_attrs:
            print(attr)
    else:
        print("\nNo attributes are highly correlated with 'lightavg'.")
