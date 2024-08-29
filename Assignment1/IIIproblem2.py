import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file
dfFile = pd.read_csv("IIproblem2.csv")

# Outlier Detection Function
def detOut(data):
    outliers = {}
    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            col_data = data[col].dropna().tolist()
            Q1 = np.percentile(col_data, 25)
            Q3 = np.percentile(col_data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers[col] = [(i, val) for i, val in enumerate(col_data) if val < lower_bound or val > upper_bound]
    return outliers

# replace outliers with median
def repOutMedian(data, outliers):
    for col, outlier_indices in outliers.items():
        if pd.api.types.is_numeric_dtype(data[col]):
            col_data = data[col].dropna().tolist()
            sorted_data = sorted(col_data)
            n = len(sorted_data)
            median = sorted_data[n//2] if n % 2 != 0 else (sorted_data[(n//2)-1] + sorted_data[n//2]) / 2
            for idx, _ in outlier_indices:
                data.at[idx, col] = median
    return data

outDetected = detOut(dfFile)

dfNoOutliers = repOutMedian(dfFile.copy(), outDetected)
dfNoOutliers.to_csv("IIproblem2_corrected.csv", index=False)

# Boxplot of Attributes after Replacing Outliers
plt.figure(figsize=(12, 8))
dfNoOutliers.boxplot()
plt.title('Boxplot of Attributes after Replacing Outliers')
plt.xticks(rotation=45)
plt.show()
