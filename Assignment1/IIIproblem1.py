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

outlierDet = detOut(dfFile)

plt.figure(figsize=(12, 8))

# print(outlierDet)

dfFile.boxplot()
plt.title('Boxplot of Attributes with Outliers')
plt.xticks(rotation=45)
plt.show()
