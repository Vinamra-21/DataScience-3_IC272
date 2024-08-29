import pandas as pd
import numpy as np

# Load the CSV file
dfFile = pd.read_csv("IIproblem2_corrected.csv")

# Standardization function
def stdrd(df):
    dfStd = df.copy()
    for col in dfStd.columns[2:]:
        if pd.api.types.is_numeric_dtype(dfStd[col]):
            mean = np.mean(dfStd[col])
            stdDev = np.std(dfStd[col])
            if stdDev != 0:  # Check to avoid division by zero
                dfStd[col] = (dfStd[col] - mean) / stdDev
            else:
                dfStd[col] = 0 
        else:
            print(f"Skipped non-num column: {col}")
    return dfStd

# Apply standardization
dfStd = stdrd(dfFile)

# Mean and standard deviation before and after standardization
print("Mean and Standard Deviation Before Standardization:")
for col in dfFile.columns[2:]:
    if pd.api.types.is_numeric_dtype(dfFile[col]):
        mean_b4 = np.mean(dfFile[col])
        std_b4 = np.std(dfFile[col])
        print(f"{col} - Mean: {mean_b4}, Std Dev: {std_b4}")

print("\nMean and Standard Deviation After Standardization:")
for col in dfStd.columns[2:]:
    if pd.api.types.is_numeric_dtype(dfStd[col]):
        mean_af = np.mean(dfStd[col])
        std_af = np.std(dfStd[col])
        print(f"{col} - Mean: {mean_af:.6f}, Std Dev: {std_af:.6f}")
