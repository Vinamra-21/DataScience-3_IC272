import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file
dfOriginal = pd.read_csv("landslide_data_original.csv")
dfMissing = pd.read_csv("landslide_data_miss.csv")

# linear interpolation
def linearInterpol(data):
    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            for i in range(len(data)):
                if pd.isna(data.at[i, col]):
                    prev_value = None
                    next_value = None

                    # previous non-missing value
                    for j in range(i-1, -1, -1):
                        if not pd.isna(data.at[j, col]):
                            prev_value = data.at[j, col]
                            break

                    # next non-missing value
                    for k in range(i+1, len(data)):
                        if not pd.isna(data.at[k, col]):
                            next_value = data.at[k, col]
                            break
                    # interpolate
                    if prev_value is not None and next_value is not None:
                        interpolated_value = prev_value + (next_value - prev_value) * ((i-j)/(k-j))
                        data.at[i, col] = interpolated_value
                    elif prev_value is not None:
                        data.at[i, col] = prev_value
                    elif next_value is not None:
                        data.at[i, col] = next_value
    return data

dfFill = linearInterpol(dfMissing)

dfFill.to_csv("IIproblem2.csv", index=False)

# mean, median, and std dev
def calcStats(data):
    stats = {}
    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            col_data = data[col].dropna().tolist()
            mean = sum(col_data) / len(col_data)
            col_data_sorted = sorted(col_data)
            median = col_data_sorted[len(col_data_sorted) // 2] if len(col_data_sorted) % 2 != 0 else \
                (col_data_sorted[len(col_data_sorted) // 2 - 1] + col_data_sorted[len(col_data_sorted) // 2]) / 2
            variance = sum((x - mean) ** 2 for x in col_data) / len(col_data)
            std_dev = variance ** 0.5
            stats[col] = {"mean": mean, "median": median, "std_dev": std_dev}
    return stats

# Calculate statistics 
origStats = calcStats(dfOriginal)
fillStats = calcStats(dfFill)

# Print statistics 
for col in origStats:
    print(f"Attribute: {col}")
    print(f"Original: Mean = {origStats[col]['mean']:.3f}, Median = {origStats[col]['median']:.3f}, Std Dev = {origStats[col]['std_dev']:.3f}")
    print(f"After Interpolation: Mean = {fillStats[col]['mean']:.3f}, Median = {fillStats[col]['median']:.3f}, Std Dev = {fillStats[col]['std_dev']:.3f}")
    print()

# Root Mean Square Error (RMSE) bw original and filled values
def calcRMSE(original, filled):
    RMSEval = {}
    for col in original.columns:
        if pd.api.types.is_numeric_dtype(original[col]):
            origCol = original[col].dropna().tolist()
            fillCol = filled[col].dropna().tolist()
            rmse = np.sqrt(sum((o - f) ** 2 for o, f in zip(origCol, fillCol)) / len(origCol))
            RMSEval[col] = rmse
    return RMSEval

RMSEval = calcRMSE(dfOriginal, dfFill)

print(RMSEval)

# plot
plt.figure(figsize=(12, 6))
plt.bar(RMSEval.keys(), RMSEval.values(), color='blue')
plt.xlabel('Attributes')
plt.ylabel('RMSE')
plt.title('RMSE between Original and Filled Values for Each Attribute')
plt.xticks(rotation=45)
plt.show()
