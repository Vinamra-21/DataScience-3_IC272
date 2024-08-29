import pandas as pd

# Load the CSV file
dfFile = pd.read_csv("IIproblem2_corrected.csv")

# Normalization function
def normFn(df, lower=5, upper=12):
    df_nrm = df.copy()
    for col in df_nrm.columns[2:]: 
        if pd.api.types.is_numeric_dtype(df_nrm[col]):
            min_val = df_nrm[col].min()
            max_val = df_nrm[col].max()
            if min_val != max_val:
                df_nrm[col] = (df_nrm[col] - min_val) / (max_val - min_val) * (upper - lower) + lower
            else:
                df_nrm[col] = lower 
        else:
            print(f"Skipping non-numeric column: {col}")
    return df_nrm

# Apply normalization
df_nrm = normFn(dfFile, lower=5, upper=12)

# Min and Max Values Before and After Normalization
print("Min and Max Values Before Normalization:")
print(dfFile.min(numeric_only=True))
print(dfFile.max(numeric_only=True))

print("\nMin and Max Values After Normalization:")
print(df_nrm.min(numeric_only=True))
print(df_nrm.max(numeric_only=True))
