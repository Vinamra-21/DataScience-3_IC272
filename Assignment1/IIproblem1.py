import pandas as pd

# load CSV
df = pd.read_csv("landslide_data_miss.csv")
df = df.dropna(subset=['stationid'])

# drop rows with missing values more than 1/3 of the columns
limit = len(df.columns) // 3 
df = df.dropna(thresh=len(df.columns) - limit, axis=0)

df.to_csv("IIproblem1.csv", index=False)
