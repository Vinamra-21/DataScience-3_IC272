import pandas as pd

# mean
def calc_mean(data):
    sum=0
    for i in range(len(data)):
        sum+=data[i]

    return sum / len(data)

# Minimum
def calc_min(data):
    min_val = data[0]
    for i in data:
        if i < min_val:
            min_val = i
    return min_val

# Maximum
def calc_max(data):
    max_val = data[0]
    for i in data:
        if i > max_val:
            max_val = i
    return max_val

# Median
def calc_median(data):
    sort_data = sorted(data)
    n = len(sort_data)
    if (n % 2):
        return (sort_data[n // 2 - 1] + sort_data[n // 2]) / 2
    else:
        return sort_data[n // 2]
        

# Std Dev
def Calc_StdDev(data, mean):
    return (sum((x - mean) ** 2 for x in data) / len(data))**(0.5)

#Data

df = pd.read_csv("landslide_data_original.csv")
temp = df['temperature'].tolist()

#Results
mean_temp = calc_mean(temp)
min_temp = calc_min(temp)
max_temp = calc_max(temp)
median_temp = calc_median(temp)
std_temp = Calc_StdDev(temp, mean_temp)

print(f"The statistical measures of Temperature attribute are:")
print(f"Mean: {mean_temp:.2f}")
print(f"Maximum: {max_temp:.2f}")
print(f"Minimum: {min_temp:.2f}")
print(f"Median: {median_temp:.2f}")
print(f"Standard Deviation: {std_temp:.2f}")
