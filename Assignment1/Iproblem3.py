import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV
df = pd.read_csv("landslide_data_original.csv")

df_t12 = df[df['stationid'] == 't12']  # 'humidity' column 
humidData = df_t12['humidity'].dropna()
humidData = humidData.astype(int)

# bin size
bin_size = 5

minHumid = int(humidData.min())
maxHumid = int(humidData.max())

# range of humidity values
humidRange = range(minHumid, maxHumid + bin_size, bin_size)

histogram = [0] * (len(humidRange) - 1)  # Initialize the histogram with zeros

for humidity in humidData:
    bin_index = (humidity - minHumid) // bin_size
    histogram[bin_index] += 1

# histogram plot
plt.figure(figsize=(10, 6))
plt.bar(humidRange[:-1], histogram, width=5, edgecolor='black', align='edge')

plt.title('Histogram of Humidity for Station ID = t12')
plt.xlabel('Humidity')
plt.ylabel('Frequency')

plt.show()
