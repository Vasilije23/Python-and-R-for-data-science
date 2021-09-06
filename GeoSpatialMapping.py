import folium
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("Police_Department_Incidents_-_Previous_Year__2016_.csv")

limit = 100
df = data.iloc[0:limit, :]

map = folium.Map(location=[37.7749, -122.44194], zum_start = 12)

for index, df in df.iterrows():
    location = [df['Y'], df['X']]
    folium.Marker(location).add_to(map)

map.save('GeoSpatialMapping.html')

