import geopandas as gpd
import pandas as pd
import pulp
from pulp import *
from shapely.geometry import Point
from tqdm import tqdm

counties_new = gpd.read_file('2022_Passenger_OD_Annual_Counties.shp')
poi = gpd.read_file('poi_data.shp')

poi_gas = poi[poi['category'] == 'gas_station']
poi_parking = poi[poi['category'] == 'parking']

counties_new = counties_new.to_crs(epsg=2240)
poi_gas = poi_gas.to_crs(epsg=2240) # 4755 gas stations
poi_parking = poi_parking.to_crs(epsg=2240) # 191 parking lots

demand_points = counties_new.copy()
demand_points['geometry'] = counties_new.centroid
demand_points['demand'] = counties_new['total_weig']

demand_points = demand_points.reset_index(drop=True)
poi_gas = poi_gas.reset_index(drop=True)

distances = demand_points.geometry.apply(lambda g: poi_gas.geometry.distance(g))
max_dist = 5000 

coverage = distances <= max_dist
demand = demand_points['demand'].values

n_facilities = 1800



prob = LpProblem("EV_Charging_Station", LpMaximize)
x = LpVariable.dicts("facility", range(len(poi_gas)), cat='Binary')
y = LpVariable.dicts("demand", range(len(demand_points)), cat='Binary')

prob += lpSum([demand[i] * y[i] for i in range(len(demand_points))]), "Sum_of_covered_demand"
for i in range(len(demand_points)):
    prob += lpSum([coverage.iloc[i, j] * x[j] for j in range(len(poi_gas))]) >= y[i], f"Coverage_constraint_{i}"
prob += lpSum([x[j] for j in range(len(poi_gas))]) == n_facilities, "Total_facilities"

prob.solve()

selected_indices = [j for j in range(len(poi_gas)) if x[j].value() > 0.5]
selected_facilities = poi_gas.iloc[selected_indices]