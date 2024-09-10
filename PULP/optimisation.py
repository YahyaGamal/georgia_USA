import geopandas as gpd
import pandas as pd
import pulp
from pulp import *
from shapely.geometry import Point
from tqdm import tqdm
import pickle
import fiona
from dbfread import DBF
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_weighted_demand_coverage(results, demand_points):
    assert isinstance(demand_points, pd.DataFrame)
    assert all(col in demand_points.columns for col in ['Register', 'total_weig', 'normalised'])
    total_demand_points = len(demand_points)

    coverage_data = []
    
    for i, result in enumerate(results):
        coverage_count = np.zeros(total_demand_points, dtype=int)
        for covered_points in result['facility_coverage'].values():
            for point in covered_points:
                coverage_count[point] += 1

        weighted_register = np.sum(coverage_count * demand_points['Register'])
        weighted_total_weight = np.sum(coverage_count * demand_points['total_weig'])
        weighted_sum = np.sum(coverage_count * demand_points['normalised'])
        
        coverage_data.append({
            'iteration': i,
            'max_dist': result['max_dist'],
            'n_facilities': result['n_facilities'],
            'weighted_register': weighted_register,
            'weighted_total_weight': weighted_total_weight,
            'weighted_sum': weighted_sum,
            **{f'demand_point_{j}': count for j, count in enumerate(coverage_count)}
        })
    
    df_coverage = pd.DataFrame(coverage_data)
    
    return df_coverage

def plot_metric(df, metric):
    plt.figure(figsize=(12, 8))
    for max_dist in sorted(df['max_dist'].unique()):
        subset = df[df['max_dist'] == max_dist]
        plt.plot(subset['n_facilities'], subset[metric], marker='o', label=f'Max Dist: {max_dist}')
    
    plt.title(f'Changes in {metric} with Number of Facilities at Different Max Distances')
    plt.xlabel('Number of Facilities')
    plt.ylabel(metric)
    plt.legend(title="Max Distances")
    plt.grid(True)
    plt.show()

### import datasets 
counties_new = gpd.read_file('2022_Passenger_OD_Annual_Counties.shp')
poi = gpd.read_file('poi_data.shp')
table = DBF("counties_regis.dbf")
records = list(table)
registration = pd.DataFrame(records)
outage = pd.read_pickle('pickle_2022.pkl')

counties_new['id'] = counties_new['id'].astype(str)
registration['id'] = registration['id'].astype(str)

counties_with_reg = counties_new.merge(registration, on='id', how='left')
outage['fips_code'] = outage['fips_code'].astype(str)
counties_final = counties_with_reg.merge(outage.rename(columns={'fips_code': 'id'}), on='id', how='left')

counties_new = counties_final 

poi_gas = poi[poi['category'] == 'gas_station']
poi_parking = poi[poi['category'] == 'parking']

counties_new = counties_new.to_crs(epsg=2240)
poi_gas = poi_gas.to_crs(epsg=2240) # 4755 gas stations
poi_parking = poi_parking.to_crs(epsg=2240) # 191 parking lots
poi_combined = pd.concat([poi_gas, poi_parking], ignore_index=True)


demand_points = counties_new.copy()
demand_points['geometry'] = counties_new.centroid

demand_points = demand_points.reset_index(drop=True)
poi_combined = poi_combined.reset_index(drop=True)

demand_points = demand_points.to_crs(epsg=26916)
poi_combined = poi_combined.to_crs(epsg=26916)

distances = demand_points.geometry.apply(lambda g: poi_combined.geometry.distance(g))

# define max distance and number of facilities 
max_dist_values = [2000, 3000, 4000, 5000, 6000]
n_facilities_values = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]

### optimisation 
results = []
facility_coverage = {}
for max_dist in tqdm(max_dist_values, desc="Max Distance"):
    coverage = distances <= max_dist
    demand_total_weight = demand_points['total_weig'].values
    demand_register = demand_points['Register'].values
    demand_sum = demand_points['normalised'].values
    
    for n_facilities in tqdm(n_facilities_values, desc="Num Facilities", leave=False):
        prob = LpProblem("EV_Charging_Station", LpMaximize)
        
        x = LpVariable.dicts("facility", range(len(poi_combined)), cat='Binary')
        y = LpVariable.dicts("demand", range(len(demand_points)), cat='Binary')
        
        z1 = LpVariable("total_weight_coverage", lowBound=0)
        z2 = LpVariable("register_coverage", lowBound=0)
        z3 = LpVariable("sum_minimization", lowBound=0)
        
        prob += z1 + z2 - z3, "Multi_objective_function"
        
        prob += z1 == lpSum([demand_total_weight[i] * y[i] for i in range(len(demand_points))]) / sum(demand_total_weight), "Total_weight_coverage"
        prob += z2 == lpSum([demand_register[i] * y[i] for i in range(len(demand_points))]) / sum(demand_register), "Register_coverage"
        prob += z3 == lpSum([demand_sum[i] * y[i] for i in range(len(demand_points))]) / sum(demand_sum), "Sum_minimization"
        
        for i in range(len(demand_points)):
            prob += lpSum([coverage.iloc[i, j] * x[j] for j in range(len(poi_combined))]) >= y[i], f"Coverage_constraint_{i}"
        
        prob += lpSum([x[j] for j in range(len(poi_combined))]) == n_facilities, "Total_facilities"
        
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        selected_indices = [j for j in range(len(poi_combined)) if x[j].value() > 0.5]
        selected_facilities = poi_combined.iloc[selected_indices]
        
        coverage_total_weight = z1.value() * 100
        coverage_register = z2.value() * 100
        coverage_sum = z3.value() * 100

        facility_coverage = {j: [] for j in selected_indices}
        for i in range(len(demand_points)):
            for j in selected_indices:
                if coverage.iloc[i, j]:
                    facility_coverage[j].append(i)
        
        results.append({
            'max_dist': max_dist,
            'n_facilities': n_facilities,
            'coverage_total_weight': coverage_total_weight,
            'coverage_register': coverage_register,
            'coverage_sum': coverage_sum,
            'selected_facilities': selected_facilities,
            'facility_coverage': facility_coverage
        })
        
        print(f"Max Distance: {max_dist}, Num Facilities: {n_facilities}")
        print(f"Coverage trip: {coverage_total_weight:.2f}%")
        print(f"Coverage register: {coverage_register:.2f}%")
        print(f"Coverage outage: {coverage_sum:.2f}%")
        print("--------------------")

# calculate statistisc 
df_coverage = analyze_weighted_demand_coverage(results, demand_points)
demand_point_columns = [col for col in df_coverage.columns if col.startswith('demand_point_')]
df_coverage['total_demand_points_covered'] = df_coverage[demand_point_columns].sum(axis=1)

df_coverage_new = df_coverage.copy()
df_normalized = df_coverage_new.rename(columns={
    'weighted_register': 'register',
    'weighted_total_weight': 'travel',
    'weighted_sum': 'outage'
})
for column in ['register', 'travel', 'outage']:
    min_col = df_normalized[column].min()
    max_col = df_normalized[column].max()
    df_normalized[column] = (df_normalized[column] - min_col) / (max_col - min_col)

# Plotting each metric
plot_metric(df_normalized, 'register')
plot_metric(df_normalized, 'travel')
plot_metric(df_normalized, 'outage')
plot_metric(df_normalized, 'total_demand_points_covered')

# save the facilities 
for item in results:
    gdf = gpd.GeoDataFrame(
        item['selected_facilities'],
        geometry=gpd.points_from_xy(item['selected_facilities']['lon'], item['selected_facilities']['lat'])
    )

    filename = f"D:\\Zixin\\ACM_trial\\outputs\\facilities_{item['max_dist']}_{item['n_facilities']}.shp"
    gdf.to_file(filename)