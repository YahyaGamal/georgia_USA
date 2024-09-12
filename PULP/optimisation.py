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
combinations = [
    (5000, 90),
    (2500, 1000),
    (1000, 2000)]


### optimisation 
results = []
facility_coverage = {}

for max_dist, n_facilities in tqdm(combinations, desc="Optimizing Combinations"):
    coverage = distances <= max_dist
    demand_total_weight = demand_points['total_weig'].values
    demand_register = demand_points['Register'].values
    demand_sum = demand_points['normalised'].values
    
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
        'selected_facilities': selected_facilities
    })
    
    print(f"Max Distance: {max_dist}, Num Facilities: {n_facilities}")
    print(f"Coverage trip: {coverage_total_weight:.2f}%")
    print(f"Coverage register: {coverage_register:.2f}%")
    print(f"Coverage outage: {coverage_sum:.2f}%")
    print("--------------------")

general_info = pd.DataFrame([{
    'max_dist': item['max_dist'],
    'n_facilities': item['n_facilities'],
    'coverage_trip': item['coverage_total_weight'],
    'coverage_register': item['coverage_register'],
    'coverage_outage': item['coverage_sum'],
    'selected_facilities': item['selected_facilities']
} for item in results])

general_info.to_csv("demand_coverage_3_indicators_0912.csv", index=False)

def process_facilities(general_info):

    phase1 = general_info.iloc[0]['selected_facilities'].copy()
    phase1['phase'] = 1

    phase2 = general_info.iloc[1]['selected_facilities'].copy()
    phase2 = phase2[~phase2.geometry.isin(phase1.geometry)]
    phase2['phase'] = 2

    phase3 = general_info.iloc[2]['selected_facilities'].copy()
    phase3 = phase3[~phase3.geometry.isin(phase1.geometry) & 
                    ~phase3.geometry.isin(phase2.geometry)]
    phase3['phase'] = 3

    all_facilities = pd.concat([phase1, phase2, phase3])

    return all_facilities

gdf_facilities = process_facilities(general_info)
gdf_facilities.to_file('station_deploy.shp')