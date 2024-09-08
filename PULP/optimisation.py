import geopandas as gpd
import pandas as pd
import pulp
from pulp import *
from shapely.geometry import Point
from tqdm import tqdm
import pickle
import fiona
from dbfread import DBF

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

distances = demand_points.geometry.apply(lambda g: poi_combined.geometry.distance(g))

max_dist_values = [2000, 3000, 4000, 5000, 6000]
n_facilities_values = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]

results = []
for max_dist in tqdm(max_dist_values, desc="Max Distance"):
    coverage = distances <= max_dist
    demand_total_weight = demand_points['total_weig'].values
    demand_register = demand_points['Register'].values
    demand_sum = demand_points['sum'].values
    
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
    'coverage_outage': item['coverage_sum']
} for item in results])

general_info.to_csv("demand_coverage_3_indicators.csv", index=False)

for item in results:
    gdf = gpd.GeoDataFrame(
        item['selected_facilities'],
        geometry=gpd.points_from_xy(item['selected_facilities']['lon'], item['selected_facilities']['lat'])
    )

    filename = f"D:\\Zixin\\ACM_trial\\outputs\\facilities_{item['max_dist']}_{item['n_facilities']}.shp"
    gdf.to_file(filename)