import pandas as pd
import geopandas as gpd
import osrm 
import requests
from shapely.geometry import Point, LineString
from tqdm import tqdm

def get_route(origin, destination):
    url = f"http://router.project-osrm.org/route/v1/driving/{origin[0]},{origin[1]};{destination[0]},{destination[1]}?overview=full&geometries=geojson"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"{response.status_code}")
        return None
    data = response.json()
    if "routes" in data and len(data["routes"]) > 0:
        return LineString(data["routes"][0]["geometry"]["coordinates"])
    else:
        print("No route found")
        return None

def calculate_intersections(county_geometry, routes):
    intersecting_routes = routes[routes.intersects(county_geometry)]
    return len(intersecting_routes), intersecting_routes['weight'].sum()


passenger_od_2022 = pd.read_csv('2022_Passenger_OD_Annual_Data.csv')
counties = gpd.read_file('georgia_counties.shp')

counties['centroid'] = counties['geometry'].centroid
centroid_dict = dict(zip(counties['id'], counties['centroid']))

osrm_client = osrm.Client(host='http://router.project-osrm.org')

routes = []
for _, row in tqdm(passenger_od_2022.iterrows(), total=passenger_od_2022.shape[0], desc="Processing routes"):
    try:
        origin = centroid_dict[row['origin_zone_id']]
        destination = centroid_dict[row['destination_zone_id']]
        route = get_route((origin.x, origin.y), (destination.x, destination.y))
        if route:
            routes.append({
                'geometry': route,
                'weight': row['annual_total_trips']
            })
    except KeyError as e:
        print(f"KeyError: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

routes_gdf = gpd.GeoDataFrame(routes)

routes_gdf = routes_gdf.set_crs(epsg=4326)
routes_gdf = routes_gdf.to_crs(counties.crs)

counties['num_routes'] = 0
counties['total_weight'] = 0

for idx, county in tqdm(counties.iterrows(), total=len(counties)):
    num_routes, total_weight = calculate_intersections(county.geometry, routes_gdf)
    counties.at[idx, 'num_routes'] = num_routes
    counties.at[idx, 'total_weight'] = total_weight

counties_new = counties.drop(columns=['centroid'])
counties_new.to_file('2022_Passenger_OD_Annual_Counties.shp')