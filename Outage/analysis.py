import pandas as pd
import os
from functions import analyse_df, create_geodf, clean_shp, plot_geodf

python_directory = os.path.dirname(os.path.realpath(__file__))

## run once to clean the csv files and the Gerogia map shp file
# df_dict = {
#     "2014": pd.read_csv(rf"{python_directory}\Data\eaglei_outages_2014.csv"),
#     "2015": pd.read_csv(rf"{python_directory}\Data\eaglei_outages_2015.csv"),
#     "2016": pd.read_csv(rf"{python_directory}\Data\eaglei_outages_2016.csv"),
#     "2017": pd.read_csv(rf"{python_directory}\Data\eaglei_outages_2017.csv"),
#     "2018": pd.read_csv(rf"{python_directory}\Data\eaglei_outages_2018.csv"),
#     "2019": pd.read_csv(rf"{python_directory}\Data\eaglei_outages_2019.csv"),
#     "2020": pd.read_csv(rf"{python_directory}\Data\eaglei_outages_2020.csv"),
#     "2021": pd.read_csv(rf"{python_directory}\Data\eaglei_outages_2021.csv"),
#     "2022": pd.read_csv(rf"{python_directory}\Data\eaglei_outages_2022.csv")
# }

# df_temp = analyse_df(df_dict)
# clean_shp()

## create geodataframes for each year
df_dict = {
    "2014": pd.read_pickle(rf"{python_directory}\Outputs\pickle_2014.pkl"),
    "2015": pd.read_pickle(rf"{python_directory}\Outputs\pickle_2015.pkl"),
    "2016": pd.read_pickle(rf"{python_directory}\Outputs\pickle_2016.pkl"),
    "2017": pd.read_pickle(rf"{python_directory}\Outputs\pickle_2017.pkl"),
    "2018": pd.read_pickle(rf"{python_directory}\Outputs\pickle_2018.pkl"),
    "2019": pd.read_pickle(rf"{python_directory}\Outputs\pickle_2019.pkl"),
    "2020": pd.read_pickle(rf"{python_directory}\Outputs\pickle_2020.pkl"),
    "2021": pd.read_pickle(rf"{python_directory}\Outputs\pickle_2021.pkl"),
    "2022": pd.read_pickle(rf"{python_directory}\Outputs\pickle_2022.pkl")
}
population_df = pd.read_csv(rf"{python_directory}\Data\population_georgia.csv")
geodf_dict = create_geodf(df_dict, population_df=population_df)

## plot pngs
plot_geodf(geodf_dict)
