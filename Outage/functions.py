import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

python_directory = os.path.dirname(os.path.realpath(__file__))

def analyse_df (df_dict, save_directory=rf"{python_directory}/Outputs"):
    """
    Analyse a file of type `Pandas.DataFrame` and save the outcome as a pickle file in the `directory`. Indices in the input dictionary `df_dict` are used as the name of the pickle files.

    Inputs
    ------
        - `df`: `Pandas.DataFrame` file
        - `save_directory`: Folder to save the outputs to

    Returns
    -------
        - `Pandas.DataFrame`
    """
    dict_output = {}
    for i in df_dict:
        print(f"Analysing {i}")
        df = df_dict[i]
        save_as = i
        df_gerogia = df[df["state"] == "Georgia"]
        df_output = df_gerogia.groupby("fips_code", as_index=False)["sum"].sum()
        df_output.to_pickle(rf"{save_directory}\pickle_{save_as}.pkl")
        dict_output[i] = df_output
    return (dict_output)

def clean_geodf(geo_df=gpd.read_file(rf"{python_directory}\Data\georgia_counties.shp")):
    geo_df["id"] = [int(e) for e in geo_df["id"]]
    geo_df.to_pickle(rf"{python_directory}\Outputs\georgia_counties.pkl")
    return(geo_df)

def create_geodf (df_dict, geo_df=False, save_directory=rf"{python_directory}\Outputs"):
    """
    Creates a geodf and saves as shape file

    Inputs
    ------
        - `df_dict`: a dictionary of `Pandas.DataFrame` files to join to the geopanadas DataFrame
        - `geo_df`: `gpd.DataFrame` file map of Georgia, USA
        - `save_directory`: Folder to save the outputs to

    Returns
    -------
        - `gpd.DataFrame` for Georgia, UK
    
    """
    if geo_df == False: geo_df = pd.read_pickle(rf"{python_directory}\Outputs\georgia_counties.pkl")
    dict_output = {}
    for i in df_dict:
        df = df_dict[i]
        geodf_output = geo_df.merge(right= df, left_on= "id", right_on= "fips_code")
        dict_output[i] = geodf_output
    return(dict_output)


def plot_geodf(geo_df_dict, save_directory=rf"{python_directory}\Plots"):
    """
    Plot maps to `\Plots`. Indices in the input dictionary `geo_df_dict` are used as the name of the png files.

    Inputs
    ------
        - `geo_df_dict`: dictionary with geopandas dataframes
        - `save_directory`: Folder to save the outputs to
    """
    for i in geo_df_dict:
        geo_df = geo_df_dict[i]
        title = i
        geo_df.plot("sum", legend=True)
        plt.title(i)
        plt.savefig(rf"{save_directory}\{title}.png")
        plt.close()