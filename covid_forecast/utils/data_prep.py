import pandas as pd
import numpy as np

loc_type_vname = "types"


def reduce_mem(df):
    """
    Function to reduce memory footprint

    Args:
        df (pd.DataFrame): input

    Returns:
        pd.DataFrame: with memory efficient dtypes and modes
    """
    # Date time cols
    dt_col_vnames = df.select_dtypes('datetime').columns.values

    # Object dtypes
    obj_col_vnames = list(df.select_dtypes("O").columns.values)
    obj_col_vnames = [obj_col for obj_col in obj_col_vnames if obj_col not in dt_col_vnames]

    if len(obj_col_vnames) > 0:
        df[obj_col_vnames] = df[obj_col_vnames].astype("category")

    # Numerical dtypes
    num_col_vnames = df.select_dtypes(np.number).columns.values
    if len(num_col_vnames) > 0:
        for col in num_col_vnames:
            if df[col].max() < np.finfo(np.float32).max:
                try:
                    if np.array_equal(df[col], df[col].astype(int)):
                        df[col] = df[col].pipe(pd.to_numeric, downcast="integer")
                    else:
                        df[col] = df[col].pipe(pd.to_numeric, downcast="float")
                except ValueError:
                    df[col] = df[col].pipe(pd.to_numeric, downcast="float")
    return df


def bulk_reduce_mem(dict_of_dfs):
    """
    Function to reduce memory in bulk.

    Args:
        dict_of_dfs (dict): of pd.DataFrame objects

    Returns:
        dict_of_dfs (dict): of memory efficient pd.DataFrames
    """
    for df in dict_of_dfs:
        dict_of_dfs[df] = dict_of_dfs[df].pipe(reduce_mem)
    return dict_of_dfs


def find_admin_loc(target_loc, admin_txt_1='administrative_area_level_1',
                   admin_txt_2='administrative_area_level_2'):
    """
    Finds province/state from given target_loc
    target_loc = gmaps.reverse_geocode((lat, lng))[0]['address_components']

    Args:
        target_loc(json): response from gmaps.reverse_geocode
        admin_txt_1 (str): 'administrative_area_level_1'
        admin_txt_2 (str): 'administrative_area_level_2'

    Returns:
        int: admin area level 1 (preferred) or 2
    """
    admin_lvl_1, admin_lvl_2 = None, None

    for i in range(len(target_loc)):
        if admin_txt_1 in target_loc[i][loc_type_vname]:
            admin_lvl_1 = i
            return admin_lvl_1
        elif admin_txt_2 in target_loc[i][loc_type_vname]:
            admin_lvl_2 = i
            return admin_lvl_2
