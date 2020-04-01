import numpy as np
import pandas as pd
# from IPython.display import display

loc_type_vname = "types"
country_region_vname = "Country/Region"
province_state_vname = "Province/State"
s_0_vname = "s_0"


def normalise_str(series, upper=True, str_replace_rgx=r"_|-|\s{1,}|\*", str_val_rgx=" "):
    """
    Normalises strings using optional parameters.
    E.g. case using parameter upper, and regex replacements.

    Args:
        series (pd.Series): obvious
        upper (bool): True for uppercase
        str_replace_rgx (str): to look for replacement in regex
        str_val_rgx (str): replacement value, leave both fields "" for no change

    Returns:

    """
    if upper:
        series_out = series.str.upper().str.replace(str_replace_rgx, str_val_rgx).str.strip()
    elif not upper:
        series_out = series.str.lower().str.replace(str_replace_rgx, str_val_rgx).str.strip()
    else:
        series_out = series.str.replace(str_replace_rgx, str_val_rgx).str.strip()
    return series_out


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
        target_loc(list): response from gmaps.reverse_geocode
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
    return admin_lvl_1


def fillna_or_add_val(self, search_srt, fill_val, search_col=country_region_vname, fill_col=s_0_vname):
    """
    Fills value with given based on string.
    If no results are found, a new line is inserted.

    Args:
        self (pd.DataFrame): input dataframe
        search_srt (str): search string in search columns
        fill_val: any fill value
        search_col (str): column name to search, default country_region_vname
        fill_col (str): col name to fill in value, default s_0_vname

    Returns:

    """
    df_copy = self.copy()
    if len(df_copy.loc[(df_copy[search_col] == search_srt), fill_col]) > 0:
        df_copy.loc[(df_copy[search_col] == search_srt), fill_col] = fill_val
    else:
        df_copy = df_copy.append({search_col: search_srt, fill_col: fill_val}, ignore_index=True)
    return df_copy


def safe_merge_cols(df_left, df_right, col_add=s_0_vname, on=country_region_vname):
    df_left = df_left.copy()
    if col_add not in df_left.columns:
        temp_df = df_left.merge(df_right, how="left", on=on)
        df_in_shp, df_out_shp = df_left.shape, temp_df.shape
        if (df_in_shp[0] == df_out_shp[0]) & (df_in_shp[1] == df_out_shp[1]-1):
            cols= temp_df.columns.tolist()
            cols = cols[-1:] + cols[:-1]
            print("INFO: successfully merged")
            return temp_df[cols]
        else:
            print("ERROR: resulting shape not as expected")
            return df_left
    else:
        print("INFO: already joined", col_add)
        return df_left


# The SIR model differential equations.
def deriv(y, t, N, beta, gamma):
    """
    Differential equations for the SIR model

    Args:
        y (tuple): initial conditions vector (y0 = S0, I0, R0)
        t (np.ndarray): grid of time points
        N (int): total population
        beta (float): contract rate (1/days)
        gamma (float): mean recovery rate (1/days)

    Returns:
        tuple: ordinary differential equation results for S, I, R
    """
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I

    return dSdt, dIdt, dRdt
