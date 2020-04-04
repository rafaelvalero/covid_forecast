#!/usr/bin/env python
# coding: utf-8

# In[317]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import gc
import os
import sys
# import time
from pprint import pprint

# import googlemaps
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from scipy.integrate import odeint

main_dir = os.path.abspath(os.pardir)
sys.path.insert(0, main_dir)
from analysis.download_data import jh_git_url, population_data_url, export_foldname, all_files
from covid_forecast.utils import data_prep as dp
from covid_forecast.utils.data_prep import country_region_vname, s_0_vname, province_state_vname
from analysis.estimate_sir_params import estimate_sir_params, estimate_sir_params_v2, s_0_vname, population_df_vnames

pd.options.display.max_rows = 999
pd.options.display.max_columns = 100
pd.options.display.precision = 8
pd.options.display.float_format = '{:,.3f}'.format


# In[318]:


export_dir = os.path.join(main_dir, "data", "raw"+os.sep)
countries_fname = "countries.csv"
print(main_dir+"\n"+export_dir)


# # Open files as dataframes
# Credit to dgrechka for locations_population. Data [source](https://www.kaggle.com/dgrechka/covid19-global-forecasting-locations-population/metadata)

# In[319]:


population_data = pd.read_excel(population_data_url)
friendly_fnames = [file.replace("time_series_covid19_", "").replace(".csv", "") for file in all_files]
filename_dct = dict(zip(friendly_fnames, all_files))
pprint(filename_dct)


# In[320]:


df_dct = dict()
for file in filename_dct.keys():
    df_dct[file] = pd.read_csv(jh_git_url+filename_dct[file])  # reads csv from github repo
    print(file, df_dct[file].shape)
    display(df_dct[file].head(), df_dct[file].describe(include="all"), df_dct[file].dtypes)


# In[321]:


filename_dct


# In[322]:


df_dct = dp.bulk_reduce_mem(df_dct)
# population_df = population_df.pipe(dp.reduce_mem)
gc.collect()


# # Global measures dataframes

# In[323]:


country_lvl_df_dct = dict()

for df_name in df_dct.keys():
    country_lvl_df_dct[df_name] = df_dct[df_name].drop(columns=[province_state_vname, "Lat", "Long"])
    country_lvl_df_dct[df_name] = country_lvl_df_dct[df_name].groupby(country_region_vname).sum().astype(int).reset_index(drop=False)
    country_lvl_df_dct[df_name][country_region_vname] = country_lvl_df_dct[df_name][country_region_vname].pipe(dp.normalise_str)


# In[324]:


country_lvl_df_dct["confirmed_global"].head(2)


# In[325]:


population_df = population_data.copy().rename(columns={"countriesAndTerritories": country_region_vname, "popData2018": s_0_vname})
population_df = population_df.sort_values("dateRep", ascending=False).reset_index(drop=True).drop_duplicates(country_region_vname)
population_df[country_region_vname] = population_df[country_region_vname].pipe(dp.normalise_str)
population_df[country_region_vname] = population_df[country_region_vname].replace(
    ["UNITED STATES OF AMERICA", "BRUNEI DARUSSALAM", "UNITED REPUBLIC OF TANZANIA", "CAPE VERDE", "DEMOCRATIC REPUBLIC OF THE CONGO",
     "CONGO", "COTE DIVOIRE", "CZECH REPUBLIC", "SOUTH KOREA"],
    ["US", "BRUNEI", "TANZANIA", "CABO VERDE", "CONGO (KINSHASA)",
     "CONGO (BRAZZAVILLE)", "COTE D'IVOIRE", "CZECHIA", "KOREA, SOUTH"], regex=False)

population_df = population_df.pipe(dp.fillna_or_add_val, "ERITREA", 5750433)
population_df = population_df.pipe(dp.fillna_or_add_val, "BURMA", 53708395)
population_df = population_df.pipe(dp.fillna_or_add_val, "DIAMOND PRINCESS", 3700)
population_df = population_df.pipe(dp.fillna_or_add_val, "MS ZAANDAM", 1243)
population_df = population_df.pipe(dp.fillna_or_add_val, "WEST BANK AND GAZA", 4685000)
population_df = population_df.pipe(dp.fillna_or_add_val, "ANGUILLA", 14731)
population_df = population_df.pipe(dp.fillna_or_add_val, "FALKLAND ISLANDS (MALVINAS)", 3398)

# population_df.head()


# In[326]:


population_df[population_df[s_0_vname].isna()]


# In[327]:


country_lvl_df_dct["confirmed_global"][country_region_vname][country_lvl_df_dct["confirmed_global"][country_region_vname].isna()]


# In[328]:


# Fill in missing province/state
# # Start timer
# start_time = pd.to_datetime("now")
# time_fmt = "%d/%m/%Y %H:%M:%S"
# print("INFO: start time", start_time.strftime(time_fmt))

# # Filling missing province/state where possible
# gmaps = googlemaps.Client(key=os.getenv("gmaps_api_key"))

# fillna_keys = ["Province/State", "Country/Region"]
# loc_admin_long_vname = "long_name"

# # Start with the first df with missing province/state values
# temp_df = confirmed_global[match_keys][confirmed_global[prov_state_vname].isna()].copy()

# for i in temp_df.index:
#     row = temp_df.loc[i]

#     # Look up an address with reverse geocoding
#     try:
#         target_loc = gmaps.reverse_geocode((row["Lat"], row["Long"]))[0]['address_components']
#     except IndexError:
#         continue
#     admin_ind = dp.find_admin_loc(target_loc)
#     if admin_ind is not None:
#         temp_df.loc[i, prov_state_vname] = target_loc[dp.find_admin_loc(target_loc)][loc_admin_long_vname]
    
#     # Fill missing with country/region
#     temp_df[prov_state_vname] = temp_df[prov_state_vname].fillna(temp_df[country_region_vname])
    
#     # Gmaps anti-throttling tactic
#     time.sleep(1)  # wait 1 second before the next iteration

# # Fillna missing values
# for df in df_dct.keys():
#     df_dct[df][fillna_keys] = confirmed_global[fillna_keys].fillna(temp_df[fillna_keys])

# # Finish time
# finish_time = pd.to_datetime("now")
# elapsed_time_min = round((finish_time-start_time).total_seconds()/60, 2)
# print("INFO: finished. This took", elapsed_time_min, "minutes.")


# # SIR
# [Source code](https://scipython.com/book/chapter-8-scipy/additional-examples/the-sir-epidemic-model/)

# In[329]:


# # Total population, N.
# N = 1000

# # Initial number of infected and recovered individuals, I0 and R0.
# I0, R0 = 1, 0

# # Everyone else, S0, is susceptible to infection initially.
# S0 = N - I0 - R0

# # Contact rate (aka beta), and mean recovery rate aka (gamma) both (in 1/days).
# beta, gamma = 0.2, 1./10

# # A grid of time points (in days)
# days = 160
# t = np.linspace(0, days, days)

# # Initial conditions vector
# y0 = S0, I0, R0
# # Integrate the SIR equations over the time grid, t.
# ret = odeint(dp.deriv, y0, t, args=(N, beta, gamma))
# S, I, R = ret.T

# # Plot the data on three separate curves for S(t), I(t) and R(t)
# fig = plt.figure(facecolor='w')
# ax = fig.add_subplot(111, axisbelow=True)
# ax.plot(t, S/1000, 'b', alpha=0.5, lw=2, label='Susceptible')
# ax.plot(t, I/1000, 'r', alpha=0.5, lw=2, label='Infected')
# ax.plot(t, R/1000, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
# ax.set_facecolor('#dddddd')
# ax.set_xlabel('Time /days')
# ax.set_ylabel('Number (1000s)')
# ax.set_ylim(0,1.2)
# ax.yaxis.set_tick_params(length=0)
# ax.xaxis.set_tick_params(length=0)
# ax.grid(b=True, which='major', c='w', lw=2, ls='-')
# legend = ax.legend()
# legend.get_frame().set_alpha(0.5)
# for spine in ('top', 'right', 'bottom', 'left'):
#     ax.spines[spine].set_visible(False)
# plt.show()


# In[330]:


for df_name in country_lvl_df_dct.keys():
    country_lvl_df_dct[df_name] = country_lvl_df_dct[df_name].pipe(dp.safe_merge_cols, population_df[[country_region_vname, s_0_vname]])
    temp_var = country_lvl_df_dct[df_name][country_lvl_df_dct[df_name][s_0_vname].isna()]
    print("WARNING: the following countries have no population values and are excluded", ", ".join(list(temp_var[country_region_vname])))
    country_lvl_df_dct[df_name] = country_lvl_df_dct[df_name].drop(index=temp_var.index)
    country_lvl_df_dct[df_name] = country_lvl_df_dct[df_name].sort_values(s_0_vname)
    exec(df_name+" = country_lvl_df_dct['"+df_name+"']")  # creates a reference for friendly_names

country_lvl_df_dct = dp.bulk_reduce_mem(country_lvl_df_dct)
gc.collect()


# ## Estimating SIR beta and gamma parameters
# [Credit Lewuathe - source](https://github.com/Lewuathe/COVID19-SIR)

# In[354]:


countries = confirmed_global[confirmed_global[country_region_vname].str.contains(
    r"cyprus|italy|spain|united kingdom|^us$|india|china", case=False)].drop_duplicates()
display(countries[population_df_vnames].head())


# In[ ]:





# In[344]:


countries = confirmed_global[confirmed_global[country_region_vname].str.contains(
    r"spain", case=False)].drop_duplicates()
display(countries[population_df_vnames].head())


# In[345]:


country_lvl_df_dct.keys()


# In[346]:


country_region_vname


# In[347]:


get_ipython().run_cell_magic('time', '', 'results_df = estimate_sir_params(countries=countries[country_region_vname], df_dct=country_lvl_df_dct, predict_range=180, plot_option=True)')


# In[355]:


get_ipython().run_cell_magic('time', '', 'results_df, report_results = estimate_sir_params_v2(countries=countries[country_region_vname], df_dct=country_lvl_df_dct, predict_range=180, plot_option=True)')


# In[349]:


results_df


# In[350]:


results_df


# ## Export results

# In[351]:


results_df.to_csv(export_foldname+"sir_params_per_country.csv")


# In[352]:


report_results.to_csv(export_foldname+"full_report.csv")


# In[353]:


# TODO: increase estimation speed or use other SIR methods: https://www.kaggle.com/lisphilar/covid-19-data-with-sir-model
# TODO: (future) return date or date_range with beta and gamma

# Resources:
# TODO: investigate ICL report and code https://www.imperial.ac.uk/news/196234/covid19-imperial-researchers-model-likely-impact/
# https://www.imperial.ac.uk/media/imperial-college/medicine/sph/ide/gida-fellowships/Imperial-College-COVID19-Europe-estimates-and-NPI-impact-30-03-2020.pdf
# https://github.com/ImperialCollegeLondon/covid19model/releases/tag/v1.0


# In[ ]:




