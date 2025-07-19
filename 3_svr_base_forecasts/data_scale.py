# process_no_opt.py - Modified for true online learning with OnlineSVR
# Sequential processing of data points to maintain continuity
#%%
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.gofplots import qqplot
import matplotlib.pyplot as plt
import pandas as pd
import py_online_forecast.core_main_fix_svr3 as c
from py_online_forecast.core_main_fix_svr3 import *
import py_online_forecast.hierarchies as h
from py_online_forecast.hierarchies import *
np.random.seed(42)


# Load and prepare data
data_raw_df = pd.read_csv("veks.csv") # Renamed to avoid conflict
data = data_raw_df[["HC.f", "Ta.f", "W.f", "GR.f", "ds.tod", "hh"]].copy()
data.insert(0, "t", range(len(data)))
data['t'] = pd.to_datetime(data['t'], unit='h', origin='2024-01-01')
data.rename(columns={"Ta.f": "Taobs"}, inplace=True) # ds.hh remains as is for FourierSeries

target_var_name = "HC.f"
# For unscaling later, get min/max BEFORE scaling and one-hot encoding modifies ranges
orig_min = data[[target_var_name, 'Taobs']].min().copy()
orig_max = data[[target_var_name, 'Taobs']].max().copy()

data = pd.get_dummies(data, columns=['ds.tod'], prefix='tod', dtype=float)
# Expected new columns: tod_1, tod_2, tod_3 (or tod_1.0, etc. if original was float)

data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)


cols_scale = [target_var_name, 'Taobs', 'W.f', 'GR.f']
# Ta.k columns are not present, so they are not in cols_scale

cols_to_scale_present = [col for col in cols_scale if col in data.columns]
data_min_scaled = data[cols_to_scale_present].min() # Renamed
data_max_scaled = data[cols_to_scale_present].max() # Renamed
data_range_scaled = data_max_scaled - data_min_scaled # Renamed
data_range_scaled[data_range_scaled == 0] = 1

data[cols_to_scale_present] = (data[cols_to_scale_present] - data_min_scaled) / data_range_scaled

data.fc.convert(separator=".k")


data.to_csv("veks_svr.csv", index = False)


# save the min/max values for later unscaling
data_min_scaled.to_frame().T.to_csv("veks_svr_min.csv", index=False)
data_max_scaled.to_frame().T.to_csv("veks_svr_max.csv", index=False)

#%%

df_min = pd.read_csv("veks_svr_min.csv")
df_max = pd.read_csv("veks_svr_max.csv")
orig_min = df_min.iloc[0]    # Series with index ['HC.f','Taobs','W.f','GR.f']
orig_max = df_max.iloc[0]


orig_min = pd.read_csv("veks_svr_min.csv").T.squeeze()
orig_max = pd.read_csv("veks_svr_max.csv").T.squeeze()