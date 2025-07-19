# process.py 

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
import py_online_forecast.core_main as c
from py_online_forecast.core_main import *
import py_online_forecast.hierarchies as h
from py_online_forecast.hierarchies import *

# Define plot function
def plot_predictions(predictions, var_est = None, observations = None, alpha = 0.05, t_vec = None, figsize = (80, 120), num_ticks = 5):
        
    n_series = len(predictions.columns)
    if var_est is None:
        lagged_predictions = lag_and_extend(predictions)
    else:
        lagged_predictions, lagged_var = lag_and_extend(predictions, var_est)
        ci = get_normal_confidence_interval(lagged_predictions, lagged_var, alpha)
    fig, axes = plt.subplots(n_series, 1, figsize=figsize, sharex=True)  # Create a grid of vertically stacked plots
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    for i, col in enumerate(lagged_predictions):

        # Plot observations if available
        if observations is not None:
            if isinstance(observations, pd.Series):
                axes[i].plot(observations, label=observations.name, linewidth=2)
            else:
                axes[i].plot(observations.fc[col[0]], label=col[0], linewidth=2)

        # Plot predictions
        axes[i].plot(lagged_predictions[col], label=col[0] + ": " + str(col[1]), linewidth=2) 

        # Plot intervals
        if not var_est is None:
            lo = pd.to_numeric(ci[col[0]][col[1]]["lo"], errors='coerce')
            hi = pd.to_numeric(ci[col[0]][col[1]]["hi"], errors='coerce')
            axes[i].fill_between(lagged_predictions[col].index, lo, hi, color='gray', alpha=0.3)  # Add confidence interval

        # Add labels and grid
        axes[i].grid(True)
        axes[i].legend()

    # Set xticks
    if not t_vec is None:
        num_ticks = min(len(t_vec), num_ticks)
        n_spacing  = len(t_vec) // (num_ticks- 1)  
        labels, indices = zip(*[(t_vec[i], i) for i in t_vec.index[::n_spacing]])
        axes[-1].set_xticks(indices)
        axes[-1].set_xticklabels(labels)


    return fig, axes

def rrmse_series(pred_series: pd.Series, obs_series: pd.Series) -> float:
    residuals = obs_series - pred_series
    rmse_val = np.sqrt(np.mean(residuals**2))
    return rmse_val / np.mean(obs_series)

#%%
# Read csv file
data = pd.read_csv("veks.csv")
data = data[data.index >= 5676]  # Filter out the first 5676 rows

# Keep columns HC.f, Ta.f
data = data[["HC.f", "Ta.f", "W.f", "GR.f", "hh"]]
# Add column t 
data.insert(0, "t", list(range(len(data))))
# Convert column t to datetime
data['t'] = pd.to_datetime(data['t'], unit='h', origin='2024-01-01')
# Rename column Ta.f to Ta
data.rename(columns={"Ta.f": "Taobs"}, inplace=True)
# Simulate temperature forecasts
forecast_horizons = list(range(1, 37))
for horizon in forecast_horizons:
    forecast_column = f'Ta.k{horizon}'
    data[forecast_column] = data['Taobs'].shift(-horizon) + np.random.normal(0, 5, len(data))
# Smooth the forecasts using a rolling mean
for horizon in forecast_horizons:
    forecast_column = f'Ta.k{horizon}'
    data[forecast_column] = data[forecast_column].rolling(window=10, min_periods=1).mean()
# Drop the rows with NaN values due to shifting
data.dropna(inplace=True)
# Convert to forecast matrix format
data.fc.convert(separator=".k")




#%% 1. Prepare and fit model for bottom level (1-hour data)
# Use the data from analysis_rls_a.py
hourly_data = data.fc.subset(kseq = ("NA", 1, ), end_index=2300)


# Initialize RLS model for 1-hour data
model1 = ForecastModel(RLS, estimate_variance=False, kseq=(1,))
model1.add_outputs("HC.f")
model1.add_inputs(
    ar_0=AR("HC.f", order=0),         # HC.f_t
    ar_1=AR("HC.f", order=1),         # HC.f_{t−1}
    ar_23=AR("HC.f", order=23),       # HC.f_{t−23}
    ar_36=AR("HC.f", order=36),       # HC.f_{t−36}
    ar_47=AR("HC.f", order=47),       # HC.f_{t−47}
    lp_Taobs=LowPass("Taobs", ta=0.9),
    W_f_0=AR("W.f", order=0),
    time_of_day_har=FourierSeries("hh", 5),
    mu=One()
)
model1.update_params(rls_lambda=0.97)

# Set bounds for optimization
model1.set_regprm(target="predictor", rls_lambda=(0.97, 0.999))
model1.set_regprm(target="lp_Taobs", ta=(0.1, 0.9))



# Optimize with bounds-respecting method
result1, optim_result1, temp_values1 = model1.optim(
    hourly_data,
    rmse,
    set_params=True,
    store_intermediate=True,
    method="Nelder-Mead", #"L-BFGS-B", 
    burn_in=24
)

# Fit and proceed
res1 = model1.fit(hourly_data)

#%% If model saved
#model1 = load_model("models/hierarchy/model_hourly")
#res1, var_est1 = model1.fit(hourly_data)

#%%
# Simply update this line to filter out burn-in period
burn_in = 300  # Same as used in optimization
plot_predictions(
    res1.iloc[burn_in:], 
    #var_est=var_est1.iloc[burn_in:] if var_est1 is not None else None, 
    observations=hourly_data.iloc[burn_in:],
    figsize=(20, 12)  # Using a more reasonable figsize here
)


