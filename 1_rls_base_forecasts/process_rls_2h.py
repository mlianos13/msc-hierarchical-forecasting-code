# process_rls_2h.py

#%% Imports and setup
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

# Read CSV file
data = pd.read_csv("veks.csv")


# Keep only relevant columns
data = data[["HC.f", "Ta.f", "W.f", "GR.f", "ds.tod", "hh"]]
# Add a time index column 't'
data.insert(0, "t", list(range(len(data))))
# Convert 't' to a datetime index (hourly from 2024-01-01)
data['t'] = pd.to_datetime(data['t'], unit='h', origin='2024-01-01')
# Rename "Ta.f" to "Taobs"
data.rename(columns={"Ta.f": "Taobs"}, inplace=True)
# Convert to "forecast‐matrix" format (MultiIndex: (variable, horizon))
data.fc.convert(separator=".k")

#
# 1. Prepare data
#

# First get the hourly data
hourly_data = data.fc.subset(kseq=("NA", 1), end_index=5876)
# Drop the first 312 rows → start at index 312
hourly_data = hourly_data[hourly_data.index >= 3576]

# Create the final DataFrame for the model
two_hourly_data = c.new_fc()

# ##-- NEW: STEP 1 - FEATURE ENGINEERING --##
# Create a new feature: the sum of the PAST 2 hours.
# The value at index 't' is the sum of HC.f from t-1 to t.
# We use min_periods=2 to ensure we only have sums over full windows.
two_hourly_data[("HC.f_past2h", "NA")] = hourly_data[("HC.f", "NA")].rolling(window=2, min_periods=2).sum()

# Also create a smoothed temperature feature
two_hourly_data[("Taobs_smooth", "NA")] = hourly_data[("Taobs", "NA")].rolling(window=2, min_periods=2).mean()


# Create the TARGET variable: the sum of the NEXT 2 hours
# This is a rolling sum of the NEXT 2 hours (t+1 to t+2)
sum_target_2h = hourly_data[("HC.f", "NA")].shift(-2).rolling(window=2).sum()
two_hourly_data[("HC.f_future2h", "NA")] = sum_target_2h


# Drop all rows with NaN values which were created by the rolling windows
two_hourly_data.dropna(inplace=True)


#
# 2. Model setup with the new target output and new inputs
#
model2h = ForecastModel(RLS, estimate_variance=False, kseq=tuple(range(1, 25)))

# The output is the forward-looking sum
model2h.add_outputs("HC.f_future2h")

# ##-- NEW: STEP 2 - MODEL DEFINITION --##
# Use the new backward-rolling sum feature as input
model2h.add_inputs(
    # AR1: The value of HC.f_past2h from t-1. This is the sum over |t-2..t-1|
    ar1_past_sum=AR("HC.f_past2h", order=1),

    # AR2: The value of HC.f_past2h from t-2. This is the sum over |t-3..t-2|
    ar2_past_sum=AR("HC.f_past2h", order=2),
    
    # Smoothed Temperature
    ta_smooth=AR("Taobs_smooth", order=1),

    # Intercept
    mu=One()
)
# Set forgetting factor λ=0.995
model2h.update_params(rls_lambda=0.995)

# Fit the model on the new data
res2h = model2h.fit(two_hourly_data)


#
# 3. Save raw predictions to CSV
#
plot_dir = "../7_results/Case_1/1_rls/res2"
os.makedirs(plot_dir, exist_ok=True)

if isinstance(res2h, ForecastTuple) and len(res2h) > 0 and isinstance(res2h[0], pd.DataFrame):
    predictions_df = res2h[0]
elif isinstance(res2h, pd.DataFrame):
    predictions_df = res2h
else:
    raise RuntimeError("Unexpected format for res2h from model2h.fit")

out_path_multiindex = os.path.join(plot_dir, "rls_res2h.csv")
predictions_df.to_csv(out_path_multiindex)
print(f"\nPredictions with MultiIndex saved to {out_path_multiindex}")

observations_df = two_hourly_data.fc[["HC.f_future2h"]].loc[predictions_df.index]
obs_path = os.path.join(plot_dir, "observations_HC_2h.csv")
observations_df.to_csv(obs_path)
print(f"Observations saved to {obs_path}")

# ##-- ADDED: Save combined observations and predictions --##
combined_df = pd.concat([observations_df, predictions_df], axis=1)
combined_path = os.path.join(plot_dir, "rls_res2h_with_obs.csv")
combined_df.to_csv(combined_path)
print(f"Combined predictions and observations saved to {combined_path}")


#
# 4. OLS Analysis and Plotting
#
horizon = 1
X_full, Y_full = model2h.transform_data(two_hourly_data, use_recursive_pars=False)
kseq_for_h1 = ("NA", horizon)
X_h1 = X_full.fc.subset(kseq=kseq_for_h1).copy()
X_h1_lagged = X_h1.shift(horizon).dropna()

y_target_col = ("HC.f_future2h", "NA")
y_target = two_hourly_data[y_target_col].copy()

common_idx = X_h1_lagged.index.intersection(y_target.index)
X_ols_aligned = X_h1_lagged.loc[common_idx]
y_ols_aligned = y_target.loc[common_idx]

if isinstance(X_ols_aligned.columns, pd.MultiIndex):
    X_ols_aligned.columns = [f"{var}__{h}" for (var, h) in X_ols_aligned.columns.to_list()]

if not any('mu' in col.lower() for col in X_ols_aligned.columns):
    X_ols_final = sm.add_constant(X_ols_aligned, has_constant='add')
else:
    X_ols_final = X_ols_aligned.copy()

ols_model = sm.OLS(y_ols_aligned, X_ols_final).fit()

# ##-- ADDED: Save AIC/BIC results to file --##
print("\n=== OLS Results for 2h-Aggregated Model (Rolling Inputs) ===")
print(f"AIC: {ols_model.aic:.4f}")
print(f"BIC: {ols_model.bic:.4f}")
print("\nP-values of regressors:")
print(ols_model.pvalues)

results_file_path = os.path.join(plot_dir, "AIC_BIC_pv_2h.txt")
with open(results_file_path, "w") as f:
    f.write("=== OLS Results for 2h-Aggregated Model (Rolling Inputs) ===\n")
    f.write(f"AIC: {ols_model.aic:.4f}\n")
    f.write(f"BIC: {ols_model.bic:.4f}\n")
    f.write("\nP-values of regressors:\n")
    f.write(ols_model.summary().tables[1].as_text())
print(f"\nResults saved to {results_file_path}")

#
# 5. Produce and save prediction & diagnostic plots ONLY for horizon 1
#
pred_col = ("HC.f_future2h", 1)
predicted_h1 = predictions_df[[pred_col]].copy()
if isinstance(predicted_h1.index, pd.DatetimeIndex):
    predicted_h1.index = pd.DatetimeIndex([idx + pd.Timedelta(hours=horizon) for idx in predicted_h1.index])
else:
    predicted_h1.index = predicted_h1.index + horizon

obs_col = ("HC.f_future2h", "NA")
observations_h1 = two_hourly_data[[obs_col]].copy()
common_idx_h1 = observations_h1.index.intersection(predicted_h1.index)
aligned_obs = observations_h1.loc[common_idx_h1][obs_col]
aligned_pred = predicted_h1.loc[common_idx_h1][pred_col]
merged_h1 = pd.DataFrame({"Obs": aligned_obs.values.squeeze(), "Pred": aligned_pred.values.squeeze()}, index=common_idx_h1)

# ##-- ADDED: RMSE Calculation --##
burn_in = 96
valid_h1 = merged_h1.iloc[burn_in:].dropna()

if not valid_h1.empty:
    residuals_h1 = valid_h1["Obs"] - valid_h1["Pred"]
    rmse_h1 = np.sqrt((residuals_h1 ** 2).mean())
    print(f"\nHorizon {horizon} → RMSE (post burn-in): {rmse_h1:.4f}")
else:
    print(f"\nHorizon {horizon} → No valid data after burn-in for RMSE computation")

# Plot forecast vs. actual
fig, ax = plt.subplots(figsize=(16, 4))
ax.plot(merged_h1.index, merged_h1["Obs"], label="Actual 2h Sum", color="black", linewidth=2.0)
ax.plot(merged_h1.index, merged_h1["Pred"], label=f"Pred 2h Sum (k={horizon})", color="red", linestyle="-", linewidth=0.8)
ax.set_title(f"Online RLS 2-Hour Forecast with Rolling Inputs (k={horizon})")
ax.legend(); ax.grid(True)
plt.tight_layout()
fig.savefig(os.path.join(plot_dir, f"rls_forecast_2h_{horizon}_step.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Forecast plot saved to {os.path.join(plot_dir, f'rls_forecast_2h_{horizon}_step.png')}")

# ##-- ADDED: Diagnostic plots for residuals --##
if not valid_h1.empty:
    residuals = valid_h1["Obs"] - valid_h1["Pred"]
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 8))
    
    # ACF Plot
    plot_acf(residuals, lags=min(24, len(residuals)-1), ax=ax1)
    ax1.set_title("ACF of Residuals")
    
    # PACF Plot
    plot_pacf(residuals, lags=min(24, len(residuals)//2 -1), ax=ax2)
    ax2.set_title("PACF of Residuals")
    
    # Q-Q Plot
    qqplot(residuals, line="s", ax=ax3)
    ax3.set_title("Q-Q Plot of Residuals")
    
    # Residuals over time
    ax4.plot(residuals)
    ax4.set_title("Residuals Over Time")
    
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"rls_diagnostics_2h_{horizon}_step.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Diagnostic plots saved for k={horizon} → {os.path.join(plot_dir, f'rls_diagnostics_2h_{horizon}_step.png')}")
