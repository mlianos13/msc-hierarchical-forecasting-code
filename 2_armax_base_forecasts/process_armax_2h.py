# process_armax_2h.py

#%% Imports and setup
import sys
import os
import warnings
import numpy as np
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.gofplots import qqplot
import matplotlib.pyplot as plt
import pandas as pd

# Append parent directory to path to access py_online_forecast library
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import py_online_forecast.core_main as c
from py_online_forecast.core_main import *
import py_online_forecast.hierarchies as h
from py_online_forecast.hierarchies import *

# Suppress pandas performance warnings
warnings.filterwarnings('ignore', message='indexing past lexsort depth may impact performance')

# --- Data Loading and Preparation ---
try:
    data = pd.read_csv("veks.csv")
except FileNotFoundError:
    print("Error: 'veks.csv' not found. Ensure it is in the script's directory.")
    sys.exit()

# Select and preprocess data
data = data[["HC.f", "Ta.f", "W.f", "GR.f", "ds.tod", "hh"]]
data.insert(0, "t", list(range(len(data))))
data['t'] = pd.to_datetime(data['t'], unit='h', origin='2024-01-01')
data.rename(columns={"Ta.f": "Taobs"}, inplace=True)
if hasattr(data, 'fc'):
    data.fc.convert(separator=".k")

#
# 1. Prepare 2-hourly aggregated data
#
hourly_data = data.fc.subset(kseq=("NA", 1), end_index=5876)
hourly_data = hourly_data[hourly_data.index >= 3576]

two_hourly_data = c.new_fc()

# Feature Engineering: Create rolling features for 2-hour aggregation
# Input feature: Sum of the past 2 hours of HC.f
two_hourly_data[("HC.f_past2h", "NA")] = hourly_data[("HC.f", "NA")].rolling(window=2, min_periods=2).sum()
# Input feature: Smoothed average of the past 2 hours of Taobs
two_hourly_data[("Taobs_smooth", "NA")] = hourly_data[("Taobs", "NA")].rolling(window=2, min_periods=2).mean()

# Target variable: Sum of the next 2 hours of HC.f
sum_target_2h = hourly_data[("HC.f", "NA")].shift(-2).rolling(window=2).sum()
two_hourly_data[("HC.f_future2h", "NA")] = sum_target_2h

# Remove rows with NaN values resulting from rolling window operations
two_hourly_data.dropna(inplace=True)

# --- Model Definition and Fitting ---
# Define exogenous feature names for the ARMAX model
armax_exog_feature_names = [
    'HC.f_past2h', 'Taobs_smooth'
]

# Initialize the ForecastModel with the ARMAX predictor
model_armax_2h = ForecastModel(
    ARMAX,
    predictor_init_params={
        'order': (2, 1),  # (p, q) order for AR and MA terms
        'exog_order': 12,
        'exog_columns': armax_exog_feature_names,
    },
    predictor_params={'lambda_val': 0.99}, # Forgetting factor
    estimate_variance=False,
    kseq=tuple(range(1, 25)) # Predict up to 24 steps (48 hours / 2h step)
)

# Define model output and inputs
model_armax_2h.add_outputs("HC.f_future2h")
model_armax_2h.add_inputs(
    "HC.f_past2h",
    "Taobs_smooth"
)

# Fit the model
res2h = model_armax_2h.fit(two_hourly_data.copy())
print("ARMAX 2h model fitting completed.")

# --- Save Raw Predictions to CSV ---
plot_dir = "../7_results/Case_1/2_armax/res2"
os.makedirs(plot_dir, exist_ok=True)

# Ensure predictions are in the expected DataFrame format
if isinstance(res2h, tuple) and len(res2h) > 0 and isinstance(res2h[0], pd.DataFrame):
    predictions_df = res2h[0]
elif isinstance(res2h, pd.DataFrame):
    predictions_df = res2h
else:
    raise RuntimeError("Unexpected format for res2h from model_armax_2h.fit")

# Save predictions and observations
out_path_multiindex = os.path.join(plot_dir, "armax_res2h.csv")
predictions_df.to_csv(out_path_multiindex)
print(f"\nPredictions with MultiIndex saved to {out_path_multiindex}")

observations_df = two_hourly_data.fc[["HC.f_future2h"]].loc[predictions_df.index]
obs_path = os.path.join(plot_dir, "observations_HC_2h.csv")
observations_df.to_csv(obs_path)
print(f"Observations saved to {obs_path}")

combined_df = pd.concat([observations_df, predictions_df], axis=1)
combined_path = os.path.join(plot_dir, "armax_res2h_with_obs.csv")
combined_df.to_csv(combined_path)
print(f"Combined predictions and observations saved to {combined_path}")

# --- OLS Analysis for h=1 ---
horizon = 1
X_full, Y_full = model_armax_2h.transform_data(two_hourly_data, use_recursive_pars=False)

# Prepare data for OLS
kseq_for_h1 = ("NA", horizon)
X_h1 = X_full.fc.subset(kseq=kseq_for_h1).copy()
X_h1_lagged = X_h1.shift(horizon).dropna()

y_target_col = ("HC.f_future2h", "NA")
y_target = two_hourly_data[y_target_col].copy()

common_idx = X_h1_lagged.index.intersection(y_target.index)
X_ols_aligned = X_h1_lagged.loc[common_idx]
y_ols_aligned = y_target.loc[common_idx]

# Flatten MultiIndex columns for statsmodels
if isinstance(X_ols_aligned.columns, pd.MultiIndex):
    X_ols_aligned.columns = [f"{var}__{h}" for (var, h) in X_ols_aligned.columns.to_list()]

# Add a constant if not present
if not any('mu' in col.lower() for col in X_ols_aligned.columns):
    X_ols_final = sm.add_constant(X_ols_aligned, has_constant='add')
else:
    X_ols_final = X_ols_aligned.copy()

# Fit OLS model and save results
ols_model = sm.OLS(y_ols_aligned, X_ols_final).fit()

print("\n=== OLS Results for 2h-Aggregated ARMAX Model ===")
print(f"AIC: {ols_model.aic:.4f}")
print(f"BIC: {ols_model.bic:.4f}")
print("\nP-values of regressors:")
print(ols_model.pvalues)

results_file_path = os.path.join(plot_dir, "AIC_BIC_pv_2h.txt")
with open(results_file_path, "w") as f:
    f.write("=== OLS Results for 2h-Aggregated ARMAX Model ===\n")
    f.write(f"AIC: {ols_model.aic:.4f}\n")
    f.write(f"BIC: {ols_model.bic:.4f}\n")
    f.write("\nP-values of regressors:\n")
    f.write(ols_model.summary().tables[1].as_text())
print(f"\nOLS results saved to {results_file_path}")

# --- Prediction and Diagnostic Plots for h=1 ---
pred_col = ("HC.f_future2h", horizon)
predicted_h1 = predictions_df[[pred_col]].copy()

# Align prediction timestamps
if isinstance(predicted_h1.index, pd.DatetimeIndex):
    predicted_h1.index = pd.DatetimeIndex([idx + pd.Timedelta(hours=horizon) for idx in predicted_h1.index])
else:
    predicted_h1.index = predicted_h1.index + horizon

# Align observations with predictions
obs_col = ("HC.f_future2h", "NA")
observations_h1 = two_hourly_data[[obs_col]].copy()
common_idx_h1 = observations_h1.index.intersection(predicted_h1.index)
aligned_obs = observations_h1.loc[common_idx_h1][obs_col]
aligned_pred = predicted_h1.loc[common_idx_h1][pred_col]
merged_h1 = pd.DataFrame({"Obs": aligned_obs.values.squeeze(), "Pred": aligned_pred.values.squeeze()}, index=common_idx_h1)

# Calculate RMSE and get data after a burn-in period
burn_in = 96
valid_h1 = merged_h1.iloc[burn_in:].dropna()

if not valid_h1.empty:
    residuals_h1 = valid_h1["Obs"] - valid_h1["Pred"]
    rmse_h1 = np.sqrt((residuals_h1 ** 2).mean())
    print(f"\nHorizon {horizon} -> RMSE (post burn-in): {rmse_h1:.4f}")
else:
    print(f"\nHorizon {horizon} -> No valid data for RMSE computation post burn-in.")

# Plot forecast vs. actual (POST BURN-IN)
fig, ax = plt.subplots(figsize=(16, 4))
ax.plot(valid_h1.index, valid_h1["Obs"], label="Actual 2h Sum", color="black", linewidth=2.0)
ax.plot(valid_h1.index, valid_h1["Pred"], label=f"Pred 2h Sum (k={horizon})", color="red", linestyle="-", linewidth=0.8)
ax.set_title(f"Online ARMAX 2-Hour Forecast (k={horizon}) - Post Burn-in")
ax.legend()
ax.grid(True)
plt.tight_layout()
fig_path = os.path.join(plot_dir, f"armax_forecast_2h_{horizon}_step_post_burn_in.png")
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Forecast plot (post burn-in) saved to {fig_path}")

# Diagnostic plots for residuals (POST BURN-IN)
if not valid_h1.empty:
    # Note: 'residuals_h1' is already calculated from the post-burn-in 'valid_h1' DataFrame
    residuals = residuals_h1
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 8))
    
    plot_acf(residuals, lags=min(24, len(residuals)-1), ax=ax1)
    ax1.set_title("ACF of Residuals (Post Burn-in)")
    
    plot_pacf(residuals, lags=min(24, len(residuals)//2 - 1), ax=ax2)
    ax2.set_title("PACF of Residuals (Post Burn-in)")
    
    qqplot(residuals, line="s", ax=ax3)
    ax3.set_title("Q-Q Plot of Residuals (Post Burn-in)")
    
    ax4.plot(residuals.index, residuals)
    ax4.set_title("Residuals Over Time (Post Burn-in)")
    
    plt.tight_layout()
    diag_fig_path = os.path.join(plot_dir, f"armax_diagnostics_2h_{horizon}_step.png")
    fig.savefig(diag_fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Diagnostic plots saved to {diag_fig_path}")

