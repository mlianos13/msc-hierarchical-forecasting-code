# process_rls_1h.py

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
import time

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
# 1. Prepare and fit the 1-hour RLS model
#

# Subset hourly data for all 48 horizons
hourly_data = data.fc.subset(kseq=("NA",), end_index=5876)
hourly_data = hourly_data[hourly_data.index >= 3576]

t0 = time.perf_counter()

# ##-- CHANGED: Model now predicts 48 horizons --##
# Initialize RLS-based ForecastModel for k=1 through k=48
model1 = ForecastModel(RLS, estimate_variance=False, kseq=tuple(range(1, 2)))
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
# Set forgetting factor λ=0.995
model1.update_params(rls_lambda=0.995)



# Fit the model on hourly_data
res1 = model1.fit(hourly_data)
t_elapsed = time.perf_counter() - t0
print(f"\nModel fit completed in {t_elapsed:.2f} seconds")

# 2. Save raw predictions to CSV
#

plot_dir = "../7_results/Case_1/1_rls/res1"
os.makedirs(plot_dir, exist_ok=True)

if isinstance(res1, ForecastTuple) and len(res1) > 0 and isinstance(res1[0], pd.DataFrame):
    predictions_df = res1[0]
elif isinstance(res1, pd.DataFrame):
    predictions_df = res1
else:
    raise RuntimeError("Unexpected format for res1 from model1.fit")

# Save multiindexed predictions
out_path_multiindex = os.path.join(plot_dir, "rls_res1h.csv")
predictions_df.to_csv(out_path_multiindex)
print(f"\nPredictions with MultiIndex saved to {out_path_multiindex}")

# Save the observed (HC.f at horizon NA) for the same indices
observations_df = hourly_data.fc[["HC.f"]].loc[predictions_df.index]
obs_path = os.path.join(plot_dir, "observations_HC.csv")
observations_df.to_csv(obs_path)
print(f"Observations saved to {obs_path}")

# Optionally save combined file (observations + predictions)
combined_df = pd.concat([observations_df, predictions_df], axis=1)
combined_path = os.path.join(plot_dir, "rls_res1h_with_obs.csv")
combined_df.to_csv(combined_path)
print(f"Combined predictions and observations saved to {combined_path}")

#
# 3. Compute AIC, BIC, and p-values for h = 1 via OLS
#

# Reconstruct the full transformed feature matrix
horizon = 1
X_full, Y_full = model1.transform_data(hourly_data, use_recursive_pars=False)

# Extract features for horizon 1 prediction and apply lag
kseq_for_h1 = ("NA", horizon)
X_h1 = X_full.fc.subset(kseq=kseq_for_h1).copy()
X_h1_lagged = X_h1.shift(horizon).dropna()

# Extract target variable
if ("HC.f", "NA") in hourly_data.columns:
    y_target = hourly_data[("HC.f", "NA")].copy()
else:
    y_target = Y_full.iloc[:, 0].copy()

# Align data
common_idx = X_h1_lagged.index.intersection(y_target.index)
X_ols_aligned = X_h1_lagged.loc[common_idx]
y_ols_aligned = y_target.loc[common_idx]

# Flatten column names
if isinstance(X_ols_aligned.columns, pd.MultiIndex):
    X_ols_aligned.columns = [f"{var}__{h}" for (var, h) in X_ols_aligned.columns.to_list()]

# Handle multicollinearity
for col in X_ols_aligned.columns:
    if X_ols_aligned[col].nunique() <= 1:
        X_ols_aligned = X_ols_aligned.drop(columns=[col])

if not any('mu' in col.lower() for col in X_ols_aligned.columns):
    X_ols_final = sm.add_constant(X_ols_aligned, has_constant='add')
else:
    X_ols_final = X_ols_aligned.copy()

# Fit OLS
ols_model = sm.OLS(y_ols_aligned, X_ols_final).fit()

# Print results
print("\n=== OLS Results for 1h Model (horizon 1) ===")
print(f"AIC: {ols_model.aic:.4f}")
print(f"BIC: {ols_model.bic:.4f}")
print("\nP-values of regressors:")
print(ols_model.pvalues)

# Save results to file
results_file_path = os.path.join(plot_dir, "AIC_BIC_pv.txt")
with open(results_file_path, "w") as f:
    f.write("=== OLS Results for horizon 1 ===\n")
    f.write(f"AIC: {ols_model.aic:.4f}\n")
    f.write(f"BIC: {ols_model.bic:.4f}\n")
    f.write("\nP-values of regressors:\n")
    f.write(ols_model.summary().tables[1].as_text())
print(f"\nResults saved to {results_file_path}")

#
# 4. Produce and save prediction & diagnostic plots ONLY for horizon 1
#

h = 1
pred_col_name = "HC.f"
pred_col = (pred_col_name, h)
if pred_col not in predictions_df.columns:
    raise KeyError(f"No prediction column found for {pred_col_name} at horizon {h}")

# Extract predicted values
predicted_h1 = predictions_df[[pred_col]].copy()
if isinstance(predicted_h1.index, pd.DatetimeIndex):
    predicted_h1.index = pd.DatetimeIndex([idx + pd.Timedelta(hours=h) for idx in predicted_h1.index])
else:
    predicted_h1.index = predicted_h1.index + h

# Extract actual observations
obs_col = (pred_col_name, "NA")
if obs_col not in hourly_data.columns:
    raise KeyError(f"Observation column {obs_col} not found in `hourly_data`")
observations_h1 = hourly_data[[obs_col]].copy()

# Align predicted vs. observed indices
common_idx_h1 = observations_h1.index.intersection(predicted_h1.index)
aligned_obs = observations_h1.loc[common_idx_h1][obs_col]
aligned_pred = predicted_h1.loc[common_idx_h1][pred_col]

merged_h1 = pd.DataFrame({
    "Obs": aligned_obs.values.squeeze(),
    "Pred": aligned_pred.values.squeeze()
}, index=common_idx_h1)

# Compute RMSE
burn_in = 96
valid_h1 = merged_h1.iloc[burn_in:].dropna()

if not valid_h1.empty:
    residuals_h1 = valid_h1["Obs"] - valid_h1["Pred"]
    rmse_h1 = np.sqrt((residuals_h1 ** 2).mean())
    print(f"\nHorizon {h} → RMSE (post burn-in): {rmse_h1:.4f}")
else:
    print(f"\nHorizon {h} → No valid data after burn-in for RMSE computation")

# Plot forecast vs. actual (h=1)
fig, ax = plt.subplots(figsize=(16, 4))
ax.plot(merged_h1.index, merged_h1["Obs"], label="Actual", color="black", linewidth=2.0)
ax.plot(merged_h1.index, merged_h1["Pred"], label=f"Pred k={h}", color="red", linestyle="-", linewidth=0.8)

if burn_in > 0 and len(merged_h1) > burn_in:
    burn_in_idx = merged_h1.index[burn_in]
    ax.axvline(x=burn_in_idx, color="gray", linestyle="--")

ax.set_xlabel("Time Index")
ax.set_ylabel("HC.f")
ax.set_title(f"Online RLS Forecast vs Actual (k={h})")
ax.legend()
ax.grid(True)
plt.tight_layout()
fig.savefig(os.path.join(plot_dir, f"rls_forecast_{h}_step.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Forecast vs Actual plot saved for k={h} → {os.path.join(plot_dir, f'rls_forecast_{h}_step.png')}")

# Diagnostic plots for residuals (only for h=1)
if h == 1 and not valid_h1.empty:
    residuals = valid_h1["Obs"] - valid_h1["Pred"]
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 8))
    plot_acf(residuals, lags=min(24, len(residuals)-1), ax=ax1)
    ax1.set_title("ACF of Residuals")
    plot_pacf(residuals, lags=min(24, len(residuals)//2 -1), ax=ax2)
    ax2.set_title("PACF of Residuals")
    qqplot(residuals, line="s", ax=ax3)
    ax3.set_title("Q-Q Plot of Residuals")
    ax4.plot(residuals)
    ax4.set_title("Residuals Over Time")
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "rls_diagnostics_1_step.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Diagnostic plots saved for k={h} → {os.path.join(plot_dir, 'rls_diagnostics_1_step.png')}")