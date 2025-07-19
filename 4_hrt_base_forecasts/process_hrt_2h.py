# process_arfr_2h.py
#%%
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.gofplots import qqplot

# Append parent directory to path to access the custom library
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import py_online_forecast.core_main as c
from py_online_forecast.core_main import *
import py_online_forecast.hierarchies as h
from py_online_forecast.hierarchies import *
from river.tree import HoeffdingTreeRegressor

np.random.seed(42)

# --- Data Loading and Preparation (1-hour scaled data) ---

try:
    # Load and prepare data using the Forecast class
    data_raw = c.read_forecast_csv("veks_svr.csv", multiindex=True)
    data_raw.fc.convert(separator=".k")
except FileNotFoundError:
    print("Error: 'veks_svr.csv' not found. Make sure it's in the correct directory.")
    sys.exit()

# 1. Prepare 1-hourly data subset
# Using a simplified kseq as we mainly need the 'NA' horizon for aggregation
hourly_data = data_raw.fc.subset(kseq = tuple(['NA']), end_index=5876)
hourly_data = hourly_data[hourly_data.index >= 3576]

# 2. Aggregate to 2-hourly data
two_hourly_data = c.new_fc()

# Target variable: Sum of the next 2 hours of HC.f (what we want to predict)
sum_target_2h = hourly_data[("HC.f", "NA")].shift(-2).rolling(window=2, min_periods=2).sum()
two_hourly_data[("HC.f_future2h", "NA")] = sum_target_2h

# Input feature: Sum of the past 2 hours of HC.f (for autoregressive terms)
two_hourly_data[("HC.f_past2h", "NA")] = hourly_data[("HC.f", "NA")].rolling(window=2, min_periods=2).sum()
# Input feature: Smoothed average of the past 2 hours of Taobs
two_hourly_data[("Taobs_smooth", "NA")] = hourly_data[("Taobs", "NA")].rolling(window=2, min_periods=2).mean()
# Input feature: Smoothed average of the past 2 hours of W.f
two_hourly_data[("W.f_smooth", "NA")] = hourly_data[("W.f", "NA")].rolling(window=2, min_periods=2).mean()
# Input feature: Smoothed average of the past 2 hours of GR.f
two_hourly_data[("GR.f_smooth", "NA")] = hourly_data[("GR.f", "NA")].rolling(window=2, min_periods=2).mean()

# Remove rows with NaN values resulting from rolling window operations
two_hourly_data.dropna(inplace=True)


# --- ARFR Model Setup for 2-hour data ---
ARFR.params = ['n_trees','max_features','lambda_value', 'delta_warning','delta_drift','burn_in',
               'grace_period','max_depth','split_confidence', 'model_selector_decay', 'leaf_prediction',
               'target_model_horizon']

# Horizons for 2-hour steps (e.g., 1, 2, 3 steps ahead correspond to 2, 4, 6 hours)
model2 = ForecastModel(ARFR, estimate_variance=False, kseq=tuple(range(1, 25)))
model2.add_outputs("HC.f_future2h") # Predict the 2-hour sum
model2.add_inputs(
    # Autoregressive terms on the 2-hour sum
    ar_0=AR("HC.f_past2h", order=0),
    ar_1=AR("HC.f_past2h", order=1),
    # Daily lag is 12 steps for 2-hour data (24h / 2h)
    ar_6=AR("HC.f_past2h", order=6),
    ar_11=AR("HC.f_past2h", order=11),
    ar_12=AR("HC.f_past2h", order=12),
    # Exogenous variables on smoothed 2-hour data
    Taobs_0=AR("Taobs_smooth", order=0),
    Taobs_1=AR("Taobs_smooth", order=1),
    W_f_0=AR("W.f_smooth", order=0),
    GR_f_0=AR("GR.f_smooth", order=0),
)

# Best Parameters from previous optimization
model2.update_params(
    n_trees=100,
    max_features=12,
    lambda_value=6.0,
    grace_period=20,
    max_depth=10,
    split_confidence=1e-07,
    model_selector_decay=0.95,
    leaf_prediction='adaptive'
)

# Fit the model to get scaled predictions
res2_scaled = model2.fit(two_hourly_data)
print("ARFR 2h model fitting completed.")


# --- Unscale and Save Results ---

# 1. Create a directory for plots and results
plot_dir = "../7_results/Case_1/4_ohdt/res2"
os.makedirs(plot_dir, exist_ok=True)

# 2. Load min/max values for unscaling
try:
    orig_min = pd.read_csv("veks_svr_min.csv").T.squeeze()
    orig_max = pd.read_csv("veks_svr_max.csv").T.squeeze()
except FileNotFoundError:
    print("Error: Min/Max scaling files ('veks_svr_min.csv', 'veks_svr_max.csv') not found.")
    sys.exit()

def unscale_sum(series_or_df, mn, mx):
    """
    Unscales a pandas Series or DataFrame that represents a sum of two scaled values.
    Derivation:
    y_sum_unscaled = y1_unscaled + y2_unscaled
                   = (y1_scaled * (mx - mn) + mn) + (y2_scaled * (mx - mn) + mn)
                   = (y1_scaled + y2_scaled) * (mx - mn) + 2 * mn
                   = y_sum_scaled * (mx - mn) + 2 * mn
    """
    return series_or_df * (mx - mn) + (2 * mn)

hc_f_min = orig_min['HC.f']
hc_f_max = orig_max['HC.f']

# 3. Get scaled predictions and observations
if isinstance(res2_scaled, tuple) and len(res2_scaled) > 0 and isinstance(res2_scaled[0], pd.DataFrame):
    predictions_scaled_df = res2_scaled[0]
elif isinstance(res2_scaled, pd.DataFrame):
    predictions_scaled_df = res2_scaled
else:
    raise RuntimeError("Unexpected format for res2_scaled from model2.fit")

observations_scaled_df = two_hourly_data.fc[["HC.f_future2h"]].loc[predictions_scaled_df.index]

# 4. Unscale the data using the correct formula for sums
predictions_unscaled_df = unscale_sum(predictions_scaled_df, hc_f_min, hc_f_max)
observations_unscaled_df = unscale_sum(observations_scaled_df, hc_f_min, hc_f_max)

# 5. Save the unscaled predictions, observations, and combined file
out_path_preds = os.path.join(plot_dir, "arfr_res2h.csv")
predictions_unscaled_df.to_csv(out_path_preds)
print(f"\nUnscaled predictions saved to {out_path_preds}")

obs_path = os.path.join(plot_dir, "observations_HC_2h.csv")
observations_unscaled_df.to_csv(obs_path)
print(f"Unscaled observations saved to {obs_path}")

combined_unscaled_df = pd.concat([observations_unscaled_df, predictions_unscaled_df], axis=1)
combined_path = os.path.join(plot_dir, "arfr_res2h_with_obs.csv")
combined_unscaled_df.to_csv(combined_path)
print(f"Combined unscaled predictions and observations saved to {combined_path}")


# --- Evaluation and Plotting ---

# 6. Prepare DataFrame for evaluation (using horizon k=1, which is a 2-hour ahead forecast)
horizon = 1
comparison_df = pd.DataFrame({
    'Obs': combined_unscaled_df[('HC.f_future2h', 'NA')],
    'Pred': combined_unscaled_df[('HC.f_future2h', horizon)]
})

# 7. Calculate RMSE and get data after a burn-in period
burn_in = 48  # Define a burn-in period (e.g., 48 * 2h steps = 96 hours)
valid_df = comparison_df.iloc[burn_in:].dropna()

if not valid_df.empty:
    residuals = valid_df["Obs"] - valid_df["Pred"]
    rmse = np.sqrt((residuals ** 2).mean())
    print(f"\nHorizon {horizon} (2 hours) -> RMSE (post burn-in): {rmse:.4f}")
else:
    print(f"\nHorizon {horizon} -> No valid data for RMSE computation post burn-in.")

# 8. Plot forecast vs. actual (POST BURN-IN)
fig, ax = plt.subplots(figsize=(16, 4))
ax.plot(valid_df.index, valid_df["Obs"], label="Actual 2h Sum (Unscaled)", color="black", linewidth=2.0)
ax.plot(valid_df.index, valid_df["Pred"], label=f"Predicted 2h Sum (Unscaled, k={horizon})", color="blue", linestyle="-", linewidth=0.8)
ax.set_title(f"Online ARFR 2-Hour Forecast (k={horizon}) - Post Burn-in")
ax.legend()
ax.grid(True)
plt.tight_layout()
fig_path = os.path.join(plot_dir, f"arfr_forecast_2h_{horizon}_step_post_burn_in.png")
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Forecast plot (post burn-in) saved to {fig_path}")

# 9. Diagnostic plots for residuals (POST BURN-IN)
if not valid_df.empty:
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 8))
    
    # ACF Plot (24 lags represent 48 hours)
    plot_acf(residuals, lags=min(24, len(residuals)-1), ax=ax1)
    ax1.set_title("ACF of Residuals (Post Burn-in)")
    
    # PACF Plot
    plot_pacf(residuals, lags=min(24, len(residuals)//2 - 1), ax=ax2)
    ax2.set_title("PACF of Residuals (Post Burn-in)")
    
    # Q-Q Plot
    qqplot(residuals, line="s", ax=ax3)
    ax3.set_title("Q-Q Plot of Residuals (Post Burn-in)")
    
    # Residuals Over Time
    ax4.plot(residuals.index, residuals)
    ax4.set_title("Residuals Over Time (Post Burn-in)")
    
    plt.tight_layout()
    diag_fig_path = os.path.join(plot_dir, f"arfr_diagnostics_2h_{horizon}_step.png")
    fig.savefig(diag_fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Diagnostic plots saved to {diag_fig_path}")