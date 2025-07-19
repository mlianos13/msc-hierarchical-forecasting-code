# process_arfr_1h.py
#%%
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.gofplots import qqplot
import time

# Append parent directory to path to access the custom library
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import py_online_forecast.core_main as c
from py_online_forecast.core_main import *
import py_online_forecast.hierarchies as h
from py_online_forecast.hierarchies import *
from river.tree import HoeffdingTreeRegressor

# Suppress pandas performance warnings
warnings.filterwarnings('ignore', message='indexing past lexsort depth may impact performance')

np.random.seed(42)

# --- Data Loading and Preparation ---
try:
    # Load and prepare data using the Forecast class
    data_raw = c.read_forecast_csv("veks_svr.csv", multiindex=True)
    data_raw.fc.convert(separator=".k")
except FileNotFoundError:
    print("Error: 'veks_svr.csv' not found. Make sure it's in the correct directory.")
    sys.exit()


# Prepare data subset for the model
# Using a simplified kseq as we mainly need the 'NA' horizon for current exogenous variable values
hourly_data = data_raw.fc.subset(kseq = tuple(['NA']), end_index=5876)
hourly_data = hourly_data[hourly_data.index >= 3576]

# --- ARFR Model Setup ---
ARFR.params = ['n_trees','max_features','lambda_value', 'delta_warning','delta_drift','burn_in',
               'grace_period','max_depth','split_confidence', 'model_selector_decay', 'leaf_prediction',
               'target_model_horizon']

model1 = ForecastModel(ARFR, estimate_variance=False, kseq=tuple(range(1, 2)))
model1.add_outputs("HC.f")
model1.add_inputs(
    ar_0=AR("HC.f", order=0),
    ar_1=AR("HC.f", order=1),
    ar_12=AR("HC.f", order=12),
    ar_23=AR("HC.f", order=23),
    ar_24=AR("HC.f", order=24),
    Taobs_0=AR("Taobs", order=0),
    Taobs_1=AR("Taobs", order=1),
    W_f_0=AR("W.f", order=0),
    GR_f_0=AR("GR.f", order=0),
)

# Best Parameters from previous optimization
model1.update_params(
    n_trees=100,
    max_features=12,
    lambda_value=6.0,
    grace_period=20,
    max_depth=10,
    split_confidence=1e-07,
    model_selector_decay=0.95,
    leaf_prediction='adaptive'
)
t0 = time.perf_counter()
# Fit the model to get scaled predictions
res1_scaled = model1.fit(hourly_data)
print("ARFR 1h model fitting completed.")
t_elapsed = time.perf_counter() - t0
print(f"\nModel fit completed in {t_elapsed:.2f} seconds")



# --- Unscale and Save Results ---

# 1. Create a directory for plots and results
plot_dir = "../7_results/Case_1/4_ohdt/res1"
os.makedirs(plot_dir, exist_ok=True)

# 2. Load min/max values for unscaling
try:
    orig_min = pd.read_csv("veks_svr_min.csv").T.squeeze()
    orig_max = pd.read_csv("veks_svr_max.csv").T.squeeze()
except FileNotFoundError:
    print("Error: Min/Max scaling files ('veks_svr_min.csv', 'veks_svr_max.csv') not found.")
    sys.exit()

def unscale(series_or_df, mn, mx):
    """Unscales a pandas Series or DataFrame."""
    return series_or_df * (mx - mn) + mn

hc_f_min = orig_min['HC.f']
hc_f_max = orig_max['HC.f']

# 3. Get scaled predictions and observations
if isinstance(res1_scaled, tuple) and len(res1_scaled) > 0 and isinstance(res1_scaled[0], pd.DataFrame):
    predictions_scaled_df = res1_scaled[0]
elif isinstance(res1_scaled, pd.DataFrame):
    predictions_scaled_df = res1_scaled
else:
    raise RuntimeError("Unexpected format for res1_scaled from model1.fit")

observations_scaled_df = hourly_data.fc[["HC.f"]].loc[predictions_scaled_df.index]

# 4. Unscale the data
predictions_unscaled_df = unscale(predictions_scaled_df, hc_f_min, hc_f_max)
observations_unscaled_df = unscale(observations_scaled_df, hc_f_min, hc_f_max)

# 5. Save the unscaled predictions, observations, and combined file
# Save unscaled predictions
out_path_preds = os.path.join(plot_dir, "arfr_res1h.csv")
predictions_unscaled_df.to_csv(out_path_preds)
print(f"\nUnscaled predictions saved to {out_path_preds}")

# Save unscaled observations
obs_path = os.path.join(plot_dir, "observations_HC.csv")
observations_unscaled_df.to_csv(obs_path)
print(f"Unscaled observations saved to {obs_path}")

# Combine unscaled predictions and observations and save
combined_unscaled_df = pd.concat([observations_unscaled_df, predictions_unscaled_df], axis=1)
combined_path = os.path.join(plot_dir, "arfr_res1h_with_obs.csv")
combined_unscaled_df.to_csv(combined_path)
print(f"Combined unscaled predictions and observations saved to {combined_path}")


# --- Evaluation and Plotting ---

# 6. Prepare DataFrame for evaluation (using horizon k=1)
horizon = 1
comparison_df = pd.DataFrame({
    'Obs': combined_unscaled_df[('HC.f', 'NA')],
    'Pred': combined_unscaled_df[('HC.f', horizon)]
})

# 7. Calculate RMSE and get data after a burn-in period
burn_in = 96  # Define a burn-in period (e.g., 96 hours)
valid_df = comparison_df.iloc[burn_in:].dropna()

if not valid_df.empty:
    residuals = valid_df["Obs"] - valid_df["Pred"]
    rmse = np.sqrt((residuals ** 2).mean())
    print(f"\nHorizon {horizon} -> RMSE (post burn-in): {rmse:.4f}")
else:
    print(f"\nHorizon {horizon} -> No valid data for RMSE computation post burn-in.")

# 8. Plot forecast vs. actual (POST BURN-IN)
fig, ax = plt.subplots(figsize=(16, 4))
ax.plot(valid_df.index, valid_df["Obs"], label="Actual Unscaled", color="black", linewidth=2.0)
ax.plot(valid_df.index, valid_df["Pred"], label=f"Predicted Unscaled (k={horizon})", color="blue", linestyle="-", linewidth=0.8)
ax.set_title(f"Online ARFR 1-Hour Forecast (k={horizon}) - Post Burn-in")
ax.legend()
ax.grid(True)
plt.tight_layout()
fig_path = os.path.join(plot_dir, f"arfr_forecast_1h_{horizon}_step_post_burn_in.png")
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Forecast plot (post burn-in) saved to {fig_path}")

# 9. Diagnostic plots for residuals (POST BURN-IN)
if not valid_df.empty:
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 8))
    
    # ACF Plot
    plot_acf(residuals, lags=min(48, len(residuals)-1), ax=ax1)
    ax1.set_title("ACF of Residuals (Post Burn-in)")
    
    # PACF Plot
    plot_pacf(residuals, lags=min(48, len(residuals)//2 - 1), ax=ax2)
    ax2.set_title("PACF of Residuals (Post Burn-in)")
    
    # Q-Q Plot
    qqplot(residuals, line="s", ax=ax3)
    ax3.set_title("Q-Q Plot of Residuals (Post Burn-in)")
    
    # Residuals Over Time
    ax4.plot(residuals.index, residuals)
    ax4.set_title("Residuals Over Time (Post Burn-in)")
    
    plt.tight_layout()
    diag_fig_path = os.path.join(plot_dir, f"arfr_diagnostics_1h_{horizon}_step.png")
    fig.savefig(diag_fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Diagnostic plots saved to {diag_fig_path}")