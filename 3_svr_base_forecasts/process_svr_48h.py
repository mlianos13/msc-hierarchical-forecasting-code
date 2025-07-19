# process_svr_48h.py (Direct Multi-step SVR for 48-hour aggregated data)
#%%
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.gofplots import qqplot
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import py_online_forecast.core_main as c
from py_online_forecast.core_main import *

np.random.seed(42)


# --- 1. Load and prepare raw 1-hour data ---
data_raw = c.read_forecast_csv("veks_svr.csv", multiindex=True)
data_raw.fc.convert(separator=".k")


# --- 2. Build 1-hourly DataFrame for aggregation ---
hourly_data = (data_raw.fc.subset(kseq=('NA',), end_index=7976))
hourly_data = hourly_data[hourly_data.index >= 5676]

# --- 3. Aggregate to 48-hourly features & target ---
fortyeight_hourly = c.new_fc()

# target: sum of the *next* 48 hours of HC.f
fortyeight_hourly[('HC.f_future48h','NA')] = (
    hourly_data[('HC.f','NA')]
    .shift(-48)
    .rolling(window=48, min_periods=48)
    .sum()
)

# autoregressive feature: sum of the *past* 48 hours
fortyeight_hourly[('HC.f_past48h','NA')] = (
    hourly_data[('HC.f','NA')]
    .rolling(window=48, min_periods=48)
    .sum()
)

# smoothed exogenous features: 48-h rolling mean
for var in ['Taobs','W.f','GR.f']:
    fortyeight_hourly[(f"{var}_smooth",'NA')] = (
        hourly_data[(var,'NA')]
        .rolling(window=48, min_periods=48)
        .mean()
    )

# drop any rows with NaNs from the rolling operations
fortyeight_hourly.dropna(inplace=True)

# --- 4. SVR Hyperparameters (unchanged) ---
OnlineSVR.params = ['C', 'epsilon', 'gamma', 'threshold']
svr_init_params  = {'kernel': 'rbf', 'threshold': 500}
svr_hyperparams = {'C': 6.870340, 'epsilon': 0.002054, 'gamma': 0.580457}

# --- 5. Direct multi-step SVR over 48-hour series ---
# horizon 1 → 48h ahead
horizons_to_predict = [1]
max_lag = 48  # keep last 48 blocks of 48h for the rolling window

all_preds = []
print("Starting direct multi-step SVR 48-hour training & prediction...")
for h in horizons_to_predict:
    print(f"  Horizon {h} (forecast {48*h}h ahead)…")
    model = ForecastModel(
        OnlineSVR,
        predictor_init_params=svr_init_params.copy(),
        estimate_variance=False,
        kseq=(h,)
    )
    model.add_outputs('HC.f_future48h')
    model.add_inputs(
        ar_0    = AR('HC.f_past48h', order=0),
        ar_1    = AR('HC.f_past48h', order=1),
        Taobs_0 = AR('Taobs_smooth', order=0),
        Taobs_1 = AR('Taobs_smooth', order=1),
        W_f_0   = AR('W.f_smooth', order=0),
        GR_f_0  = AR('GR.f_smooth', order=0),
    )
    model.update_params(**svr_hyperparams)

    preds = []
    for i in range(len(fortyeight_hourly) - h):
        batch      = fortyeight_hourly.iloc[i:i+1].copy()
        future_val = fortyeight_hourly[('HC.f_future48h','NA')].iloc[i+h]
        if np.isnan(future_val):
            continue
        batch[('HC.f_future48h','NA')] = future_val

        out = model.update(batch, n_keep=max_lag)
        df_out = out[0] if isinstance(out, tuple) else out
        if isinstance(df_out, pd.DataFrame) and not df_out.empty:
            val = df_out.iloc[0][('HC.f_future48h', h)]
            ts  = pd.Series(
                [val],
                index=[fortyeight_hourly.index[i+h]],
                name=('HC.f_future48h', h)
            )
            preds.append(ts)

    if preds:
        all_preds.append(pd.concat(preds))

# combine all horizons into one DataFrame
if all_preds:
    combined_pred = pd.concat(all_preds, axis=1)
    if not isinstance(combined_pred.columns, pd.MultiIndex):
        combined_pred.columns = pd.MultiIndex.from_tuples(
            [s.name for s in all_preds],
            names=['Variable','Horizon']
        )
    combined_pred = combined_pred.loc[:, ~combined_pred.columns.duplicated()]
    print("All predictions combined.")
else:
    combined_pred = pd.DataFrame()
    print("No predictions generated.")

# --- 6. Unscale & Save CSVs ---
orig_min = pd.read_csv("veks_svr_min.csv").T.squeeze()
orig_max = pd.read_csv("veks_svr_max.csv").T.squeeze()
hc_f_min, hc_f_max = orig_min['HC.f'], orig_max['HC.f']

def unscale_sum(x, mn, mx, n=48):
    # for sums of n scaled values: x*(mx-mn) + n*mn
    return x * (mx - mn) + (n * mn)

plot_dir = "../7_results/Case_2/3_svr/res48"
os.makedirs(plot_dir, exist_ok=True)

# --- Predictions (h=48) ---
preds_unscaled = unscale_sum(combined_pred, hc_f_min, hc_f_max, n=48)
# inject the two level-names so the CSV header reads Variable / Horizon
preds_unscaled.columns.names = ['Variable', 'Horizon']
preds_unscaled.to_csv(
    os.path.join(plot_dir, "svr_res48h.csv"),
    index_label=['', '']    # blank out the top-left cell
)
print(f"Predictions saved → {os.path.join(plot_dir, 'svr_res48h.csv')}")

# --- Observations (NA-horizon) ---
obs_scaled   = fortyeight_hourly[('HC.f_future48h','NA')]
obs_unscaled = unscale_sum(obs_scaled, hc_f_min, hc_f_max, n=48).to_frame()
# rebuild a one-column MultiIndex so you get the two-row header
obs_unscaled.columns = pd.MultiIndex.from_tuples(
    [('HC.f','NA')],
    names=['Variable','Horizon']
)
obs_unscaled.to_csv(
    os.path.join(plot_dir, "observations_HC_48h.csv"),
    index_label=['', '']
)
print(f"Observations saved → {os.path.join(plot_dir, 'observations_HC_48h.csv')}")

# --- Combined (obs + preds) ---
combined_df = pd.concat([obs_unscaled, preds_unscaled], axis=1)
combined_df.to_csv(
    os.path.join(plot_dir, "svr_res48h_with_obs.csv"),
    index_label=['', '']
)
print(f"Combined saved → {os.path.join(plot_dir, 'svr_res48h_with_obs.csv')}")

# --- 7. Evaluation & Plotting for first 48h horizon (h=1) ---
burn_in = 48  # skip first 48 blocks for metrics
for var, h in combined_pred.columns:
    if var != 'HC.f_future48h':
        continue
    pred_s = combined_pred[(var, h)].dropna()
    if pred_s.empty:
        continue

    y_pred = unscale_sum(pred_s, hc_f_min, hc_f_max, n=48)
    y_obs  = unscale_sum(obs_scaled, hc_f_min, hc_f_max, n=48)

    aligned_pred, aligned_obs = y_pred.align(y_obs, join='inner')
    df = pd.DataFrame({'pred': aligned_pred, 'actual': aligned_obs}).dropna()
    df_post = df.iloc[burn_in:] if burn_in < len(df) else pd.DataFrame()

    if df_post.empty:
        print(f"No data post burn-in at h={h}")
        continue

    y_true, y_hat = df_post['actual'], df_post['pred']
    rmse = np.sqrt(((y_true - y_hat)**2).mean())
    mape = mean_absolute_percentage_error(y_true, y_hat) * 100
    r2v  = r2_score(y_true, y_hat)
    print(f"Horizon {h} (48h ahead) → RMSE={rmse:.4f}, MAPE={mape:.2f}%, R2={r2v:.4f}")

    if h == 1:
        # forecast vs actual
        fig, ax = plt.subplots(figsize=(16, 4))
        ax.plot(y_true.index, y_true, label='Actual 48h Sum', linewidth=2.0)
        ax.plot(y_hat.index, y_hat, label='Predicted 48h Sum', linewidth=0.8)
        ax.set_title("SVR 48-Hour Forecast (h=1) – Post Burn-in")
        ax.legend(); ax.grid(True)
        fig.savefig(
            os.path.join(plot_dir, "svr_forecast_48h_1_step_post_burn_in.png"),
            dpi=300, bbox_inches='tight'
        )
        plt.close(fig)

        # diagnostics
        res = y_true - y_hat
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 8))
        plot_acf(res, lags=min(2, len(res)-1), ax=ax1); ax1.set_title("ACF of Residuals")
        plot_pacf(res, lags=min(2, len(res)//2-1), ax=ax2); ax2.set_title("PACF of Residuals")
        qqplot(res, line="s", ax=ax3); ax3.set_title("Q-Q Plot of Residuals")
        ax4.plot(res.index, res); ax4.set_title("Residuals Over Time")
        plt.tight_layout()
        fig.savefig(
            os.path.join(plot_dir, "svr_diagnostics_48h_1_step.png"),
            dpi=300, bbox_inches='tight'
        )
        plt.close(fig)

print("Processing complete.")