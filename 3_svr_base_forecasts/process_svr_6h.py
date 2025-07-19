# process_svr_6h.py (Direct Multi-step SVR for 6-hour aggregated data)
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
hourly_data = data_raw.fc.subset(kseq=('NA',), end_index=7976)
hourly_data = hourly_data[hourly_data.index >= 5676]

# --- 3. Aggregate to 6-hourly features & target ---
six_hourly = c.new_fc()

# target: sum of the *next* 6 hours of HC.f
six_hourly[('HC.f_future6h','NA')] = (
    hourly_data[('HC.f','NA')]
    .shift(-6)
    .rolling(window=6, min_periods=6)
    .sum()
)

# autoregressive feature: sum of the *past* 6 hours
six_hourly[('HC.f_past6h','NA')] = (
    hourly_data[('HC.f','NA')]
    .rolling(window=6, min_periods=6)
    .sum()
)

# smoothed exogenous features: 6-h rolling mean
for var in ['Taobs','W.f','GR.f']:
    six_hourly[(f"{var}_smooth",'NA')] = (
        hourly_data[(var,'NA')]
        .rolling(window=6, min_periods=6)
        .mean()
    )

# drop any rows with NaNs from the rolling operations
six_hourly.dropna(inplace=True)

# --- 4. SVR Hyperparameters (same as 2h version) ---
OnlineSVR.params = ['C', 'epsilon', 'gamma', 'threshold']
svr_init_params = {'kernel': 'rbf', 'threshold': 500}
svr_hyperparams = {'C': 6.870340, 'epsilon': 0.002054, 'gamma': 0.580457}

# --- 5. Direct multi-step SVR over 6-hour series ---
horizons_to_predict = list(range(1, 9))  # 1→6h, …, 8→48h
max_lag = 48  # keep last 48 six-hour blocks for the rolling window

all_preds = []
print("Starting direct multi-step SVR 6-hour training & prediction...")
for h in horizons_to_predict:
    print(f"  Horizon {h} (forecast {6*h}h ahead)…")
    model = ForecastModel(
        OnlineSVR,
        predictor_init_params=svr_init_params.copy(),
        estimate_variance=False,
        kseq=(h,)
    )
    model.add_outputs('HC.f_future6h')
    model.add_inputs(
        ar_0    = AR('HC.f_past6h', order=0),
        ar_1    = AR('HC.f_past6h', order=1),
        ar_4    = AR('HC.f_past6h', order=4),
        ar_6    = AR('HC.f_past6h', order=6),
        ar_8    = AR('HC.f_past6h', order=8),
        Taobs_0 = AR('Taobs_smooth', order=0),
        Taobs_1 = AR('Taobs_smooth', order=1),
        W_f_0   = AR('W.f_smooth', order=0),
        GR_f_0  = AR('GR.f_smooth', order=0),
    )
    model.update_params(**svr_hyperparams)

    preds = []
    # slide a 1-row window through six_hourly
    for i in range(len(six_hourly) - h):
        batch = six_hourly.iloc[i:i+1].copy()
        future_val = six_hourly[('HC.f_future6h','NA')].iloc[i+h]
        if np.isnan(future_val):
            continue
        # inject true future for the online update
        batch[('HC.f_future6h','NA')] = future_val

        out = model.update(batch, n_keep=max_lag)
        df_out = out[0] if isinstance(out, tuple) else out
        if isinstance(df_out, pd.DataFrame) and not df_out.empty:
            val = df_out.iloc[0][('HC.f_future6h', h)]
            ts = pd.Series([val],
                           index=[six_hourly.index[i+h]],
                           name=('HC.f_future6h', h))
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

def unscale_sum(x, mn, mx, n=6):
    # for sums of n scaled values: x*(mx-mn) + n*mn
    return x * (mx - mn) + (n * mn)

plot_dir = "../7_results/Case_2/3_svr/res6"
os.makedirs(plot_dir, exist_ok=True)

# --- Predictions (h=6) ---
preds_unscaled = unscale_sum(combined_pred, hc_f_min, hc_f_max, n=6)
# inject the two level-names so the CSV header reads Variable / Horizon
preds_unscaled.columns.names = ['Variable', 'Horizon']
preds_unscaled.to_csv(
    os.path.join(plot_dir, "svr_res6h.csv"),
    index_label=['', '']    # blank out the top-left cell
)
print(f"Predictions saved → {os.path.join(plot_dir, 'svr_res6h.csv')}")

# --- Observations (NA-horizon) ---
obs_scaled   = six_hourly[('HC.f_future6h','NA')]
obs_unscaled = unscale_sum(obs_scaled, hc_f_min, hc_f_max, n=6).to_frame()
# rebuild a one-column MultiIndex so you get the two-row header
obs_unscaled.columns = pd.MultiIndex.from_tuples(
    [('HC.f','NA')],
    names=['Variable','Horizon']
)
obs_unscaled.to_csv(
    os.path.join(plot_dir, "observations_HC_6h.csv"),
    index_label=['', '']
)
print(f"Observations saved → {os.path.join(plot_dir, 'observations_HC_6h.csv')}")

# --- Combined (obs + preds) ---
combined_df = pd.concat([obs_unscaled, preds_unscaled], axis=1)
combined_df.to_csv(
    os.path.join(plot_dir, "svr_res6h_with_obs.csv"),
    index_label=['', '']
)
print(f"Combined saved → {os.path.join(plot_dir, 'svr_res6h_with_obs.csv')}")

# --- 7. Evaluation & Plotting for first 6h horizon (h=1) ---
burn_in = 48  # skip first 48 six-hour steps for metrics
for var, h in combined_pred.columns:
    if var != 'HC.f_future6h':
        continue
    pred_s = combined_pred[(var, h)].dropna()
    if pred_s.empty:
        continue

    y_pred = unscale_sum(pred_s, hc_f_min, hc_f_max, n=6)
    y_obs  = unscale_sum(obs_scaled, hc_f_min, hc_f_max, n=6)

    aligned_pred, aligned_obs = y_pred.align(y_obs, join='inner')
    df = pd.DataFrame({'pred': aligned_pred, 'actual': aligned_obs}).dropna()
    df_post = df.iloc[burn_in:] if burn_in < len(df) else pd.DataFrame()

    if df_post.empty:
        print(f"No data for post burn-in at h={h}")
        continue

    y_true, y_hat = df_post['actual'], df_post['pred']
    rmse = np.sqrt(((y_true - y_hat)**2).mean())
    mape = mean_absolute_percentage_error(y_true, y_hat) * 100
    r2v  = r2_score(y_true, y_hat)
    print(f"Horizon {h} (6h ahead) → RMSE={rmse:.4f}, MAPE={mape:.2f}%, R2={r2v:.4f}")

    if h == 1:
        # forecast vs actual
        fig, ax = plt.subplots(figsize=(16, 4))
        ax.plot(y_true.index, y_true, label='Actual 6h Sum',   linewidth=2.0)
        ax.plot(y_hat.index, y_hat, label='Predicted 6h Sum', linewidth=0.8)
        ax.set_title("SVR 6-Hour Forecast (h=1) – Post Burn-in")
        ax.legend(); ax.grid(True)
        fig.savefig(
            os.path.join(plot_dir, "svr_forecast_6h_1_step_post_burn_in.png"),
            dpi=300, bbox_inches='tight'
        )
        plt.close(fig)

        # diagnostic plots
        res = y_true - y_hat
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 8))
        plot_acf(res, lags=min(16, len(res)-1), ax=ax1); ax1.set_title("ACF of Residuals")
        plot_pacf(res, lags=min(16, len(res)//2-1), ax=ax2); ax2.set_title("PACF of Residuals")
        qqplot(res, line="s", ax=ax3); ax3.set_title("Q-Q Plot of Residuals")
        ax4.plot(res.index, res); ax4.set_title("Residuals Over Time")
        plt.tight_layout()
        fig.savefig(
            os.path.join(plot_dir, "svr_diagnostics_6h_1_step.png"),
            dpi=300, bbox_inches='tight'
        )
        plt.close(fig)

print("Processing complete.")