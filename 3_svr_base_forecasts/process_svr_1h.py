# process_svr_1h.py (Direct Multi-step SVR with RMSE, MAPE, R-squared & NaN Fix)
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
import time
np.random.seed(42)

# 1. Load and prepare data
data_raw = c.read_forecast_csv("veks_svr.csv", multiindex=True)
data_raw.fc.convert(separator=".k")
data_raw = data_raw[data_raw.index >= 5676]
#
# 2. SVR hyperparameters
OnlineSVR.params = ['C', 'epsilon', 'gamma', 'threshold']
svr_init_params = {'kernel': 'rbf', 'threshold': 500}
svr_hyperparams = {'C': 3.320506, 'epsilon': 0.003792, 'gamma': 0.489677}




# 3. Forecast settings
horizons_to_predict = list(range(1, 49))
max_lag = 48
end_index = 2300
print("Starting direct multi-step SVR training & prediction...")

t0_global = time.perf_counter()

# 4. Generate predictions for each horizon
all_preds = []
for h in horizons_to_predict:
    t0_h = time.perf_counter()
    print(f"  Horizon {h}...")
    model = ForecastModel(
        OnlineSVR,
        predictor_init_params=svr_init_params.copy(),
        estimate_variance=False,
        kseq=(h,)
    )
    model.add_outputs('HC.f')
    model.add_inputs(
        ar_0=AR("HC.f", order=0), taobs_0=AR("Taobs", order=0),
        ar_1=AR("HC.f", order=1), ar_12=AR("HC.f", order=11), ar_2=AR("HC.f", order=2),
        ar_3=AR("HC.f", order=3), ar_4=AR("HC.f", order=4), ar_5=AR("HC.f", order=5),
        ar_48=AR("HC.f", order=47), ar_36=AR("HC.f", order=35),
        ar_24=AR("HC.f", order=23), W_f_0=AR("W.f", order=0),
        GR_f_0=AR("GR.f", order=0)
    )
    model.update_params(**svr_hyperparams)
    preds = []
    for i in range(end_index - h):
        batch = data_raw.fc.subset(start_index=i, end_index=i+1)
        future = data_raw.fc.subset('HC.f', kseq=('NA',), start_index=i+h, end_index=i+h+1)
        if future.empty or future[('HC.f','NA')].isnull().all():
            continue
        batch[('HC.f','NA')] = future.iloc[0][('HC.f','NA')]
        out = model.update(batch, n_keep=max_lag)
        df_out = out[0] if isinstance(out, tuple) else out
        if isinstance(df_out, pd.DataFrame) and not df_out.empty:
            val = df_out.iloc[0][('HC.f', h)]
            ts = pd.Series([val], index=[data_raw.index[i+h]], name=('HC.f', h))
            preds.append(ts)
    if preds:
        all_preds.append(pd.concat(preds))

    elapsed_h = time.perf_counter() - t0_h
    print(f"done in {elapsed_h:.2f} s")

t_total = time.perf_counter() - t0_global
print(f"\nAll {len(horizons_to_predict)} horizons done in {t_total:.2f} seconds")

# 5. Combine predictions
if all_preds:
    combined_pred = pd.concat(all_preds, axis=1)
    if not isinstance(combined_pred.columns, pd.MultiIndex):
        combined_pred.columns = pd.MultiIndex.from_tuples(
            [s.name for s in all_preds], names=['Variable','Horizon']
        )
    combined_pred = combined_pred.loc[:, ~combined_pred.columns.duplicated()]
    print("All predictions combined.")
else:
    combined_pred = pd.DataFrame(); print("No predictions generated.")

# 6. Prepare observations & scaling
scaled_obs = data_raw.fc.subset('HC.f', kseq=('NA',), end_index=end_index + max(horizons_to_predict, default=0))
orig_min = pd.read_csv("veks_svr_min.csv").T.squeeze()
orig_max = pd.read_csv("veks_svr_max.csv").T.squeeze()

def unscale(x, mn, mx):
    return x * (mx - mn) + mn

# 7. Save raw CSVs (matching RLS format)
plot_dir = "../7_results/Case_2/3_svr/res1"
os.makedirs(plot_dir, exist_ok=True)

# --- 7a. Predictions ---
preds_unscaled = unscale(combined_pred, orig_min['HC.f'], orig_max['HC.f'])
# give the MultiIndex the same level‐names as in the RLS code
preds_unscaled.columns.names = ['Variable', 'Horizon']
preds_unscaled.to_csv(os.path.join(plot_dir, 'svr_res1h.csv'),
                      index_label=['', ''])  # keep the left index‐label blank

print(f"Predictions saved → {os.path.join(plot_dir, 'svr_res1h.csv')}")

# --- 7b. Observations ---
obs_unscaled = (
    unscale(scaled_obs[('HC.f','NA')], orig_min['HC.f'], orig_max['HC.f'])
    .to_frame()
)
# it's just one column, but we still want the same two‐row header format:
obs_unscaled.columns = pd.MultiIndex.from_tuples(
    [('HC.f', 'NA')], names=['Variable','Horizon']
)
obs_unscaled.to_csv(os.path.join(plot_dir, 'observations_HC.csv'),
                    index_label=['', ''])
print(f"Observations saved → {os.path.join(plot_dir, 'observations_HC.csv')}")

# --- 7c. Combined (obs + preds) ---
combined_df = pd.concat([obs_unscaled, preds_unscaled], axis=1)
combined_df.to_csv(os.path.join(plot_dir, 'svr_res1h_with_obs.csv'),
                   index_label=['', ''])
print(f"Combined saved → {os.path.join(plot_dir, 'svr_res1h_with_obs.csv')}")

# 8. Evaluate & plot per-horizon (no summary metrics saved/plotted)
burn_in = 96
for var, h in combined_pred.columns:
    if var != 'HC.f': continue
    print(f"\nEvaluating h={h}...")
    pred_s = combined_pred[(var,h)].dropna()
    if pred_s.empty:
        print("  no predictions"); continue

    y_pred = unscale(pred_s, orig_min['HC.f'], orig_max['HC.f'])
    obs_s  = unscale(scaled_obs[('HC.f','NA')], orig_min['HC.f'], orig_max['HC.f'])
    aligned_pred, aligned_obs = y_pred.align(obs_s, join='inner')
    df = pd.DataFrame({'pred': aligned_pred, 'actual': aligned_obs}).dropna()
    df_post = df.iloc[burn_in:] if burn_in < len(df) else pd.DataFrame()
    if df_post.empty:
        print("  insufficient data after burn-in"); continue

    y_true, y_hat = df_post['actual'], df_post['pred']
    rmse = np.sqrt(((y_true - y_hat)**2).mean())
    mape = mean_absolute_percentage_error(y_true, y_hat) * 100
    r2v  = r2_score(y_true, y_hat)
    print(f"  RMSE={rmse:.4f}, MAPE={mape:.2f}%, R2={r2v:.4f}")

    # --- only save forecast plot for h = 1 ---
    if h == 1:
        fig, ax = plt.subplots(figsize=(16,4))
        ax.plot(y_true.index, y_true, label='Actual')
        ax.plot(y_hat.index, y_hat, label=f'Pred h={h}')
        ax.axvline(x=y_true.index[0], color='gray', linestyle='--')
        ax.set_title(f'SVR Forecast vs Actual (h={h})')
        ax.legend(); ax.grid(True)
        fig.savefig(os.path.join(plot_dir, f'svr_forecast_{h}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)

    # Diagnostics for h=1
    if h == 1 and len(y_true) > 10:
        res = y_true - y_hat
        fig, axes = plt.subplots(2,2, figsize=(16,8))
        plot_acf(res, lags=min(24, len(res)-1), ax=axes[0,0])
        axes[0,0].set_title('ACF')
        plot_pacf(res, lags=min(24, len(res)//2-1), ax=axes[0,1])
        axes[0,1].set_title('PACF')
        qqplot(res, line='s', ax=axes[1,0])
        axes[1,0].set_title('Q-Q')
        axes[1,1].plot(res.index, res, marker='o', linestyle='-')
        axes[1,1].set_title('Residuals')
        plt.tight_layout()
        fig.savefig(os.path.join(plot_dir, 'svr_diagnostics_1_step.png'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)

    

print("\nProcessing complete.")