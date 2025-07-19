# process_arfr_opt_1h.py
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
from py_online_forecast.core_main import * # pyright: ignore[reportWildcardImportFromLibrary]
import py_online_forecast.hierarchies as h
from py_online_forecast.hierarchies import * # pyright: ignore[reportWildcardImportFromLibrary]
from river.tree import HoeffdingAdaptiveTreeRegressor
import itertools # Added for Grid Search
import copy      # Added for deep copying model if needed

np.random.seed(42)


# Define plotting function (original from user)
def plot_predictions(predictions, var_est=None, observations=None, alpha=0.05, t_vec=None,
                     figsize=(20, 12), num_ticks=5):
    # Function implementation unchanged
    n_series = len(predictions.columns)
    if var_est is None:
        lagged_predictions = lag_and_extend(predictions)
    else:
        lagged_predictions, lagged_var = lag_and_extend(predictions, var_est)
        ci = get_normal_confidence_interval(lagged_predictions, lagged_var, alpha)
    fig, axes = plt.subplots(n_series, 1, figsize=figsize, sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    for i, col in enumerate(lagged_predictions):
        if observations is not None:
            if col[0] in observations.columns.get_level_values(0):
                axes[i].plot(observations.fc[col[0]], label=f"Actual {col[0]}", linewidth=2)
            else:
                print(f"Warning: Observation column for {col[0]} not found.")
        axes[i].plot(lagged_predictions[col], label=f"Pred {col[0]}: Lag {col[1]}", linewidth=2)
        if var_est is not None:
            lo = pd.to_numeric(ci[col[0]][col[1]]['lo'], errors='coerce')
            hi = pd.to_numeric(ci[col[0]][col[1]]['hi'], errors='coerce')
            axes[i].fill_between(lagged_predictions[col].index, lo, hi, alpha=0.3)
        axes[i].grid(True)
        axes[i].legend()
    if t_vec is not None:
        num_ticks = min(len(t_vec), num_ticks)
        spacing = len(t_vec) // (num_ticks - 1) if num_ticks > 1 else 1
        labels_indices = t_vec.index[::spacing]
        axes[-1].set_xticks(labels_indices)
        axes[-1].set_xticklabels(labels_indices)
    return fig, axes

# Load and prepare data
data_raw = pd.read_csv("veks.csv")
# Select initial columns, excluding Ta.k for now
data_df = data_raw[["HC.f", "Ta.f", "W.f", "GR.f", "ds.tod", "hh"]].copy()
data_df.insert(0, "t", range(len(data_df)))
data_df['t'] = pd.to_datetime(data_df['t'], unit='h', origin='2024-01-01')
data_df.rename(columns={"Ta.f": "Taobs"}, inplace=True)

target_var_name = "HC.f"

# One-hot encode 'ds.tod'
data_df = pd.get_dummies(data_df, columns=['ds.tod'], prefix='tod', dtype=float)


data_df.dropna(inplace=True) 
data_df.reset_index(drop=True, inplace=True)

# Scale data
cols_scale = [target_var_name, 'Taobs', 'W.f', 'GR.f']

cols_to_scale_present = [col for col in cols_scale if col in data_df.columns]
data_min_val = data_df[cols_to_scale_present].min()
data_max_val = data_df[cols_to_scale_present].max()
data_range_val = data_max_val - data_min_val
data_range_val[data_range_val == 0] = 1

data_df[cols_to_scale_present] = (data_df[cols_to_scale_present] - data_min_val) / data_range_val
data_df.fc.convert(separator=".k") 

hourly_data_full_range = data_df.fc.subset(kseq = tuple(['NA'] + list(range(1,25))))
# CORRECTED: Use the first 312 samples for optimization: 96 for training, 216 for testing.
optimization_data = hourly_data_full_range.iloc[:312].copy()


# Define ARFR's expected parameters
ARFR.params = ['n_trees','max_features','lambda_value', 'burn_in',
               'grace_period','max_depth','split_confidence', 'model_selector_decay', 'leaf_prediction',
               'target_model_horizon']

# --- Grid Search Setup ---
param_grid = {
    'n_trees': [25, 50, 100],
    'max_features': [None, 5, 12, 25], 
    'lambda_value': [6.0],
    'grace_period': [10, 20],
    'max_depth': [10, 20, None],
    'split_confidence': [1e-7, 1e-4],
    'model_selector_decay': [0.8, 0.95],
    'leaf_prediction': ['adaptive'] 
}

default_arfr_params = {
    'n_trees': 100, 'max_features': None, 'lambda_value': 6.0,
    'burn_in': 1, 
    'grace_period': 20, 
    'max_depth': None,
    'split_confidence': 1e-4,
    'model_selector_decay': 0.95, 
    'leaf_prediction': 'adaptive',
    'target_model_horizon': None
}

param_names = list(param_grid.keys())
all_combinations = list(itertools.product(*(param_grid[name] for name in param_names)))

best_score = float('inf')
best_params_combo = None
grid_search_results = []

# CORRECTED: Define train/test split sizes for evaluation
train_size = 96
test_size = 216

print(f"--- Starting Grid Search with {len(all_combinations)} combinations ---")
print(f"Optimization data size: {len(optimization_data)} (Train: {train_size}, Test: {test_size})")

for i, combo in enumerate(all_combinations):
    current_param_values = dict(zip(param_names, combo))
    print(f"\nEvaluating combination {i+1}/{len(all_combinations)}: {current_param_values}")

    model_grid = ForecastModel(
        ARFR,
        predictor_init_params=copy.deepcopy(default_arfr_params),
        estimate_variance=False,
        kseq=(1,) 
    )
    model_grid.add_outputs(target_var_name)
    model_grid.add_inputs(
        ar_1=AR("HC.f", order=1), ar_24=AR("HC.f", order=24),
        Taobs_0=AR("Taobs", order=0), Taobs_1=AR("Taobs", order=1),
        lp_Taobs=LowPass("Taobs", ta=0.9), W_f=LowPass("W.f", ta=0.9),
        W_f_0=AR("W.f", order=0), W_f_1=AR("W.f", order=1),
        GR_fs=LowPass("GR.f", ta=0.95), GR_ff=LowPass("GR.f", ta=0.85),
        GR_f_0=AR("GR.f", order=0), GR_f_1=AR("GR.f", order=1),
        time_of_day_har=FourierSeries("hh", 5),
        tod_1=AR("tod_1", order=0), 
        tod_2=AR("tod_2", order=0), 
        tod_3=AR("tod_3", order=0)
    )

    try:
        model_grid.update_params(**current_param_values)
        
        # CORRECTED: Fit the model on the entire optimization dataset (312 points)
        predictions_scaled = model_grid.fit(optimization_data.copy())

        score = float('inf')

        if predictions_scaled is None or predictions_scaled.empty:
            print("  No predictions generated.")
        else:
            actuals_scaled = optimization_data.fc.subset(target_var_name, kseq=['NA'])
            
            pred_series = None
            if isinstance(predictions_scaled, ForecastTuple):
                pred_df_to_use = predictions_scaled[0] if predictions_scaled and not predictions_scaled[0].empty else pd.DataFrame()
            elif isinstance(predictions_scaled, pd.DataFrame):
                pred_df_to_use = predictions_scaled
            else:
                pred_df_to_use = pd.DataFrame()

            if not pred_df_to_use.empty:
                pred_series = pred_df_to_use.iloc[:, 0] 
                actual_shifted = actuals_scaled.shift(-1) # Y(t+1) is now at index t
                
                common_indices = pred_series.index.intersection(actual_shifted.index)
                
                pred_eval = pred_series.loc[common_indices]
                actual_eval = actual_shifted.loc[common_indices].iloc[:,0]

                # --- CORRECTED: Evaluation Logic ---
                # The first `train_size` points are for training.
                # The rest are for testing.
                if len(common_indices) <= train_size:
                    print(f"  Not enough common data points ({len(common_indices)}) to form a test set (training size: {train_size}).")
                else:
                    pred_test = pred_eval.iloc[train_size:]
                    actual_test = actual_eval.iloc[train_size:]

                    if actual_test.empty or pred_test.empty:
                        print("  Data for evaluation (test set) is empty after slicing.")
                    elif actual_test.isnull().all() or pred_test.isnull().all():
                        print("  Actuals or predictions in the test set are all NaN.")
                    else:
                        res_values = actual_test.values - pred_test.values
                        valid_residuals = res_values[np.isfinite(res_values)]
                        if valid_residuals.size > 0:
                            # CORRECTED: Calculate RMSE directly instead of using an undefined function
                            score = np.sqrt((valid_residuals ** 2).mean())
                        else:
                            print("  All residual values in the test set are NaN or infinite after filtering.")
            else:
                 print("  Prediction DataFrame is empty.")
            
        print(f"  Score (RMSE): {score}")
        grid_search_results.append({'params': current_param_values, 'score': score})

        if score < best_score:
            best_score = score
            best_params_combo = current_param_values
            print(f"  --- New best score found! ---")

    except Exception as e:
        print(f"  Error evaluating combination {current_param_values}: {e}")
        import traceback
        traceback.print_exc()
        grid_search_results.append({'params': current_param_values, 'score': float('inf')})

print("\n--- Grid Search Finished ---")
print("Best Score (RMSE):", best_score)
print("Best Parameters:", best_params_combo)

# Save results to a file
results_df = pd.DataFrame(grid_search_results)
results_df.to_csv("hyper_params/1_hour_arfr/arfr_opt_1h.csv", index=False)
print("\nGrid search results saved to hyper_params/1_hour_arfr/arfr_opt_1h.csv")


if best_params_combo:
    print("\nConfiguring final model with best parameters...")
    final_model_arfr_params = copy.deepcopy(default_arfr_params)
    final_model_arfr_params.update(best_params_combo)

    model1_final = ForecastModel(
        ARFR, 
        predictor_init_params=final_model_arfr_params,
        estimate_variance=False, 
        kseq=(1,)
    )
    model1_final.add_outputs(target_var_name)
    model1_final.add_inputs( 
        ar_0=AR("HC.f", order=0),
        ar_1=AR("HC.f", order=1), ar_24=AR("HC.f", order=24),
        Taobs_0=AR("Taobs", order=0), Taobs_1=AR("Taobs", order=1),
        #lp_Taobs=LowPass("Taobs", ta=0.9), W_f=LowPass("W.f", ta=0.9),
        W_f_0=AR("W.f", order=0), W_f_1=AR("W.f", order=1),
        #GR_fs=LowPass("GR.f", ta=0.95), GR_ff=LowPass("GR.f", ta=0.85),
        GR_f_0=AR("GR.f", order=0), GR_f_1=AR("GR.f", order=1),
        #time_of_day_har=FourierSeries("hh", 5),
        #tod_1=AR("tod_1", order=0),
        #tod_2=AR("tod_2", order=0),
        #tod_3=AR("tod_3", order=0)
    )
    print("Final model configured. You can now fit it on the desired dataset.")

else:
    print("\nNo best parameters found (all combinations might have resulted in errors or NaN scores).")

#%%