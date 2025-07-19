# process3.py - Modified for true online learning with OnlineSVR
# Sequential processing of data points to maintain continuity

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.gofplots import qqplot
import matplotlib.pyplot as plt
import pandas as pd
import py_online_forecast.core_main_fix_svr3 as c
from py_online_forecast.core_main_fix_svr3 import *
import py_online_forecast.hierarchies as h
from py_online_forecast.hierarchies import *
np.random.seed(42)

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
data_raw = pd.read_csv("Project/PyOnlineForecast-main/veks.csv")

#%%
# Keep columns HC.f, Ta.f
data = data_raw[["HC.f", "Ta.f", "ds.hh", "ds.tod", "ds.dow", "W.f", "GR.f"]].copy()
data.insert(0, "t", list(range(len(data))))
data['t'] = pd.to_datetime(data['t'], unit='h', origin='2024-01-01')
data.rename(columns={"Ta.f": "Taobs"}, inplace=True)

# --- Store original data range for unscaling ---
# Store before creating synthetic data and scaling
scaling_columns = ["HC.f", "Taobs"] # Columns that will be scaled and need unscaling later
original_min = data[scaling_columns].min()
original_max = data[scaling_columns].max()

# --- Create synthetic future data (BEFORE SCALING) - Method 1 ---
forecast_horizons = list(range(1, 173))

# Create all forecast columns in a dictionary first
forecast_data = {}
for horizon in forecast_horizons:
    forecast_column = f'Ta.k{horizon}'
    forecast_data[forecast_column] = data['Taobs'].shift(-horizon) + np.random.normal(0, 2, len(data))

# Add all columns at once
data = pd.concat([data, pd.DataFrame(forecast_data)], axis=1)

# Apply rolling mean to synthetic future data
for horizon in forecast_horizons:
    forecast_column = f'Ta.k{horizon}'
    data[forecast_column] = data[forecast_column].rolling(window=6, min_periods=1).mean()

# Drop rows with NaNs introduced by shifting
data.dropna(inplace=True)

# --- Min-Max Scaling (AFTER creating synthetic data) ---
# Select all numeric columns except 't' for scaling
cols_to_scale = ["HC.f", "Taobs", 'W.f', 'GR.f'] # Only scale these
data[cols_to_scale] = (data[cols_to_scale] - data[cols_to_scale].min()) / (data[cols_to_scale].max() - data[cols_to_scale].min())

data.fc.convert(separator=".k")

#%% Split data into training and testing segments for evaluation (not for sequential processing)
training_end_idx = 120
train_indices = range(training_end_idx)  # First 120 data points for training
test_indices = range(training_end_idx, 144)  # Data points 120-480 for testing

#%% Initialize a smaller training dataset for parameter optimization
tuning_data = data.fc.subset(end_index=training_end_idx)  # First part for tuning













#%% Initialize OnlineSVR Model for parameter optimization with AR terms
OnlineSVR.params = ['C', 'epsilon', 'gamma', 'threshold']

model1 = ForecastModel(OnlineSVR, predictor_init_params={'kernel': 'rbf'}, estimate_variance=True, kseq=(1,))
model1.add_outputs("HC.f")
model1.add_inputs(
    "HC.f", "Taobs", "W.f", "GR.f",
        #"HC.f", "Taobs", #"W.f", "GR.f",
        ar_0=AR("HC.f", order=0), # AR(0) for the target variable
        taobs_0=AR("Taobs", order=0), # AR(0) for Taobs
        ar_1=AR("HC.f", order=1), ar_12=AR("HC.f", order=11),
        ar_24=AR("HC.f", order=23), #Taobs_1=AR("Taobs", order=1), #Taobs_2=AR("Taobs", order=2),
        #W_f_1=AR("W.f", order=1),
        #GR_f_1=AR("GR.f", order=1),
           
)
model1.update_params(C=5.309, epsilon=0.00523375, gamma=0.266, threshold=500)

# Set bounds for optimization
model1.set_regprm(target="predictor", C=(1, 7), epsilon=(0.001, 0.01), gamma=(0.1, 0.6))
# LowPass bounds (these names match the input labels)

#%% Optimize model parameters on tuning data
result1, optim_result1, temp_values1, _ = model1.optim_pso(
    tuning_data,
    rmse,
    set_params=True,
    store_intermediate=True,
    n_particles=50,  
    max_iter=50,
    burn_in=24,  # Set burn-in to match highest AR lag
    seed=42
)

# Get optimized parameters
optimized_params = model1.get_params()["predictor"]
print("Optimized parameters:", optimized_params)

#%% Now initialize the FINAL model for true sequential online learning
full_model = ForecastModel(OnlineSVR, predictor_init_params={'kernel': 'rbf'}, estimate_variance=True, kseq=(1,))
full_model.add_outputs("HC.f")
full_model.add_inputs(
    "HC.f", "Taobs", "W.f", "GR.f",
        #"HC.f", "Taobs", #"W.f", "GR.f",
        ar_0=AR("HC.f", order=0), # AR(0) for the target variable
        taobs_0=AR("Taobs", order=0), # AR(0) for Taobs
        ar_1=AR("HC.f", order=1), ar_12=AR("HC.f", order=11),
        ar_24=AR("HC.f", order=23), #Taobs_1=AR("Taobs", order=1), #Taobs_2=AR("Taobs", order=2),
        #W_f_1=AR("W.f", order=1),
        #GR_f_1=AR("GR.f", order=1),   
)

# Apply optimized parameters
full_model.update_params(
    C=optimized_params["C"], 
    epsilon=optimized_params["epsilon"],
    gamma=optimized_params["gamma"],
    threshold=500  # Keep threshold as is
)

#%% TRUE ONLINE LEARNING - Process all data sequentially through a single model
# Preallocate containers to store all predictions and actual values
all_predictions = []
all_observations = []

max_lag = 24
# Define batch size for processing (for efficiency)
batch_size = 1 # Process this many data points at once

# Process data in small batches, but in sequential order
data_length = 240  # Total data points to process
num_batches = (data_length + batch_size - 1) // batch_size

for batch_idx in range(num_batches):
    # Calculate start and end indices for this batch
    start_idx = batch_idx * batch_size
    end_idx = min(start_idx + batch_size, data_length)
    
    # Get current batch of data
    current_batch = data.fc.subset(start_index=start_idx, 
                                  end_index=end_idx)
    
    # Update model with this batch and get predictions
    batch_predictions = full_model.update(current_batch, n_keep=max_lag)
    
    # Store predictions (ensuring we get a DataFrame)
    if isinstance(batch_predictions, ForecastTuple):
        batch_pred_df = batch_predictions[0]  # Extract DataFrame
    else:
        batch_pred_df = batch_predictions  # Already a DataFrame
    
    all_predictions.append(batch_pred_df)
    all_observations.append(current_batch)
    
    # Print progress
    if batch_idx % 20 == 0 or batch_idx == num_batches - 1:
        print(f"Processed batch {batch_idx+1}/{num_batches} (data points {start_idx}-{end_idx})")

# Combine all predictions and observations
combined_predictions = pd.concat(all_predictions)
combined_observations = pd.concat(all_observations)

# Separate into training and testing periods for evaluation
train_predictions = combined_predictions.iloc[list(train_indices)]
test_predictions = combined_predictions.iloc[list(test_indices)]
train_observations = combined_observations.iloc[list(train_indices)]
test_observations = combined_observations.iloc[list(test_indices)]

#%% Calculate RMSE for training and testing periods
output_var = "HC.f"

# Extract relevant prediction and observation series
train_pred_series = train_predictions[output_var][1]  # Horizon 1
train_obs_series = train_observations[output_var]['NA']
test_pred_series = test_predictions[output_var][1]  # Horizon 1  
test_obs_series = test_observations[output_var]['NA']

# Filter out NaN values
valid_idx_train = ~(np.isnan(train_pred_series) | np.isnan(train_obs_series))
valid_idx_test = ~(np.isnan(test_pred_series) | np.isnan(test_obs_series))

# Calculate RMSE
if valid_idx_train.any():
    train_rmse = np.sqrt(np.mean((train_pred_series[valid_idx_train] - train_obs_series[valid_idx_train])**2))
    print(f"Training RMSE: {train_rmse:.4f}")

if valid_idx_test.any():
    test_rmse = np.sqrt(np.mean((test_pred_series[valid_idx_test] - test_obs_series[valid_idx_test])**2))
    print(f"Test RMSE: {test_rmse:.4f}")

#%% Unscale predictions and observations for plotting
# Function to unscale data
def unscale_data(scaled_data, min_value, max_value, column_name):
    unscaled_data = scaled_data.copy()
    value_range = max_value - min_value
    
    if value_range != 0:
        for column in unscaled_data.columns:
            if column[0] == column_name:
                unscaled_data[column] = unscaled_data[column] * value_range + min_value
    
    return unscaled_data

# Unscale predictions
train_preds_unscaled = unscale_data(train_predictions, 
                                    original_min[output_var], 
                                    original_max[output_var], 
                                    output_var)

test_preds_unscaled = unscale_data(test_predictions, 
                                  original_min[output_var], 
                                  original_max[output_var], 
                                  output_var)

# Combine predictions
combined_preds_unscaled = pd.concat([train_preds_unscaled, test_preds_unscaled])

# Unscale observations
train_obs_unscaled = train_observations.copy()
test_obs_unscaled = test_observations.copy()

obs_col_to_unscale = (output_var, 'NA')
value_range_obs = original_max[output_var] - original_min[output_var]

if value_range_obs != 0:
    # Unscale training observations
    if obs_col_to_unscale in train_obs_unscaled.columns:
        train_obs_unscaled[obs_col_to_unscale] = train_obs_unscaled[obs_col_to_unscale] * value_range_obs + original_min[output_var]
    
    # Unscale test observations
    if obs_col_to_unscale in test_obs_unscaled.columns:
        test_obs_unscaled[obs_col_to_unscale] = test_obs_unscaled[obs_col_to_unscale] * value_range_obs + original_min[output_var]

# Combine observations
combined_obs_unscaled = pd.concat([train_obs_unscaled, test_obs_unscaled])

#%% Plot results with vertical line separating training and testing periods
# Create figure
fig, ax = plt.subplots(figsize=(19, 7))

# Plot observations
ax.plot(combined_obs_unscaled.index, combined_obs_unscaled[output_var]['NA'], 'k-', label='Actual Values', linewidth=2)

# Plot predictions
ax.plot(combined_preds_unscaled.index, combined_preds_unscaled[output_var][1], 'r-', label='Predictions', linewidth=1.5)

# Add vertical line to separate training and test periods
ax.axvline(x=training_end_idx, color='blue', linestyle='--', linewidth=1.5, 
           label='Train-Test Split')

# Shade the training and testing regions
ax.axvspan(0, training_end_idx, alpha=0.2, color='green', label='Training Region')
ax.axvspan(training_end_idx, 120, alpha=0.2, color='red', label='Testing Region')

# Add text annotations for RMSE
ax.text(20, ax.get_ylim()[1]*0.9, f'Training RMSE: {train_rmse:.4f}', 
        fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
ax.text(training_end_idx + 30, ax.get_ylim()[1]*0.9, f'Testing RMSE: {test_rmse:.4f}', 
        fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

# Add labels and legend
ax.set_xlabel('Time Step', fontsize=12)
ax.set_ylabel(f'{output_var} (Unscaled)', fontsize=12)
ax.set_title('Online SVR', fontsize=14)
ax.legend(loc='lower right')
ax.grid(True)

# Ensure the plots directory exists
plot_dir = "Vault/PyOnlineForecast-main/3_svr_base_forecasts/plots"
os.makedirs(plot_dir, exist_ok=True)

# Save the figure
plot_path = os.path.join(plot_dir, "opt.png")
fig.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {plot_path}")

# Print summary
print("\nFORECASTING SUMMARY:")
print(f"Training period: RRMSE = {train_rmse:.4f}")
print(f"Testing period: RRMSE = {test_rmse:.4f}")
print(f"Model parameters: C={optimized_params['C'][0]:.6f}, " +
      f"epsilon={optimized_params['epsilon'][0]:.6f}, gamma={optimized_params['gamma'][0]:.6f}")