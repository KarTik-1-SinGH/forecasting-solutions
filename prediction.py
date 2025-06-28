import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

# Create Results folder if not exist
os.makedirs("Results", exist_ok=True)

# Custom metrics functions
def calculate_mape(actual, predicted):
    mask = actual != 0
    if mask.sum() == 0:
        return float('inf')
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

def calculate_smape(actual, predicted):
    denominator = (np.abs(actual) + np.abs(predicted)) / 2
    mask = denominator != 0
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs(actual[mask] - predicted[mask]) / denominator[mask]) * 100

def calculate_all_metrics(actual, predicted, model_name):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = calculate_mape(actual, predicted)
    smape = calculate_smape(actual, predicted)
    return {
        'Model': model_name,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'SMAPE': smape,
        'MSE': mse
    }

def main():
    # Load and preprocess minute-level data
    data_path = os.path.join("data", "testing_data.csv")
    df = pd.read_csv(data_path, parse_dates=['time'])

    # Rename columns for consistency
    df = df.rename(columns={"time": "timestamp", "Eact": "eact"})

    # Resample to hourly average
    df.set_index("timestamp", inplace=True)
    df_hourly = df.resample("H").mean().reset_index()
    df_hourly = df_hourly.rename(columns={"timestamp": "hour", "eact": "avg_eact"})

    # Extract values
    values = df_hourly['avg_eact'].values
    hours_per_day = 24
    context_length = 4 * hours_per_day  # 96 hours (4 days)
    prediction_length = hours_per_day   # 24 hours (1 day)

    if len(values) < context_length + prediction_length:
        print("❌ Not enough data: Need at least 120 hours (4 days context + 1 day prediction)")
        return

    ctx = torch.tensor(values[:context_length], dtype=torch.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # Load Chronos model
    try:
        chronos_pipeline = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-large",
            device_map=device,
            torch_dtype=dtype,
        )
        print("✓ Chronos model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading Chronos model: {e}")
        return

    # Predict
    try:
        forecast = chronos_pipeline.predict(ctx, prediction_length)
        chronos_forecast = np.median(forecast[0].cpu().numpy(), axis=0)
        print("✓ Chronos prediction completed")
    except Exception as e:
        print(f"✗ Chronos prediction failed: {e}")
        return

    # Actual values and time
    actual = values[context_length:context_length + prediction_length]
    times = df_hourly['hour'][context_length:context_length + prediction_length]

    # Calculate metrics
    metrics = calculate_all_metrics(actual, chronos_forecast, "Chronos T5-Large")
    print("\n📊 Metrics:")
    for k, v in metrics.items():
        if k != 'Model':
            print(f"  {k}: {v:.4f}")

    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv("Results/chronos_metrics.csv", index=False)

    # =============================
    # 🌟 Enhanced Visualization 🌟
    # =============================

    # Day-wise ticks
    full_range = context_length + prediction_length
    full_times = df_hourly['hour'][:full_range]
    full_values = values[:full_range]

    num_days = full_range // hours_per_day
    day_labels = [f"Day {i+1}" for i in range(num_days)]
    day_ticks = [i * hours_per_day for i in range(num_days)]

    # Plot: Context + Actual + Predicted
    plt.figure(figsize=(12, 6))
    plt.plot(range(context_length), values[:context_length], label="Context (History)", color="#1f77b4", linewidth=2)
    plt.plot(range(context_length, context_length + prediction_length), actual, label="Actual", color="#2ca02c", marker='o', linewidth=2)
    plt.plot(range(context_length, context_length + prediction_length), chronos_forecast, label="Predicted", color="#ff7f0e", marker='x', linewidth=2)

    plt.title("Chronos Forecasting: Context, Actual vs Predicted", fontsize=16, fontweight='bold')
    plt.xlabel("Timeline (Day-wise)", fontsize=13)
    plt.ylabel("avg_eact", fontsize=13)
    plt.xticks(day_ticks, day_labels, rotation=0)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("Results/chronos_prediction_plot.png", dpi=300)
    plt.close()

    # Plot: Error (Actual - Predicted)
    errors = actual - chronos_forecast
    plt.figure(figsize=(12, 4))
    plt.plot(range(prediction_length), errors, label="Prediction Error", color='crimson', marker='.', linewidth=1.5)
    plt.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.6)
    plt.title("Forecast Error (Actual - Predicted)", fontsize=15, fontweight='bold')
    plt.xlabel("Prediction Horizon (Hourly - Day 5)", fontsize=13)
    plt.ylabel("Error", fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Results/chronos_error_plot.png", dpi=300)
    plt.close()

    print("\n📁 Plots and metrics saved in 'Results/' folder.")

if __name__ == "__main__":
    main()
