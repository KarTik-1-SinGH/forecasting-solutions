import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.offline import plot
import os
import argparse

# ----------------------------
# Load and preprocess the data
# ----------------------------
def load_data(eact_path, missing_info_path):
    eact_df = pd.read_csv(eact_path, parse_dates=['time'])
    missing_df = pd.read_csv(missing_info_path, parse_dates=['gap_start_time', 'gap_end_time'])
    eact_df.set_index('time', inplace=True)
    return eact_df, missing_df

# ----------------------------
# Fill missing periods with predicted values (NaN)
# ----------------------------
def insert_missing_periods(eact_df, missing_df, freq='1min'):
    predicted_rows = []

    for _, row in missing_df.iterrows():
        time_range = pd.date_range(start=row['gap_start_time'], end=row['gap_end_time'], freq=freq)
        for t in time_range:
            if t not in eact_df.index:
                predicted_rows.append({
                    'time': t,
                    'Eact': np.nan,
                    'source': 'predicted'
                })

    predicted_df = pd.DataFrame(predicted_rows).set_index('time')
    eact_df = eact_df.copy()
    eact_df['source'] = 'actual'
    combined = pd.concat([eact_df, predicted_df])
    combined = combined[~combined.index.duplicated()].sort_index()
    
    # Linear interpolation for predicted values
    combined['Eact'] = combined['Eact'].interpolate(method='time')
    combined['source'] = combined['source'].fillna('predicted')
    
    return combined

# ----------------------------
# Resample data to selected granularity
# ----------------------------
def resample_data(df, granularity):
    rule_map = {
        'minute': '1min',
        '15min': '15min',
        '30min': '30min',
        'hourly': 'H',
        'daily': 'D',
        'weekly': 'W'
    }

    if granularity not in rule_map:
        raise ValueError(f"Invalid granularity: {granularity}. Choose from {list(rule_map.keys())}")
    
    rule = rule_map[granularity]

    # Resample only numeric values (Eact), grouped by source
    resampled = (
        df[['Eact']].groupby(df['source'])
        .resample(rule)
        .mean()
        .reset_index()
    )

    return resampled

# ----------------------------
# Plot using Plotly
# ----------------------------
def plot_data(df, output_html):
    fig = go.Figure()

    for source in ['actual', 'predicted']:
        sub_df = df[df['source'] == source]
        fig.add_trace(go.Scatter(
            x=sub_df['time'],
            y=sub_df['Eact'],
            mode='lines+markers',
            name=source,
            line=dict(shape='spline'),
            opacity=0.9 if source == 'actual' else 0.6
        ))

    fig.update_layout(
        title="Eact with Predicted Data Highlighted",
        xaxis_title="Time",
        yaxis_title="Eact (kWh)",
        hovermode='x unified',
        template="plotly_white"
    )

    plot(fig, filename=output_html, auto_open=True)

# ----------------------------
# Main
# ----------------------------
def main(granularity='daily'):
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, 'data')
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    eact_csv = os.path.join(data_dir, 'complete_eact.csv')
    missing_csv = os.path.join(data_dir, 'missing_data_info.csv')
    output_html = os.path.join(results_dir, 'eact_plot.html')

    eact_df, missing_df = load_data(eact_csv, missing_csv)
    combined_df = insert_missing_periods(eact_df, missing_df)
    resampled_df = resample_data(combined_df, granularity)
    plot_data(resampled_df, output_html)

# ----------------------------
# CLI interface
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot Eact with interpolated predicted values.')
    parser.add_argument('--granularity', type=str, default='daily',
                        choices=['minute', '15min', '30min', 'hourly', 'daily', 'weekly'],
                        help='Granularity of the data (default: daily)')
    args = parser.parse_args()

    main(args.granularity)
