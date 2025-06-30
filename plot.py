import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.offline import plot
import os
import argparse

# ----------------------------
# Load and tag data
# ----------------------------
def load_and_label_data(eact_path, missing_info_path):
    eact_df = pd.read_csv(eact_path, parse_dates=['time'])
    missing_df = pd.read_csv(missing_info_path, parse_dates=['gap_start_time', 'gap_end_time'])

    # Mark each row as 'predicted' or 'actual' based on gap periods
    eact_df['source'] = 'actual'
    for _, row in missing_df.iterrows():
        mask = (eact_df['time'] >= row['gap_start_time']) & (eact_df['time'] <= row['gap_end_time'])
        eact_df.loc[mask, 'source'] = 'predicted'

    return eact_df

# ----------------------------
# Resample to daily granularity
# ----------------------------
def aggregate_daily(df):
    df.set_index('time', inplace=True)

    # Separate actual and predicted data
    actual_df = df[df['source'] == 'actual'][['Eact']]
    predicted_df = df[df['source'] == 'predicted'][['Eact']]

    # Resample both individually
    daily_actual = actual_df.resample('D').mean().reset_index()
    daily_actual['source'] = 'actual'

    daily_pred = predicted_df.resample('D').mean().reset_index()
    daily_pred['source'] = 'predicted'

    # Combine, prioritize actual where both exist
    combined = pd.concat([daily_pred, daily_actual])  # actual last so it overwrites
    combined = combined.sort_values('time').drop_duplicates(subset='time', keep='last')

    return combined


# ----------------------------
# Plot cleanly
# ----------------------------
def plot_data(df, output_html):
    fig = go.Figure()

    for src, color, dash in [('actual', 'blue', 'solid'), ('predicted', 'orange', 'dot')]:
        seg = df.copy()
        seg.loc[seg['source'] != src, 'Eact'] = None  # mask other values
        fig.add_trace(go.Scatter(
            x=seg['time'],
            y=seg['Eact'],
            mode='lines',
            name=src.capitalize(),
            line=dict(color=color, width=2, dash=dash),
            connectgaps=False
        ))

    fig.update_layout(
        title="Daily Aggregated Eact with Predicted Segments Highlighted",
        xaxis_title="Date",
        yaxis_title="Eact (kWh)",
        hovermode='x unified',
        template="plotly_white"
    )

    plot(fig, filename=output_html, auto_open=True)

# ----------------------------
# Main
# ----------------------------
def main():
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, 'data')
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    eact_path = os.path.join(data_dir, 'complete_eact.csv')
    missing_path = os.path.join(data_dir, 'missing_data_info.csv')
    output_html = os.path.join(results_dir, 'eact_plot.html')

    full_df = load_and_label_data(eact_path, missing_path)
    daily_df = aggregate_daily(full_df)
    plot_data(daily_df, output_html)

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    main()
