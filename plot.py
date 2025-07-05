import pandas as pd
import plotly.graph_objs as go
import numpy as np
import os

# Create Results directory if it doesn't exist
os.makedirs("Results", exist_ok=True)

# Load minute-level energy data
eact_df = pd.read_csv("data/complete_eact.csv", parse_dates=["time"])
eact_df['hour'] = eact_df['time'].dt.floor('H')
eact_df['date'] = eact_df['time'].dt.date

# Load missing data information
missing_df = pd.read_csv("data/missing_data_info.csv", parse_dates=["gap_start_time", "gap_end_time"])

def create_colored_line_segments(x_data, y_data, color_condition, normal_color, missing_color):
    """Create line segments with different colors based on condition"""
    segments = []
    current_segment_x = []
    current_segment_y = []
    current_is_missing = None

    for x, y, is_missing in zip(x_data, y_data, color_condition):
        if current_is_missing is None:
            current_is_missing = is_missing
            current_segment_x = [x]
            current_segment_y = [y]
        elif current_is_missing == is_missing:
            current_segment_x.append(x)
            current_segment_y.append(y)
        else:
            segments.append({
                'x': current_segment_x.copy(),
                'y': current_segment_y.copy(),
                'color': missing_color if current_is_missing else normal_color,
                'is_missing': current_is_missing
            })
            current_segment_x = [current_segment_x[-1], x]
            current_segment_y = [current_segment_y[-1], y]
            current_is_missing = is_missing

    if current_segment_x:
        segments.append({
            'x': current_segment_x,
            'y': current_segment_y,
            'color': missing_color if current_is_missing else normal_color,
            'is_missing': current_is_missing
        })

    return segments

# ------------------------------------------------------------
# DAILY AGGREGATION + Large Gaps (≥ 1440 mins)
# ------------------------------------------------------------

daily_avg = eact_df.groupby(['virtual_meter_id', 'date'])['Eact'].mean().reset_index()
large_gaps = missing_df[missing_df['gap_length_minutes'] >= 1440]

# Build set of missing days
missing_days = set()
for _, row in large_gaps.iterrows():
    gap_days = pd.date_range(start=row['gap_start_time'].date(), end=row['gap_end_time'].date(), freq='D')
    missing_days.update([d.date() for d in gap_days])

# Daily plots for each VM
for vm_id in daily_avg['virtual_meter_id'].unique():
    vm_data = daily_avg[daily_avg['virtual_meter_id'] == vm_id].sort_values('date').reset_index(drop=True)
    is_missing_array = [date in missing_days for date in vm_data['date']]

    segments = create_colored_line_segments(
        vm_data['date'], 
        vm_data['Eact'], 
        is_missing_array, 
        'blue', 
        'red'
    )

    fig = go.Figure()
    for segment in segments:
        fig.add_trace(go.Scatter(
            x=segment['x'],
            y=segment['y'],
            mode='lines',
            line=dict(color=segment['color'], width=2),
            name=f"VM {vm_id}",
            showlegend=False,
            hovertemplate=f'<b>VM {vm_id}</b><br>Date: %{{x}}<br>Eact: %{{y:.4f}}<extra></extra>'
        ))

    fig.update_layout(
        title=f'Daily Average Eact - VM {vm_id} (Blue: Normal, Red: Missing)',
        xaxis_title='Date',
        yaxis_title='Average Eact',
        hovermode='closest',
        template='plotly_white'
    )

    output_path = f"Results/eact_daily_VM_{vm_id}.html"
    fig.write_html(output_path)
    print(f"✅ Saved daily plot for VM {vm_id} → {output_path}")

# ------------------------------------------------------------
# HOURLY AGGREGATION + Medium Gaps (60 to <1440 mins)
# ------------------------------------------------------------

hourly_avg = eact_df.groupby(['virtual_meter_id', 'hour'])['Eact'].mean().reset_index()
medium_gaps = missing_df[(missing_df['gap_length_minutes'] >= 60) & (missing_df['gap_length_minutes'] < 1440)]

# Build set of missing hours
missing_hours = set()
for _, row in medium_gaps.iterrows():
    start_hour = row['gap_start_time'].floor('H')
    end_hour = row['gap_end_time'].floor('H')
    gap_hours = pd.date_range(start=start_hour, end=end_hour, freq='H')
    missing_hours.update(gap_hours)

    if row['gap_end_time'] > end_hour:
        missing_hours.add(end_hour + pd.Timedelta(hours=1))

# Hourly plots for each VM
for vm_id in hourly_avg['virtual_meter_id'].unique():
    vm_data = hourly_avg[hourly_avg['virtual_meter_id'] == vm_id].sort_values('hour').reset_index(drop=True)
    is_missing_array = [hour in missing_hours for hour in vm_data['hour']]

    segments = create_colored_line_segments(
        vm_data['hour'], 
        vm_data['Eact'], 
        is_missing_array, 
        'green', 
        'orange'
    )

    fig = go.Figure()
    for segment in segments:
        fig.add_trace(go.Scatter(
            x=segment['x'],
            y=segment['y'],
            mode='lines',
            line=dict(color=segment['color'], width=2),
            name=f"VM {vm_id}",
            showlegend=False,
            hovertemplate=f'<b>VM {vm_id}</b><br>Hour: %{{x}}<br>Eact: %{{y:.4f}}<extra></extra>'
        ))

    fig.update_layout(
        title=f'Hourly Average Eact - VM {vm_id} (Green: Normal, Orange: Missing)',
        xaxis_title='Hour',
        yaxis_title='Average Eact',
        hovermode='closest',
        template='plotly_white'
    )

    output_path = f"Results/eact_hourly_VM_{vm_id}.html"
    fig.write_html(output_path)
    print(f"✅ Saved hourly plot for VM {vm_id} → {output_path}")
