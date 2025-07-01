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
    
    for i, (x, y, is_missing) in enumerate(zip(x_data, y_data, color_condition)):
        if current_is_missing is None:
            # First point
            current_is_missing = is_missing
            current_segment_x = [x]
            current_segment_y = [y]
        elif current_is_missing == is_missing:
            # Same type, continue segment
            current_segment_x.append(x)
            current_segment_y.append(y)
        else:
            # Color change, finish current segment and start new one
            segments.append({
                'x': current_segment_x.copy(),
                'y': current_segment_y.copy(),
                'color': missing_color if current_is_missing else normal_color,
                'is_missing': current_is_missing
            })
            # Start new segment with previous point to ensure continuity
            current_segment_x = [current_segment_x[-1], x] if current_segment_x else [x]
            current_segment_y = [current_segment_y[-1], y] if current_segment_y else [y]
            current_is_missing = is_missing
    
    # Add final segment
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

# Aggregate to daily average
daily_avg = eact_df.groupby(['virtual_meter_id', 'date'])['Eact'].mean().reset_index()

# Filter gaps ≥ 1440 min (1 day)
large_gaps = missing_df[missing_df['gap_length_minutes'] >= 1440]

# Build set of missing days
missing_days = set()
for _, row in large_gaps.iterrows():
    gap_days = pd.date_range(start=row['gap_start_time'].date(), end=row['gap_end_time'].date(), freq='D')
    missing_days.update([d.date() for d in gap_days])

# Plot daily with single line per meter
fig_daily = go.Figure()

for vm_id in daily_avg['virtual_meter_id'].unique():
    vm_data = daily_avg[daily_avg['virtual_meter_id'] == vm_id].sort_values('date').reset_index(drop=True)
    
    # Create color condition array
    is_missing_array = [date in missing_days for date in vm_data['date']]
    
    # Create colored segments
    segments = create_colored_line_segments(
        vm_data['date'], 
        vm_data['Eact'], 
        is_missing_array, 
        'blue', 
        'red'
    )
    
    # Add each segment as a separate trace
    legend_added = False
    for i, segment in enumerate(segments):
        show_legend = not legend_added and not segment['is_missing']
        fig_daily.add_trace(go.Scatter(
            x=segment['x'],
            y=segment['y'],
            mode='lines',
            line=dict(color=segment['color'], width=2),
            name=f"VM {vm_id}",
            showlegend=show_legend,
            legendgroup=f"vm_{vm_id}",
            hovertemplate=f'<b>VM {vm_id}</b><br>Date: %{{x}}<br>Eact: %{{y:.4f}}<extra></extra>'
        ))
        if show_legend:
            legend_added = True

fig_daily.update_layout(
    title='Daily Average Eact (Blue: Normal, Red: Missing Data)',
    xaxis_title='Date',
    yaxis_title='Average Eact',
    hovermode='closest',
    template='plotly_white',
    legend_title="Virtual Meter ID"
)

fig_daily.write_html("Results/eact_daily_plot.html")
print("✅ Daily plot saved to Results/eact_daily_plot.html")

# ------------------------------------------------------------
# HOURLY AGGREGATION + Medium Gaps (60 to <1440 mins)
# ------------------------------------------------------------

# Aggregate to hourly average
hourly_avg = eact_df.groupby(['virtual_meter_id', 'hour'])['Eact'].mean().reset_index()

# Filter gaps between 60 and <1440 min
medium_gaps = missing_df[(missing_df['gap_length_minutes'] >= 60) & (missing_df['gap_length_minutes'] < 1440)]

# Build set of missing hours - include all hours within each gap period
missing_hours = set()
for _, row in medium_gaps.iterrows():
    # Get the start and end hours (floor to hour)
    start_hour = row['gap_start_time'].floor('H')
    end_hour = row['gap_end_time'].floor('H')
    
    # Generate all hours from start to end (inclusive)
    gap_hours = pd.date_range(start=start_hour, end=end_hour, freq='H')
    missing_hours.update(gap_hours)
    
    # Also include the hour containing the end time if it's not already included
    if row['gap_end_time'] > end_hour:
        missing_hours.add(end_hour + pd.Timedelta(hours=1))

# Plot hourly with single line per meter
fig_hourly = go.Figure()

for vm_id in hourly_avg['virtual_meter_id'].unique():
    vm_data = hourly_avg[hourly_avg['virtual_meter_id'] == vm_id].sort_values('hour').reset_index(drop=True)
    
    # Create color condition array
    is_missing_array = [hour in missing_hours for hour in vm_data['hour']]
    
    # Create colored segments
    segments = create_colored_line_segments(
        vm_data['hour'], 
        vm_data['Eact'], 
        is_missing_array, 
        'green', 
        'orange'
    )
    
    # Add each segment as a separate trace
    legend_added = False
    for i, segment in enumerate(segments):
        show_legend = not legend_added and not segment['is_missing']
        fig_hourly.add_trace(go.Scatter(
            x=segment['x'],
            y=segment['y'],
            mode='lines',
            line=dict(color=segment['color'], width=2),
            name=f"VM {vm_id}",
            showlegend=show_legend,
            legendgroup=f"vm_{vm_id}",
            hovertemplate=f'<b>VM {vm_id}</b><br>Hour: %{{x}}<br>Eact: %{{y:.4f}}<extra></extra>'
        ))
        if show_legend:
            legend_added = True

fig_hourly.update_layout(
    title='Hourly Average Eact (Green: Normal, Orange: Missing Data)',
    xaxis_title='Hour',
    yaxis_title='Average Eact',
    hovermode='closest',
    template='plotly_white',
    legend_title="Virtual Meter ID"
)

fig_hourly.write_html("Results/eact_hourly_plot.html")
print("✅ Hourly plot saved to Results/eact_hourly_plot.html")