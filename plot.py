import os
import math
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output

# Replace with the path to your continuously updating CSV file
CSV_FILE = 'output/Fold_0/Fold_0_cdc_Training_Metrics.log'

app = Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='live-update-graph'),
    # Refresh the plot every 2000 ms (2 seconds)
    dcc.Interval(
        id='interval-component',
        interval=2000,
        n_intervals=0
    )
])

@app.callback(
    Output('live-update-graph', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_graph_live(n):
    if os.path.exists(CSV_FILE):
        # Read CSV with semicolon delimiter
        df = pd.read_csv(CSV_FILE, delimiter=';')
    else:
        return {}

    # Identify training metrics (exclude 'epoch' and any columns starting with 'val_')
    metrics = [col for col in df.columns if col != 'epoch' and not col.startswith('val_')]
    num_metrics = len(metrics)
    if num_metrics == 0:
        return {}

    # Define grid layout: fixed number of columns (e.g., 2) and compute number of rows
    n_cols = 3
    n_rows = math.ceil(num_metrics / n_cols)

    # Create a grid of subplots
    fig = make_subplots(rows=n_rows, cols=n_cols, 
                        subplot_titles=metrics,
                        horizontal_spacing=0.05,
                        vertical_spacing=0.1,
                        shared_xaxes=False)

    # Add traces for each metric into the corresponding subplot grid cell
    for i, metric in enumerate(metrics):
        row = i // n_cols + 1
        col = i % n_cols + 1
        # Add training metric trace
        fig.add_trace(
            go.Scatter(x=df['epoch'], y=df[metric],
                       mode='lines', name=f"Train {metric}"),
            row=row, col=col
        )
        # Add validation metric trace if it exists
        val_metric = f"val_{metric}"
        if val_metric in df.columns:
            fig.add_trace(
                go.Scatter(x=df['epoch'], y=df[val_metric],
                           mode='lines', name=f"Val {metric}"),
                row=row, col=col
            )

    # Adjust overall figure layout size based on the grid dimensions
    fig.update_layout(height=200*n_rows, width=450*n_cols,
                      title_text="T1, 100 epochs, binary CE loss, in-memory augmentation with 16 rotations, 10x10 translations, 0.1 zoom")
    
    fig.update_layout(
    margin=dict(l=20, r=20, t=60, b=20))
    return fig

if __name__ == '__main__':
    app.run(debug=True)
