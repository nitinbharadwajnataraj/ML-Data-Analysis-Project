import dash
from dash import dcc, html
import pandas as pd
import plotly.graph_objects as go
from data_model import data_model
from create_fake_data import create_fake_dataset
from Compute_fit import compute_fit, count_yes_no

# Initialize the Dash app
app = dash.Dash(__name__)

def df_return():
    df = create_fake_dataset()
    return df

def box_plot():
    df = df_return()
    fig = go.Figure()
    for column in df.columns[1:]:  # Exclude 'ID' column
        fig.add_trace(go.Box(y=df[column], name=column))
    fig.update_layout(title='Box Plot (Fake data understanding)', xaxis_title='Parameters', yaxis_title='Values')
    return dcc.Graph(figure=fig)

def bar_chart():
    df_1, count_yes, count_no = count_yes_no()
    data = {
        'Category': ['yes', 'no'],
        'Value': [count_yes, count_no]
    }
    df = pd.DataFrame(data)

    # Create bar chart
    fig = go.Figure(data=[go.Bar(x=df['Category'], y=df['Value'])])
    fig.update_layout(title='Bar Chart', xaxis_title='Category', yaxis_title='Value')
    return dcc.Graph(figure=fig)

app = dash.Dash(__name__, suppress_callback_exceptions=True)

def scatter_plot():
    df_1 = compute_fit()

    # Allow users to choose a shape for the scatter plot
    shape_options = [{'label': shape, 'value': shape} for shape in ["Box", "Cylinder"]]
    shape_dropdown = dcc.Dropdown(
        id='shape-dropdown',
        options=shape_options,
        value='Box'
    )

    # Create scatter plot
    scatter_plot_output = html.Div(id='scatter-plot-output')

    return html.Div([
        html.Label('Select shape for scatter plot:'),
        shape_dropdown,
        scatter_plot_output
    ])

@app.callback(
    dash.dependencies.Output('scatter-plot-output', 'children'),
    [dash.dependencies.Input('shape-dropdown', 'value')]
)
def update_scatter_plot(selected_shape):
    df_1 = compute_fit()

    # Filter columns of df_1 based on the selected shape
    if selected_shape == "Box":
        related_columns = ["box_hole_diameter", "box_hole_depth"]
    elif selected_shape == "Cylinder":
        related_columns = ["cylinder_diameter", "cylinder_height"]
    else:
        related_columns = []

    # Allow users to choose a column for the scatter plot
    column_options = [{'label': column, 'value': column} for column in related_columns]
    column_dropdown = dcc.Dropdown(
        id='column-dropdown',
        options=column_options,
        value=related_columns[0] if related_columns else None
    )

    # Create scatter plot figure
    scatter_plot_figure = dcc.Graph(id='scatter-plot')

    return html.Div([
        html.Label('Select a column for scatter plot:'),
        column_dropdown,
        scatter_plot_figure
    ])

@app.callback(
    dash.dependencies.Output('scatter-plot', 'figure'),
    [dash.dependencies.Input('shape-dropdown', 'value'),
     dash.dependencies.Input('column-dropdown', 'value')]
)
def update_scatter_plot_figure(selected_shape, selected_column):
    df_1 = compute_fit()

    if selected_column:
        # Set color based on 'check' value
        colors = ['#FF7F0E' if val == 'yes' else 'grey' for val in df_1["check"]]

        # Create scatter plot figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_1.index, y=df_1[selected_column], mode='markers', name=selected_column, marker=dict(color=colors)))

        # Add horizontal lines for minimum, maximum, and target values
        min_value = data_model[selected_shape][selected_column]["min_value"]
        max_value = data_model[selected_shape][selected_column]["max_value"]
        target_value = data_model[selected_shape][selected_column]["target_value"]
        fig.add_shape(type="line", x0=df_1.index.min(), y0=min_value, x1=df_1.index.max(), y1=min_value,
                      line=dict(color="red", width=1.5, dash="solid"))
        fig.add_shape(type="line", x0=df_1.index.min(), y0=max_value, x1=df_1.index.max(), y1=max_value,
                      line=dict(color="red", width=1.5, dash="solid"))
        fig.add_shape(type="line", x0=df_1.index.min(), y0=target_value, x1=df_1.index.max(), y1=target_value,
                      line=dict(color="blue", width=1.5, dash="solid"))

        # Add annotations for the lines
        annotation_offset = 0.75  # Adjust this value to position the annotations
        fig.update_layout(annotations=[
            dict(
                x=df_1.index.min(),
                y=min_value + annotation_offset,
                xref="x",
                yref="y",
                text="Minimum Value",
                showarrow=False,
                font=dict(
                    size=12,
                    color="red"
                )
            ),
            dict(
                x=df_1.index.min(),
                y=max_value + annotation_offset,
                xref="x",
                yref="y",
                text="Maximum Value",
                showarrow=False,
                font=dict(
                    size=12,
                    color="red"
                )
            ),
            dict(
                x=df_1.index.min(),
                y=target_value + annotation_offset,
                xref="x",
                yref="y",
                text="Target Value",
                showarrow=False,
                font=dict(
                    size=12,
                    color="blue"
                )
            )
        ])

        fig.update_layout(
            title='Scatter Plot',
            xaxis_title='Index',
            yaxis_title=selected_column
        )
        return fig


# Define the layout of the Dash app
app.layout = html.Div([
    html.H1('Box and Cylinder Analysis', style={'text-align': 'center'}),
    scatter_plot(),
    box_plot(),
    bar_chart()
])

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
