import random
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, dash_table
import functools as ft


# Function to generate fake data
def generate_fake_data(num_rows, process_data_model):
    data = []
    for i in range(num_rows):
        row_data = {"ID": i + 1}  # Basic ID for each item
        for param, values in process_data_model.items():
            target_value = values["target_value"]
            min_value = values["min_value"]
            max_value = values["max_value"]
            factor = values["factor"]
            # Generate fake data based on defined target, min, and max values
            value = np.random.normal(loc=target_value, scale=(max_value - min_value) / 2 * factor)
            if values["min_eq_target"]:
                value = max(value, min_value)
            row_data[param] = value
        data.append(row_data)
    return data


# Function to generate dataset
def generate_dataset(num_rows, data_model):
    dfs = []
    for process, process_data_model in data_model.items():
        df = pd.DataFrame(generate_fake_data(num_rows, process_data_model))
        dfs.append(df)

    # Add a column indicating the paired item for each item
    df_final = ft.reduce(lambda left, right: pd.merge(left, right, on='ID'), dfs)

    paired_item = [i + 1 if i % 2 == 0 else i - 1 for i in range(1, 501)]
    df_final["Paired_Item"] = paired_item
    return df_final


# Preloaded dictionaries containing target, min, max values, and factors for each process
data_model = {
    "Cube": {
        "Cube_Diameter": {"target_value": 30, "min_value": 28, "max_value": 32, "min_eq_target": False,
                          "factor": 1.0},
        "Hole_Depth": {"target_value": 40, "min_value": 37, "max_value": 43, "min_eq_target": False,
                       "factor": 1.0}
    },
    "Cylinder": {
        "Cylinder_Diameter": {"target_value": 40, "min_value": 38.5, "max_value": 41.5, "min_eq_target": False,
                              "factor": 1.0},
        "Cylinder_Height": {"target_value": 30, "min_value": 27.5, "max_value": 32.5, "min_eq_target": False,
                            "factor": 1.0}
    },
    "Alignment": {
        "Joining_Force": {"target_value": 10, "min_value": 9, "max_value": 11, "min_eq_target": False,
                          "factor": 1.0}
    }
}

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("PVM4"),
    html.Div(id="synthetic-data", children=[
        html.H2("Prepared Data"),
        html.Div(id="synthetic-data-table-container"),
        # Dummy input component to trigger the callback
        dcc.Input(id='dummy-input', style={'display': 'none'})
    ])
])


@app.callback(
    Output("synthetic-data-table-container", "children"),
    [Input("dummy-input", "value")]
)
def update_synthetic_data_table(dummy_input):
    fake_data = generate_dataset(500, data_model)
    table = dash_table.DataTable(
        id="synthetic-data-table",
        columns=[{"name": col, "id": col} for col in fake_data.columns],
        data=fake_data.to_dict("records"),
        style_table={"overflowX": "scroll"},
    )
    return table


if __name__ == "__main__":
    app.run_server(debug=True)
