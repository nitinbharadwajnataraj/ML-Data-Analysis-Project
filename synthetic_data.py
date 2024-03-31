import random
import pandas as pd
import functools as ft
import streamlit as st

from data_model import data_model as cc_data_model


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
            value = random.normalvariate(target_value, (max_value - min_value)/2 * factor)
            if values["min_eq_target"]:
                value = max(value, min_value)
            row_data[param] = value
        data.append(row_data)
    return data


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
