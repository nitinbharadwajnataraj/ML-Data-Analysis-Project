import random
import pandas as pd
import streamlit as st
import functools as ft
import os
from data_model import data_model as hairpin_data_model, data_model

def generate_fake_data(num_rows, process_data_model):
    data = []
    for i in range(num_rows):
        row_data = {"ID": i + 1}
        for param, values in process_data_model.items():
            target_value = values["target_value"]
            min_value = values["min_value"]
            max_value = values["max_value"]
            # Removed "factor" from calculation as it's not needed
            value = random.normalvariate(target_value, (max_value - min_value)/2)
            row_data[param] = value
        data.append(row_data)
    return data

def generate_dataset(num_rows, data_model):
    dfs = []
    for process, process_data_model in data_model.items():
        # Generate fake data for each process based on the specified number of rows
        df = pd.DataFrame(generate_fake_data(num_rows, process_data_model))
        dfs.append(df)
    # Merge all dataframes
    df_final = ft.reduce(lambda left, right: pd.merge(left, right, on='ID'), dfs)
    df_final["Material_seller"] = [random.randint(0, 2) for _ in range(len(df_final))]
    return df_final

def create_fake_dataset():
    num_rows = 1000  # Adjust the number of rows as needed
    fake_dataset = generate_dataset(num_rows, data_model)
    return fake_dataset

fake_data = create_fake_dataset()
path = os.path.join(os.path.expanduser('~'),'Downloads')
file_path = os.path.join(path, 'fake_data.csv')
fake_data.to_csv(file_path, index=False)
# print(fake_data)
