import streamlit as st
import pandas as pd
from data_model import data_model as hairpin_data_model, data_model
from create_fake_data import create_fake_dataset
import os

def compute_fit():
    df = create_fake_dataset()
    box_hole_diameter_error_allowance = 2
    box_hole_depth_error_allowance = 3
    cylinder_diameter_error_allowance = 1.5
    cylinder_depth_error_allowance = 2.5
    df_1 = df
    for column in df_1.columns:
        if column == "box_hole_diameter":
            target_value = data_model["Box"]["box_hole_diameter"]["target_value"]
            error_allowance = box_hole_diameter_error_allowance
        elif column == "box_hole_depth":
            target_value = data_model["Box"]["box_hole_depth"]["target_value"]
            error_allowance = box_hole_depth_error_allowance
        elif column == "cylinder_diameter":
            target_value = data_model["Cylinder"]["cylinder_diameter"]["target_value"]
            error_allowance = cylinder_diameter_error_allowance
        elif column == "cylinder_height":
            target_value = data_model["Cylinder"]["cylinder_height"]["target_value"]
            error_allowance = cylinder_depth_error_allowance
        else:
            continue

        df_1["check"] = df_1.apply(lambda row: 'yes' if abs(row[column] - target_value) < error_allowance else 'no', axis=1)
    print(df_1)
    path = os.path.join(os.path.expanduser('~'),'Downloads')
    file_path = os.path.join(path, 'compute_fit.csv')
    df_1.to_csv(file_path, index=False)
    return df_1
def count_yes_no():
    df_1 = compute_fit()
    count_yes = 0
    for val in df_1["check"]:
        if val == "yes":
            count_yes += 1
    count_no = len(df_1) - count_yes
    return df_1,count_yes, count_no

compute_fit()