import streamlit as st
import pandas as pd
from data_model import data_model as hairpin_data_model, data_model
from create_fake_data import create_fake_dataset
import os


def compute_fit():
    df = pd.read_excel("fake_data.xlsx")
    box_hole_diameter_error_allowance = 2
    box_hole_depth_error_allowance = 3
    cylinder_diameter_error_allowance = 1.5
    cylinder_depth_error_allowance = 2.5
    df_1 = df
    for index, row in df_1.iterrows():
        check = 'yes'  # Assume all conditions are met initially
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

            # Check if condition is not met for any column, set check to 'no' and break the loop
            if abs(row[column] - target_value) >= error_allowance:
                check = 'no'
                break

        # Set the check value for the row
        df_1.at[index, "check"] = check

    # path = os.path.join(os.path.expanduser('~'),'Downloads')
    # file_path = os.path.join(path, 'compute_fit.csv')
    # df_1.to_csv(file_path, index=False)
    return df_1


def count_yes_no():
    df_1 = compute_fit()
    count_yes, count_no = 0,0
    for val in df_1["check"]:
        if val == "yes":
            count_yes += 1
        else:
            count_no += 1
    return df_1,count_yes, count_no


compute_fit()
