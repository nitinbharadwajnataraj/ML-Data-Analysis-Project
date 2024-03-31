import streamlit as st
from components.navigation import nav_button
from synthetic_data import generate_dataset
from data_model import data_model as cc_data_model

st.set_page_config(layout="wide")

sections = {
    'PVM4': 'section-1',
    'Prepared Data': 'synthetic-data'
}

with st.sidebar:
    st.title('Navigation')
    for title, section in sections.items():
        nav_button(title, section)

st.header("PVM4", anchor=sections['PVM4'])
st.subheader('Prepared Data', anchor=sections['Prepared Data'], )
st.info("The below data shows the synthetic dataset prepared for analysis.")


with st.expander('Show Synthetic Data', expanded=True):
    fake_data = generate_dataset(500, {"Cube": cc_data_model["Cube"], "Cylinder": cc_data_model["Cylinder"]})
    st.write(fake_data.head())
