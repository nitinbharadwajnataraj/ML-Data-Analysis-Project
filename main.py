import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from data_model import data_model as hairpin_data_model, data_model
from create_fake_data import create_fake_dataset
from Compute_fit import compute_fit,count_yes_no

def df_return():
    df = create_fake_dataset()
    return df

def scatter_plot():
    st.markdown(
        f'<h1 style="text-align: center;">Box and Cylinder Analysis</h1>',
        unsafe_allow_html=True
    )
    df = df_return()
    df_1 = compute_fit()
    df_1 = df_1.drop(columns=['ID'], axis=1)

    # Allow users to choose columns for the scatter plot
    selected_columns = st.multiselect("Select columns for scatter plot", df_1.columns)

    # Define colors for "yes" and "no" values
    yes_color = 'green'
    no_color = 'red'

    # Create scatter plot if at least two columns are selected
    if len(selected_columns) >= 2:
        st.write("Scatter Plot:")
        fig = go.Figure()
        for col in selected_columns:
            # Set color based on 'check' value
            colors = [yes_color if val == 'yes' else no_color for val in df_1["check"]]
            fig.add_trace(go.Scatter(x=df_1.index, y=df_1[col], mode='markers', name=col, marker=dict(color=colors)))
        st.plotly_chart(fig)
    else:
        st.warning("Please select at least two columns for the scatter plot.")

# def scatter_plot():
#     # Center the title
#
#     df = df_return()
#     st.write(df)
#     # Allow users to choose columns for the scatter plot
#     # selected_columns = st.multiselect("Select columns for scatter plot", df.columns)
#
#     # Create scatter plot if at least two columns are selected
#     for column in df.columns[1:]:  # Exclude 'ID' column
#         st.write("Scatter Plot:")
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(x=df.index, y=df[column], mode='markers', name=column))
#         st.plotly_chart(fig)
#         return fig

def box_plot():
    df = df_return()
    fig = go.Figure()
    for column in df.columns[1:]:  # Exclude 'ID' column
        fig.add_trace(go.Box(y=df[column], name=column))
    fig.update_layout(title='Box Plot(Fake_data understanding)', xaxis_title='Parameters', yaxis_title='Values')
    st.plotly_chart(fig)
    return fig

def bar_chart():
    df_1, count_yes, count_no = count_yes_no()
    distinct_check_value = list(set(df_1['check']))
    data = {
        'Category': distinct_check_value,
        'Value': [count_yes, count_no]
    }
    df = pd.DataFrame(data)

    # Create bar chart
    fig = go.Figure(data=[go.Bar(x=df['Category'], y=df['Value'])])
    fig.update_layout(title='Bar Chart', xaxis_title='Category', yaxis_title='Value')

    # Display the chart in Streamlit
    st.plotly_chart(fig)



def main():
    scatter_plot()
    box_plot()
    bar_chart()

    # compute_fit()


if __name__ == "__main__":
    main()
