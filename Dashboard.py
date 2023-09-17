import pandas as pd
import streamlit as st
from datetime import date
import numpy as np
import matplotlib.pyplot as plt


st.set_page_config(layout="wide")
st.title("Dashboard")
st.divider()
st.header("Please upload a file")
uploaded_file = st.file_uploader("Choose a file")
dataset = None
if  uploaded_file is None:
    dataset =  pd.DataFrame(columns=['Date of Completion','Proposed Solutions', "Reasons for Incompletion", "Incomplete Topics", "Current Achievement Rate", "Attached Evidence of Achievements", "Completed Topics", "Sub"])
else:
    if ".xlsx" in uploaded_file.name:
        dataset = pd.read_excel(uploaded_file)
    elif ".csv" in uploaded_file.name:
        dataset = pd.read_csv(uploaded_file)
    else:
        st.error('Please upload an csv or excel file', icon="🚨")


st.divider()
st.header("Data view")


edited_date = st.data_editor(dataset, use_container_width=True, num_rows = "dynamic", hide_index = False, width = 500, height = 500
                             , column_config={
                                                "Current Achievement Rate": st.column_config.ProgressColumn(
                                                    "Current Achievement Ratee",
                                                    min_value=0,
                                                    max_value=1,
                                                ),
                                                "Date of Completion": st.column_config.DateColumn(
                                                    "Date of Completion"
                                                ),
                                                "Completed Topics": st.column_config.NumberColumn(
                                                "Completed Topics",
                                                min_value=0,
                                                max_value=1000,
                                                step=1
                                                ),
                                                "Incomplete Topics": st.column_config.NumberColumn(
                                                "Incomplete Topics",
                                                min_value=0,
                                                max_value=1000,
                                                step=1
                                                ),
                                                "Attached Evidence of Achievements": st.column_config.ImageColumn(
                                                "Attached Evidence of Achievements"
                                            )
                                            })

def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')


csv = convert_df(edited_date)

st.download_button(
   "Press to Download",
   csv,
   "file.csv",
   key='download-csv'
)
st.divider()
st.header("Current Achievment Rate & Reasons for Incompletion")
button_1 = st.button("Show Bar Chart", type="primary", key = "Button1")


if button_1:
    chart_data = pd.DataFrame({
        'Current Achievement Rate' : edited_date["Current Achievement Rate"],
        'Reasons for Incompletion' : edited_date["Reasons for Incompletion"]
    })




    fig1, ax1 = plt.subplots()
    ax1.pie(edited_date["Current Achievement Rate"], labels=edited_date["Reasons for Incompletion"], autopct='%1.1f%%',
            shadow=False, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig1, use_container_width = False)
st.divider()
st.header("Date of Completion & Completed Topics")
button_2 = st.button("Show Line Chart", type="primary", key = "Button2")

if button_2:
    chart_data = pd.DataFrame({
        'Date of Completion' : edited_date["Date of Completion"],
        'Completed Topics' : edited_date["Completed Topics"]
    })


    st.line_chart(
        chart_data,
        x='Date of Completion',
        y='Completed Topics',
        height= 500
    )
st.divider()
st.header("Date of Completion & Incomplete Topics")
button_4 = st.button("Show Line Chart", type="primary", key = "Button4")

if button_4:
    chart_data = pd.DataFrame({
        'Date of Completion' : edited_date["Date of Completion"],
        'Incomplete Topics' : edited_date["Incomplete Topics"]
    })


    st.line_chart(
        chart_data,
        x='Date of Completion',
        y='Incomplete Topics',
        height= 500
    )
st.divider()
st.header("Incomplete Topics & Reasons for Incompletion")
button_3 = st.button("Show Bar Chart", type="primary", key = "Button3")

if button_3:
    chart_data = pd.DataFrame({
        'Incomplete Topics' : edited_date["Incomplete Topics"],
        'Reasons for Incompletion' : edited_date["Reasons for Incompletion"]
    })


    st.bar_chart(
        chart_data,
        x='Reasons for Incompletion',
        y= 'Incomplete Topics',
        height= 500
    )
