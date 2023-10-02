import pandas as pd

import matplotlib.pyplot as plt

import streamlit as st
import streamlit.components.v1 as components
import matplotlib as mpl

from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    modify = st.checkbox("Add filters")

    if not modify:
        return df

    df = df.copy()

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df


st.set_page_config(layout="wide")


data_url = fr"C:\Users\Omar\Documents\Python Files\Streamlit\Dummy DATA - Sheet1 (3).csv"

df = pd.read_csv(data_url)


uploaded_file = st.file_uploader("Please Enter a File")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)


main_df = df.set_index('Projects', append=True).swaplevel(0,1)





prj_df = df.dropna()

main_df = main_df.fillna(method='ffill', axis=0)
project_df = df[["Activities", "Status", "Achievement", "Cost", "Department"]]



options = st.selectbox("Select Project",prj_df["Projects"])

all_prj_index = prj_df['Projects'].index.values.tolist()

prj_index = df[df['Projects']== options].index.values[0]


min_prj_index = df.index[prj_index]
max_prj_index = all_prj_index.index(prj_index) + 1

try: 
    max_prj_index = all_prj_index[max_prj_index]

except:
    max_prj_index = len(df.index)



two_dataset = filter_dataframe(project_df.iloc[min_prj_index:max_prj_index]),
print(two_dataset[0])
df_prj_plt = pd.DataFrame(two_dataset[0])

#labels = ["Achieved Activities Equal or Over  70% in the reporting period", "Achieved Activities Between 40%-<70% in the reporting period", "Achieved Activities Under 40% in the reporting period"]


plt_70 = df_prj_plt[df_prj_plt['Achievement'] >= 70].index.values.tolist()
plt_40 = df_prj_plt[df_prj_plt['Achievement'] >= 40].index.values.tolist()
plt_0 = df_prj_plt[df_prj_plt['Achievement']  >= 0].index.values.tolist()

df_ach_plt = df_prj_plt["Achievement"].copy()
    
plt_40 = [x for x in plt_40 if x not in plt_70]
plt_0 = [x for x in plt_0 if x not in plt_40 if x not in plt_70]



for i in plt_70:

    df_ach_plt[i] = 2

for i in plt_40:

    df_ach_plt[i] = 1
for i in plt_0:

    df_ach_plt[i] = 0
df_ach_plt = pd.DataFrame(df_ach_plt)

labels = []



for i in df_ach_plt.value_counts().index:
    print(i[0])

    index1 = i[0]
    if index1 == 0:
        labels += ["Achieved Activities Under 40% in the reporting period"]
    if index1 == 1:
        labels += ["Achieved Activities Between 40%-<70% in the reporting period"]
    if index1 == 2:
        labels += ["Achieved Activities Equal or Over  70% in the reporting period"]
print(labels)


st.divider()

col1, col2 = st.columns(2)

with col1:
    st.header("PROJECT STATUS REPORT")
    st.divider()
    subcol1, subcol2, subcol3 = st.columns(3)
    st.divider()
    subcol4, subcol5, subcol6, subcol7= st.columns(4)
    with subcol1:
        st.subheader("Management:")
        st.subheader("Project:")
        st.subheader("Project Manager:")
        st.subheader("Project Supervisor:")

    with subcol2:
        st.text_input(value  = prj_df["Management"][prj_index], label = "8", label_visibility = "collapsed")
        st.text_input(value  = prj_df["Projects"][prj_index], label = "7", label_visibility = "collapsed")
        st.text_input(value  = prj_df["Project Manager"][prj_index], label = "6", label_visibility = "collapsed")
        st.text_input(value  = prj_df["Project Supervisor"][prj_index], label = "5", label_visibility = "collapsed")
    with subcol4:
        st.subheader("NO. Activities:")
        st.text_input(value  = prj_df["Activities Type"][prj_index], label = "4", label_visibility = "collapsed", key = "key5")
    with subcol5:
        st.text_input(value  = prj_df["Number of Activities"][prj_index], label = "3", label_visibility = "collapsed", key = "key6")
    with subcol6:
        st.subheader("Budget:")
        st.subheader("Total Cost:")
    with subcol7:
        st.text_input(value  = prj_df["Budget"][prj_index], label = "2", label_visibility = "collapsed", key = "key7")
        st.text_input(value  = prj_df["Total Cost"][prj_index], label = "1", label_visibility = "collapsed", key = "key8")
with col2:

    colors = ['#CC3B33','#3b70c4','#43bc51','#ee8711']
    fig1, ax1 = plt.subplots()
    ax1.pie(df_ach_plt.value_counts(), labels= labels, autopct='%1.1f%%',colors = colors,
            shadow=False, startangle=90)
    

    ax1.axis("scaled") 

    ax1.legend((df_ach_plt.value_counts()/df_ach_plt.value_counts().sum() * 100).round(1), loc="lower right", )
    st.pyplot(fig1)

st.divider()



kpi_df = df[["KPI", "Actual Performance", "Target"]]
kpi_df = kpi_df.iloc[min_prj_index:max_prj_index]
kpi_df = kpi_df.dropna()
st.data_editor(kpi_df, use_container_width = True, hide_index  = True)

#print(df.head())
st.divider()





st.data_editor(df_prj_plt, use_container_width = True, hide_index  = True)