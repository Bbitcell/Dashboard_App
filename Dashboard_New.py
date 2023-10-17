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
st.set_page_config(layout="wide")
if 'key' not in st.session_state:
    st.session_state['key'] = 'value'

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
                    default=list(df[column].unique())
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

def plot_diagram(df):
        try:
            data = df[["Activities performed", "Level of Achievement by Dep."]]
            colors = ['#CC3B33','#3b70c4','#43bc51','#ee8711']
            fig1, ax1 = plt.subplots()
            ax1.pie( data["Level of Achievement by Dep."] , labels= data["Activities performed"], autopct='%1.1f%%',colors = colors,
                    shadow=False, startangle=90)
            

            ax1.axis("scaled") 
            #ax1.legend((data["Level of Achievement by Dep."].value_counts()/data["Level of Achievement by Dep."].value_counts().sum() * 100).round(1), loc="lower right", )
            plot_dr = st.pyplot(fig1)
            return plot_dr
        except ValueError:
            print("Select appropriate column")



data_url = fr"Dummy DATA New.csv"

df = pd.read_csv(data_url)


uploaded_file = st.file_uploader("Please Enter a File")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)


main_df = df.set_index('Projects', append=True).swaplevel(0,1)





prj_df = df["Projects"].dropna()

prj_df = pd.DataFrame(prj_df)

main_df = df.fillna(method='ffill', axis=0)
project_df = df[["Activities performed", "Department", "Cost", "Total Spended Budget", "Level of Achievement by Dep."]]



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

df_prj_plt = pd.DataFrame(two_dataset[0])

#labels = ["Achieved Activities Equal or Over  70% in the reporting period", "Achieved Activities Between 40%-<70% in the reporting period", "Achieved Activities Under 40% in the reporting period"]

kpi_df = df[["KPI", " Total Actual Performance", "Targeted Performance"]]
kpi_df = kpi_df.iloc[min_prj_index:max_prj_index]
kpi_df = kpi_df[kpi_df['KPI'].notna()]

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.header("PROJECT STATUS REPORT")
    st.divider()
    subcol1, subcol2, subcol3 = st.columns(3)
    st.divider()
    subcol4, subcol5, subcol6= st.columns(3)
    with subcol1:
        st.subheader("Project:")
        st.subheader("Unit in charge:")

#        st.subheader("Expected Activities:")
#        st.subheader("Alocated Budget:")

    with subcol2:
        project_name = st.text_input(value  = df["Projects"][prj_index], label = "7", label_visibility = "collapsed")
        unit  = st.text_input(value  = df["Unit in charge"][prj_index], label = "8", label_visibility = "collapsed")

#        st.text_input(value  = prj_df["Project Manager"][prj_index], label = "6", label_visibility = "collapsed")
#        st.text_input(value  = prj_df["Alocated Budget:"][prj_index], label = "5", label_visibility = "collapsed")
    with subcol4:
#        st.subheader("NO. Activities:")
#        st.text_input(value  = prj_df["Activities Type"][prj_index], label = "4", label_visibility = "collapsed", key = "key5")
        st.subheader("Allocated Budget:")
        st.subheader("Total Spended Budget:")
    with subcol5:
#        st.text_input(value  = prj_df["Number of Activities"][prj_index], label = "3", label_visibility = "collapsed", key = "key6")
        al_budget = st.text_input(value  = df["Allocated Budget"][prj_index], label = "2", label_visibility = "collapsed", key = "key7")
        spended_budget = st.text_input(value  = df["Total Spended Budget"][prj_index], label = "1", label_visibility = "collapsed", key = "key8")
#with col2:
#    with subcol6:
#        st.subheader("Allocated Budget:")
#        st.subheader("Total Spended Budget:")
#    with subcol7:
#        st.text_input(value  = df["Allocated Budget"][prj_index], label = "2", label_visibility = "collapsed", key = "key7")
#        st.text_input(value  = df["Total Spended Budget"][prj_index], label = "1", label_visibility = "collapsed", key = "key8")


st.session_state["df2"] = st.data_editor(kpi_df, use_container_width = True, hide_index  = True, column_config={"Targeted Performance": st.column_config.SelectboxColumn( "Targeted Performance", width="medium", options=["Not Started" ,20, 40, 60, 80, 100], required=True), " Total Actual Performance": st.column_config.SelectboxColumn( " Total Actual Performance", width="medium", options=["Not Started" ,20, 40, 60, 80, 100], required=True)})


#print(df.head())
st.divider()


df_final = df[["Projects","Unit in charge","Total Spended Budget", "Allocated Budget"]]
df_final["Projects"].loc[prj_index] = project_name
df_final["Unit in charge"].loc[prj_index] = unit
df_final["Allocated Budget"].loc[prj_index] = al_budget
df_final["Total Spended Budget"].loc[prj_index] = spended_budget




st.session_state["df3"] = st.data_editor(df_prj_plt, use_container_width = True, hide_index  = True)

with col2:

    try:
        plot_dr = plot_diagram(st.session_state["df3"])
    except:
        plot_dr = plot_diagram(df_prj_plt.notna())
st.divider()


final_dataframe = pd.concat([df_final, st.session_state["df3"], st.session_state["df2"]], axis=1)
data_as_csv= final_dataframe.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download data as CSV",
    data=data_as_csv,
    file_name='large_df.csv',
    mime='text/csv',
)
