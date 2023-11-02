import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import matplotlib as mpl
from io import BytesIO
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

            try:
                df[column] = pd.to_numeric(df[column])
            except:
                df[column] = df[column].astype('category')

            if is_categorical_dtype(df[column]):
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

def plot_diagram(df, op1, op2):
        try:
            
            data = df[[op1, op2]]
            fig1, ax1 = plt.subplots()
            if pd.notna(data).any().any():
                data = data.dropna()
                try:
                    df[op1] = pd.to_numeric(df[op1])
                except:
                    pass

                try:
                    df[op2] = pd.to_numeric(df[op2])
                except:
                    pass
                condtion1 = is_numeric_dtype(df[op1])
                condtion2 = is_numeric_dtype(df[op2])
                print(data, condtion1, condtion2)
                
                if condtion1 and not condtion2:
                    colors = ['#CC3B33','#3b70c4','#43bc51','#ee8711']

                    ax1.pie( data[op1] , labels= data[op2], autopct='%1.1f%%',colors = colors,
                            shadow=False, startangle=90)
                    
                elif not condtion1 and condtion2:
                    
                    colors = ['#CC3B33','#3b70c4','#43bc51','#ee8711']

                    ax1.pie( data[op2] , labels= data[op1], autopct='%1.1f%%',colors = colors,
                            shadow=False, startangle=90)
                elif condtion1 and condtion2:

                    plt.xlabel(op1)
                    plt.ylabel(op2)
                    x = np.array(data[op1])
                    a, b = np.polyfit(data[op1], data[op2], 1)
                    ax1.scatter(data[op1], data[op2], linewidth=2.0)
                    ax1.plot(x, a * x + b)
            plt.figure(figsize=(500, 500))
                #ax1.legend((data["Level of Achievement by Dep."].value_counts()/data["Level of Achievement by Dep."].value_counts().sum() * 100).round(1), loc="lower right", )
                
            buf = BytesIO()
            fig1.savefig(buf, format="png")
            plot_dr = st.image(buf, width = 800, use_column_width  = "never")
            return plot_dr
        except ValueError:
            print("Select appropriate column")




if 'main' not in st.session_state:
    st.session_state['main'] = 'value'

    uploaded_file = st.file_uploader("Please Enter a File")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        data_url = fr"Dummy DATA New.csv"
        df = pd.read_csv(data_url)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df["Projects"] = pd.Series(df["Projects"]).fillna(method='ffill')
    st.session_state["main"] = df.copy()

    columns = df.columns




project_created = False

project_creation = st.button("Add New Project", type="primary")
if project_creation:
    project_created = True
    df2 = pd.DataFrame(columns=st.session_state["main"].columns)
    df2 = pd.concat([df2, {'Projects': 'New Project'}])
    st.session_state["main"] = df2

bk2 = st.session_state["main"].copy()
columns = bk2.columns
prj_df = st.session_state["main"]["Projects"]

prj_df = pd.DataFrame(prj_df)



if project_created:
    

    index = prj_df["Projects"].unique()
    print(len(index))
    options = st.selectbox("Select Project", prj_df["Projects"].unique(), index = len(index) - 1)

    options = "New Project"
    

else:
    options = st.selectbox("Select Project",prj_df["Projects"].unique(), index = 0)
project_created = False
all_prj_index = prj_df['Projects'].index.values.tolist()

prj_index = st.session_state["main"][st.session_state["main"]['Projects']== options].index.values[0]

mprj = st.session_state["main"]['Projects'].value_counts()[options]

min_prj_index = st.session_state["main"].index[prj_index]
max_prj_index = mprj + min_prj_index

bk = st.session_state["main"].copy()


two_dataset = filter_dataframe(st.session_state["main"][min_prj_index:max_prj_index]),

df_prj_plt = pd.DataFrame(two_dataset[0])

#labels = ["Achieved Activities Equal or Over  70% in the reporting period", "Achieved Activities Between 40%-<70% in the reporting period", "Achieved Activities Under 40% in the reporting period"]

kpi_df = df_prj_plt


#df_prj_plt = df_prj_plt[["Activities performed", "Department", "Cost", "Total Spended Budget", "Level of Achievement by Dep."]]




#kpi_df = kpi_df[["KPI", " Total Actual Performance", "Targeted Performance"]]
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
        st.subheader("Expected Activity:")
        st.subheader("Unit in charge:")

#        st.subheader("Expected Activities:")
#        st.subheader("Alocated Budget:")

    with subcol2:
        df9 = st.session_state["main"].dropna(subset = "Projects")
        st.session_state["main"]["Projects"][prj_index] = st.text_input(value  = df9["Projects"][prj_index], label = "7", label_visibility = "collapsed")
        st.session_state["main"]["Expected Activities "][prj_index]  = st.text_input(value  = df9["Expected Activities "][prj_index], label = "10", label_visibility = "collapsed")
        st.session_state["main"]["Unit in charge"][prj_index]  = st.text_input(value  = df9["Unit in charge"][prj_index], label = "8", label_visibility = "collapsed")

#        st.text_input(value  = prj_df["Project Manager"][prj_index], label = "6", label_visibility = "collapsed")
#        st.text_input(value  = prj_df["Alocated Budget:"][prj_index], label = "5", label_visibility = "collapsed")
    with subcol4:
#        st.subheader("NO. Activities:")
#        st.text_input(value  = prj_df["Activities Type"][prj_index], label = "4", label_visibility = "collapsed", key = "key5")
        st.subheader("Allocated Budget:")
        st.subheader("Total Spended Budget:")
    with subcol5:
#        st.text_input(value  = prj_df["Number of Activities"][prj_index], label = "3", label_visibility = "collapsed", key = "key6")
        st.session_state["main"]["Allocated Budget"][prj_index] = st.text_input(value  = df9["Allocated Budget"][prj_index], label = "2", label_visibility = "collapsed", key = "key7")
        st.session_state["main"]["Total Spended Budget"][prj_index] = st.text_input(value  = df9["Total Spended Budget"][prj_index], label = "1", label_visibility = "collapsed", key = "key8")
#with col2:
#    with subcol6:
#        st.subheader("Allocated Budget:")
#        st.subheader("Total Spended Budget:")
#    with subcol7:
#        st.text_input(value  = df["Allocated Budget"][prj_index], label = "2", label_visibility = "collapsed", key = "key7")
#        st.text_input(value  = df["Total Spended Budget"][prj_index], label = "1", label_visibility = "collapsed", key = "key8")


st.session_state["main"] = st.data_editor(kpi_df, key= "df1", column_order = ("KPI", " Total Actual Performance", "Targeted Performance"), num_rows = "dynamic", use_container_width = True, hide_index  = True, column_config={"Targeted Performance": st.column_config.SelectboxColumn( "Targeted Performance", width="medium", options=["Not Started" ,20, 40, 60, 80, 100], required=True), " Total Actual Performance": st.column_config.SelectboxColumn( " Total Actual Performance", width="medium", options=["Not Started" ,20, 40, 60, 80, 100], required=True)})




#print(df.head())
st.divider()


st.session_state["main"] = st.data_editor(st.session_state["main"], key=  "df2", column_order= ("Activities performed", "Department", "Cost", "Level of Achievement by Dep."), use_container_width = True, num_rows = "dynamic", hide_index  = True)


with col2:
    st.session_state["df4"] = st.session_state["main"][["Activities performed", "Department", "Cost", "Total Spended Budget", "Level of Achievement by Dep.", "KPI", " Total Actual Performance", "Targeted Performance"]]
    subcol7, subcol8 = st.columns(2)
    
    with subcol7:
        option1 = st.selectbox(
        'Select First Column to Compare',
        (st.session_state["df4"].columns))
    with subcol8:
        option2 = st.selectbox(
        'Select Second Column to Compare',
        (st.session_state["df4"].columns))



    plot_dr = plot_diagram(st.session_state["df4"], option1, option2)






#print(st.session_state["main"])
st.divider()

bk2.update(st.session_state["main"][pd.notna(st.session_state["main"])])


st.session_state["main"] = bk2.copy()



#print(st.session_state["main"] ["Projects"])

if pd.notna(st.session_state["main"].iloc[-1][st.session_state["main"].columns[5:]]).any().any():
    project_name = st.session_state["main"]["Projects"][prj_index]

    st.session_state["main"] = pd.concat([st.session_state["main"], {'Projects': project_name}])
if not st.session_state["main"].equals(bk):
    st.rerun()



data_as_csv= st.session_state["main"].to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download data as CSV",
    data=data_as_csv,
    file_name='large_df.csv',
    mime='text/csv',
)

