import pandas as pd
import plotly.express as px
import numpy as np
import streamlit as st
from PIL import Image
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

import warnings
warnings.filterwarnings("ignore")

import base64
st.set_page_config(layout="wide")
@st.cache_data()
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    div[data-testid="stSelectbox"] {
    background-color: rgba(255,255,255,1);}
    div[data-testid="stMultiSelect"] {
    background-color: rgba(255,255,255,1);}
    div[data-testid="column"] {
    background-color: rgba(255,255,255,1);

    text-align: center;}
    div[data-testid="stCheckbox"] {
    background-color: rgba(255,255,255,1);}
    div[data-testid="stDataframe"] {
    background-color: rgba(255,255,255,1);}
    div[data-testid="stFileUploader"] {
    background-color: rgba(255,255,255,1);
    }
        div[data-baseweb="tab-list"] {
    background-color: rgba(255,255,255,1);
    height: 110px;
    }
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
    background-color: rgba(255,255,255,1);
    text-align: center;}
    .stTabs [data-baseweb="tab"] {


		background-color: rgba(255,255,255,1);
		border-radius: 4px 4px 0px 0px;
		gap: 1px;

        height: 110px;
        width: 210px;
		padding-bottom: 10px;
        
    }
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size: 60px;
    }


    
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg(fr"banner-cropped.png")


style_image1 = """
width: 1745px;
max-width: 1745px;
height: auto;
max-height: 750px;
display: block;
justify-content: center;
"""

file_ = open(fr"stamp9.gif", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()

st.markdown(
    f'<img src="data:image/gif;base64,{data_url}" alt="cat gif" style="{style_image1}">',
    unsafe_allow_html=True,
)
#image = Image.open(fr"stamp9.gif")
#img2 = st.image(video_file, use_column_width  = "always", width = 10)
tab1, tab2 = st.tabs(["Main", "Pie Chart"])



if 'key' not in st.session_state:
    st.session_state['key'] = 'value'

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:



    global modify, container
    with container:
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
                    if not df[~df[column].isin(user_cat_input)].empty:
                        st.session_state['bk3'] = df[~df[column].isin(user_cat_input)]
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
                    if not df[~df[column].between(*user_num_input)].empty:
                        st.session_state['bk3'] = df[~df[column].between(*user_num_input)]
                    df = df[df[column].between(*user_num_input)]
                else:
                    user_text_input = right.text_input(
                        f"Substring or regex in {column}",
                    )
                    if user_text_input:
                        if not df[~df[column].astype(str).str.contains(user_text_input)].empty:
                            st.session_state['bk3'] = df[~df[column].astype(str).str.contains(user_text_input)]
                        df = df[df[column].astype(str).str.contains(user_text_input)]

    return df

def plot_diagram(df, op1, op2):
        try:
            
            data = df[[op1, op2]]


            #fig1, ax1 = plt.subplots()
            fig = px.pie()
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
                #print(data, condtion1, condtion2)
                
                if condtion1 and not condtion2:
                    colors = ['#CC3B33','#3b70c4','#43bc51','#ee8711']

                    #ax1.pie( data[op1] , labels= data[op2], autopct='%1.1f%%',colors = colors,
                    #        shadow=False, startangle=90)
                    fig = px.pie(data, values = op1, names= op2)
                elif not condtion1 and condtion2:

                    #colors = ['#CC3B33','#3b70c4','#43bc51','#ee8711']
                    #ax1.pie( data[op2] , labels= data[op1], autopct='%1.1f%%',colors = colors,
                    #        shadow=False, startangle=90)
                    
                    fig = px.pie(data, values = op2, names= op1)
                    
                elif condtion1 and condtion2:

                    #plt.xlabel(op1)
                    #plt.ylabel(op2)
                    #x = np.array(data[op1])
                    #a, b = np.polyfit(data[op1], data[op2], 1)
                    #ax1.scatter(data[op1], data[op2], linewidth=2.0)
                    
                    fig = px.line(df, x= op1, y= op2, markers=True)
                    #ax1.plot(x, a * x + b)
            #plt.figure(figsize=(500, 500))
                #ax1.legend((data["Level of Achievement by Dep."].value_counts()/data["Level of Achievement by Dep."].value_counts().sum() * 100).round(1), loc="lower right", )
                
            #buf = BytesIO()
            #fig1.savefig(buf, format="png")
            #plot_dr = st.image(buf, width = 800, use_column_width  = "never")
            
            plot_dr = st.plotly_chart(fig, theme="streamlit", use_container_width=True, key = "dfplot")
            return plot_dr
        except ValueError:
            print("Select appropriate column")
def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)


def diagram_page():
    df1 = pd.read_csv(fr"Temp_Data.csv")
    data = df1[["Department", "Level of Achievement by Dep.", "Projects"]] 
    counter = 0
    data = data.dropna(how = "all")

    unique_prj = data["Projects"].unique()
    #fig2, ax2 = plt.subplots(len(unique_prj))

    colors = ['#CC3B33','#3b70c4','#43bc51','#ee8711']
    if len(data["Projects"].unique()) > 1:
        for i in data["Projects"].unique():
            
            rslt_df = data[data['Projects'].isin([i])] 





            if not rslt_df.empty:

                #ax2[counter].pie( rslt_df["Level of Achievement by Dep."] , labels= rslt_df["Department"], autopct='%1.1f%%',colors = colors,
                #        shadow=False, startangle=90)
                #ax2[counter].title.set_text(arabic_reshaper.reshape(i))

                fig = px.pie(rslt_df, values = "Level of Achievement by Dep.", names='Department', title= i)
                st.plotly_chart(fig, theme="streamlit", use_container_width=True, key = "df" + i)
            counter+=1
    else:
        rslt_df = data[data['Projects'].isin([data["Projects"].unique()])]
        if not rslt_df.empty:
            
            #ax2.pie( rslt_df["Level of Achievement by Dep."] , labels= rslt_df["Department"], autopct='%1.1f%%',colors = colors,
            #    shadow=False, startangle=90)
            

            fig = px.pie(rslt_df, values = "Level of Achievement by Dep.", names='Department', title= data["Projects"].unique()[0])
            st.plotly_chart(fig, theme="streamlit", use_container_width=True, key = "df1" + data["Projects"].unique())
    #buf1 = BytesIO()
    #fig2.savefig(buf1, format="png")
    #plot_dr2 = st.image(buf1, width = 1200, use_column_width  = "never")
def edit():

    global condition
    condition = True

def main_page():
    
    
    global condition, container
    condition = False



    container = st.container()
   
    if 'main' not in st.session_state:
        st.session_state['main'] = 'value'



        data_url = fr"Dummy DATA New 3.csv"
        df = pd.read_csv(data_url)
        df["Projects"] = pd.Series(df["Projects"]).fillna(method='ffill')
        df = df.sort_values(by= ["Projects", "Unit in charge", "Allocated Budget", "Total Spended Budget", "Expected Activities "]).reindex()

        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        st.session_state["main"] = df.copy()
        bk2 = st.session_state["main"].copy()

        st.session_state["main"].to_csv(fr"Temp_Data.csv")
            


        columns = df.columns
    else:
        data_url = fr"Temp_Data.csv"
        bk2 = pd.read_csv(data_url)
        #bk2["Projects"] = pd.Series(bk2["Projects"]).fillna(method='ffill')
        bk2 = bk2.sort_values(by= ["Projects", "KPI"]).reindex()
        bk2["Allocated Budget"] = bk2["Allocated Budget"].astype("Int64")
        bk2["Total Spended Budget"] = bk2["Total Spended Budget"].astype("Int64")
        bk2 = bk2.loc[:, ~bk2.columns.str.contains('^Unnamed')]
        #bk2.sort_values('Projects').reindex()
        st.session_state["main"] = bk2.copy()

    project_created = False
    with container:
        project_creation = st.button("Add New Project", type="primary")


    if 'count' not in st.session_state:
        st.session_state.count = 0



    if project_creation:
        project_created = True
        #df2 = pd.DataFrame(columns=st.session_state["main"].columns)
        st.session_state["main"] = st.session_state["main"].append({'Projects': 'New Project'}, ignore_index = True)
        
        st.session_state["main"] = st.session_state["main"].sort_values(by= ["Projects"]).reindex()
        st.session_state["main"].reindex().to_csv(fr"Temp_Data.csv")

        st.rerun()
        
    if 'bk3' not in st.session_state:
        st.session_state['bk3'] = pd.DataFrame(columns=columns)
    
    #bk2 = st.session_state["main"].copy()



    columns = bk2.columns
    prj_df = st.session_state["main"]["Projects"]

    prj_df = pd.DataFrame(prj_df)



    if project_created:
        

        index =prj_df["Projects"].unique()
        print(len(index))
        with container:
            options = st.selectbox("Select Project", prj_df["Projects"].unique(), index = len(index) - 1)

        options = "New Project"
        
    
    else:
        with container:
            options = st.selectbox("Select Project", prj_df["Projects"].unique(), index = 0)
    project_created = False
    all_prj_index = prj_df['Projects'].index.values.tolist()

    prj_index = st.session_state["main"][st.session_state["main"]['Projects']== options].index.values[0]

    mprj = st.session_state["main"]['Projects'].value_counts()[options]
    
    min_prj_index = st.session_state["main"].index[prj_index]
    max_prj_index = mprj + min_prj_index

    

    two_dataset = filter_dataframe(st.session_state["main"][st.session_state["main"]["Projects"] == options]),

    df_prj_plt = pd.DataFrame(two_dataset[0])
    bk = df_prj_plt.copy()
    #labels = ["Achieved Activities Equal or Over  70% in the reporting period", "Achieved Activities Between 40%-<70% in the reporting period", "Achieved Activities Under 40% in the reporting period"]
    #st.write(two_dataset[0].reset_index(drop=True))
    #st.write(bk[bk["Projects"] == options].reset_index(drop=True))
    #st.write(len(two_dataset[0].index) == len(bk[bk["Projects"] == options].index))

    #st.write(two_dataset[0].reset_index(drop=True).isin(bk.reset_index(drop=True)))
    if not modify:
        st.session_state.count += 1
        st.session_state["bk3"] = st.session_state["bk3"].iloc[0:0]
    else:
        st.session_state.count = 0
    kpi_df = df_prj_plt
    #st.write(st.session_state["bk3"])
    #st.write(st.session_state.count)
    #df_prj_plt = df_prj_plt[["Activities performed", "Department", "Cost", "Total Spended Budget", "Level of Achievement by Dep."]]
    
    
    
    st.session_state["bk3"] = st.session_state["bk3"].dropna(how = "all", subset = ["Unit in charge", "Allocated Budget", "Total Spended Budget", "Expected Activities ", "Activities performed", "Department", "Cost", "Total Spended Budget", "Level of Achievement by Dep.", "KPI", " Total Actual Performance", "Targeted Performance"])
    
    #st.write(kpi_df.iloc[-1,1:].isnull().any())

    if not kpi_df.iloc[-1,5:].isnull().all():


        kpi_df = kpi_df.append({'Projects': options}, ignore_index=True)

    #kpi_df = kpi_df[["KPI", " Total Actual Performance", "Targeted Performance"]]
    
    
    
    st.divider()

    col1, col2 = st.columns(2)
    #local_css("style.css")


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
            df9 = st.session_state["main"]
            
            kpi_df["Projects"].iloc[0] = st.text_input(value  = kpi_df["Projects"].sort_values().iloc[0], label = "7", label_visibility = "collapsed")
            kpi_df["Expected Activities "].iloc[0]  = st.text_input(value  = kpi_df["Expected Activities "].sort_values().iloc[0], label = "10", label_visibility = "collapsed")
            
            kpi_df["Unit in charge"].iloc[0]  = st.text_input(value  = kpi_df["Unit in charge"].sort_values().iloc[0], label = "8", label_visibility = "collapsed")
            
    #        st.text_input(value  = prj_df["Project Manager"][prj_index], label = "6", label_visibility = "collapsed")
    #        st.text_input(value  = prj_df["Alocated Budget:"][prj_index], label = "5", label_visibility = "collapsed")
        with subcol4:
    #        st.subheader("NO. Activities:")
    #        st.text_input(value  = prj_df["Activities Type"][prj_index], label = "4", label_visibility = "collapsed", key = "key5")
            st.subheader("Allocated Budget:")
            st.subheader("Total Spended Budget:")
        with subcol5:
    #        st.text_input(value  = prj_df["Number of Activities"][prj_index], label = "3", label_visibility = "collapsed", key = "key6")
            
            try:
                kpi_df["Allocated Budget"].iloc[0] = st.number_input(value  = kpi_df["Allocated Budget"].sort_values().iloc[0], label = "2", label_visibility = "collapsed", key = "key7")
            except:
                kpi_df["Allocated Budget"].iloc[0] = st.number_input(label = "2", label_visibility = "collapsed", key = "key7")

            try:
                kpi_df["Total Spended Budget"].iloc[0] = st.number_input(value  = kpi_df["Total Spended Budget"].sort_values().iloc[0], label = "1", label_visibility = "collapsed", key = "key8")
            except:
                kpi_df["Total Spended Budget"].iloc[0] = st.number_input(label = "1", label_visibility = "collapsed", key = "key8")
                
    #with col2:
    #    with subcol6:
    #        st.subheader("Allocated Budget:")
    #        st.subheader("Total Spended Budget:")
    #    with subcol7:
    #        st.text_input(value  = df["Allocated Budget"][prj_index], label = "2", label_visibility = "collapsed", key = "key7")
    #        st.text_input(value  = df["Total Spended Budget"][prj_index], label = "1", label_visibility = "collapsed", key = "key8")

    st.divider()
    st.session_state["main"] = st.data_editor(kpi_df, key= "df1", on_change = edit(), column_order = ("KPI", " Total Actual Performance", "Targeted Performance"), num_rows = "dynamic", use_container_width = True, hide_index  = True, column_config={"Targeted Performance": st.column_config.SelectboxColumn( "Targeted Performance", width="medium", options=["Not Started" ,20, 40, 60, 80, 100], required=True), " Total Actual Performance": st.column_config.SelectboxColumn( " Total Actual Performance", width="medium", options=["Not Started" ,20, 40, 60, 80, 100], required=True)})

    


    #print(df.head())
    st.divider()


    st.session_state["main"] = st.data_editor(st.session_state["main"], on_change = edit(), key=  "df2", column_order= ("Activities performed", "Department", "Cost", "Level of Achievement by Dep."), use_container_width = True, num_rows = "dynamic", hide_index  = True)

    
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
    st.session_state["main"] = st.session_state["main"].replace("nan", np.nan)
    try:
        st.session_state["main"]["Total Spended Budget"].iloc[0] = int(st.session_state["main"]["Total Spended Budget"].sort_values().iloc[0])
        st.session_state["main"]["Allocated Budget"].iloc[0] = int(st.session_state["main"]["Allocated Budget"].sort_values().iloc[0])
    except ValueError:
        pass

    #st.write(st.session_state["main"])
    #bk2[bk2['Projects'].isin([options])].merge(st.session_state["main"][pd.notna(st.session_state["main"])], how = "inner")
    st.session_state["main"]["Projects"] = pd.Series(st.session_state["main"]["Projects"]).fillna(method='ffill')
    
    if not bk2[bk2== options].equals(st.session_state["main"][st.session_state["main"]['Projects']== options]):
        condition = False

    bk2 = bk2[~bk2.Projects.isin(st.session_state["main"]["Projects"])]
    bk2 = bk2.append(st.session_state["main"], ignore_index=True)

    #bk2.update(st.session_state["main"][pd.notna(st.session_state["main"])])

    if not pd.isna(bk["Allocated Budget"].iloc[0]):
        bk["Allocated Budget"].iloc[0] = int(float(bk["Allocated Budget"].iloc[0]))


    if not pd.isna(bk["Total Spended Budget"].iloc[0]):
        bk["Total Spended Budget"].iloc[0] = int(float(bk["Total Spended Budget"].iloc[0]))



    st.session_state["main"] = bk2.copy()

    #print(st.session_state["main"])

    #print(st.session_state["main"] ["Projects"])

    #if pd.notna(st.session_state["main"].iloc[-1][st.session_state["main"].columns[5:]]).any().any():
    #    project_name = st.session_state["main"]["Projects"][prj_index]

    #    st.session_state["main"] = st.session_state["main"].append({'Projects': project_name}, ignore_index = True)
    #bk = pd.concat([bk, st.session_state["bk3"]]).drop_duplicates(keep=False)
    #st.write(st.session_state["bk3"][st.session_state["bk3"]['Projects']== options].reset_index(drop=True))
    #st.write(bk2[bk2['Projects']== options].reset_index(drop=True))
    #st.write(bk[bk['Projects']== options].reset_index(drop=True))    

    #st.write(((bk2[bk2['Projects']== options].reset_index(drop=True)).equals(bk[bk['Projects']== options].reset_index(drop=True)))) 
    if not (((bk2[bk2['Projects']== options].reset_index(drop=True))).applymap(str).equals((bk[bk['Projects']== options].reset_index(drop=True)).applymap(str))):


        if not st.session_state["bk3"].empty:
            
            st.session_state["main"] = st.session_state["main"].append(st.session_state["bk3"], ignore_index=True)
            st.session_state["bk3"] = st.session_state["bk3"].iloc[0:0]
        
        st.session_state["main"] = st.session_state["main"].sort_values(by= ["Projects"]).reindex()
        st.session_state["main"].reindex().to_csv(fr"Temp_Data.csv")
        condition = False
    
    
        st.rerun()



    save = st.button(
        label="Download data as CSV")
    
    if save:
        st.session_state["main"].to_csv(fr"Dummy DATA New 3.csv")
        save = False

with tab1:
    uploaded_file = st.file_uploader("Please Enter a File")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.to_csv(fr"Dummy DATA New 3.csv")

    main_page()

with tab2:
    
    diagram_page()

