import streamlit as st
import pandas as pd
import numpy as np
import io
from io import StringIO, BytesIO
import streamlit.components.v1 as components
import sys
import st_aggrid_pro
from st_aggrid_pro import AgGridPro
from streamlit_extras.switch_page_button import switch_page
import plotly.express as px
import base64
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go


# To have Wide page
st.set_page_config(page_title="Tranining",
                   layout="wide",
                   initial_sidebar_state="collapsed",
                   page_icon= '✌')

# Hide the Sidebar
st.markdown(
    """
<style>
    [data-testid="collapsedControl"] {
        display: none
    }
</style>
""",
    unsafe_allow_html=True,
)

# css for page text

css = """
<style>
    .custom-container {
    height: 1%;
    width: 1%;
    border: 1px solid black;
    }
    
    .welcome-font{
        color: red;
        font-size: 50px;
        text-align: center;
    }

    .intro-font{
        font-size: 29px;
        font-family: Rockwell;
    }
    
    .lead-font{
        font-size: 30px;
        color: red;
        font-family: Rockwell;
    }

    .page-font{
        font-family: Comic Sans MS !important;
        font-weight: bold;
        font-size: 19px;
    }

    .color{
        color: red;
    }
    
    .small-font{
    font-family: small-font;
    font-size: 16px;
    font-weight: bold;
    color: red;
    }
    
</style>
"""
# To make css true for st.markdown...
st.markdown(css, unsafe_allow_html=True)

# JavaScript for buttons
def changebtn(label, font_size, marginleft, width):

    btn = f"""
        </style>
        <script>
            var elements = window.parent.document.querySelectorAll('button');
            for (var i = 0; i < elements.length; ++i) {{ 
                if (elements[i].innerText == '{label}') {{ 
                    elements[i].style.fontSize = '{font_size}';
                    elements[i].style.color = 'white';
                    elements[i].style.textAlign = 'center';
                    elements[i].style.padding = '20px';
                    elements[i].style.marginLeft = '{marginleft}';
                    elements[i].style.backgroundColor = 'rgb(97, 153, 222)';
                    elements[i].style.height = '100%';
                    elements[i].style.width = '{width}';
                    elements[i].style.border = 'solid rgb(89, 94, 101)';
                    elements[i].style.borderRadius = '7px';
                    elements[i].style.transition = 'all 0.6s ease';
                    elements[i].style.cursor = 'pointer';
                    
                    elements[i].onmouseover = function() {{
                        this.style.backgroundColor = 'transparent';
                        this.style.color = 'black';
                        this.style.transform = 'scale(1.1)';
                    }}
                    elements[i].onmouseout = function() {{
                        this.style.backgroundColor = 'rgb(97, 153, 222)';
                        this.style.color = 'white';
                        this.style.transform = 'scale(1)';
                    }}
                }}
            }}
        </script>
        """
    components.html(f"{btn}", height=0)

# Display the balloons
if 'balloons_shown' not in st.session_state:
    st.session_state['balloons_shown'] = False

if not st.session_state['balloons_shown']:
    st.balloons()
    st.balloons()
    st.session_state['balloons_shown'] = True

# ... balloon end


# Create a markdown string for the welcome message and intro
welcome_message = """
<h1 class='welcome-font'>Welcome to the Training Phase! 🙋‍♂️</h1><br>
"""
st.markdown(welcome_message, unsafe_allow_html=True)
st.markdown("<div class = 'intro-font'><b><span class = 'color'>'The Titanic' 🚢⚓️</span> Dataset is a popular choice for Data Analysts in Training - it's like the Training wheels of Data Analysis!</b></div><br>", unsafe_allow_html=True)
st.markdown("<div class = 'lead-font'>I Believe in You, let's Do This! 🌟. Together We Can Achieve Anything! 🤝. <b>Let's Start.</b></div><br><hr>", unsafe_allow_html=True)

# ... intro end


# load dataset button

st.markdown("<div class = 'page-font' > Press <span class = 'color'>Load Dataset</span> Button, To Add a Dataset for Analysis.</div>", unsafe_allow_html=True)
st.write('')

# session_state for load data button
if 'dataset_bnt' not in st.session_state:
    st.session_state.dataset_bnt = False
dataset_bnt = False

# sesstion_state for statistics button
if 'stat' not in st.session_state:
    st.session_state.stat = False
stat = False

# sesstion_state for duplicate button
if 'duplicate' not in st.session_state:
    st.session_state.duplicate = False
duplicate = False

# session_state for drop_null button
if 'drop_null' not in st.session_state:
    st.session_state.drop_null = False
drop_null = False

# sesstion_state for replace_with_mean button
if 'replace_with_mean' not in st.session_state:
    st.session_state.replace_with_mean = False
replace_with_mean = False

# sesstion_state for edit button
if 'edit' not in st.session_state:
    st.session_state.edit = False
edit = False

# sesstion_state for see_change button
if 'see_change' not in st.session_state:
    st.session_state.see_change = False
see_change = False

# sesstion_state for save button
if 'save' not in st.session_state:
    st.session_state.save = False
save = False

# sesstion_state for groubby button
if 'groubby' not in st.session_state:
    st.session_state.groubby = False
groubby = False

# sesstion_state for pivot_table button
if 'pivot_table' not in st.session_state:
    st.session_state.pivot_table = False
pivot_table = False

# sesstion_state for save_pivot table button
if 'savep' not in st.session_state:
    st.session_state.savep = False
savep = False

# sesstion_state for save_Groub by button
if 'saveg' not in st.session_state:
    st.session_state.saveg = False
saveg = False

# sesstion_state for single column grapy button
if 'single' not in st.session_state:
    st.session_state.single = False
single = False

# sesstion_state for 2d graphes button
if 'd2' not in st.session_state:
    st.session_state.d2 = False
d2 = False

# sesstion_state for 3d graphes button
if 'd3' not in st.session_state:
    st.session_state.d3 = False
d3 = False

col1, col2, col3 = st.columns(3)
with col1:
    st.write('')
    st.write('')
    dataset_bnt = st.button('# Load Dataset')
    changebtn("Load Dataset", "25px", "0", "96%")
    if dataset_bnt:
        st.session_state.dataset_bnt = True

if st.session_state.dataset_bnt:
    with col2:
        st.file_uploader('Choose a file', type= ['csv', 'xlsx', 'xls', 'txt', 'json', 'html', 'xml'])
    with col3:
        st.write('')
        st.write('')
        st.write('')
        st.markdown("<div class = 'page-font'>Dataset? No problem, we'll just Titanic it and make waves 🌊🚢😉.</div>", unsafe_allow_html= True)
st.write('---')
# ... load data button end

# to see head and tail of data set
data = pd.read_csv('pages/titanic.csv')
data.index.name = 'Index'

if st.session_state.dataset_bnt:
    st.markdown("<div class = 'page-font' style = 'text-align: center;'>🎯 Here's where we Begin and End - the <span class = 'color'>Head & Tail</span> of our dataset! 🎲", unsafe_allow_html= True)
    st.write('')
    st.table(data.head())
    st.table(data.tail())
    st.text('See! The Head & Tail. Notice the <NA> ("None" or "null") in Head and nan ("Not a Number") in tail, also true/false in Head and 0/1 in Tail. Both means the same.\nThe purpose to Differencite both is for better understanding with the Data')
    st.write('---')

    # showing statistics button
    st.markdown("<div class = 'page-font' style = 'text-align: center;'>Unlock the Secrets of your Data with <span class = 'color'>Statistics</span>, it's like a Treasure Map! 🗺️</div>", unsafe_allow_html=True)
    st.markdown("<div class='stat-button'></div>", unsafe_allow_html=True)
    stat = st.button('# Statistics')
    changebtn("Statistics", "30px", "20%", "60%")


if stat or st.session_state.stat:
    st.session_state.stat = True
    col1, col2, col3= st.columns((4, 0.0001, 2.1))
    with col1:
        st.markdown("<div class = 'small-font'>You can Analyze the Statistics of Data (only Numerical Data). Using dataset.describe()</div>", unsafe_allow_html= True)
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        formatted_data = data.describe().applymap(lambda x: "{:.1f}".format(x))
        st.table(formatted_data)
        st.text('You can see the Statistics of the Dataset.\nCount is the number of rows. Mean, standard deviation, Minimun and Maximum of each column.\nYou can also Analyse the 4-Quartile. 1st Quartile from 0-25%. 2nd Quartile from 25-50%.\n3rd Quartile from 50-75%. 4th Quartile form 75-100%')

    with col3:
        st.markdown("<div class = 'small-font'>All Information of the Dataset. Using dataset.info()</div>", unsafe_allow_html= True)
        st.write('')
        def get_info(df):
            buf = io.StringIO()
            df.info(buf=buf)
            return buf.getvalue()
        st.text(get_info(data))

if stat or st.session_state.stat:
    st.session_state.stat = True
    '---'
    st.write('')
    st.markdown("<div class = 'page-font' style = 'text-align: center;'>The important part is to see for <span class = 'color'>NULL valuse</span> and <span class = 'color'>Duplicates</span> in the Dataset.</div>", unsafe_allow_html=True)
    st.write('')
    col1, col2 = st.columns((1,1))
    with col1:
        nan_data = pd.DataFrame(data.isna().sum()).reset_index()
        nan_data.columns = ['Column Name', 'Number of NULL values']
        st.markdown("<div class = 'small-font'>You can analyze the NULL values or Empty cells. Using dataset.isna()</div>", unsafe_allow_html=True)
        st.write('')
        st.write('')
        st.table(nan_data)

    with col2:
        st.markdown("<div class = 'small-font'>Keep Remember the Data Type of Each column, it will help you further. Using dataset.dtype()</div>", unsafe_allow_html= True)
        data_types = pd.DataFrame(data.dtypes, columns=['Data Type'])
        st.write('')
        st.write('')
        st.table(data_types)

if stat or st.session_state.stat:
    st.session_state.stat = True
    '---'
    col1, col2, col3 = st.columns((2,0.001,1))
    with col1:
        st.markdown("<div class = 'small-font'>Duplicate Values. Using datset.duplicated()</div>", unsafe_allow_html= True)
        st.write('')
        duplicates = data[data.duplicated()]
        st.dataframe(duplicates)

    with col3:
        st.markdown("<div class = 'small-font'>Shape of Dataset. Using datset.shape()</div>", unsafe_allow_html= True)
        st.write('')
        initial_shape = data.shape
        shape_after_duplicates_dropped = data.drop_duplicates().shape
        shape_after_nulls_dropped = data.dropna().shape
        num_nulls = data.isna().sum().sum()
        num_duplicates = data.duplicated().sum()
        col1, col2, col3 = st.columns(3)
        # Use the st.metric function to display these values
        with col2:
            st.metric(label="**Initial Dataset Shape**", value=str(initial_shape))
        col1, col2, col3 = st.columns((3,0.001, 3))
        with col1:
            st.write('')
            st.write('')
            st.write('')
            st.metric(label="**Number of Null Values**", value=str(num_nulls))
        with col3:
            st.write('')
            st.write('')
            st.write('')
            st.metric(label="**Number of Duplicates**", value=str(num_duplicates))
        with col1:
            st.write('')
            st.write('')
            st.write('')
            st.metric(label="**Shape After Dropping Duplicates**", value=str(shape_after_duplicates_dropped), delta=f"-{initial_shape[0] - shape_after_duplicates_dropped[0]} rows")
        with col3:
            st.write('')
            st.write('')
            st.write('')
            st.metric(label="**Shape After Dropping Null Values**", value=str(shape_after_nulls_dropped), delta=f"-{initial_shape[0] - shape_after_nulls_dropped[0]} rows", delta_color= 'inverse')
    st.write('---')

# ... Stat button end


    # Removing Duplicates
    st.markdown("<div class = 'page-font' style = 'text-align: center'>Let's Remove the Duplicates as Duplicates lead to <span class = 'color'>Inaccurate Results</span> and cause of <span class = 'color'>Bias in Machine Learning Models</span>.", unsafe_allow_html= True)
    st.write('')
    duplicate = st.button('# Drop Duplicates')
    changebtn("Drop Duplicates", "25px", "20%", "60%")
    if duplicate or st.session_state.duplicate:
        st.session_state.duplicate = True
        ndata = data.drop_duplicates()
        st.text('Duplicates removed Successfully')
    st.write('---')

# ... duplibicate button end

if duplicate or st.session_state.duplicate:
    st.session_state.duplicate = True
    # Three columns. Deletion, Imputation, Understanding the Reason
    st.markdown("<div class = 'page-font' style = 'text-align: center'>Let's go through the <span class = 'color'>NULL Cells</span>. As <span class = 'color'>NULL Cells</span> lead to <span class = 'color'>Inaccuracy in Results</span>, cause of <span class = 'color'>Incompatibility with Machine Learning Algorithms</span>, and <span class = 'color'>Misrepresentation of Data</span>.", unsafe_allow_html= True)
    st.text('There are several ways to handle the NULL Cells.')

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class = 'page-font' style = 'color: red;'>Deletion:</div>", unsafe_allow_html= True)
        st.write('If the proportion of null values is small, you might choose to simply delete the rows or columns that contain them. However, this can lead to loss of information.')
        

    with col2:
        st.markdown("<div class = 'page-font' style = 'color: red;'>Imputation:</div>", unsafe_allow_html= True)
        st.write('Imputation means filling in the null values with some value. This could be a central tendency measure like the mean or median, or a prediction from a machine learning model, or simply a constant value like zero.')
        


    with col3:
        st.markdown("<div class = 'page-font' style = 'color: red;'>Understanding the Reason:</div>", unsafe_allow_html= True)
        st.write("Try to understand why the data is missing. Is it missing at random, or is there a pattern to the missing data? If it's the latter, the fact that a value is missing might in itself be informative.")

if duplicate or st.session_state.duplicate:
    st.session_state.duplicate = True
    # Three buttons. Drop NuLL, replace with mean, edit your self
    col1, col2, col3 = st.columns(3)

    with col1:
        drop_null = st.button('# Drop NULL')
        changebtn("Drop NULL", "20px", "5%", "90%")

    with col2:
        replace_with_mean = st.button('# Replace with Mean')
        changebtn("Replace with Mean", "20px", "5%", "90%")

    with col3:
        edit = st.button('# Edit Yourself')
        changebtn("Edit Yourself", "20px", "5%", "90%")

    # Button actions and initial states
    if drop_null:
        st.session_state.drop_null = True
        st.session_state.replace_with_mean = False
        st.session_state.edit = False
        st.session_state.see_change = True  # ready to see changes

    elif replace_with_mean:
        st.session_state.replace_with_mean = True
        st.session_state.drop_null = False
        st.session_state.edit = False
        st.session_state.see_change = True  # ready to see changes 

    elif edit or st.session_state.edit:
        st.session_state.edit = True
        st.session_state.drop_null = False
        st.session_state.replace_with_mean = False
        columns_with_nulls = ndata.columns[ndata.isnull().any()]
        filtered_data = ndata[columns_with_nulls]
        edited_data = AgGridPro(filtered_data, editable=True, fit_columns_on_grid_load=True, update_mode="value_changed", cache=True, height= 400)
        # Now we map the edited data back to original dataframe
        for column in columns_with_nulls:
            ndata[column] = edited_data['data'][column]
        st.session_state.see_change = True  # ready to see changes

    if st.session_state.drop_null:
        ndata = ndata.dropna(axis=0)
        st.text('NULL Cells droped')

    elif st.session_state.replace_with_mean:
        numeric_cols = ndata.select_dtypes(include=[np.number])
        # We calculate the mean of numeric_cols and fill NA values in original dataframe
        ndata.fillna(numeric_cols.mean(), inplace=True)
        st.warning('🧨 Only Numerical Data is Replaced.')
        st.text('Successfully replaced with mean')

    # Check session state and display the 'See Changes' button
    if replace_with_mean or drop_null or edit or st.session_state.drop_null or st.session_state.replace_with_mean or st.session_state.edit:
        st.markdown("<div class = 'page-font' style = 'text-align: center'>Let's See the <span class = 'color'> Changes </span> in the <span class = 'color'> Dataset</span>.", unsafe_allow_html= True)
        ('')
        see_change = st.button("# See Changes!")
        changebtn("See Changes!", "25px", "20%", "60%")
        
        if see_change:
            ndata = AgGridPro(ndata, fit_columns_on_grid_load=True, cache=True, height= 400)
    st.write('---')
# ... Null part end


# save data button

def save_data(df, file_format):
    if file_format == 'CSV':
        data = df.to_csv(index=False)
        st.download_button("Download CSV", data, file_name="data.csv")
    elif file_format == 'Excel':
        towrite = io.BytesIO()
        downloaded_file = df.to_excel(towrite, index=False, engine='openpyxl')
        towrite.seek(0)
        st.download_button("Download Excel", towrite, file_name="data.xlsx")
    elif file_format == 'Text':
        data = df.to_csv(sep='\t', index=False)
        st.download_button("Download Text", data, file_name="data.txt")
    elif file_format == 'Pickle':
        towrite = io.BytesIO()
        pickle.dump(df, towrite)
        towrite.seek(0)
        st.download_button("Download Pickle", towrite, file_name="data.pkl")
    elif file_format == 'Python':
        data = df.to_csv(index=False)
        st.download_button("Download Python", data, file_name="data.py")
    else:
        st.error('Select a file format to save data.')

if st.session_state.see_change:
    st.markdown("<div class = 'page-font' style = 'text-align: center'>Store Data for <span class = 'color'> Future Use</span>, in any Format.", unsafe_allow_html= True)
    ('')
    save = st.button('# Save Data!', key = 'cleandata')
    changebtn("Save Data!", "25px", "20%", "60%")
    if save:
        st.session_state.save = True

if st.session_state.save:
    col1, col2, col3 = st.columns((1 ,1 , 1))
    with col2:
        fformat = st.selectbox('Select the file format for saving:', ['CSV', 'Excel', 'Text', 'Pickle', 'Python'], key="save_format")
    col1, col2, col3 = st.columns((1.5 ,1 , 1))
    with col2:
        save_data(ndata, fformat)
    st.write('---')

# ... Save for clean data

if st.session_state.edit or st.session_state.drop_null or st.session_state.replace_with_mean:
    st.markdown("<div class = 'page-font' style = 'text-align: center'>Check, How to Create <span class = 'color'> Pivot Table </span>?", unsafe_allow_html= True)
    ('')
    pivot_table = st.button('# Pivot Table')
    changebtn("Pivot Table", "25px", "20%", "60%")

    if pivot_table:
        st.session_state.pivot_table = True

    if st.session_state.pivot_table:
        st.text('Select columns for Pivot Table')
        col1, col2, col3, col4 = st.columns((1, 1, 1, 1))
        with col1:
            values = st.multiselect('Select one or more columns for the pivot table data', ndata.columns.tolist(), default=ndata.columns[0], key="pivot_values")
        with col2:
            index = st.multiselect('Select one or more index columns', ndata.columns.tolist(), default=ndata.columns[1], key="pivot_index")
        with col3:
            columns = st.multiselect('Select one or more columns for the pivot table columns', ndata.columns.tolist(), default=ndata.columns[2], key="pivot_columns")
        with col4:
            aggfunc = st.selectbox('Choose an aggregation function', ['mean', 'sum', 'count'], 0, key="pivot_aggfunc")

        pivot = pd.pivot_table(data, values=values, index=index, columns=columns, aggfunc=aggfunc)
        st.dataframe(pivot)

        c1, c2, c3 = st.columns((1,1,1))

        with c2:
            # File format selection and download button
            fformat = st.selectbox('Select the file format for saving:', ['CSV', 'Excel', 'Text', 'Pickle', 'Python'], key="pivot_save_format")

        col1, col2, col3 = st.columns((1.4 ,1 , 1))
        with col2:
            savep = st.button(f'Save as {fformat}', key="pivot_save_button")

        if savep:
            st.session_state.savep = True
            save_data(pivot, fformat)
        st.write('---')



# ... pivot tables end

# Group by button

if st.session_state.edit or st.session_state.drop_null or st.session_state.replace_with_mean:
    st.markdown("<div class = 'page-font' style = 'text-align: center'>You can<span class = 'color'> GroupBy </span> Data by any column.", unsafe_allow_html= True)
    ('')
    groubby = st.button('# GroupBy')
    changebtn("GroupBy", "25px", "20%", "60%")

if groubby:
    st.session_state.groubby = True

if st.session_state.groubby:
    st.text('Select columns for Group By')
    col1, col2, col3 = st.columns((1,1,1))
    with col1:
        group_cols = st.multiselect('Select one or more group columns', ndata.columns.tolist())
    with col2:
        aggregate_cols = st.multiselect('Select one or more aggregate columns', ndata.select_dtypes(include=['int', 'float']).columns.tolist())
    with col3:
        aggregate_func = st.selectbox('Choose an aggregation function', ['mean', 'sum', 'count', 'max', 'min'])


    if len(group_cols) > 0 and len(aggregate_cols) > 0:
        group_data = ndata.groupby(group_cols)[aggregate_cols].agg(aggregate_func)
        st.dataframe(group_data)

        c1, c2, c3 = st.columns((1,1,1))

        with c2:
            fformat = st.selectbox('Select the file format for saving:', ['CSV', 'Excel', 'Text', 'Pickle', 'Python'], key="group")

        c1, c2, c3 = st.columns((1.5,1,1))
        with c2:    
            saveg = st.button(f'Save as {fformat}')
    
        if saveg:
            st.session_state.saveg = True
            save_data(group_data, fformat)
    else:
        st.error('Please select at least one group column and one aggregate column.')
    st.write('---')

# ... GroupBy End



# single column Graphes

def charts(ndata, col):
    c1,c3, c2 = st.columns((1.8, 0.1, 2.1))
    if ndata[col].dtype in ['int64', 'float64']:
        with c1:
            sns.histplot(ndata[col])
            st.pyplot(plt.gcf())
        
        with c2:
            fig = px.box(ndata, y=col)
            st.plotly_chart(fig)
        
        with c1:
            fig = px.line(ndata, x=ndata.index, y=col)
            st.plotly_chart(fig)
        
    elif ndata[col].dtype == 'object':
        with c1:
            sns.histplot(ndata[col])
            st.pyplot(plt.gcf())

        with c2:
            fig = px.pie(ndata, names=col)
            st.plotly_chart(fig)
            
        with c2:
            st.write(f"{col} has an unsupported data type {ndata[col].dtype}")
        
    elif ndata[col].dtype == 'bool':
        with c1:
            sns.histplot(ndata[col])
            st.pyplot(plt.gcf())
        
        with c2:
            fig = px.pie(ndata, names=col)
            st.plotly_chart(fig)
            
        with c2:
            st.write(f"{col} has an unsupported data type {ndata[col].dtype}")
    else:
        st.write(f"{col} has an unsupported data type {ndata[col].dtype}")

st.write('')
st.write('')


if st.session_state.edit or st.session_state.drop_null or st.session_state.replace_with_mean:
    st.markdown("<div class = 'page-font' style = 'text-align: center'>Go Through the Interesting Part of the Data.<span class = 'color'> The Visualizes.</span> Single Column Visualizes.", unsafe_allow_html= True)
    ('')
    single = st.button('# Graph of single column')
    changebtn("Graph of single column", "25px", "20%", "60%")


if single:
    st.session_state.single = True

if st.session_state.single:
    c1, c2, c3 = st.columns(3)
    columns_list = ndata.columns.tolist()
    with c2:
        col = st.selectbox("Select a column to visualize", options=columns_list)

    charts(ndata, col)
    st.write('---')

# ... single chart end


if st.session_state.edit or st.session_state.drop_null or st.session_state.replace_with_mean:
    st.markdown("<div class = 'page-font' style = 'text-align: center'>Go Through the Interesting Part of the Data.<span class = 'color'> The Visualizes.</span> 2D Visualizes.", unsafe_allow_html= True)
    ('')
    d2 = st.button('# Graph of two columns')
    changebtn("Graph of two columns", "25px", "20%", "60%")


if d2:
    st.session_state.d2 = True

if st.session_state.d2:

    st.write('')
    st.write('')
    col1, col2, col3 = st.columns((1.8, 0.1, 1.3))
    with col3:
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        column1 = st.selectbox('Select the first column (X - Axis)', ndata.columns.tolist())
        column2 = st.selectbox('Select the second column (Y - Axis)', ndata.columns.tolist())

    # Check if the selected columns are not empty
    if column1 and column2:
        with col3:
            plot_type = st.selectbox("Select type of plot", ["scatter", "bar", "box", "area", "pie"])
        
        
        # Plot based on the selected plot type
        if plot_type == "scatter":
            fig = px.scatter(ndata, x=column1, y=column2)
        elif plot_type == "bar":
            fig = px.bar(ndata, x=column1, y=column2)
        elif plot_type == "box":
            fig = px.box(ndata, x=column1, y=column2)
        elif plot_type == "area":
            fig = px.area(ndata, x=column1, y=column2)
        elif plot_type == "pie":
            fig = px.pie(ndata, values=column2, names=column1)
        
        with col1:
            # Display the plot
            st.plotly_chart(fig)

    else:
        st.write("No columns selected. Please select columns.")
    st.write('---')


# ... 2D Graphes End

if st.session_state.edit or st.session_state.drop_null or st.session_state.replace_with_mean:
    st.markdown("<div class = 'page-font' style = 'text-align: center'>Go Through the Interesting Part of the Data.<span class = 'color'> The Visualizes.</span> 3D Visualizes.", unsafe_allow_html= True)
    ('')
    d3 = st.button('# 3D Graphes')
    changebtn("3D Graphes", "25px", "20%", "60%")

if d3:
    st.session_state.d3 = True

if st.session_state.d3:
    st.write('')
    st.write('')

    col1, col2, col3 = st.columns((1.8, 0.1, 2.2))

    with col1:
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
    
        columns = st.multiselect('Select the columns you want to plot', ndata.columns.tolist(), default=ndata.columns[:3].tolist())

    # Check if the selected columns list is not empty
    if columns:
        # Check if more than two column is selected
        if len(columns) > 2:
            # Select type of plot
            with col1:
                plot_type = st.selectbox("Select type of plot", ["3D Scatter", "3D Line", "3D Surface", "3D Bubble", "3D Cone"])

            fig = None  # Initialize fig with a default value

            # Plot based on the selected plot type
            if plot_type == "3D Scatter":
                fig = px.scatter_3d(ndata, x=columns[0], y=columns[1], z=columns[2])
            elif plot_type == "3D Line":
                fig = px.line_3d(ndata, x=columns[0], y=columns[1], z=columns[2])
            elif plot_type == "3D Surface":
                # Please ensure that the selected columns can form a valid surface plot
                fig = go.Figure(data=[go.Surface(z=ndata[columns].values)])
            elif plot_type == "3D Bubble":
                fig = px.scatter_3d(ndata, x=columns[0], y=columns[1], z=columns[2], size=ndata[columns[0]], color=ndata[columns[1]], hover_name=ndata[columns[2]], symbol=ndata[columns[0]], opacity=0.7)
            elif plot_type == "3D Cone":
                fig = go.Figure(data=[go.Cone(x=ndata[columns[0]], y=ndata[columns[1]], z=ndata[columns[2]], u=ndata[columns[0]], v=ndata[columns[1]], w=ndata[columns[2]])])
            # If fig is not None, display the plot
            if fig is not None:
                with col3:
                    st.plotly_chart(fig)
            else:
                st.write("Please select a valid plot type.")
        else:
            st.write("Please select at least three columns to plot.")
    else:
        st.write("No columns selected. Please select columns.")
    ('---')

# ... 3D End

# Back button

back = st.button('< Back')
if back:
    switch_page('main')

# ... Back button End


# Footer
# Edit footer
footer="""<style>
footer {
	
	visibility: hidden;
	
	}

a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: none;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: none !important;
transform: scale(1.04);
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: none !important;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p style = 'font-size: 10px;'>Developed by ❤<a style='display: block; text-align: center;' href="https://www.linkedin.com/in/hazrat-bilal-3642b4228/" target="_blank">Hazrat Bilal</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)

# ... Footer End
