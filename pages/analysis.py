import streamlit as st
import pandas as pd
import numpy as np
import io
from io import StringIO, BytesIO
import streamlit.components.v1 as components
import sys
from st_aggrid_pro import AgGridPro
from streamlit_extras.switch_page_button import switch_page
import plotly.express as px
import base64
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics as sm
from ast import literal_eval
from streamlit_pandas_profiling import st_profile_report
import pandas_profiling


# To have Wide page
st.set_page_config(page_title="Analytics & Predictions",
                   layout="wide",
                   initial_sidebar_state="collapsed",
                   page_icon= '☠')

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
<h1 class='welcome-font'>Welcome <br> To Analytics & Predictions Phase! 🙋‍♂️</h1><br>
"""
st.markdown(welcome_message, unsafe_allow_html=True)

st.markdown("<div class = 'page-font' ><span class = 'color'>Load Dataset</span>, To Add a Dataset for Analysis.</div>", unsafe_allow_html=True)
st.write('')

# session_state for load data button
if 'dataset_bnt' not in st.session_state:
    st.session_state.dataset_bnt = False
dataset_bnt = False

# session_state for file load button
if 'file' not in st.session_state:
    st.session_state.file = False
file = False

# sesstion_state for statistics button
if 'stat' not in st.session_state:
    st.session_state.stat = False
stat = False

# sesstion_state for statistics button
if 'duplicate' not in st.session_state:
    st.session_state.duplicate = False
duplicate = False

# sesstion_state for worning message
if 'wor' not in st.session_state:
    st.session_state.wor = False
wor = False

# sesstion_state for null session
if 'null' not in st.session_state:
    st.session_state.null = False
null = False

# sesstion_state for null session
if 'null2' not in st.session_state:
    st.session_state.null2 = False
null2 = False

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

# sesstion_state for 3d graphes button
if 'regression' not in st.session_state:
    st.session_state.regression = False
regression = False

# sesstion_state for 3d graphes button
if 'deep' not in st.session_state:
    st.session_state.deep = False
deep = False

# defining some variables
data = None


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
        file = st.file_uploader('Choose a file', type=['csv', 'xlsx', 'xls', 'txt', 'json', 'html', 'xml'])
    if file:
        with col3:
            st.write('')
            st.write('')
            st.markdown("<div class = 'page-font'>Dataset Successfully Uploaded.</div>", unsafe_allow_html=True)
            st.write(file)
        data = pd.read_csv(file, encoding='latin-1')
st.write('---')

# Head and Tail
if st.session_state.file:
    st.markdown("<div class = 'page-font'><span class = 'color'>Head</span>", unsafe_allow_html= True)
    st.write('')
    st.table(data.head())
    st.markdown("<div class = 'page-font'><span class = 'color'>Tail</span>", unsafe_allow_html= True)
    st._legacy_table(data.tail())
    st.write('---')
    
    # showing statistics button
    st.markdown("<div class='stat-button'></div>", unsafe_allow_html=True)
    stat = st.button('# Statistics')
    changebtn("Statistics", "30px", "20%", "60%")

# ... load data set end


# statistics button start

if stat:
    st.session_state.stat =True

if st.session_state.stat:
    col1, col2, col3= st.columns((0.1, 1, 0.1))
    with col2:
        ('')
        st.markdown("<div class = 'page-font'><span class = 'color'>Statistics.</div>", unsafe_allow_html=True)
        st.write('')
        formatted_data = data.describe().applymap(lambda x: "{:.1f}".format(x))
        st._legacy_table(formatted_data)
        st.text('You can see the Statistics of the Dataset. Count is the number of rows. Mean, standard deviation, Minimun and Maximum of each column.\nYou can also Analyse the 4-Quartile. 1st Quartile from 0-25%. 2nd Quartile from 25-50%.\n3rd Quartile from 50-75%. 4th Quartile form 75-100%')
        
    with col2:
        st.markdown("<div class = 'page-font'><span class = 'color'>Information About Dataset.</span></div>", unsafe_allow_html=True)
        st.write('')
        def get_info(df):
            buf = io.StringIO()
            df.info(buf=buf)
            return buf.getvalue()
        st.text(get_info(data))


if stat or st.session_state.stat:
    st.session_state.stat = True
    '---'
    col1, col2, col3 = st.columns((0.6,2,0.1))
    with col2:
        ('')
        ('')
        st.markdown("<div class = 'page-font'><span class = 'color'>Shape Of Dataset.</div>", unsafe_allow_html=True)
        ('')
    with col2:
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
            st.metric(label="**Shape After Dropping Null Values**", value=str(shape_after_nulls_dropped), delta=f"-{initial_shape[0] - shape_after_nulls_dropped[0]} rows", delta_color= 'inverse')
        with col3:
            st.write('')
            st.write('')
            st.write('')
            st.metric(label="**Shape After Dropping Duplicates**", value=str(shape_after_duplicates_dropped), delta=f"-{initial_shape[0] - shape_after_duplicates_dropped[0]} rows")
    st.write('---')

if stat or st.session_state.stat:
    st.session_state.stat = True
    st.write('')
    st.markdown("<div class='page-font' style='text-align: center;'><span class='color'>NULL values</span> and <span class='color'>Duplicates</span>.</div>", unsafe_allow_html=True)
    st.write('')
    col1, col2 = st.columns((1, 1))
    with col1:
        nan_data = pd.DataFrame(data.isna().sum()).reset_index()
        nan_data.columns = ['Column Name', 'Number of NULL values']
        st.write('')
        st.write('')
        st.table(nan_data)

    with col2:
        data_types = pd.DataFrame(data.dtypes, columns=['Data Type'])
        st.write('')
        st.write('')
        st.table(data_types)

# ... stat button 

# Duplicates start
    ('---')
    st.markdown("<div class = 'page-font'><span class = 'color'>Duplicates.</div>", unsafe_allow_html=True)
    duplicates = data[data.duplicated()]
    if duplicates.empty:
        ('')
        st.warning('Congrates! There is no Duplicates in Dataset')
        ('---')
        st.session_state.null = True
        
    else:
        ('')
        ('')
        AgGridPro(duplicates, fit_columns_on_grid_load=True, cache=True, height= 200)
        ('')
        ('')

        if st.session_state.stat:
            duplicate = st.button('# Drop Duplicates')
            changebtn("Drop Duplicates", "25px", "20%", "60%")
           

        if duplicate:
            st.session_state.duplicate = True
            data = data.drop_duplicates()
            st.text('Duplicates removed Successfully')
            st.session_state.null = True
        st.write('---')

# ... Duplicates button end

if st.session_state.null:

    if nan_data.iloc[:, 1].sum() == 0:
        st.success('Congrats! There are no null values in the dataset.')
        st.write('---')
        st.session_state.null2 = True
    else:
        st.markdown("<div class='page-font' style='text-align: center'>Let's go through the <span class='color'>NULL Cells</span>. NULL Cells can lead to <span class='color'>inaccuracy in results</span>, cause <span class='color'>incompatibility with machine learning algorithms</span>, and <span class='color'>misrepresentation of data</span>.</div>", unsafe_allow_html=True)
        st.write('There are several ways to handle NULL Cells.')

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<div class='page-font' style='color: red;'>Deletion:</div>", unsafe_allow_html=True)
            st.write('If the proportion of null values is small, you might choose to simply delete the rows or columns that contain them. However, this can lead to a loss of information.')

        with col2:
            st.markdown("<div class='page-font' style='color: red;'>Imputation:</div>", unsafe_allow_html=True)
            st.write('Imputation means filling in the null values with some value. This could be a central tendency measure like the mean or median, or a prediction from a machine learning model, or simply a constant value like zero.')

        with col3:
            st.markdown("<div class='page-font' style='color: red;'>Understanding the Reason:</div>", unsafe_allow_html=True)
            st.write("Try to understand why the data is missing. Is it missing at random, or is there a pattern to the missing data? If it's the latter, the fact that a value is missing might itself be informative.")

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
            columns_with_nulls = data.columns[data.isnull().any()]
            filtered_data = data[columns_with_nulls]
            edited_data = AgGridPro(filtered_data, editable=True, fit_columns_on_grid_load=True, update_mode="value_changed", cache=True, height=400)
            for column in columns_with_nulls:
                data[column] = edited_data['data'][column]
            st.session_state.see_change = True  # ready to see changes

        if st.session_state.drop_null:
            data = data.dropna(axis=0)
            st.text('NULL Cells dropped')

        elif st.session_state.replace_with_mean:
            numeric_cols = data.select_dtypes(include=[np.number])
            data.fillna(numeric_cols.mean(), inplace=True)
            st.warning('🧨 Only Numerical Data is Replaced.')
            st.text('Successfully replaced with mean')

        if replace_with_mean or drop_null or edit or st.session_state.drop_null or st.session_state.replace_with_mean or st.session_state.edit:
            see_change = st.button("# See Changes!")
            changebtn("See Changes!", "25px", "20%", "60%")
            
            if see_change:
                data = AgGridPro(data, fit_columns_on_grid_load=True, cache=True, height=400)

        st.write('---')

# ... Null part end


# Save Data
    
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

if st.session_state.see_change or st.session_state.null2:
    save = st.button('# Save Data!', key = 'cleandata')
    changebtn("Save Data!", "25px", "20%", "60%")
    if save:
        st.session_state.save = True

if st.session_state.save:
    col1, col2, col3 = st.columns((1 ,1 , 1))
    with col2:
        fformat = st.selectbox('Select the file format for saving:', ['CSV', 'Excel', 'Text', 'Pickle', 'Python'], key="save_format")
    col1, col2, col3 = st.columns((1.4 ,1 , 1))
    with col2:
        save_data(data, fformat)
    st.write('---')

# ... Save data End




if st.session_state.null2 or st.session_state.edit or st.session_state.drop_null or st.session_state.replace_with_mean:
    pivot_table = st.button('# Pivot Table')
    changebtn("Pivot Table", "25px", "20%", "60%")

    if pivot_table:
        st.session_state.pivot_table = True

    if st.session_state.pivot_table:
        st.text('Select columns for Pivot Table')
        col1, col2, col3, col4 = st.columns((1, 1, 1, 1))
        with col1:
            values = st.multiselect('Select one or more columns for the pivot table data', data.columns.tolist(), default=data.columns[0], key="pivot_values")
        with col2:
            index = st.multiselect('Select one or more index columns', data.columns.tolist(), default=data.columns[1], key="pivot_index")
        with col3:
            columns = st.multiselect('Select one or more columns for the pivot table columns', data.columns.tolist(), default=data.columns[2], key="pivot_columns")
        with col4:
            aggfunc = st.selectbox('Choose an aggregation function', ['mean', 'sum', 'count'], 0, key="pivot_aggfunc")

        pivot = pd.pivot_table(data, values=values, index=index, columns=columns, aggfunc=aggfunc)
        st.dataframe(pivot, use_container_width=True)

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

if st.session_state.null2 or st.session_state.edit or st.session_state.drop_null or st.session_state.replace_with_mean:
    groubby = st.button('# GroupBy')
    changebtn("GroupBy", "25px", "20%", "60%")

if groubby:
    st.session_state.groubby = True

if st.session_state.groubby:
    st.text('Select columns for Group By')
    col1, col2, col3 = st.columns((1,1,1))
    with col1:
        group_cols = st.multiselect('Select one or more group columns', data.columns.tolist())
    with col2:
        aggregate_cols = st.multiselect('Select one or more aggregate columns', data.select_dtypes(include=['int', 'float']).columns.tolist())
    with col3:
        aggregate_func = st.selectbox('Choose an aggregation function', ['mean', 'sum', 'count', 'max', 'min'])


    if len(group_cols) > 0 and len(aggregate_cols) > 0:
        group_data = data.groupby(group_cols)[aggregate_cols].agg(aggregate_func)
        st.dataframe(group_data, use_container_width=True)

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

def charts(data, col):
    c1,c3, c2 = st.columns((1.8, 0.1, 2.1))
    if data[col].dtype in ['int64', 'float64']:
        with c1:
            sns.histplot(data[col])
            st.pyplot(plt.gcf())
        
        with c2:
            fig = px.box(data, y=col)
            st.plotly_chart(fig)
        
        with c1:
            fig = px.line(data, x=data.index, y=col)
            st.plotly_chart(fig)
        
    elif data[col].dtype == 'object':
        with c1:
            sns.histplot(data[col])
            st.pyplot(plt.gcf())

        with c2:
            fig = px.pie(data, names=col)
            st.plotly_chart(fig)
            
        with c2:
            st.write(f"{col} has an unsupported data type {data[col].dtype}")
        
    elif data[col].dtype == 'bool':
        with c1:
            sns.histplot(data[col])
            st.pyplot(plt.gcf())
        
        with c2:
            fig = px.pie(data, names=col)
            st.plotly_chart(fig)
            
        with c2:
            st.write(f"{col} has an unsupported data type {data[col].dtype}")
    else:
        st.write(f"{col} has an unsupported data type {data[col].dtype}")

st.write('')
st.write('')


if st.session_state.null2 or st.session_state.edit or st.session_state.drop_null or st.session_state.replace_with_mean:
    single = st.button('# Graph of single column')
    changebtn("Graph of single column", "25px", "20%", "60%")


if single:
    st.session_state.single = True

if st.session_state.single:
    st.markdown("<div class='page-font'>Goodbye Boredom, Hello <span class='color'>Visual Delight!</span> 🎉📊</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    columns_list = data.columns.tolist()
    with c2:
        col = st.selectbox("Select a column to visualize", options=columns_list)

    charts(data, col)
    st.write('---')

# ... single chart end


if st.session_state.null2 or st.session_state.edit or st.session_state.drop_null or st.session_state.replace_with_mean:
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
        column1 = st.selectbox('Select the first column (X - Axis)', data.columns.tolist())
        column2 = st.selectbox('Select the second column (Y - Axis)', data.columns.tolist())

    # Check if the selected columns are not empty
    if column1 and column2:
        with col3:
            plot_type = st.selectbox("Select type of plot", ["scatter", "bar", "box", "area", "pie"])
        
        
        # Plot based on the selected plot type
        if plot_type == "scatter":
            fig = px.scatter(data, x=column1, y=column2)
        elif plot_type == "bar":
            fig = px.bar(data, x=column1, y=column2)
        elif plot_type == "box":
            fig = px.box(data, x=column1, y=column2)
        elif plot_type == "area":
            fig = px.area(data, x=column1, y=column2)
        elif plot_type == "pie":
            fig = px.pie(data, values=column2, names=column1)
        
        with col1:
            # Display the plot
            st.plotly_chart(fig)

    else:
        st.write("No columns selected. Please select columns.")
    st.write('---')


# ... 2D Graphes End

if st.session_state.null2 or st.session_state.edit or st.session_state.drop_null or st.session_state.replace_with_mean:
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
    
        columns = st.multiselect('Select the columns you want to plot', data.columns.tolist(), default=data.columns[:3].tolist())

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
                fig = px.scatter_3d(data, x=columns[0], y=columns[1], z=columns[2])
            elif plot_type == "3D Line":
                fig = px.line_3d(data, x=columns[0], y=columns[1], z=columns[2])
            elif plot_type == "3D Surface":
                # Please ensure that the selected columns can form a valid surface plot
                fig = go.Figure(data=[go.Surface(z=data[columns].values)])
            elif plot_type == "3D Bubble":
                fig = px.scatter_3d(data, x=columns[0], y=columns[1], z=columns[2], size=data[columns[0]], color=data[columns[1]], hover_name=data[columns[2]], symbol=data[columns[0]], opacity=0.7)
            elif plot_type == "3D Cone":
                fig = go.Figure(data=[go.Cone(x=data[columns[0]], y=data[columns[1]], z=data[columns[2]], u=data[columns[0]], v=data[columns[1]], w=data[columns[2]])])
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

# Deep Dive button

if st.session_state.null2 or st.session_state.edit or st.session_state.drop_null or st.session_state.replace_with_mean:
    deep = st.button('# Deep Drive into Dataset')
    changebtn("Deep Drive into Dataset", "25px", "20%", "60%")

if deep:
    st.session_state.deep = True

if st.session_state.deep:
    
    pr = data.profile_report()
    st_profile_report(pr)

# ... Deep Dive Button end

# supervies Machine Learning Algos.

def linear_regression(data):
    ('')
    ('')
    ('')
    column_list = list(data.columns)
    col1, col2, col3 = st.columns(3)

    with col1:
        # Create a dropdown widget for selecting X column
        x_column = st.selectbox("Select X column:", column_list, key='x_column')

    with col2:
        # Create a dropdown widget for selecting Y column
        y_column = st.selectbox("Select Y column:", column_list, key='y_column')

    # Prepare X and Y data
    
    X = data[[x_column]]
    Y = data[y_column]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    model = LinearRegression()
    model = model.fit(X_train, y_train)

    with col3:
    # User input for prediction
        value = st.text_input("Enter a number or an array for prediction:")
        prediction = None

    try:
        # Try converting the input to a float (for a single number)
        value = float(value)
        prediction = model.predict([[value]])
    except ValueError:
        # If conversion to a float fails, assume it's an array
        try:
            # Try evaluating the input as a literal array
            value = literal_eval(value)
            value = np.array(value).reshape(-1, 1)  # Reshape to a 2D array
            prediction = model.predict(value)
        except (SyntaxError, ValueError):
            pass

    # Display the prediction result
    if prediction is not None:
        ('')
        ('')
        col1, col2, col3 = st.columns((0.6, 1, 0.1))
        with col2:
            st.write("Prediction:", prediction)

        # Perform prediction on the test data
        prediction = model.predict(X_test)

        # Display model evaluation scores
        st.subheader("Model Evaluation Scores")
        ('')
        col1, col2 = st.columns(2)
        ('')
        ('')
        with col1:
            st.metric(":blue[Score in Testing]", f"{model.score(X_test, y_test) * 100:.2f}%")
        with col2:
            st.metric(":blue[Score in Training]", f"{model.score(X_train, y_train) * 100:.2f}%")
        ('---')
        col1, col2, col3 = st.columns(3)
        ('')
        ('')
        with col1:
            st.metric(":green[Mean Absolute Error (MAE)]", round(sm.mean_absolute_error(y_test, prediction), 3), delta_color="off")
        with col2:
            st.metric(":green[Mean Squared Error (MSE)]", round(sm.mean_squared_error(y_test, prediction), 3), delta_color="off")
        with col3:
            st.metric(":green[Median Absolute Error (MAE)]", round(sm.median_absolute_error(y_test, prediction), 3), delta_color="off")
        with col1:
            st.metric(":green[Explained Variance Score (EVS)]", round(sm.explained_variance_score(y_test, prediction), 3), delta_color="off")
        with col3:
            st.metric(":green[R^2 (R-square Score)]", round(sm.r2_score(y_test, prediction), 3), delta_color="off")




if st.session_state.null2 or st.session_state.edit or st.session_state.drop_null or st.session_state.replace_with_mean:
    
    regression = st.button('# Linear Regression')
    changebtn("Linear Regression", "25px", "20%", "60%")

if regression:
    st.session_state.regression = True
    st.session_state.deep = False

if st.session_state.regression:
    AgGridPro(data, fit_columns_on_grid_load=True, cache=True, height= 200)
    # Call the linear_regression function
    linear_regression(data)

# ... Regression End

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
