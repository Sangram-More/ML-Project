from statistics import correlation
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# --------------- Data Analystics code --------------------------
df_uncleaned = pd.read_csv(r"App/Tabs/Datasets/Merged_Data.csv")
df = pd.read_csv(r"App/Tabs/Datasets/finaldataset.csv")

df['date'] = pd.to_datetime(df['date'])



#--------------- Custom CSS Styling -----------------------------

# Custom CSS for text justification
st.markdown(
    """
    <style>
    .justified-text {
        text-align: justify;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# st.markdown("<p class='justified-text'></p>", unsafe_allow_html=True)

st.title("Data Preperation.")
st.header("Lets see how to gather data.")

# ---------------------------------------------------------------
# Section 1

st.markdown("<p class='justified-text'>The data for this project was gathered from the Federal Reserve Bank of St. Louis (FRED) API, a widely used economic data source. The researcher utilized the API to collect key economic indicators, including interest rates, GDP, inflation, unemployment rates, and consumer price indices. The data was retrieved programmatically to ensure accuracy, consistency, and real-time updates. The selection of FRED as the data source was driven by its reliability, comprehensive historical records, and accessibility. The collected data was then processed and cleaned to maintain uniformity across datasets, ensuring seamless integration for analysis. This dataset serves as the foundation for predicting U.S. Federal Reserve interest rates using machine learning techniques.</p>", unsafe_allow_html=True)

# Inserting code template:
st.markdown("#### Python Code Template:")

Api_code = '''import requests
import pandas as pd


def api_data_retrival(series_id, name):

    # API Key
    api_key = "d45a04206227ded2f814f8046869de4f"

    # Series ID for Dataset
    series_id = series_id

    # FRED API URL
    url = f"https://api.stlouisfed.org/fred/series/observations"

    # Parameters for API request
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",  # JSON response
    }

    # Making API request
    response = requests.get(url, params=params)

    # Checking if the response is successful
    if response.status_code == 200:
        # Parsing JSON data
        data = response.json()
        observations = data.get("observations", [])
        
        # Convert to DataFrame
        df = pd.DataFrame(observations)
        
        # Save to CSV
        df.to_csv(f"{name}.csv", index=False)
        print(f"Data saved to '{name}.csv'")
    else:
        print(f"Failed to retrieve data: {response.status_code} - {response.text}")



# ----- Initialing Series ID to fetch different datasets ----------

FEDRates = "FEDFUNDS"
UnemployemenrRate = "UNRATE"
GDP = "GDP"
RealGDP = "GDPC1"
RealPotentialGDP = "GDPPOT"
RealGDPPerCapita = "A939RX0Q048SBEA"
InflationConsumerPrice = "FPCPITOTLZGUSA"
ConsumerPriceIndexAllItems = "CPALTT01USM657N"
MedianConsumerPriceIndex = "MEDCPIM158SFRBCLE"

# --------- Calling the function to use API ----------
api_data_retrival(FEDRates, "FEDRates")
api_data_retrival(UnemployemenrRate, "UnemployemenrRate")
api_data_retrival(GDP, "GDP")
api_data_retrival(RealGDP, "RealGDP")
api_data_retrival(RealPotentialGDP, "RealPotentialGDP")
api_data_retrival(RealGDPPerCapita, "RealGDPPerCapita")
api_data_retrival(InflationConsumerPrice, "InflationConsumerPrice")
api_data_retrival(ConsumerPriceIndexAllItems, "ConsumerPriceIndexAllItems")
api_data_retrival(MedianConsumerPriceIndex, "MedianConsumerPriceIndex")'''

st.code(Api_code, language="python")

# ---------------------------------------------------------------
# Section 2
st.markdown("###")
st.markdown("<p class='justified-text'>The datasets obtained from the API are stored as separate CSV files, each containing specific financial or economic data. To facilitate analysis, these individual files need to be merged into a single comprehensive dataset. This consolidation process ensures that all relevant data points are organized in a structured format, enabling efficient processing and model training. A script is used to automate this merging process, combining multiple CSV files into one unified file for further analysis.</p>", unsafe_allow_html=True)

st.markdown("<p class='justified-text'>Here is a code snippet to do just the same.", unsafe_allow_html=True)

Merge_data_code = '''
import pandas as pd
import glob

# List of file paths (update with actual paths if running locally)
file_paths = [
    "Cleaned/ConsumerPriceIndexAllItemsCleaned.csv",
    "Cleaned/FEDRatesCleaned.csv",
    "Cleaned/GDPCleaned.csv",
    "Cleaned/InflationConsumerPriceCleaned.csv",
    "Cleaned/MedianConsumerPriceIndexCleaned.csv",
    "Cleaned/RealGDPCleaned.csv",
    "Cleaned/RealGDPPerCapitaCleaned.csv",
    "Cleaned/RealPotentialGDPCleaned.csv",
    "Cleaned/UnemployemenrRateCleaned.csv",
]

# Read CSV files, drop 'Unnamed: 0' if it exists, and rename value columns
dfs = []
for file in file_paths:
    df = pd.read_csv(file, parse_dates=["date"])  # Ensure 'date' is in datetime format
    df = df.drop(columns=[col for col in df.columns if "Unnamed" in col], errors="ignore")  # Drop unnecessary columns
    value_col = [col for col in df.columns if col != "date"][0]  # Identify the column with values
    df = df.rename(columns={value_col: file.replace("Cleaned.csv", "").replace(".csv", "")})  # Rename column
    dfs.append(df)

# Merge all DataFrames on 'date' using an outer join
merged_df = dfs[0]
for df in dfs[1:]:
    merged_df = pd.merge(merged_df, df, on="date", how="outer")

# Convert NaN values to empty strings for better visualization
merged_df = merged_df.fillna("")

# Sort by date
merged_df = merged_df.sort_values(by="date")

# Save the merged DataFrame to a CSV file (optional)
merged_df.to_csv("Cleaned/Merged_Data.csv", index=False)

# Display the merged DataFrame
print(merged_df.head())  # You can replace this with any other method to visualize the data
'''

st.code(Merge_data_code, language="python")
st.divider()

# ---------------------------------------------------------------
# Section 3

st.subheader("Here is what you might see after merging all the datasets")
st.write(df_uncleaned.iloc[89:99].head(10))
st.markdown("<p class='justified-text'>This dataset is currently uncleaned dataset as you can see a lot of its cells contain NaN values. Below are the steps and methodologies used for cleaning the dataset.</p>", unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------
# Section 4

st.header("Data Cleaning Steps:")

st.subheader("Step 1: Loading and Exploring dataset.")
st.markdown("<p class='justified-text'>Using pandas library, the dataset is loaded and explored.</p>", unsafe_allow_html=True)

st.subheader("Step 2: Removing all entries where there is no value in Fed Rates column.")
st.markdown("<p class='justified-text'>Dataset for fed rates is between the timeframe of 1954 to 2024. Some variables do have values which are pre 1954 years and post 2024 years. Thus as now missing value was found betwwen the timeframe of 1954 to 2024, we remove all the extra entries that are pre 1954 and post 2024 as they do not contain any information regarding Fed Rates.</p>", unsafe_allow_html=True)

st.subheader("Step 3: Replacing Nan(Missing values).")
st.markdown("<p class='justified-text'>As the dataset we are working with is a time series dataset, thus we can't simplify replace nan values with mean, median or mode of the respective column. For example, values for GDP are released quaterly, thus for a given year we just have 4 vlaues for GDP and 12 values for Fed Rates. We use pandas fillna() function with parameter methid='ffill' for fill in the missing values. Now in this case, thr missing values for GPD will get filled with the value just before it (forward fill). This will help to maintain infromation in the dataset.</p>", unsafe_allow_html=True)

st.subheader("Step 3: Checking for relevant datatypes.")
st.markdown("<p class='justified-text'>Using the info function in pandas library, we check for the datatypes of every column.</p>", unsafe_allow_html=True)
st.image(r"App/Tabs/Images/uncleanedinfo.png", caption="Uncleaned Data Info")
st.markdown("<p class='justified-text'>As we can see that the column GDP has a datatype of object. In order to convert this to a float datatype, pandas to_numeric() function is used. Thus after procesing the info section looks like this:</p>", unsafe_allow_html=True)
st.image(r"App/Tabs/Images/cleanedinfo.png", caption="Cleaned Data Info")

st.subheader("Step 3: Use of Pandas describe function to compute statistical measures.")
st.write(df.describe())

st.divider()

# ---------------------------------------------------------------
# Section 5

st.subheader("Here is a summary of entire dataset using Visualizations:")


# Set the target variable
target_variable = "FEDRates"

# Streamlit App Title
st.title("Exploratory Data Analysis (EDA) - US Fed Rates Prediction")

# Sidebar for navigation
st.sidebar.title("Select a Plot to View")

plot_options = [
    "Histogram",
    "Boxplot",
    "Pairplot",
    "Correlation Heatmap",
    "Scatter Plot",
    "Violin Plot",
    "KDE Plot",
    "Countplot",
    "Time Series",
    "Interactive Scatter Plot",
    "Interactive Histogram",
    "Interactive Line Chart",
    "Interactive Heatmap",
    "Interactive Box Plot",
    "Interactive Parallel Coordinates"
]

selected_plot = st.sidebar.selectbox("Choose a Plot", plot_options)

# Generate and Display Plots
st.subheader(selected_plot)

if selected_plot == "Histogram":
    fig, ax = plt.subplots()
    sns.histplot(df[target_variable], bins=30, kde=True, ax=ax)
    ax.set_title(f"Histogram of {target_variable}")
    st.pyplot(fig)

elif selected_plot == "Boxplot":
    fig, ax = plt.subplots()
    sns.boxplot(y=df[target_variable], ax=ax)
    ax.set_title(f"Boxplot of {target_variable}")
    st.pyplot(fig)

elif selected_plot == "Pairplot":
    fig = sns.pairplot(df.select_dtypes(include=['number']), diag_kind="kde")
    st.pyplot(fig.fig)

elif selected_plot == "Correlation Heatmap":
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

elif selected_plot == "Scatter Plot":
    correlations = df.corr()[target_variable].abs().sort_values(ascending=False)
    top_feature = correlations.index[1]  # The most correlated feature
    fig, ax = plt.subplots()
    sns.scatterplot(x=df[top_feature], y=df[target_variable], ax=ax)
    ax.set_title(f"Scatter Plot of {target_variable} vs {top_feature}")
    st.pyplot(fig)

elif selected_plot == "Violin Plot":
    fig, ax = plt.subplots()
    sns.violinplot(y=df[target_variable], ax=ax)
    ax.set_title(f"Violin Plot of {target_variable}")
    st.pyplot(fig)

elif selected_plot == "KDE Plot":
    fig, ax = plt.subplots()
    sns.kdeplot(df[target_variable], ax=ax, shade=True)
    ax.set_title(f"KDE Plot of {target_variable}")
    st.pyplot(fig)

elif selected_plot == "Countplot":
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        fig, ax = plt.subplots()
        sns.countplot(x=df[categorical_cols[0]], ax=ax)
        ax.set_title(f"Countplot of {categorical_cols[0]}")
        st.pyplot(fig)
    else:
        st.warning("No categorical columns found.")

elif selected_plot == "Time Series":
    date_cols = [col for col in df.columns if 'date' in col.lower()]
    if len(date_cols) > 0:
        df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], errors='coerce')
        df_sorted = df.sort_values(by=date_cols[0])
        fig, ax = plt.subplots()
        sns.lineplot(x=df_sorted[date_cols[0]], y=df_sorted[target_variable], ax=ax)
        ax.set_title(f"Time Series Trend of {target_variable}")
        st.pyplot(fig)
    else:
        st.warning("No date column found.")

elif selected_plot == "Interactive Scatter Plot":
    correlations = df.corr()[target_variable].abs().sort_values(ascending=False)
    top_feature = correlations.index[1]
    fig = px.scatter(df, x=top_feature, y=target_variable,
                     title=f"Interactive Scatter Plot of {target_variable} vs {top_feature}",
                     trendline="ols")
    st.plotly_chart(fig)

elif selected_plot == "Interactive Histogram":
    fig = px.histogram(df, x=target_variable, title=f"Interactive Histogram of {target_variable}", nbins=30, marginal="box")
    st.plotly_chart(fig)

elif selected_plot == "Interactive Line Chart":
    if len(date_cols) > 0:
        df_sorted = df.sort_values(by=date_cols[0])
        fig = px.line(df_sorted, x=date_cols[0], y=target_variable, title=f"Interactive Line Chart of {target_variable}")
        st.plotly_chart(fig)
    else:
        st.warning("No date column found.")

elif selected_plot == "Interactive Heatmap":
    fig = px.imshow(df.corr(), text_auto=True, title="Interactive Correlation Heatmap")
    st.plotly_chart(fig)

elif selected_plot == "Interactive Box Plot":
    fig = px.box(df, y=target_variable, title=f"Interactive Box Plot of {target_variable}")
    st.plotly_chart(fig)

elif selected_plot == "Interactive Parallel Coordinates":
    top_features = correlation.index[1:6]  # Top 5 most correlated features
    fig = px.parallel_coordinates(df, dimensions=[target_variable] + list(top_features), title="Parallel Coordinates Plot")
    st.plotly_chart(fig)
