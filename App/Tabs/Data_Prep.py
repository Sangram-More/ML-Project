from statistics import correlation
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

# --------------- Data Analystics code --------------------------
df_uncleaned = pd.read_csv(r"Tabs/Datasets/Merged_Data.csv")
df = pd.read_csv(r"Tabs/Datasets/finaldataset.csv")

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

# ---------------------------------------------------------------
# Section 2
st.markdown("###")
st.markdown("<p class='justified-text'>The datasets obtained from the API are stored as separate CSV files, each containing specific financial or economic data. To facilitate analysis, these individual files need to be merged into a single comprehensive dataset. This consolidation process ensures that all relevant data points are organized in a structured format, enabling efficient processing and model training. A script is used to automate this merging process, combining multiple CSV files into one unified file for further analysis.</p>", unsafe_allow_html=True)

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
st.image(r"Tabs/Images/uncleanedinfo.png", caption="Uncleaned Data Info")
st.markdown("<p class='justified-text'>As we can see that the column GDP has a datatype of object. In order to convert this to a float datatype, pandas to_numeric() function is used. Thus after procesing the info section looks like this:</p>", unsafe_allow_html=True)
st.image(r"Tabs/Images/cleanedinfo.png", caption="Cleaned Data Info")

st.subheader("Step 3: Use of Pandas describe function to compute statistical measures.")
st.write(df.describe())

st.subheader("Here is the snapshot of Dataset after cleaning the data.")
st.write(df.iloc[89:99].head(10))

st.divider()

# ---------------------------------------------------------------
# Section 5


# Convert date columns to datetime
df['date'] = pd.to_datetime(df['date'], errors='coerce')

target_variable = "FEDRates"

# Streamlit App Title
st.title("Exploratory Data Analysis - US Fed Rates Prediction")
st.subheader("Explore different visualizations:")

# 1. Histogram
fig, ax = plt.subplots(facecolor='#ebf0ef')
sns.histplot(df[target_variable], bins=30, kde=True, ax=ax, color='#54bebe', edgecolor='black')
ax.set_title(f"Histogram of {target_variable}", fontsize=14, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, linestyle='--', alpha=0.5, color='gray')
st.pyplot(fig)
st.markdown("<p class='justified-text'>The histogram of FEDRates reveals a bimodal distribution, with two significant peaks around 0% and 5%, indicating that these interest rate levels have been the most frequent in the dataset. The distribution is right-skewed, showing that while most interest rates remain below 7.5%, there are instances where they have surged beyond 15%, though such occurrences are rare. The presence of a high concentration at 0% suggests periods of economic downturns where the Federal Reserve implemented near-zero interest rates, likely in response to recessions, such as the 2008 financial crisis and COVID-19 pandemic. Conversely, the second peak around 5% signifies phases of economic stability or inflation control where the Fed adopted a more balanced monetary policy. The KDE curve further reinforces these patterns, highlighting two primary clusters of FEDRates. Overall, the graph suggests that interest rate decisions are cyclical, influenced by macroeconomic conditions such as inflation, GDP growth, and unemployment rates.</p>", unsafe_allow_html=True)
st.markdown("###")

# 2. Boxplot
fig, ax = plt.subplots(facecolor='#ebf0ef')
sns.boxplot(y=df[target_variable], ax=ax, palette=['#54bebe', '#54bebe', '#54bebe', '#54bebe'])
ax.set_title(f"Boxplot of {target_variable}", fontsize=14, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, linestyle='--', alpha=0.5, color='gray')
st.pyplot(fig)
st.markdown("<p class='justified-text'>The boxplot of FEDRates provides insights into the distribution and variability of interest rates over time. The median FEDRate appears to be around 4-5%, with the interquartile range (IQR) spanning from approximately 2% to 7%, indicating that most interest rate values fall within this range. The whiskers extend up to around 13%, beyond which several outliers are observed, suggesting that extreme high-interest rates (above 13%) were recorded in certain periods. These outliers, reaching close to 20%, indicate instances of aggressive monetary policy tightening, likely in response to inflation surges, such as those in the 1970s and 1980s. The lower whisker extends to near 0%, reflecting periods of economic downturns when interest rates were kept minimal to boost economic activity. Overall, the boxplot suggests that while FEDRates have mostly remained moderate, there have been historical periods of extreme fluctuations driven by economic conditions.</p>", unsafe_allow_html=True)
st.markdown("###")

# 3. Correlation Heatmap
fig, ax = plt.subplots(figsize=(10, 6), facecolor='#ebf0ef')
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax, linewidths=0.8, linecolor='black', cbar=True, square=True, xticklabels=True, yticklabels=True)
ax.set_title("Correlation Heatmap", fontsize=14, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, linestyle='--', alpha=0.2, color='gray')
st.pyplot(fig)
st.markdown("<p class='justified-text'>The correlation heatmap provides valuable insights into the relationships between key economic indicators and FEDRates. The FEDRates show a strong positive correlation (0.72) with InflationConsumerPrice, indicating that higher inflation tends to be associated with increased interest rates, as the Federal Reserve raises rates to control inflationary pressures. Conversely, GDP (-0.45) and RealGDP (-0.41) have moderate negative correlations with FEDRates, suggesting that economic growth is typically linked to lower interest rates, likely due to expansionary monetary policies. Additionally, RealGDP, RealGDPPercapita, and RealPotentialGDP are highly correlated (~0.99), highlighting redundancy among these indicators. The Unemployment Rate (-0.02) shows almost no correlation with FEDRates, implying that interest rate adjustments alone do not significantly drive employment changes. Overall, the heatmap suggests that FEDRates are primarily influenced by inflation trends and economic growth metrics, reflecting the Federal Reserve’s balancing act between economic stability and inflation control.</p>", unsafe_allow_html=True)
st.markdown("###")

# 4. Scatter Plot vs Top Correlated Feature
correlations = df.corr()[target_variable].abs().sort_values(ascending=False)
top_feature = correlations.index[1]
fig, ax = plt.subplots(facecolor='#ebf0ef')
sns.scatterplot(x=df[top_feature], y=df[target_variable], ax=ax, color='#54bebe', alpha=0.7)
ax.set_title(f"Scatter Plot of {target_variable} vs {top_feature}", fontsize=14, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, linestyle='--', alpha=0.5, color='gray')
st.pyplot(fig)
st.markdown("<p class='justified-text'>The scatter plot of FEDRates vs InflationConsumerPrice highlights a positive correlation, suggesting that as inflation increases, the Federal Reserve tends to raise interest rates. The concentration of points between 0% and 6% inflation with FEDRates between 2% and 10% indicates that most historical interest rate decisions have been made within this range. However, as inflation rises above 8%, the FEDRates show a wider dispersion, implying that monetary policy responses to high inflation can vary significantly depending on broader economic conditions. The presence of clusters at lower inflation values (around 2-4%) suggests periods of relative economic stability, where the Federal Reserve maintained moderate rates. Overall, the plot reinforces that inflation plays a crucial role in interest rate decisions, with higher inflation often prompting the Fed to implement tighter monetary policies.</p>", unsafe_allow_html=True)
st.markdown("###")

# 5. Time Series Plot
fig, ax = plt.subplots(facecolor='#ebf0ef')
sns.lineplot(x=df['date'], y=df[target_variable], ax=ax, color='#54bebe', linewidth=2.5)
ax.set_title(f"Time Series Trend of {target_variable}", fontsize=14, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, linestyle='--', alpha=0.5, color='gray')
st.pyplot(fig)
st.markdown("<p class='justified-text'>The time series trend of FEDRates reveals key patterns in monetary policy decisions over decades. Interest rates were relatively low in the 1950s and 1960s, but saw a sharp increase in the 1970s and early 1980s, peaking close to 20%, likely in response to the high inflation crisis during that period. After the peak, the rates gradually declined, with fluctuations reflecting economic cycles and Federal Reserve interventions. The 2008 financial crisis led to an extended period of near-zero interest rates, a strategy aimed at stimulating economic recovery. Similarly, rates remained historically low post-2020, following the COVID-19 pandemic, before rising again due to inflationary concerns in 2022-2023. The trend suggests that FEDRates are highly cyclical, influenced by inflation, recession risks, and macroeconomic stability.</p>", unsafe_allow_html=True)
st.markdown("###")

# 6. Moving Average Plot
df['MA'] = df[target_variable].rolling(window=12).mean()
fig, ax = plt.subplots(facecolor='#ebf0ef')
sns.lineplot(x=df['date'], y=df['MA'], ax=ax, color='#54bebe', linewidth=2.5)
ax.set_title(f"12-Month Moving Average of {target_variable}", fontsize=14, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, linestyle='--', alpha=0.5, color='gray')
st.pyplot(fig)
st.markdown("<p class='justified-text'>The 12-month moving average of FEDRates smooths out short-term fluctuations, highlighting long-term trends in interest rate policies. The graph confirms a sharp rise in rates during the late 1970s and early 1980s, reaching a peak above 17%, likely due to aggressive Federal Reserve actions to curb inflation. Following this, there was a steady decline throughout the 1990s and early 2000s, reflecting a shift toward lower interest rate policies to support economic growth. The 2008 financial crisis led to near-zero rates, which remained low for an extended period until a gradual increase in the late 2010s. After another drop during the COVID-19 pandemic, rates have risen sharply again post-2022, indicating policy tightening in response to inflationary pressures. This moving average trend underscores the **cyclical nature of FEDRates, influenced by macroeconomic conditions and monetary policies.</p>", unsafe_allow_html=True)
st.markdown("###")

# 7. Trend Component
fig, ax = plt.subplots(facecolor='#ebf0ef')
decomposition = seasonal_decompose(df.set_index('date')[target_variable], model='additive', period=12)
decomposition.trend.plot(ax=ax, title="Trend", color='#54bebe')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, linestyle='--', alpha=0.5, color='gray')
st.pyplot(fig)
st.markdown("<p class='justified-text'>The trend component of FEDRates extracted from the time series decomposition highlights the long-term direction of interest rates, filtering out short-term fluctuations. The trend closely follows the historical pattern of monetary policy changes, showing a steady rise in rates during the 1960s and 1970s, peaking in the early 1980s at nearly 18% due to aggressive inflation control measures. This was followed by a gradual decline in the 1990s and early 2000s, reflecting an era of more stable economic policies and lower inflation rates. The sharp drop in rates during the 2008 financial crisis and the prolonged period of near-zero interest rates in the 2010s indicate expansionary monetary policies to stimulate economic recovery. More recently, rates have surged again post-2022, aligning with efforts to curb rising inflation. This trend confirms that interest rate policies are cyclically adjusted based on inflationary pressures, economic growth, and financial stability.</p>", unsafe_allow_html=True)
st.markdown("###")

# # 8. Seasonality Component
# fig, ax = plt.subplots(facecolor='#ebf0ef')
# decomposition.seasonal.plot(ax=ax, title="Seasonality", color='#54bebe')
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.grid(True, linestyle='--', alpha=0.5, color='gray')
# st.pyplot(fig)
# st.markdown("<p class='justified-text'></p>", unsafe_allow_html=True)

# 9. Residual Component
fig, ax = plt.subplots(facecolor='#ebf0ef')
decomposition.resid.plot(ax=ax, title="Residuals", color='#54bebe')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, linestyle='--', alpha=0.5, color='gray')
st.pyplot(fig)
st.markdown("<p class='justified-text'>The residuals plot from the time series decomposition represents the unexplained variations in FEDRates after accounting for trend and seasonality. For most of the timeline, the residuals fluctuate around zero, indicating that the model effectively captures the primary components of interest rate changes. However, the late 1970s and early 1980s exhibit significant volatility, with large deviations both above and below zero, reflecting high uncertainty and rapid fluctuations in monetary policy due to extreme inflation during that period. After the 1990s, the residuals stabilize, suggesting that interest rate changes became more predictable and policy-driven. The post-2008 financial crisis and 2020 COVID-19 period show minor fluctuations, indicating that monetary policy adjustments were more controlled despite economic shocks. Overall, this plot highlights that while major historical events have led to significant deviations in FEDRates, most variations remain within a stable range in normal economic conditions.</p>", unsafe_allow_html=True)
st.markdown("###")

# 10. Unemployment Rate vs FEDRates Scatter Plot
fig, ax = plt.subplots(facecolor='#ebf0ef')
sns.scatterplot(x=df['UnemployemenrRate'], y=df[target_variable], ax=ax, color='#54bebe', alpha=0.7)
ax.set_title("FEDRates vs Unemployment Rate", fontsize=14, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, linestyle='--', alpha=0.5, color='gray')
st.pyplot(fig)
st.markdown("<p class='justified-text'>The scatter plot of FEDRates vs. Unemployment Rate reveals an unclear or weak correlation between the two variables. The data points are widely scattered, suggesting that changes in interest rates do not have a direct or consistent impact on unemployment levels. While some higher interest rates (above 10%) appear to coincide with higher unemployment levels (above 8%), this pattern is not strong enough to indicate a definitive relationship. The concentration of points between 4% and 7% unemployment, with FEDRates mostly between 2% and 8%, suggests that most historical economic conditions operated within this range. The absence of a clear trend indicates that other factors, such as fiscal policies, inflation, and economic growth, likely play a more significant role in determining unemployment levels than interest rate changes alone.</p>", unsafe_allow_html=True)
st.markdown("###")

# 11. KDE Plot
fig, ax = plt.subplots(facecolor='#ebf0ef')
sns.kdeplot(df[target_variable], ax=ax, fill=True, color='#54bebe')
ax.set_title(f"KDE Plot of {target_variable}", fontsize=14, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, linestyle='--', alpha=0.5, color='gray')
st.pyplot(fig)
st.markdown("<p class='justified-text'>The KDE (Kernel Density Estimate) plot of FEDRates provides a smooth approximation of the probability distribution of interest rates. The distribution is right-skewed, indicating that lower interest rates have been more common historically, with most values concentrated between 0% and 7%. The peak density occurs around 4-5%, suggesting that the Federal Reserve has frequently maintained rates within this range. The distribution gradually declines beyond 10%, with a long tail extending up to 20%, reflecting historical periods of aggressive monetary tightening, such as the 1980s inflation crisis. The smooth nature of the KDE curve highlights the overall trend, confirming that higher interest rates are rare occurrences, while moderate to low rates have been the norm in most economic conditions.</p>", unsafe_allow_html=True)
st.markdown("###")

# ---------------------------------------------------------------

