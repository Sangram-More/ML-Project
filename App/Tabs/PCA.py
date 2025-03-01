import streamlit as st
from streamlit_lottie import st_lottie
import json
import os
from statistics import correlation
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# --------------- Data Analystics code --------------------------
# df_uncleaned = pd.read_csv(r"App/Tabs/Datasets/Merged_Data.csv")
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
# Implementing animation function
def animation_file(filepath: str):
    with open(filepath, "rb") as f:
        return json.load(f)

def PCA_Animation():   
    animation = st_lottie(
        animation_file(r"App/Tabs/Animations/PCA.json"),
        speed = 1,
        reverse = False,
        loop = True,
        quality = "high"
    )
    return animation

# -----------------------------------------------------------------------------
# st.markdown("<p class='justified-text'></p>", unsafe_allow_html=True)

st.title("Principal Component Analysis")
st.header("So what exactly is Principal Component Analysis (PCA)")

# -----------------------------------------------------------------------------
# Section 1

st.markdown("<p class='justified-text'>Principal Component Analysis (PCA) is a dimensionality reduction technique used to simplify large datasets while preserving as much important information as possible. It works by transforming correlated variables into a smaller set of uncorrelated variables called principal components. These components are ordered by how much variance (or information) they capture from the original data, meaning the first principal component holds the most information, the second holds slightly less, and so on. By keeping only the top few components, PCA helps in reducing complexity, improving computation speed, and even removing noise from data. It is widely used in data visualization, machine learning, and image processing when dealing with high-dimensional datasets. However, since PCA transforms the data into new axes, the new features lose their original meaning, making interpretation challenging. Despite this, PCA remains a powerful tool for simplifying data while retaining its essential patterns.</p>", unsafe_allow_html=True)

st.markdown("### Here is a demonstration of how data changes before and after application of PCA on random dataset.")

# Creating 2 columns to add animations side by side.

column1_1, column2_1 = st.columns(2, gap="large", vertical_alignment="center")

with column1_1:
    file_path = r"App/Tabs/Animations/pca_original.gif"
    st.image(file_path)

with column2_1:
    file_path = r"App/Tabs/Animations/pca_transformed.gif"
    st.image(file_path)

st.divider()

# -----------------------------------------------------------------------------
# Section 2

st.markdown("### Here is our dataset of Fed Rates with different features that we will be using to perform PCA on.")
st.write(df.iloc[89:99].head(10))
st.markdown("Click here to download cleaned dataset [link](%s)" % "https://github.com/Sangram-More/ML-Project/blob/master/Jupyter%20Lab%20Analysis/Cleaned/finaldataset.csv")

st.divider()

# -----------------------------------------------------------------------------
# Section 3

