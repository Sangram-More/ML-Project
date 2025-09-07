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
# df_uncleaned = pd.read_csv(r"Tabs/Datasets/Merged_Data.csv")
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
# Implementing animation function
def animation_file(filepath: str):
    with open(filepath, "rb") as f:
        return json.load(f)

def PCA_Animation():   
    animation = st_lottie(
        animation_file(r"Tabs/Animations/PCA.json"),
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
    file_path = r"Tabs/Animations/pca_original.gif"
    st.image(file_path)

with column2_1:
    file_path = r"Tabs/Animations/pca_transformed.gif"
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
st.markdown("### After Performing PCA on our Cleand Dataset, here is what we get to see.")
# st.markdown("<p class='justified-text'> PCA Explained Varience</p>", unsafe_allow_html=True)
file_path = r"Tabs/Images/PCA_Explained_Varience.png"
st.image(file_path)
st.markdown("<p class='justified-text'>The graph displays the Cumulative Explained Variance by Principal Component Analysis (PCA) components. It shows that the first three components capture nearly 95% of the total variance in the data, indicating that these components retain most of the useful information. Adding more components contributes only marginally to the explained variance, suggesting diminishing returns after the third component. This means that a dimensionality reduction to 3 components would effectively compress the data with minimal information loss. Such behavior is common when certain features are highly correlated, and PCA successfully captures these correlations early on. Overall, this scree plot helps determine the optimal number of components to retain for efficient modeling and visualization.</p>", unsafe_allow_html=True)

st.divider()

# -----------------------------------------------------------------------------
# Section 4

file_path = r"Tabs/Images/PCA_Projection.png"
st.image(file_path)
st.markdown("<p class='justified-text'>This graph presents the PCA projection onto the first two principal components, providing a 2D visualization of the dataset after dimensionality reduction. Each point represents an observation from the dataset, colored by its corresponding FEDRates value. The color gradient from blue to red helps us see how different ranges of FEDRates align within the principal component space. Observations with higher FEDRates are clustered towards the left side, while those with lower FEDRates spread across the middle and right side, indicating that the first two principal components capture meaningful structure related to interest rate variations. This visualization highlights that FEDRates are strongly associated with some underlying features captured by these components, supporting the effectiveness of PCA in uncovering dominant patterns in the data. Overall, this plot allows for both dimensionality reduction and interpretability in the context of economic factors influencing FEDRates.</p>", unsafe_allow_html=True)

st.divider()

# -----------------------------------------------------------------------------
# Section 5

file_path = r"Tabs/Images/PCA_Scree_Plot.png"
st.image(file_path)
st.markdown("<p class='justified-text'>The scree plot shown highlights the explained variance percentage for each principal component derived from the dataset. The first principal component (PC1) captures approximately 55.80% of the total variance, indicating that this single component holds a significant amount of the dataset's information. The second component (PC2) captures an additional 20.20%, and the third component (PC3) adds 13.78%. Together, the first three components explain over 89% of the variance, demonstrating that these three dimensions represent most of the patterns in the data. The remaining components contribute only minimal variance, with the sixth, seventh, and eighth components explaining close to zero. This plot confirms that a dimensionality reduction to 3 components would effectively preserve the majority of the dataset’s meaningful information.</p>", unsafe_allow_html=True)

st.divider()

# -----------------------------------------------------------------------------
# Section 6

file_path = r"Tabs/Images/PCA_Loading_Plot.png"
st.image(file_path)
st.markdown("<p class='justified-text'>The loading plot visualizes how each original feature contributes to the first few principal components derived through PCA. Each bar segment shows the magnitude and direction (positive or negative) of a feature’s loading on a particular component. For example, UnemploymentRate has a strong positive loading on the third principal component (PC3), indicating that this feature significantly influences that component. Similarly, features like GDP and InflationConsumerPrice contribute heavily to the first two components, implying that these economic indicators capture much of the variance in the data. The mixed colors in each bar highlight how different components rely on different combinations of features. This visualization helps explain how PCA reshapes the dataset into components that capture distinct sources of variation from the original economic factors.</p>", unsafe_allow_html=True)

st.divider()

# -----------------------------------------------------------------------------
# Section 6

st.markdown("### Here is a 3D Animation of 3 components of PCA before and after applying PCA to our dataset.")

# Creating 2 columns to add animations side by side.

column1_1, column2_1 = st.columns(2, gap="large", vertical_alignment="center")

with column1_1:
    file_path = r"Tabs/Animations/PCA_Original_Dataset.gif"
    st.image(file_path)

with column2_1:
    file_path = r"Tabs/Animations/PCA_Transformed_Dataset.gif"
    st.image(file_path)

st.divider()

# -----------------------------------------------------------------------------
# Section 6

st.subheader("Information Retained in 2D Dataset (First 2 PCs)")
st.write("The first two principal components retain **76.00%** of the total variance in the dataset. This means reducing the dataset to two dimensions preserves 76% of the original information.")

st.subheader("Information Retained in 3D Dataset (First 3 PCs)")
st.write("The first three principal components retain **89.78%** of the total variance. This 3D projection preserves nearly 90% of the original information, making it a good choice for visualization and modeling.")

st.subheader("Number of Components Needed to Retain 95% of Variance")
st.write("To retain at least **95%** of the variance, you need the first **4 principal components**, as they together capture approximately **95.61%** of the variance.")

st.subheader("Top 3 Eigenvalues")
st.write("""
- **Eigenvalue 1:** 4.4694  
- **Eigenvalue 2:** 1.6179  
- **Eigenvalue 3:** 1.1036  
""")

file_path = r"Tabs/Images/PCA_Values.png"
st.image(file_path)

st.divider()
# -----------------------------------------------------------------------------

st.markdown("Click here to download code for PCA Analysis [link](%s)" % "https://github.com/Sangram-More/ML-Project/blob/master/Jupyter%20Lab%20Analysis/PCA.ipynb")

# -----------------------------------------------------------------------------